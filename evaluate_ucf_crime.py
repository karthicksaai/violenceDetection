import torch
import cv2
import numpy as np
import sys
import os
import random
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

# Add project root to path
sys.path.append(os.getcwd())

from orchestration.violence_detector import HockeyGRU, FeatureExtractor

# Config
MODEL_PATH = "violence_detection_model.pth"
UCF_ROOT = "UFC-Crime/Anomaly_Dataset/Anomaly_Videos"
NORMAL_DIR = os.path.join(UCF_ROOT, "Normal-Videos-Part-1")
ASSAULT_DIR = os.path.join(UCF_ROOT, "Anomaly-Videos-Part-1", "Assault")

SAMPLE_SIZE = 10 # Reduced sample size for speed (Total 20 videos)
MAX_FRAMES_PER_VIDEO = 2000 # ~1 minute of video
BATCH_SIZE = 64 # Extraction batch size
WINDOW_SIZE = 20
STRIDE = 10

def process_video_batch(video_path, extractor, model, device):
    """
    1. Extract features for ALL frames in batches.
    2. Run sliding window GRU on the extracted feature sequence.
    """
    cap = cv2.VideoCapture(video_path)
    frames_buffer = []
    features_list = []
    
    frame_count = 0
    
    # --- PHASE 1: BATCH FEATURE EXTRACTION ---
    while frame_count < MAX_FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret: break
        
        # Preprocessing for OSNet (matches FeatureExtractor logic)
        # Convert BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_buffer.append(Image.fromarray(frame))
        
        # When buffer full, process batch
        if len(frames_buffer) == BATCH_SIZE:
            # Stack tensors
            batch_tensors = torch.stack([extractor.preprocess(img) for img in frames_buffer]).to(device)
            
            with torch.no_grad():
                # OSNet Inference
                embeddings = extractor.model(batch_tensors)
                # L2 Normalize (Critical for ReID features!)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
            features_list.append(embeddings.cpu()) # Move to CPU to save GPU memory
            frames_buffer = []
            
        frame_count += 1
        
    # Process remaining frames
    if frames_buffer:
        batch_tensors = torch.stack([extractor.preprocess(img) for img in frames_buffer]).to(device)
        with torch.no_grad():
            embeddings = extractor.model(batch_tensors)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        features_list.append(embeddings.cpu())
        
    cap.release()
    
    if not features_list: return 0.0
    
    # Concatenate all features -> (T, 512)
    all_features = torch.cat(features_list, dim=0)
    T = all_features.shape[0]
    
    if T < WINDOW_SIZE: return 0.0
    
    # --- PHASE 2: SLIDING WINDOW INFERENCE ---
    max_prob = 0.0
    
    # Create windows
    # We can create a batch of windows for GRU too!
    windows = []
    for i in range(0, T - WINDOW_SIZE, STRIDE):
        window = all_features[i : i+WINDOW_SIZE] # (20, 512)
        windows.append(window)
        
    if not windows: return 0.0
    
    # Process windows in batches (e.g., batch of 32 sequences)
    # GRU input: (Batch, 20, 512)
    
    dataset = torch.stack(windows) # (NumWindows, 20, 512)
    
    # Check if too big for GPU
    n_windows = dataset.shape[0]
    gru_batch_size = 32
    
    with torch.no_grad():
        for i in range(0, n_windows, gru_batch_size):
            batch = dataset[i : i+gru_batch_size].to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            
            # Get max violence prob in this batch
            batch_max = probs[:, 1].max().item()
            if batch_max > max_prob:
                max_prob = batch_max
                
            # Early exit if confirmed violence? (Optional)
            # if max_prob > 0.99: break 
            
    return max_prob

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        
    print(f"🧠 Evaluating using {device} (Batch Processing)...")
    
    # Load Model
    model = HockeyGRU(input_dim=512, hidden_dim=256).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"✅ Loaded Model: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Load Extractor
    extractor = FeatureExtractor(device=device)
    
    # 1. Get Samples
    normal_files = [os.path.join(NORMAL_DIR, f) for f in os.listdir(NORMAL_DIR) if f.endswith(".mp4")]
    assault_files = [os.path.join(ASSAULT_DIR, f) for f in os.listdir(ASSAULT_DIR) if f.endswith(".mp4")]
    
    # Ensure consistent random sampling
    random.seed(55) # New seed for variety
    sample_normal = random.sample(normal_files, min(len(normal_files), SAMPLE_SIZE))
    sample_assault = random.sample(assault_files, min(len(assault_files), SAMPLE_SIZE))
    
    print(f"📊 Sampling: {len(sample_normal)} Normal + {len(sample_assault)} Assault")
    print(f"⚙️  Settings: MaxFrames={MAX_FRAMES_PER_VIDEO}, BatchSize={BATCH_SIZE}, Stride={STRIDE}\n")
    
    y_true = []
    y_pred = []
    results = []
    
    print(f"{'Type':<10} | {'Video Name':<30} | {'Pred':<10} | {'Max Conf':<8} | {'Status'}")
    print("-" * 80)
    
    # Evaluate Normal
    for i, f in enumerate(sample_normal):
        try:
            max_prob = process_video_batch(f, extractor, model, device)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            max_prob = 0.0
            
        pred = 1 if max_prob > 0.85 else 0 
        
        y_true.append(0)
        y_pred.append(pred)
        
        status = "✅" if pred == 0 else "❌"
        fname = os.path.basename(f)
        results.append(f"Normal,{fname},{pred},{max_prob:.2f},{status}")
        print(f"[{i+1}/{SAMPLE_SIZE}] {'Normal':<10} | {fname[:30]:<30} | {'Viol' if pred else 'Norm':<10} | {max_prob:.2f}     | {status}")

    # Evaluate Assault
    for i, f in enumerate(sample_assault):
        try:
            max_prob = process_video_batch(f, extractor, model, device)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            max_prob = 0.0
            
        pred = 1 if max_prob > 0.85 else 0
        
        y_true.append(1)
        y_pred.append(pred)
        
        status = "✅" if pred == 1 else "❌"
        fname = os.path.basename(f)
        results.append(f"Assault,{fname},{pred},{max_prob:.2f},{status}")
        print(f"[{i+1}/{SAMPLE_SIZE}] {'Assault':<10} | {fname[:30]:<30} | {'Viol' if pred else 'Norm':<10} | {max_prob:.2f}     | {status}")
        
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("\n" + "="*40)
    print("📈 EVALUATION RESULTS (UCF-Crime) - Sliding Window")
    print("="*40)
    print(f"Overall Accuracy: {acc:.2%}")
    print(f"Sensitivity (Recall on Violence): {sensitivity:.2%}")
    print(f"Specificity (Recall on Normal):   {specificity:.2%}")
    print(f"Confusion Matrix: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")
    
    # Save to CSV
    with open("ucf_evaluation_results_full.csv", "w") as f:
        f.write("Type,Video,Prediction,Max_Confidence,Correctness\n")
        f.write("\n".join(results))
    print("\n📄 Detailed results saved to 'ucf_evaluation_results_full.csv'")

if __name__ == "__main__":
    evaluate()
