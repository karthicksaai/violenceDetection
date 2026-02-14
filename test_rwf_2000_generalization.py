import torch
import cv2
import numpy as np
import sys
import os
import random
from PIL import Image

# Add project root to path
sys.path.append(os.getcwd())

from orchestration.violence_detector import HockeyGRU, FeatureExtractor

def evaluate_rwf():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        
    print(f"🧠 Evaluating on RWF-2000 using {device}...")
    
    # 1. Load Model
    model = HockeyGRU(input_dim=512, hidden_dim=256).to(device)
    try:
        model.load_state_dict(torch.load("violence_detection_model.pth", map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    extractor = FeatureExtractor(device=device)
    
    # 2. Get Test List from Directory Structure
    rwf_root = "RWF-2000"
    # Try multiple common paths for RWF
    possible_roots = [
        os.path.join(rwf_root, "val"),
        os.path.join(rwf_root, "test"),
        "RWF-2000/val" 
    ]
    
    fight_dir = None
    nonfight_dir = None
    
    for r in possible_roots:
        f = os.path.join(r, "Fight")
        n = os.path.join(r, "NonFight")
        if os.path.exists(f) and os.path.exists(n):
            fight_dir = f
            nonfight_dir = n
            break
            
    if not fight_dir:
        print(f"❌ RWF-2000 test directories not found in {rwf_root}")
        return

    # 3. Sample Videos
    # RWF typically has .npy files? If so, we need to load them.
    # But user previous context implied video files or we used .npy loader.
    # Let's check if there are videos.
    fight_videos = [os.path.join(fight_dir, f) for f in os.listdir(fight_dir) if f.endswith(('.avi', '.mp4', '.npy'))]
    nonfight_videos = [os.path.join(nonfight_dir, f) for f in os.listdir(nonfight_dir) if f.endswith(('.avi', '.mp4', '.npy'))]
    
    if len(fight_videos) < 5 or len(nonfight_videos) < 5:
        print("⚠️  Not enough videos found in RWF-2000 to test properly.")
        return

    random.seed(42)
    test_fight = random.sample(fight_videos, 5)
    test_nonfight = random.sample(nonfight_videos, 5)
    
    print("\n🧐 Evaluation on RWF-2000 (Unseen Dataset)...")
    print("-" * 65)
    print(f"{'Category':<10} | {'Video Name':<25} | {'Pred':<10} | {'Conf':<6} | {'Result':<10}")
    print("-" * 65)
    
    # Helper for inference
    def predict(video_path):
        frames_buffer = []
        
        # Handle .npy vs Video
        if video_path.endswith('.npy'):
            # Load npy (T, H, W, 3) or (T, C, H, W)?
            # RWF numpy files are usually (30, 224, 224, 3) RGB
            data = np.load(video_path) 
            # Check shape
            # We need to convert to PIL images for Extractor
            for i in range(len(data)):
                frame = data[i]
                # If float, convert to uint8?
                if frame.dtype != np.uint8:
                     frame = (frame * 255).astype(np.uint8)
                frames_buffer.append(Image.fromarray(frame))
        else:
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (128, 256))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_buffer.append(Image.fromarray(frame))
                if len(frames_buffer) > 200: break 
            cap.release()
        
        if not frames_buffer: return 0.0
        
        # Batch extract
        # Process in chunks if too large
        batch_size = 32
        feats_list = []
        
        for i in range(0, len(frames_buffer), batch_size):
            chunk = frames_buffer[i:i+batch_size]
            batch = torch.stack([extractor.preprocess(img) for img in chunk]).to(device)
            with torch.no_grad():
                f = extractor.model(batch)
                f = torch.nn.functional.normalize(f, p=2, dim=1)
                feats_list.append(f)
                
        full_feats = torch.cat(feats_list, dim=0)
        
        # Inference
        seq = full_feats.unsqueeze(0) # (1, T, 512)
        with torch.no_grad():
            logits = model(seq)
            probs = torch.softmax(logits, dim=1)
            
        return probs[0][1].item()

    # Test Fight
    tp = 0
    fn = 0
    for v in test_fight:
        try:
            conf = predict(v)
            pred = "Violence" if conf > 0.5 else "Non-Viol"
            res = "✅" if pred == "Violence" else "❌"
            if pred == "Violence": tp += 1
            else: fn += 1
            print(f"{'Fight':<10} | {os.path.basename(v)[:25]:<25} | {pred:<10} | {conf:.2f}   | {res}")
        except Exception as e:
            print(f"Error {v}: {e}")

    # Test NonFight
    tn = 0
    fp = 0
    for v in test_nonfight:
        try:
            conf = predict(v)
            pred = "Violence" if conf > 0.5 else "Non-Viol"
            res = "✅" if pred == "Non-Viol" else "❌"
            if pred == "Non-Viol": tn += 1
            else: fp += 1
            print(f"{'NonFight':<10} | {os.path.basename(v)[:25]:<25} | {pred:<10} | {conf:.2f}   | {res}")
        except Exception as e:
             print(f"Error {v}: {e}")

    acc = (tp + tn) / 10
    print("-" * 65)
    print(f"✅ Overall Accuracy: {acc:.0%}")

if __name__ == "__main__":
    evaluate_rwf()
