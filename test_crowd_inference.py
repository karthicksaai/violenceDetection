import torch
import cv2
import numpy as np
import sys
import os
import argparse
from PIL import Image

# Add project root
sys.path.append(os.getcwd())

from orchestration.violence_detector import HockeyGRU, FeatureExtractor

def predict_crowd_anomaly(video_path, model_path="crowd_anomaly_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        
    print(f"🧠 Loading Crowd Model on {device}...")
    
    # Load Model
    model = HockeyGRU(input_dim=512, hidden_dim=256, num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Load Extractor
    extractor = FeatureExtractor(device=device)
    
    print(f"🎥 Processing Video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video.")
        return
        
    buffer = []
    violence_probs = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Resize for speed
        frame = cv2.resize(frame, (128, 256))
        
        # Extract
        # Note: FeatureExtractor usually expects (256, 128) but handles resize internally
        # We invoke extractor manually per frame for simplicity vs batch
        feat = extractor.extract(frame)
        buffer.append(torch.tensor(feat, dtype=torch.float32))
        
        if len(buffer) == 20: # Sliding Window = 20
            # Inference
            seq = torch.stack(buffer).unsqueeze(0).to(device) # (1, 20, 512)
            with torch.no_grad():
                logits = model(seq)
                probs = torch.softmax(logits, dim=1)
                p = probs[0][1].item()
                violence_probs.append(p)
                
            # Slide
            buffer.pop(0)
            
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...", end='\r')
            
    cap.release()
    print("\n✅ Processing Complete.")
    
    if not violence_probs:
        print("Video too short (less than 20 frames).")
        return
        
    max_prob = max(violence_probs)
    avg_prob = sum(violence_probs) / len(violence_probs)
    
    print(f"📊 Results for {os.path.basename(video_path)}:")
    print(f"Max Anomaly Score: {max_prob:.4f}")
    print(f"Avg Anomaly Score: {avg_prob:.4f}")
    
    if max_prob > 0.85:
        print("🚨 ANOMALY DETECTED! (Stampede/Pushing)")
    else:
        print("✅ Crowd Behavior Normal")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("--model", type=str, default="crowd_anomaly_model.pth", help="Path to model weights")
    args = parser.parse_args()
    
    predict_crowd_anomaly(args.video_path, args.model)
