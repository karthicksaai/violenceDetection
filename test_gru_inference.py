
import torch
import cv2
import numpy as np
import sys
import os
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import random

# Import from training script
sys.path.append(os.getcwd())
try:
    from train_hockey_gru import HockeyGRU, FeatureExtractor, extract_cnn_features
except ImportError:
    # Use fallback if running standalone and imports fail due to relative paths
    pass

def predict_single_video(video_path, model_path="violence_detection_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Loading model from {model_path}...")
    model = HockeyGRU(input_dim=512, hidden_dim=256).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    extractor = FeatureExtractor(device=device)
    
    print(f"\nPredicting for video: {os.path.basename(video_path)}")
    
    # Feature Extraction
    features = extract_cnn_features(video_path, extractor, device, max_frames=20)
    x = features.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_label = "Violence" if pred_idx == 1 else "Normal"
        conf = probs[0][pred_idx].item()
        
    print(f"Prediction: {pred_label} (Confidence: {conf:.2f})")
    return pred_label, conf

def predict_batch(model_path="violence_detection_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Loading model from {model_path}...")
    model = HockeyGRU(input_dim=512, hidden_dim=256).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    extractor = FeatureExtractor(device=device)
    # extractor.eval() is not needed as it's done in init
    
    # Define test set
    dataset_root = "HockeyDataset"
    fights = [os.path.join(dataset_root, "fights", f) for f in os.listdir(os.path.join(dataset_root, "fights")) if f.endswith(".avi")]
    nofights = [os.path.join(dataset_root, "nofights", f) for f in os.listdir(os.path.join(dataset_root, "nofights")) if f.endswith(".avi")]
    
    # Pick 10 random from each
    random.seed(42)
    test_fights = random.sample(fights, 10)
    test_nofights = random.sample(nofights, 10)
    
    print(f"\nEvaluating on 10 Fights and 10 NoFights...")
    print("-" * 60)
    print(f"{'File':<20} | {'True Label':<10} | {'Pred':<10} | {'Conf':<6} | {'Result':<10}")
    print("-" * 60)
    
    metrics = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    
    # Helper
    def eval_video(path, true_label_str):
        # Feature Extraction
        features = extract_cnn_features(path, extractor, device, max_frames=20)
        x = features.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = "Violence" if pred_idx == 1 else "Normal"
            conf = probs[0][pred_idx].item()
            
        is_correct = (pred_label == true_label_str)
        result = "CORRECT" if is_correct else "WRONG"
        
        # Color code result
        # Note: ANSI codes might not render in all artifact viewers but are good for terminal
        
        fname = os.path.basename(path)
        print(f"{fname:<20} | {true_label_str:<10} | {pred_label:<10} | {conf:.2f}   | {result}")
        
        if true_label_str == "Violence":
            if pred_label == "Violence": metrics["TP"] += 1
            else: metrics["FN"] += 1
        else:
            if pred_label == "Normal": metrics["TN"] += 1
            else: metrics["FP"] += 1

    for f in test_fights:
        eval_video(f, "Violence")
        
    for f in test_nofights:
        eval_video(f, "Normal")
        
    print("-" * 60)
    print("Summary:")
    print(f"True Positives (Detected Fights): {metrics['TP']}/10")
    print(f"False Negatives (Missed Fights):  {metrics['FN']}/10  <-- CRITICAL FOR SAFETY")
    print(f"True Negatives (Correct Normal):  {metrics['TN']}/10")
    print(f"False Positives (False Alarms):   {metrics['FP']}/10")
    
    recall = metrics['TP'] / (metrics['TP'] + metrics['FN']) if (metrics['TP'] + metrics['FN']) > 0 else 0
    print(f"\nRecall (Sensitivity): {recall:.2%}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_single_video(sys.argv[1])
    else:
        predict_batch()
