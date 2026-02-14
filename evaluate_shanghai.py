import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
import numpy as np

# Add project root
sys.path.append(os.getcwd())

from orchestration.violence_detector import HockeyGRU
from train_shanghai_fast import ListDataset

# Config
FEATURE_FILE = "shanghai_features_list.pt"
LABEL_FILE = "shanghai_labels.pt"
MODEL_PATH = "crowd_anomaly_model.pth"

def evaluate_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    
    print(f"🚀 Evaluating Crowd Model on {device}...")
    
    if not os.path.exists(FEATURE_FILE) or not os.path.exists(LABEL_FILE):
        print("❌ Feature files not found.")
        return
        
    if not os.path.exists(MODEL_PATH):
        print("❌ Model weights not found.")
        return

    # Load Data
    X_list = torch.load(FEATURE_FILE)
    y = torch.load(LABEL_FILE)
    dataset = ListDataset(X_list, y)
    
    # Use the same split logic as training (80/20)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_indices = list(range(train_size, total_size))
    val_dataset = Subset(dataset, val_indices)
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Load Model
    model = HockeyGRU(input_dim=512, hidden_dim=256, num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print(f"Testing on {len(val_dataset)} validation samples...")
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.item())
            all_labels.append(y_batch.item())
            
    # Metrics
    print("\n📊 Evaluation Results:")
    print("-" * 30)
    print(classification_report(all_labels, all_preds, target_names=["Normal", "Anomaly"]))
    
    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    # Visual check on a few samples
    print("\n🔍 Random Samples Check:")
    indices = np.random.choice(len(all_labels), min(10, len(all_labels)), replace=False)
    for idx in indices:
        true = "Anomaly" if all_labels[idx] == 1 else "Normal"
        pred = "Anomaly" if all_preds[idx] == 1 else "Normal"
        res = "✅" if true == pred else "❌"
        print(f"Sample {idx}: True={true:<8} | Pred={pred:<8} | {res}")

if __name__ == "__main__":
    evaluate_model()
