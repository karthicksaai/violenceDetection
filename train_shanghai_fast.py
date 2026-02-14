import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix
import sys
import os

# Add project root
sys.path.append(os.getcwd())

from orchestration.violence_detector import HockeyGRU

# Config
FEATURE_FILE = "shanghai_features_list.pt"
LABEL_FILE = "shanghai_labels.pt"
MODEL_SAVE_PATH = "crowd_anomaly_model.pth"
BATCH_SIZE = 1 # Variable length sequences, using batch 1 avoids padding

class ListDataset(Dataset):
    def __init__(self, feature_list, labels):
        self.features = feature_list
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # features[idx] is (T, 512)
        # We need to return it. DataLoader with batch_size=1 will wrap it in (1, T, 512)
        return self.features[idx], self.labels[idx]

def train_shanghai():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        
    print(f"🚀 Starting Training on {device}...")
    
    # 1. Load Data
    if not os.path.exists(FEATURE_FILE) or not os.path.exists(LABEL_FILE):
        print("❌ Feature files not found. Run extract_shanghai_features.py first.")
        return

    print("Loading features...")
    X_list = torch.load(FEATURE_FILE) # List of tensors
    y = torch.load(LABEL_FILE)        # Tensor (N,)
    
    # Check balance
    n_abnormal = y.sum().item()
    n_normal = len(y) - n_abnormal
    print(f"Data Distribution: {n_normal} Normal, {n_abnormal} Abnormal")
    
    dataset = ListDataset(X_list, y)
    
    # Split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Loaders (Batch Size 1 to handle variable lengths)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Model
    model = HockeyGRU(input_dim=512, hidden_dim=256, num_classes=2).to(device)
    
    # Class weights if needed
    # We use a VERY high weight for Anomaly to force the model to try predicting it.
    weights = torch.tensor([1.0, 15.0]).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Slightly higher LR
    
    epochs = 50 
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        train_preds = []
        train_labels = []
        
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Sequence Cropping for Training
            T = X_batch.shape[1]
            if T > 300:
                start = torch.randint(0, T - 300, (1,)).item()
                X_batch = X_batch[:, start:start+300, :]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            train_preds.append(predicted.item())
            train_labels.append(y_batch.item())
            
            if (i + 1) % 50 == 0:
                print(f"  Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Train Acc: {train_acc:.2f}%")
        print("Train CM:\n", confusion_matrix(train_labels, train_preds))
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                val_preds.append(predicted.item())
                val_labels.append(y_batch.item())
        
        val_acc = 100 * correct / total
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        print("Val CM:\n", confusion_matrix(val_labels, val_preds))
        
        # Save Best
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

            
    print(f"✅ Training Complete. Best Val Acc: {best_acc:.2f}%")
    print(f"💾 Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_shanghai()
