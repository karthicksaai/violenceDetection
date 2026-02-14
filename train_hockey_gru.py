
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
import argparse
import sys

# Import robust ReID Feature Extractor
try:
    from orchestration.reid.feature_extractor import FeatureExtractor
except ImportError:
    # Fallback if running from script root without package context
    sys.path.append(os.getcwd())
    from orchestration.reid.feature_extractor import FeatureExtractor


# --- Configuration ---
LABEL_MAP = {
    "nofights": 0,
    "fights": 1
}
# Hyperparameters
MAX_FRAMES = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
HIDDEN_DIM = 256 # Increased from 128 for 2048 input
NUM_LAYERS = 2
INPUT_DIM = 512 # ResNet50 Feature Dim

# Transformations (Standard ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_cnn_features(video_path, extractor, device, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames_list = []
    frames_read = 0
    
    while frames_read < max_frames:
        ret, frame = cap.read()
        if not ret: break
        
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        
        # Preprocess using Extractor's transform (CPU)
        try:
            input_tensor = extractor.preprocess(pil_img)
            frames_list.append(input_tensor)
        except Exception as e:
            print(f"Error prepping frame: {e}")
            break
            
        frames_read += 1
        
    cap.release()

    if len(frames_list) == 0:
        return torch.zeros((max_frames, INPUT_DIM))

    # Padding
    while len(frames_list) < max_frames:
        frames_list.append(frames_list[-1]) # Repeat last
        
    # Stack -> Batch (T, 3, H, W)
    batch = torch.stack(frames_list[:max_frames]).to(device)
    
    # Inference (GPU - Batched)
    with torch.no_grad():
        # extractor.model expects (B, 3, H, W)
        embeddings = extractor.model(batch)
        
    # Flatten
    features = embeddings.flatten(start_dim=1) # (T, 2048)
    
    # L2 Normalize (Essential for ReID)
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    
    return features.cpu()

# --- Dataset ---
class HockeyDataset(Dataset):
    def __init__(self, root_dir, device, max_frames=20):
        self.samples = []
        self.max_frames = max_frames
        self.root_dir = root_dir
        self.device = device
    # Initialize extractor once per dataset
        # FeatureExtractor takes device in constructor, not .to()
        try:
            self.extractor = FeatureExtractor(device=device)
        except TypeError:
             # Fallback if I got the wrong class or signature
            self.extractor = FeatureExtractor()


        print(f"Scanning {root_dir}...")
        for label_name, label_id in LABEL_MAP.items():
            folder = os.path.join(root_dir, label_name)
            if not os.path.exists(folder):
                continue
                
            count = 0
            for file in os.listdir(folder):
                if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    self.samples.append((os.path.join(folder, file), label_id))
                    count += 1
            print(f"  Found {count} samples for {label_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # On-the-fly extraction (Slow but simple)
        # For production, pre-compute these to .pt files!
        features = extract_cnn_features(path, self.extractor, self.device, self.max_frames)
        return features, label

class UCFCrimeDataset(Dataset):
    def __init__(self, root_dir, device, max_frames=20):
        self.samples = []
        self.max_frames = max_frames
        self.device = device
        try:
            self.extractor = FeatureExtractor(device=device)
        except TypeError:
            self.extractor = FeatureExtractor()
        
        print(f"Scanning UCF-Crime at {root_dir}...")
        if not os.path.exists(root_dir):
             print(f"Error: UCF Path {root_dir} does not exist")
             return

        def process_class_folder(class_folder_path, folder_name):
            if "Normal" in folder_name: 
                label_id = 0
            else:
                label_id = 1 
            
            files = [f for f in os.listdir(class_folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            for f in files:
                self.samples.append((os.path.join(class_folder_path, f), label_id))
            
            if len(files) > 0:
                print(f"  Loaded {len(files)} videos from {folder_name} (Label: {label_id})")

        for entry in os.listdir(root_dir):
            if entry.startswith('.'): continue
            path = os.path.join(root_dir, entry)
            if not os.path.isdir(path): continue
            
            subentries = [e for e in os.listdir(path) if not e.startswith('.')]
            if len(subentries) == 0: continue
            
            first_sub = os.path.join(path, subentries[0])
            if os.path.isdir(first_sub):
                print(f"  Entering Part folder: {entry}")
                for sub in subentries:
                     sub_path = os.path.join(path, sub)
                     if os.path.isdir(sub_path):
                         process_class_folder(sub_path, sub)
            else:
                if any(e.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) for e in subentries[:10]):
                     process_class_folder(path, entry)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        features = extract_cnn_features(path, self.extractor, self.device, self.max_frames)
        return features, label

# --- Model ---
class HockeyGRU(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
            num_layers=2, # Deeper for complex features
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: (B, T, 1280)
        _, h = self.gru(x) 
        # h: (num_layers, B, hidden_dim). Take last layer.
        return self.fc(h[-1]) 

# --- Main Training ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hockey", choices=["hockey", "ucf"], help="Dataset type")
    parser.add_argument("--data_root", type=str, default="HockeyDataset", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--finetune", type=str, default=None, help="Path to model weights for fine-tuning")
    parser.add_argument("--save_name", type=str, default=None, help="Filename to save model")
    args = parser.parse_args()

    # 1. Setup Dataset
    if args.dataset == "hockey":
        dataset = HockeyDataset(args.data_root, device, max_frames=MAX_FRAMES)
    else:
        dataset = UCFCrimeDataset(args.data_root, device, max_frames=MAX_FRAMES)
    
    if len(dataset) == 0:
        print("Error: No samples found.")
        exit(1)
        
    # Collate fn might be needed if tensor sizes vary, but we padding in extract_cnn_features
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    # num_workers=0 because we are using CUDA/MPS inside Dataset (not safe in subprocesses usually)

    print(f"Dataset: {args.dataset} | Total samples: {len(dataset)}")

    # 2. Setup Model
    model = HockeyGRU(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(device)
    
    if args.finetune:
        if os.path.exists(args.finetune):
            print(f"🔄 Loading weights from {args.finetune}...")
            try:
                model.load_state_dict(torch.load(args.finetune, map_location=device))
            except:
                print("⚠️  Architecture mismatch (Old model was 2-dim input, new is 1280). Starting scratch.")
        else:
            print("⚠️  Model path not found.")

    # Class Weights
    labels = [sample[1] for sample in dataset.samples]
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    if len(class_counts) == 2:
        weights = total_samples / (2 * class_counts)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"Using Class Weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 3. Train
    print(f"Starting CNN-GRU training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device) 
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            # Print every few batches because extraction is slow
            if batch_idx % 10 == 0:
                 print(f"  Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        acc = correct / total
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.2f}s")
        
        scheduler.step(avg_loss)

        # Simple Save Best by Overwriting
        if args.save_name:
            save_name = args.save_name
        else:
            save_name = f"{args.dataset}_cnn_gru.pth"
        torch.save(model.state_dict(), save_name)

    print(f"Training Complete. Saved to {save_name}")
