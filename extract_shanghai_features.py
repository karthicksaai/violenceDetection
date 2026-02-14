import torch
import cv2
import numpy as np
import os
import sys
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Add project root
sys.path.append(os.getcwd())

from orchestration.violence_detector import FeatureExtractor

# Config
DATA_ROOT = "SHANGHAI"
TRAIN_LIST = os.path.join(DATA_ROOT, "SHANGHAI_TRAIN", "SHANGHAI_train.txt")
SAVE_DIR = "shanghai_features"
BATCH_SIZE = 64

class ShanghaiDataset(Dataset):
    def __init__(self, txt_file, root_dir):
        self.root_dir = root_dir
        self.samples = []
        
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue # Skip empty lines
                
                # Line format: SHANGHAI_TRAIN/frames/01_0014 0 265 1
                rel_path = parts[0]
                label = int(parts[3])
                
                # Construct full path
                # rel_path is 'SHANGHAI_TRAIN/frames/01_0014'
                # root_dir is 'SHANGHAI'
                # Full: 'SHANGHAI/SHANGHAI_TRAIN/frames/01_0014'
                full_path = os.path.join(self.root_dir, rel_path)
                
                self.samples.append((full_path, label))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def extract_features():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        
    print(f"🚀 Starting Feature Extraction on {device}...")
    
    # 1. Setup
    dataset = ShanghaiDataset(TRAIN_LIST, DATA_ROOT)
    extractor = FeatureExtractor(device=device)
    
    print(f"Found {len(dataset)} videos.")
    
    all_features = []
    all_labels = []
    skipped = 0
    
    # 2. Iterate Videos
    for i, (video_dir, label) in enumerate(dataset):
        if not os.path.exists(video_dir):
            print(f"❌ Missing: {video_dir}")
            skipped += 1
            continue
            
        print(f"[{i+1}/{len(dataset)}] Processing {os.path.basename(video_dir)} (Label: {label})")
        
        # Load all frames
        frame_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.jpg')])
        
        if not frame_files:
            print(f"⚠️  No frames in {video_dir}")
            skipped += 1
            continue
            
        # Extract in batches
        frames_buffer = []
        video_features = []
        
        for frame_path in frame_files:
            try:
                # Load and enable Resize/Norm via Extractor
                frame = cv2.imread(frame_path)
                if frame is None: continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_buffer.append(Image.fromarray(frame))
                
                if len(frames_buffer) == BATCH_SIZE:
                    batch = torch.stack([extractor.preprocess(img) for img in frames_buffer]).to(device)
                    with torch.no_grad():
                        feats = extractor.model(batch)
                        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
                    video_features.append(feats.cpu())
                    frames_buffer = []
                    
            except Exception as e:
                print(f"Error frame {frame_path}: {e}")

        # Process remaining
        if frames_buffer:
            batch = torch.stack([extractor.preprocess(img) for img in frames_buffer]).to(device)
            with torch.no_grad():
                feats = extractor.model(batch)
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            video_features.append(feats.cpu())

        if not video_features:
            print(f"⚠️  No features extracted for {video_dir}")
            skipped += 1
            continue
            
        # Concatenate -> (T, 512)
        full_seq = torch.cat(video_features, dim=0)
        
        all_features.append(full_seq)
        all_labels.append(label)
        
    print(f"✅ Extraction Complete. Processed {len(all_features)} videos. Skipped {skipped}.")
    
    # 3. Save
    # We can't stack all_features because T varies!
    # We must save as a list of tensors or perform padding.
    # For training, it's easier to verify simply by creating a custom collate_fn or saving individually.
    # Saving as a list of tensors in one file is fine for small dataset (240 videos).
    
    torch.save(all_features, "shanghai_features_list.pt")
    torch.save(torch.tensor(all_labels), "shanghai_labels.pt")
    print("💾 Saved to shanghai_features_list.pt and shanghai_labels.pt")

if __name__ == "__main__":
    extract_features()
