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
    sys.path.append(os.getcwd())
    from orchestration.reid.feature_extractor import FeatureExtractor


# --- Configuration ---
LABEL_MAP = {
    "nofights": 0,
    "fights": 1
}
MAX_FRAMES = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
HIDDEN_DIM = 256
NUM_LAYERS = 2
INPUT_DIM = 512  # ResNet50 Feature Dim

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
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
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

    while len(frames_list) < max_frames:
        frames_list.append(frames_list[-1])

    batch = torch.stack(frames_list[:max_frames]).to(device)

    with torch.no_grad():
        embeddings = extractor.model(batch)

    features = embeddings.flatten(start_dim=1)  # (T, 512)
    features = torch.nn.functional.normalize(features, p=2, dim=1)

    return features.cpu()


# --- Dataset ---
class HockeyDataset(Dataset):
    def __init__(self, root_dir, device, max_frames=20):
        self.samples = []
        self.max_frames = max_frames
        self.root_dir = root_dir
        self.device = device
        try:
            self.extractor = FeatureExtractor(device=device)
        except TypeError:
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
            label_id = 0 if "Normal" in folder_name else 1
            files = [f for f in os.listdir(class_folder_path)
                     if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
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


# ============================================================
# LEGACY MODEL (original — kept for backward compatibility)
# ============================================================
class HockeyGRU_Legacy(nn.Module):
    """Original unidirectional GRU — no attention. Kept for comparison."""
    def __init__(self, input_dim=1280, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            batch_first=True, num_layers=2, dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: (B, T, input_dim)
        _, h = self.gru(x)
        logits = self.fc(h[-1])
        # Return (logits, None) for API consistency with BiTA model
        return logits, None


# ============================================================
# NOVELTY: Temporal Attention Module
# ============================================================
class TemporalAttention(nn.Module):
    """
    Learns a scalar importance weight for each timestep in the GRU output.

    Instead of discarding all but the last hidden state, this module
    computes a softmax distribution over all T hidden states and returns
    a weighted context vector — focusing on the frames that matter most
    (i.e., the exact moment of the violent act).

    The attention weights are also returned so they can be visualized
    to explain WHICH frames caused the violence classification.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Single linear layer: maps each hidden vector to a scalar score
        self.attention_fc = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_outputs):
        """
        Args:
            gru_outputs: (B, T, hidden_dim) — all T hidden states from GRU

        Returns:
            context:      (B, hidden_dim)  — attention-weighted sum
            attn_weights: (B, T)           — normalized per-frame importance
        """
        # Score each timestep: (B, T, 1)
        scores = self.attention_fc(gru_outputs)

        # Normalize across time dimension: (B, T, 1)
        attn_weights = torch.softmax(scores, dim=1)

        # Weighted sum over T: (B, hidden_dim)
        context = (attn_weights * gru_outputs).sum(dim=1)

        # Squeeze weights for easy return: (B, T)
        return context, attn_weights.squeeze(-1)


# ============================================================
# NOVELTY: BiGRU-TA — Bidirectional GRU + Temporal Attention
# ============================================================
class HockeyGRU_BiTA(nn.Module):
    """
    BiGRU-TA: Bidirectional GRU with Temporal Self-Attention

    Novel contributions over the baseline HockeyGRU:
    1. BIDIRECTIONAL GRU: reads the 20-frame sequence both forward
       (frame 1→20) and backward (frame 20→1) simultaneously.
       The forward pass captures build-up to violence; the backward
       pass captures the aftermath. Concatenating both gives richer
       per-frame representations (hidden_dim * 2 per timestep).

    2. TEMPORAL ATTENTION: instead of using only the final GRU hidden
       state (which blends all frames equally), a learnable attention
       layer assigns a weight to each of the 20 timesteps.
       Frames containing the violent act receive high weights;
       calm frames receive near-zero weights.

    3. INTERPRETABILITY: attn_weights (shape B x T) are returned from
       forward() — they can be overlaid on the video timeline to show
       exactly which frames triggered the classification decision.

    Architecture:
        Input (B, T, 512)
            → BiGRU (hidden=256, bidirectional) → (B, T, 512)
            → Projection FC (512 → 256)         → (B, T, 256)
            → TemporalAttention                 → context (B, 256)
            → Classifier FC (256 → 2)           → logits (B, 2)
    """
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()

        # Bidirectional GRU: output dim = hidden_dim * 2
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True          # <-- KEY NOVELTY 1
        )

        # Project bidirectional output (hidden_dim*2) back to hidden_dim
        # This keeps the attention module dimension-agnostic
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

        # Temporal Attention over all T projected hidden states
        self.attention = TemporalAttention(hidden_dim)  # <-- KEY NOVELTY 2

        # Final violence classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim) — sequence of CNN feature vectors

        Returns:
            logits:       (B, num_classes) — raw classification scores
            attn_weights: (B, T)           — per-frame importance (for visualization)
        """
        # 1. Bidirectional GRU over all T frames
        # gru_out: (B, T, hidden_dim * 2)
        gru_out, _ = self.gru(x)

        # 2. Project to hidden_dim for attention
        # projected: (B, T, hidden_dim)
        projected = self.proj(gru_out)

        # 3. Temporal attention — weighted context vector
        # context: (B, hidden_dim), attn_weights: (B, T)
        context, attn_weights = self.attention(projected)

        # 4. Classify
        logits = self.classifier(context)

        return logits, attn_weights


# Alias: default model used throughout the project
HockeyGRU = HockeyGRU_BiTA


# --- Main Training ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hockey", choices=["hockey", "ucf"])
    parser.add_argument("--data_root", type=str, default="HockeyDataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument(
        "--model", type=str, default="bita", choices=["legacy", "bita"],
        help="'legacy' = original unidirectional GRU | 'bita' = BiGRU + Temporal Attention (default)"
    )
    args = parser.parse_args()

    # 1. Dataset
    if args.dataset == "hockey":
        dataset = HockeyDataset(args.data_root, device, max_frames=MAX_FRAMES)
    else:
        dataset = UCFCrimeDataset(args.data_root, device, max_frames=MAX_FRAMES)

    if len(dataset) == 0:
        print("Error: No samples found.")
        exit(1)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Dataset: {args.dataset} | Total samples: {len(dataset)}")

    # 2. Model selection
    if args.model == "legacy":
        model = HockeyGRU_Legacy(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(device)
        print("Using LEGACY unidirectional GRU (no attention)")
    else:
        model = HockeyGRU_BiTA(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(device)
        print("Using NOVEL BiGRU-TA (Bidirectional GRU + Temporal Attention)")

    if args.finetune and os.path.exists(args.finetune):
        print(f"Loading weights from {args.finetune}...")
        try:
            model.load_state_dict(torch.load(args.finetune, map_location=device))
        except Exception:
            print("Architecture mismatch — starting from scratch.")

    # Class weights
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
    print(f"Starting {args.model.upper()} training for {args.epochs} epochs...")
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
            # Both models now return (logits, attn_weights)
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        acc = correct / total
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | "
              f"LR: {current_lr:.6f} | Time: {elapsed:.2f}s")

        scheduler.step(avg_loss)

        save_name = args.save_name if args.save_name else f"{args.dataset}_{args.model}_model.pth"
        torch.save(model.state_dict(), save_name)

    print(f"Training Complete. Model saved to: {save_name}")
