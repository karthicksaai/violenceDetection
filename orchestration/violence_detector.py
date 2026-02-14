
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import sys
import os

# Import our models
try:
    from orchestration.reid.feature_extractor import FeatureExtractor
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from orchestration.reid.feature_extractor import FeatureExtractor

# Define GRU Architecture (Must match training)
class HockeyGRU(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2, num_classes=2):
        super(HockeyGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        _, h = self.gru(x)
        # h: (num_layers, B, hidden_dim). Take last layer.
        return self.fc(h[-1])

class ViolenceDetector:
    def __init__(self, model_path="violence_detection_model.pth", device=None, threshold=0.85):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = "mps"
            
        print(f"Initializing ViolenceDetector on {self.device}...")
        
        # Load ReID Feature Extractor (ResNet50)
        self.extractor = FeatureExtractor(device=self.device)
        
        # Load GRU
        self.model = HockeyGRU(input_dim=512, hidden_dim=256).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("✅ Violence Prediction Model Loaded.")
        else:
            print(f"❌ Warning: Model {model_path} not found. Predictions will be random.")
            
        # Buffer for sliding window (20 frames)
        self.sequence_length = 20
        self.buffer = deque(maxlen=self.sequence_length)
        self.threshold = threshold

    def process_frame(self, frame_bgr):
        """
        Process a single frame:
        1. Extract features (ResNet50)
        2. Update buffer
        3. If buffer full, run GRU inference
        4. Return probability of violence
        """
        # 1. Extract Feature
        try:
            # FeatureExtractor expects BGR numpy image
            feat = self.extractor.extract(frame_bgr) # (2048,) numpy
            self.buffer.append(torch.tensor(feat, dtype=torch.float32))
        except Exception as e:
            print(f"Error extracting features: {e}")
            return 0.0

        # 2. Check Buffer
        if len(self.buffer) < self.sequence_length:
            return 0.0 # Not enough frames yet

        # 3. Prepare Batch
        # Stack frames -> (1, 20, 2048)
        sequence = torch.stack(list(self.buffer)).unsqueeze(0).to(self.device)
        
        # 4. Inference
        with torch.no_grad():
            logits = self.model(sequence)
            probs = torch.softmax(logits, dim=1)
            # Class 1 is Violence
            violence_prob = probs[0][1].item()
            
        return violence_prob
