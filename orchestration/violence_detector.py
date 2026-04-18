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
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from orchestration.reid.feature_extractor import FeatureExtractor

# -----------------------------------------------------------------
# Import the novel BiGRU-TA architecture from the training module.
# -----------------------------------------------------------------
try:
    from train_hockey_gru import HockeyGRU_BiTA, HockeyGRU_Legacy
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from train_hockey_gru import HockeyGRU_BiTA, HockeyGRU_Legacy


# =================================================================
# NOVELTY 1: CAVE Gate — Context-Aware Violence Escalation Gate
#
# A multiplicative gating mechanism that modulates the raw GRU
# violence score using crowd context features extracted from YOLO.
#
# The gate learns that a punch among 6 agitated people is more
# likely to be genuine violence than the same punch in an empty
# corridor.
#
# Crowd context vector (3 features):
#   [0] norm_count     : number of people in frame, normalised by 10
#   [1] norm_density   : mean inter-person distance (inverse), normalised
#   [2] crowd_velocity : rate of bounding-box area change between frames
# =================================================================

class CAVEGate(nn.Module):
    def __init__(self, context_dim=3, hidden=16):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, raw_score: float, context_vec: np.ndarray) -> float:
        """
        Args:
            raw_score   : float, violence probability from GRU
            context_vec : np.ndarray shape (3,)
        Returns:
            gated_score : float, context-modulated violence probability
        """
        ctx = torch.tensor(context_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            gate = self.gate_net(ctx).item()
        # Multiplicative gate: amplify when context is alarming
        # range: raw*0.5 (very calm) to raw*1.5 (very alarming)
        gated = raw_score * (0.5 + gate)
        return float(np.clip(gated, 0.0, 1.0))


class ViolenceDetector:
    """
    Real-time per-person violence classifier.

    Uses the novel BiGRU-TA (Bidirectional GRU + Temporal Attention)
    model by default. The detector maintains a sliding window buffer
    of CNN features for each tracked person and runs GRU inference
    once the buffer is full.

    Novelties integrated:
    - BiGRU-TA: Bidirectional GRU + Temporal Attention
    - CAVE Gate: Context-Aware Violence Escalation Gate
    """

    def __init__(
        self,
        model_path="violence_detection_model.pth",
        device=None,
        threshold=0.85,
        use_legacy=False
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = "mps"

        print(f"Initializing ViolenceDetector on {self.device}...")

        # Load ReID Feature Extractor (ResNet50)
        self.extractor = FeatureExtractor(device=self.device)

        # Model selection
        if use_legacy:
            self.model = HockeyGRU_Legacy(input_dim=512, hidden_dim=256).to(self.device)
            print("  Model: HockeyGRU_Legacy (original unidirectional GRU)")
        else:
            self.model = HockeyGRU_BiTA(input_dim=512, hidden_dim=256).to(self.device)
            print("  Model: HockeyGRU_BiTA (Bidirectional GRU + Temporal Attention) [NOVEL]")

        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"  Weights loaded from: {model_path}")
            except Exception as e:
                print(f"  WARNING: Could not load weights ({e}). Running with random init.")
        else:
            print(f"  WARNING: {model_path} not found — predictions will be random.")

        self.model.eval()

        # Sliding window buffer (20 frames of CNN features per person)
        self.sequence_length = 20
        self.buffer = deque(maxlen=self.sequence_length)
        self.threshold = threshold

        # CAVE Gate — context-aware violence escalation
        self.cave_gate = CAVEGate(context_dim=3, hidden=16)
        self._prev_bbox_areas: list = []

    # ------------------------------------------------------------------
    def compute_context_vector(self, all_bboxes: list) -> np.ndarray:
        """
        Compute the 3-feature crowd context vector from YOLO bounding boxes.

        Args:
            all_bboxes: list of (x1,y1,x2,y2) for ALL persons in the frame

        Returns:
            np.ndarray shape (3,): [norm_count, norm_density, crowd_velocity]
        """
        n = len(all_bboxes)

        # Feature 1: normalised person count
        norm_count = min(n / 10.0, 1.0)

        # Feature 2: mean pairwise distance (inverted → higher = denser)
        if n > 1:
            centroids = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in all_bboxes])
            dists = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    d = np.linalg.norm(centroids[i] - centroids[j])
                    dists.append(d)
            mean_dist = np.mean(dists) if dists else 500.0
            norm_density = float(np.clip(1.0 - mean_dist / 500.0, 0.0, 1.0))
        else:
            norm_density = 0.0

        # Feature 3: crowd velocity (bbox area change rate)
        curr_areas = [abs((b[2]-b[0])*(b[3]-b[1])) for b in all_bboxes]
        if self._prev_bbox_areas and len(curr_areas) == len(self._prev_bbox_areas):
            delta = np.mean(np.abs(np.array(curr_areas) - np.array(self._prev_bbox_areas)))
            crowd_velocity = float(np.clip(delta / 5000.0, 0.0, 1.0))
        else:
            crowd_velocity = 0.0
        self._prev_bbox_areas = curr_areas

        return np.array([norm_count, norm_density, crowd_velocity], dtype=np.float32)

    # ------------------------------------------------------------------
    def process_frame(self, frame_bgr, all_bboxes: list = None):
        """
        Process a single BGR frame from one tracked person's crop.

        Steps:
          1. Extract 512-d CNN feature vector (ResNet50 via FeatureExtractor)
          2. Append to the 20-frame sliding window buffer
          3. When buffer is full, run BiGRU-TA inference
          4. Apply CAVE Gate using crowd context (if all_bboxes provided)
          5. Return (violence_prob, attn_weights)

        Args:
            frame_bgr  : BGR crop of the person
            all_bboxes : list of (x1,y1,x2,y2) for ALL people in frame
                         used by CAVE Gate for crowd context

        Returns:
            violence_prob (float)           : context-modulated probability
            attn_weights  (np.ndarray|None) : per-frame attention weights
        """
        # 1. Extract CNN feature
        try:
            feat = self.extractor.extract(frame_bgr)
            self.buffer.append(torch.tensor(feat, dtype=torch.float32))
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return 0.0, None

        # 2. Not enough frames yet
        if len(self.buffer) < self.sequence_length:
            return 0.0, None

        # 3. Prepare input tensor: (1, T, 512)
        sequence = torch.stack(list(self.buffer)).unsqueeze(0).to(self.device)

        # 4. Inference
        with torch.no_grad():
            logits, attn_weights = self.model(sequence)
            probs = torch.softmax(logits, dim=1)
            violence_prob = probs[0][1].item()

        # Convert attention weights to numpy
        if attn_weights is not None:
            attn_np = attn_weights[0].cpu().numpy()
        else:
            attn_np = None

        # 5. CAVE Gate: modulate score with crowd context
        if all_bboxes is not None and len(all_bboxes) > 0:
            ctx_vec = self.compute_context_vector(all_bboxes)
            violence_prob = self.cave_gate(violence_prob, ctx_vec)

        return violence_prob, attn_np

    # ------------------------------------------------------------------
    def get_attention_heatmap(self, attn_weights, bar_width=300, bar_height=20):
        """
        Convert temporal attention weights into a colour heatmap bar
        suitable for overlaying on the video dashboard.
        """
        import cv2

        if attn_weights is None or len(attn_weights) == 0:
            return np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

        T = len(attn_weights)
        w = attn_weights - attn_weights.min()
        if w.max() > 0:
            w = w / w.max()

        heatmap = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
        cell_w = bar_width // T

        for i, weight in enumerate(w):
            g = int(255 * (1.0 - weight))
            r = int(255 * weight)
            colour = (0, g, r)
            x1 = i * cell_w
            x2 = x1 + cell_w
            heatmap[:, x1:x2] = colour

        return heatmap

    # ------------------------------------------------------------------
    def reset_buffer(self):
        """Clear the frame buffer (call when a new person track starts)."""
        self.buffer.clear()
