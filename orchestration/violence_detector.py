
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
# This ensures the inference code always matches the training code
# (single source of truth — no duplicate class definitions).
# -----------------------------------------------------------------
try:
    from train_hockey_gru import HockeyGRU_BiTA, HockeyGRU_Legacy
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from train_hockey_gru import HockeyGRU_BiTA, HockeyGRU_Legacy

class CAVEGate(nn.Module):
    """
    CAVE Gate: Context-Aware Violence Escalation Gate

    A novel multiplicative gating mechanism that modulates the raw GRU
    violence score using crowd context features extracted from YOLO detections.

    The gate learns that a punch among 6 agitated people is more likely
    to be genuine violence than the same punch in an empty corridor.

    Crowd context vector (3 features):
      [0] norm_count      : number of people in frame, normalised by 10
      [1] norm_density    : mean inter-person distance (inverse), normalised
      [2] crowd_velocity  : rate of bounding-box area change between frames
    """
    def __init__(self, context_dim=3, hidden=16):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()   # output in (0, 1) — scales the raw score
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
            gate = self.gate_net(ctx).item()   # scalar in (0,1)
        # Multiplicative gate: amplify when context is alarming
        gated = raw_score * (0.5 + gate)       # range: raw*0.5 → raw*1.5
        return float(np.clip(gated, 0.0, 1.0))

class ViolenceDetector:
    """
    Real-time per-person violence classifier.

    Uses the novel BiGRU-TA (Bidirectional GRU + Temporal Attention)
    model by default.  The detector maintains a sliding window buffer
    of CNN features for each tracked person and runs GRU inference
    once the buffer is full.

    Key upgrade over the legacy detector:
    - process_frame() now returns BOTH the violence probability AND
      the temporal attention weights, enabling frame-level explanations
      of why a detection was triggered.
    - get_attention_heatmap() converts weights to a visual colour bar
      that can be rendered on the surveillance dashboard.
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

        # ---------------------------------------------------------
        # Model selection:
        #   use_legacy=False  → BiGRU-TA (novel, default)
        #   use_legacy=True   → original unidirectional GRU
        # ---------------------------------------------------------
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

    # ------------------------------------------------------------------
    def process_frame(self, frame_bgr):
        """
        Process a single BGR frame from one tracked person's crop.

        Steps:
          1. Extract 512-d CNN feature vector (ResNet50 via FeatureExtractor)
          2. Append to the 20-frame sliding window buffer
          3. When buffer is full, run BiGRU-TA inference
          4. Return (violence_prob, attn_weights)

        Returns:
            violence_prob (float)  : probability of violence in [0, 1]
            attn_weights  (np.ndarray | None) : shape (T,) per-frame
                importance weights, or None if buffer not yet full.
                High values indicate the frames that most influenced
                the classification decision.
        """
        # 1. Extract CNN feature
        try:
            feat = self.extractor.extract(frame_bgr)  # (512,) numpy
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
            violence_prob = probs[0][1].item()  # Class 1 = Violence

        # Convert attention weights to numpy for the calling code
        if attn_weights is not None:
            attn_np = attn_weights[0].cpu().numpy()  # shape (T,)
        else:
            attn_np = None

        return violence_prob, attn_np

    # ------------------------------------------------------------------
    def get_attention_heatmap(self, attn_weights, bar_width=300, bar_height=20):
        """
        Convert temporal attention weights into a colour heatmap bar
        suitable for overlaying on the video dashboard.

        Each frame in the sliding window gets a colour ranging from
        green (low attention = calm frame) to red (high attention =
        critical / violent frame).

        Args:
            attn_weights (np.ndarray): shape (T,) from process_frame()
            bar_width  (int): pixel width of the output bar
            bar_height (int): pixel height of the output bar

        Returns:
            heatmap (np.ndarray): BGR image of shape (bar_height, bar_width, 3)
        """
        import cv2

        if attn_weights is None or len(attn_weights) == 0:
            return np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

        T = len(attn_weights)
        # Normalise to [0, 1]
        w = attn_weights - attn_weights.min()
        if w.max() > 0:
            w = w / w.max()

        heatmap = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
        cell_w = bar_width // T

        for i, weight in enumerate(w):
            # Green (low) → Red (high)
            g = int(255 * (1.0 - weight))
            r = int(255 * weight)
            colour = (0, g, r)  # BGR
            x1 = i * cell_w
            x2 = x1 + cell_w
            heatmap[:, x1:x2] = colour

        return heatmap

    # ------------------------------------------------------------------
    def reset_buffer(self):
        """Clear the frame buffer (call when a new person track starts)."""
        self.buffer.clear()
