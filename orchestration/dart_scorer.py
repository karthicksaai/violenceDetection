"""
orchestration/dart_scorer.py

DART Score: Decay-Augmented Recency Threshold

A novel temporal scoring mechanism that replaces fixed-threshold
violence classification with a per-camera adaptive decision boundary.

Two innovations:
1. DART Score: exponentially-weighted running score where recent
   GRU outputs matter more than older ones (recency bias)
2. Adaptive Threshold: each camera learns its own baseline activity
   level and raises/lowers its decision boundary accordingly —
   a calm corridor requires stronger evidence than a busy market.
"""

import numpy as np
import time
from collections import deque


class DARTScorer:
    """
    Decay-Augmented Recency Threshold Scorer.
    One instance per camera. Maintains a rolling window of GRU scores
    and computes both the DART score and the adaptive threshold.
    """

    def __init__(
        self,
        camera_id: str,
        decay_lambda: float = 0.5,
        window_size: int = 10,
        base_threshold: float = 0.50,
        alpha: float = 0.15,
        calm_window: int = 100,
    ):
        self.camera_id = camera_id
        self.lam = decay_lambda
        self.window_size = window_size
        self.base_threshold = base_threshold
        self.alpha = alpha

        self.score_history: deque = deque(maxlen=window_size)
        self.calm_history: deque = deque(maxlen=calm_window)
        self.current_threshold = base_threshold

    def update(self, gru_score: float) -> tuple:
        """
        Ingest a new raw GRU score and compute the DART score + threshold.

        Args:
            gru_score: float in [0,1] from ViolenceDetector.process_frame()

        Returns:
            dart_score  (float): exponentially-weighted recency score
            threshold   (float): adaptive per-camera decision boundary
            is_violence (bool) : dart_score >= threshold
        """
        now = time.time()
        self.score_history.append((now, gru_score))

        # --- 1. DART Score (exponential decay over history) ---
        scores = list(self.score_history)
        dart_score = 0.0
        weight_sum = 0.0
        for i, (_, s) in enumerate(reversed(scores)):
            w = np.exp(-self.lam * i)
            dart_score += w * s
            weight_sum += w
        dart_score = dart_score / weight_sum if weight_sum > 0 else gru_score

        # --- 2. Update calm baseline ---
        if gru_score < 0.30:
            self.calm_history.append(gru_score)

        # --- 3. Adaptive Threshold ---
        if len(self.calm_history) > 10:
            calm_mean = np.mean(list(self.calm_history))
            self.current_threshold = self.base_threshold + self.alpha * calm_mean
        else:
            self.current_threshold = self.base_threshold

        is_violence = dart_score >= self.current_threshold

        return dart_score, self.current_threshold, is_violence

    def reset(self):
        """Clear all history (call when stream resets)."""
        self.score_history.clear()
        self.calm_history.clear()
        self.current_threshold = self.base_threshold

    def get_state(self) -> dict:
        """Return current state dict for logging/dashboard."""
        return {
            'camera_id':   self.camera_id,
            'dart_score':  round(float(list(self.score_history)[-1][1]) if self.score_history else 0, 4),
            'threshold':   round(self.current_threshold, 4),
            'history_len': len(self.score_history),
            'calm_len':    len(self.calm_history),
        }


class DARTScorerBank:
    """
    Manages one DARTScorer instance per camera automatically.
    Drop-in replacement for a fixed threshold check.

    Usage in engine.py:
        dart_bank = DARTScorerBank()
        dart, thresh, is_violence = dart_bank.update(camera_id, gru_score)
    """
    def __init__(self, **dart_kwargs):
        self.scorers: dict = {}
        self.kwargs = dart_kwargs

    def update(self, camera_id: str, gru_score: float) -> tuple:
        if camera_id not in self.scorers:
            self.scorers[camera_id] = DARTScorer(camera_id, **self.kwargs)
        return self.scorers[camera_id].update(gru_score)

    def reset(self, camera_id: str):
        if camera_id in self.scorers:
            self.scorers[camera_id].reset()

    def get_all_states(self) -> list:
        return [s.get_state() for s in self.scorers.values()]
