"""
orchestration/attention_visualizer.py

Visualizes the Temporal Attention weights produced by the BiGRU-TA
violence classifier.  These weights tell you WHICH frames in the
20-frame sliding window most influenced the VIOLENCE classification.

Usage example (in your main loop / engine.py):

    from orchestration.attention_visualizer import AttentionVisualizer
    from orchestration.violence_detector import ViolenceDetector

    detector  = ViolenceDetector()
    visualizer = AttentionVisualizer(sequence_length=20)

    violence_prob, attn_weights = detector.process_frame(person_crop)

    if violence_prob > detector.threshold:
        annotated = visualizer.annotate_frame(
            frame       = full_frame,
            bbox        = (x1, y1, x2, y2),
            attn_weights= attn_weights,
            score       = violence_prob,
            person_id   = track_id
        )
        cv2.imshow("Control Room", annotated)
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class AttentionVisualizer:
    """
    Renders BiGRU-TA temporal attention weights onto surveillance frames.

    The attention bar is a horizontal strip drawn below the person's
    bounding box.  Each cell in the strip corresponds to one frame in
    the sliding window; its colour encodes the attention weight:
        Green  (weight ≈ 0.0) — calm / unimportant frame
        Yellow (weight ≈ 0.5) — moderate activity
        Red    (weight ≈ 1.0) — peak violence frame

    A white tick marks the frame with the highest attention weight.
    """

    def __init__(
        self,
        sequence_length: int = 20,
        bar_height: int = 14,
        bar_width: int = 120,
    ):
        self.T = sequence_length
        self.bar_height = bar_height
        self.bar_width = bar_width

    # ------------------------------------------------------------------
    def _weights_to_colours(self, attn_weights: np.ndarray) -> np.ndarray:
        """
        Map normalised attention weights → BGR colour per cell.
        Returns array of shape (T, 3).
        """
        w = attn_weights.copy()
        # Normalise to [0, 1]
        if w.max() > w.min():
            w = (w - w.min()) / (w.max() - w.min())

        colours = np.zeros((len(w), 3), dtype=np.uint8)
        for i, weight in enumerate(w):
            # Green → Yellow → Red  (BGR)
            if weight < 0.5:
                # Green → Yellow
                g = 255
                r = int(255 * weight * 2)
            else:
                # Yellow → Red
                g = int(255 * (1.0 - weight) * 2)
                r = 255
            colours[i] = (0, g, r)  # B=0, G, R
        return colours

    # ------------------------------------------------------------------
    def draw_attention_bar(
        self,
        attn_weights: np.ndarray,
        bar_width: Optional[int] = None,
        bar_height: Optional[int] = None,
    ) -> np.ndarray:
        """
        Build a standalone BGR attention bar image.

        Args:
            attn_weights: (T,) numpy array from ViolenceDetector.process_frame()
            bar_width:    override default bar width in pixels
            bar_height:   override default bar height in pixels

        Returns:
            bar: BGR image of shape (bar_height, bar_width, 3)
        """
        bw = bar_width or self.bar_width
        bh = bar_height or self.bar_height
        T  = len(attn_weights)

        colours = self._weights_to_colours(attn_weights)
        bar = np.zeros((bh, bw, 3), dtype=np.uint8)

        cell_w = bw / T
        for i, colour in enumerate(colours):
            x1 = int(i * cell_w)
            x2 = int((i + 1) * cell_w)
            bar[:, x1:x2] = colour

        # White tick on the peak-attention frame
        peak_idx = int(np.argmax(attn_weights))
        peak_x   = int((peak_idx + 0.5) * cell_w)
        cv2.line(bar, (peak_x, 0), (peak_x, bh), (255, 255, 255), 1)

        return bar

    # ------------------------------------------------------------------
    def annotate_frame(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        attn_weights: Optional[np.ndarray],
        score: float,
        person_id: int = 0,
        label_prefix: str = "VIOLENCE",
    ) -> np.ndarray:
        """
        Annotate a full surveillance frame with:
          - Red bounding box around the violent person
          - Score + person ID label
          - Attention heatmap bar below the bounding box
          - Peak-frame marker on the bar

        Args:
            frame:        BGR numpy array (H, W, 3) — the full camera frame
            bbox:         (x1, y1, x2, y2) bounding box of the person
            attn_weights: (T,) array or None
            score:        violence probability in [0, 1]
            person_id:    tracker-assigned person ID
            label_prefix: string shown in the label box

        Returns:
            Annotated copy of the frame (original is not modified).
        """
        out = frame.copy()
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # --- Bounding box ---
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # --- Label ---
        label = f"{label_prefix} P{person_id} {score:.2f}"
        label_y = max(y1 - 6, 14)
        cv2.rectangle(out, (x1, label_y - 14), (x1 + len(label) * 9, label_y + 2), (0, 0, 200), -1)
        cv2.putText(out, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # --- Attention bar ---
        if attn_weights is not None and len(attn_weights) > 0:
            bar_w = min(x2 - x1, self.bar_width)
            bar   = self.draw_attention_bar(attn_weights, bar_width=bar_w, bar_height=self.bar_height)

            bar_y1 = y2 + 2
            bar_y2 = bar_y1 + self.bar_height

            # Clip to frame bounds
            h, w = out.shape[:2]
            if bar_y2 <= h and x1 + bar_w <= w:
                out[bar_y1:bar_y2, x1:x1 + bar_w] = bar

                # Caption
                cv2.putText(
                    out, "ATTN",
                    (x1, bar_y2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1
                )

        return out

    # ------------------------------------------------------------------
    def save_attention_plot(
        self,
        attn_weights: np.ndarray,
        save_path: str = "attention_plot.png",
        title: str = "BiGRU-TA Temporal Attention Weights",
    ) -> None:
        """
        Save a matplotlib bar chart of the attention weights as a PNG.
        Useful for academic reports and presentations.

        Args:
            attn_weights: (T,) numpy array
            save_path:    output file path (e.g. 'output/attn_frame42.png')
            title:        chart title
        """
        try:
            import matplotlib
            matplotlib.use("Agg")   # Non-interactive backend (works headless)
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            T = len(attn_weights)
            w = attn_weights.copy()
            if w.max() > w.min():
                w_norm = (w - w.min()) / (w.max() - w.min())
            else:
                w_norm = w

            colours = cm.RdYlGn_r(w_norm)   # Red=high, Green=low

            fig, ax = plt.subplots(figsize=(10, 3))
            bars = ax.bar(range(T), w, color=colours, edgecolor="none")

            peak = int(np.argmax(w))
            ax.axvline(peak, color="white", linewidth=1.5, linestyle="--", label=f"Peak frame {peak}")

            ax.set_xlabel("Frame index in sliding window", fontsize=10)
            ax.set_ylabel("Attention weight", fontsize=10)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlim(-0.5, T - 0.5)
            ax.legend(fontsize=9)
            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.spines[:].set_color("#444")

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Attention plot saved to: {save_path}")

        except ImportError:
            print("matplotlib not installed — skipping attention plot save.")
