import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
import logging

from .violence_detector import HockeyGRU
from .fastreid_wrapper import FastReIDExtractor
from .reid.feature_extractor import FeatureExtractor
from .identity_manager import IdentityManager

logger = logging.getLogger("Orchestrator")


class YOLOOrchestrator:
    def __init__(
        self,
        yolo_model="yolo26n.pt",
        violence_model="violence_detection_model.pth",
        crowd_model="crowd_anomaly_model.pth",
        device=None,
    ):
        # ---------------- Device ----------------
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if torch.backends.mps.is_available():
            self.device = "mps"

        logger.info(f"🚀 Initializing YOLOOrchestrator on {self.device}...")

        # ---------------- YOLO ----------------
        self.yolo = YOLO(yolo_model)
        logger.info(f"✅ Loaded YOLO model: {yolo_model}")

        # ---------------- ReID ----------------
        self.extractor = FastReIDExtractor(device=self.device)
        self.identity_manager = IdentityManager(
            similarity_threshold=0.7,
            confirm_frames=4
        )
        logger.info("✅ Loaded FastReID + Identity Manager")

        # ---------------- OSNet for GRU ----------------
        self.osnet = FeatureExtractor(device=self.device)
        logger.info("✅ Loaded OSNet for GRU")

        # ---------------- GRU Models ----------------
        self.violence_gru = self._load_gru(
            violence_model, input_dim=512, hidden_dim=256
        )
        self.crowd_gru = self._load_gru(
            crowd_model, input_dim=512, hidden_dim=256
        )

        self.sequence_length = 20
        self.camera_buffers = {}

        self.violence_threshold = 0.75
        self.crowd_threshold = 0.85

    # =====================================================
    # Load GRU
    # =====================================================
    def _load_gru(self, path, input_dim, hidden_dim):
        model = HockeyGRU(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        logger.info(f"✅ Loaded GRU: {path}")
        return model

    # =====================================================
    # Improved Crop Gating
    # =====================================================
    def is_valid_reid_crop(self, crop):
        if crop is None or crop.size == 0:
            return False

        h, w = crop.shape[:2]
        if h * w < 10000:
            return False

        aspect = h / float(w)
        if aspect < 0.3 or aspect > 3.5:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:
            return False

        if gray.std() < 20:
            return False

        return True

    def is_valid_gru_crop(self, crop):
        if crop is None or crop.size == 0:
            return False

        h, w = crop.shape[:2]
        if h * w < 6000:
            return False

        return True

    # =====================================================
    # MAIN PIPELINE
    # =====================================================
    def process_frame(self, frame, camera_id):

        if camera_id not in self.camera_buffers:
            self.camera_buffers[camera_id] = deque(
                maxlen=self.sequence_length
            )

        frame_buffer = self.camera_buffers[camera_id]

        # ---------------- YOLO Tracking ----------------
        results = self.yolo.track(frame, persist=True, verbose=False)
        detections = results[0].boxes

        if detections is None:
            return {}

        person_boxes = []
        H, W = frame.shape[:2]

        for box in detections:
            if int(box.cls[0].item()) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                tid = int(box.id[0].item()) if box.id is not None else None

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                person_boxes.append(
                    {"bbox": (x1, y1, x2, y2), "track_id": tid}
                )

        if not person_boxes:
            return {}

        # Sort by area (largest first)
        person_boxes.sort(
            key=lambda b: (b["bbox"][2] - b["bbox"][0]) *
                          (b["bbox"][3] - b["bbox"][1]),
            reverse=True
        )

        batch_crops = []
        batch_meta = []
        primary_feature = None

        # ---------------- Crop Loop ----------------
        for p in person_boxes[:5]:
            x1, y1, x2, y2 = p["bbox"]
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # -------- ReID branch --------
            if self.is_valid_reid_crop(crop):
                batch_crops.append(crop)
                batch_meta.append(p)

            # -------- GRU branch --------
            if primary_feature is None and self.is_valid_gru_crop(crop):
                try:
                    primary_feature = self.osnet.extract(crop)
                except Exception as e:
                    logger.error(f"OSNet extraction failed: {e}")

        # ---------------- ReID Embeddings ----------------
        embeddings = []
        if batch_crops:
            embeddings = self.extractor.extract(batch_crops)

        # ---------------- Identity Matching ----------------
        detections_out = []

        for i, emb in enumerate(embeddings):
            meta = batch_meta[i]
            track_id = meta["track_id"]

            if track_id is None:
                continue

            person_id, similarity = self.identity_manager.match(track_id, emb)

            if person_id is None:
                continue

            detections_out.append(
                {
                    "track_id": track_id,
                    "person_id": person_id,
                    "similarity": similarity,
                    "embedding": emb,
                    "bbox": meta["bbox"],
                }
            )

        # ---------------- GRU Sequence ----------------
        if primary_feature is None:
            return {"detections": detections_out}

        frame_buffer.append(
            torch.tensor(primary_feature, dtype=torch.float32)
        )

        if len(frame_buffer) < 5:
            return {"detections": detections_out}

        curr_buff = list(frame_buffer)
        if len(curr_buff) < self.sequence_length:
            curr_buff = (
                [curr_buff[0]] * (self.sequence_length - len(curr_buff))
                + curr_buff
            )

        seq = torch.stack(curr_buff).unsqueeze(0).to(self.device)

        with torch.no_grad():
            v_logits = self.violence_gru(seq)
            v_prob = torch.softmax(v_logits, dim=1)[0][1].item()

            c_prob = 0.0
            if len(person_boxes) > 4:
                c_logits = self.crowd_gru(seq)
                c_prob = torch.softmax(c_logits, dim=1)[0][1].item()

        result = {
            "violence_score": v_prob,
            "crowd_score": c_prob,
            "detections": detections_out,
        }

        # ---------------- Alerts ----------------
        if v_prob > self.violence_threshold:
            result["alert"] = "VIOLENCE DETECTED"
            result["action"] = "Dispatch Police"

        elif c_prob > self.crowd_threshold:
            result["alert"] = "CROWD ANOMALY"
            result["action"] = "Alert Control Room"

        return result
