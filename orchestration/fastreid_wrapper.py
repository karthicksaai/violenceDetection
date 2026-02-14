import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FASTREID_ROOT = PROJECT_ROOT / "fast_reid_repo"

if str(FASTREID_ROOT) not in sys.path:
    sys.path.append(str(FASTREID_ROOT))

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.data import transforms as T


class FastReIDExtractor:
    def __init__(self, config_file="configs/Market1501/sbs_R50.yml", device="cpu"):
        self.cfg = get_cfg()

        config_path = FASTREID_ROOT / config_file
        self.cfg.merge_from_file(str(config_path))

        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.BACKBONE.PRETRAIN = False
        self.cfg.TEST.IMS_PER_BATCH = 32

        self.predictor = DefaultPredictor(self.cfg)

        # ✅ CORRECT TRANSFORM PIPELINE
        self.transform = T.build_transforms(self.cfg, is_train=False)

    def extract(self, img_list):
        if not img_list:
            return []

        batch = []

        for img in img_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transform(img)
            batch.append(img)

        batch = torch.stack(batch).to(self.cfg.MODEL.DEVICE)

        inputs = {"images": batch}

        self.predictor.model.eval()
        with torch.no_grad():
            feat = self.predictor.model(inputs)

        feat = feat.cpu().numpy()

        # ✅ L2 normalize
        norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-6
        feat = feat / norm

        return [f for f in feat]

import numpy as np
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity


class IdentityManager:
    def __init__(self, similarity_threshold=0.65):
        self.similarity_threshold = similarity_threshold
        self.gallery = {}
        self.next_id = 1

    def match(self, track_id, embedding):

        if not self.gallery:
            self.gallery[self.next_id] = embedding
            pid = self.next_id
            self.next_id += 1
            return pid, 1.0

        best_score = -1.0
        best_id = None

        for person_id, ref_emb in self.gallery.items():
            score = float(np.dot(ref_emb, embedding))
            if score > best_score:
                best_score = score
                best_id = person_id

        print(f"[SIM] BestSim: {best_score:.3f}")

        if best_score > self.similarity_threshold:
            return best_id, best_score

        # new person
        self.gallery[self.next_id] = embedding
        pid = self.next_id
        self.next_id += 1
        return pid, best_score