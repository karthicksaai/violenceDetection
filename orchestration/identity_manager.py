import numpy as np
from collections import deque


class IdentityManager:
    def __init__(self, similarity_threshold=0.85, confirm_frames=4):
        self.similarity_threshold = similarity_threshold
        self.confirm_frames = confirm_frames

        self.gallery = {}            # person_id -> embedding
        self.track_memory = {}       # track_id -> deque of person_ids
        self.next_id = 1

    def match(self, track_id, embedding):

        if not self.gallery:
            self.gallery[self.next_id] = embedding
            self.track_memory[track_id] = deque(maxlen=self.confirm_frames)
            self.track_memory[track_id].append(self.next_id)
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

        # -------------------------
        # MATCH CASE
        # -------------------------
        if best_score > self.similarity_threshold:

            # update streak memory
            self.track_memory.setdefault(track_id, deque(maxlen=self.confirm_frames))
            self.track_memory[track_id].append(best_id)

            # confirm after streak
            if self.track_memory[track_id].count(best_id) >= self.confirm_frames:
                return best_id, best_score

            # 🔥 IMPORTANT: return best_id even before confirmation
            return best_id, best_score

        # -------------------------
        # NEW PERSON CASE
        # -------------------------
        self.gallery[self.next_id] = embedding
        pid = self.next_id
        self.next_id += 1
        return pid, best_score
