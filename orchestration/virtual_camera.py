import cv2
import time
import os
import glob
from typing import Optional

class VirtualCamera:
    def __init__(self, camera_id: str, zone_id: str, source_path: str, location: dict):
        self.camera_id = camera_id
        self.zone_id = zone_id
        self.source_path = source_path
        self.location = location # {'lat': float, 'lng': float}
        self.is_active = False 
        self.cap = None
        self.image_files = []
        self.current_frame_idx = 0
        self.is_video_file = False

        # Determine source type
        if os.path.isdir(source_path):
            self.image_files = sorted(glob.glob(os.path.join(source_path, "*.jpg")) + 
                                      glob.glob(os.path.join(source_path, "*.png")))
            self.is_video_file = False
            # For demo variety, start different cameras at different offsets
            import random
            if self.image_files:
                self.current_frame_idx = random.randint(0, len(self.image_files) - 1)
        else:
            self.is_video_file = True

    def start_stream(self):
        """Simulate waking up the sensor"""
        # print(f"[{self.camera_id}] Waking up sensor...")
        self.is_active = True
        if self.is_video_file:
            self.cap = cv2.VideoCapture(self.source_path)

    def stop_stream(self):
        # print(f"[{self.camera_id}] Returning to sleep mode...")
        self.is_active = False
        if self.cap:
            self.cap.release()

    def get_frame(self):
        if not self.is_active:
            return None, False

        # CASE A: Playing from Image Directory (Dataset)
        if not self.is_video_file and self.image_files:
            img_path = self.image_files[self.current_frame_idx]
            frame = cv2.imread(img_path)
            
            # Advance frame
            self.current_frame_idx = (self.current_frame_idx + 1) % len(self.image_files)
            
            # Resize if needed (datasets can vary)
            if frame is not None:
                frame = cv2.resize(frame, (640, 480))
            
            time.sleep(0.05) # ~20FPS simulation
            return frame, True

        # CASE B: Playing from Video File
        if self.is_video_file:
            if not self.cap or not self.cap.isOpened():
                # FALLBACK: Return Dummy Frame
                import numpy as np
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(dummy_frame, f"MOCK: {self.camera_id}", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                time.sleep(0.1)
                return dummy_frame, True

            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            return frame, ret
            
        return None, False

    def __repr__(self):
        return f"<VirtualCamera {self.camera_id} (Zone: {self.zone_id})>"
