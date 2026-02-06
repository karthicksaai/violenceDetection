import logging
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
from .models import EventContextCluster, EventType
from .virtual_camera import VirtualCamera
from .reid.feature_extractor import FeatureExtractor
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Orchestrator")

class OrchestrationEngine:
    def __init__(self, camera_graph_path: str):
        self.active_events: Dict[str, EventContextCluster] = {}
        self.cameras: Dict[str, VirtualCamera] = {}
        self.graph: Dict[str, List[str]] = {}
        
        # Initialize ReID Model
        self.reid = FeatureExtractor()
        
        self._load_network(camera_graph_path)

    def _load_network(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Load topology
        self.graph = data.get('graph', {})
        
        # Load virtual cameras (Metadata only initially)
        for cam_data in data.get('cameras', []):
            cam = VirtualCamera(
                camera_id=cam_data['id'],
                zone_id=cam_data['zone'],
                source_path=cam_data['source'],
                location={'lat': cam_data['lat'], 'lng': cam_data['lng']}
            )
            # Initialize status from JSON
            if cam_data.get('status') == 'active':
                cam.start_stream()
            
            self.cameras[cam_data['id']] = cam

    def handle_detection(self, camera_id: str, label: str, confidence: float, bbox=None, frame=None):
        """
        bbox: [x1, y1, x2, y2]
        frame: numpy array (BGR image)
        """
        logger.info(f"Detection at {camera_id}: {label} ({confidence:.2f})")
        
        # 1. Feature Extraction (Real ReID)
        current_embedding = None
        if bbox is not None and frame is not None:
            x1, y1, x2, y2 = bbox
            # Ensure valid crop
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                current_embedding = self.reid.extract(crop)
        
        # 2. Logic: ID Assignment via Cosine Similarity
        subject_id = None
        matched_event = None
        
        # Search active events for a match
        if current_embedding is not None:
            best_score = 0.0
            
            for event in self.active_events.values():
                if event.status != 'active': continue
                
                # Check against target embedding
                target_emb = event.metadata.get('target_embedding')
                if target_emb is not None:
                    score = self.reid.cosine_similarity(current_embedding, target_emb)
                    logger.info(f"🔍 ReID Compare vs {event.metadata.get('subject_id')}: Score {score:.3f}")
                    if score > 0.65: # Lowered Threshold for testing
                        if score > best_score:
                            best_score = score
                            subject_id = event.metadata.get('subject_id')
                            matched_event = event
                            
            if subject_id:
                logger.info(f"🧬 ReID MATCH: {subject_id} (Score: {best_score:.3f})")
        
        # Fallback for new suspect (if no match or first time)
        if not subject_id:
             subject_id = f"SUSPECT-{int(time.time()) % 1000}"

        
        existing_event = matched_event if matched_event else self._find_event_for_camera(camera_id)
        
        if existing_event:
            # If we matched visually, ensure we update the correct event even if finding by camera failed
            logger.info(f"Updating existing event {existing_event.event_id}")
            
            self._activate_spatial_neighbors(existing_event, camera_id)
        else:
            new_event = EventContextCluster(
                origin_sensor=camera_id,
                event_type=EventType.VIOLENCE if label == 'Violence' else EventType.SUSPICIOUS_OBJECT
            )
            new_event.add_sensor(camera_id)
            new_event.confidence_score = confidence
            
            # Store ReID Data
            new_event.metadata['subject_id'] = subject_id
            if current_embedding is not None and label in ['Violence', 'Person']:
                 # Set the anchor embedding for this suspect
                 new_event.metadata['target_embedding'] = current_embedding
            
            self.active_events[new_event.event_id] = new_event
            logger.info(f"💥 NEW EVENT {new_event.event_id} | Subject: {subject_id}")
            
            # TRIGGER SPATIAL ORCHESTRATION
            self._activate_spatial_neighbors(new_event, camera_id)

    def _find_event_for_camera(self, camera_id: str) -> Optional[EventContextCluster]:
        for event in self.active_events.values():
            if camera_id in event.active_sensors and event.status == "active":
                return event
        return None

    def _activate_spatial_neighbors(self, event: EventContextCluster, current_cam_id: str):
        """
        Dynamic Spatial Discovery.
        Find cameras within X distance (approx .002 degrees ~ 200m)
        """
        current_cam = self.cameras.get(current_cam_id)
        if not current_cam: return

        origin_lat = current_cam.location['lat']
        origin_lng = current_cam.location['lng']
        
        neighbors = []
        RADIUS_DEG = 0.005 # Approx 500m

        # 1. Calculate Neighbors
        for cam_id, cam in self.cameras.items():
            if cam_id == current_cam_id: continue
            
            d_lat = cam.location['lat'] - origin_lat
            d_lng = cam.location['lng'] - origin_lng
            dist = (d_lat**2 + d_lng**2)**0.5
            
            if dist < RADIUS_DEG:
                neighbors.append(cam_id)

        # 2. Filter Active Neighbors
        new_neighbors = []
        for cam_id in neighbors:
            if cam_id not in event.active_sensors:
                new_neighbors.append(cam_id)
        
        # 3. Check & Log
        if not new_neighbors:
            return

        logger.info(f"📡 Neighbor Scan: Found NEW cameras near {current_cam_id} ({origin_lat}, {origin_lng}): {new_neighbors}")
        
        # 4. Activate
        for neighbor_id in new_neighbors:
            event.add_sensor(neighbor_id)
            if neighbor_id in self.cameras:
                self.cameras[neighbor_id].start_stream()

    def get_active_feeds(self):
        active_cams = []
        for event in self.active_events.values():
            active_cams.extend(event.active_sensors)
        return list(set(active_cams))
