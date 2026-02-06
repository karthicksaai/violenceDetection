import logging
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
from .models import EventContextCluster, EventType
from .virtual_camera import VirtualCamera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Orchestrator")

class OrchestrationEngine:
    def __init__(self, camera_graph_path: str):
        self.active_events: Dict[str, EventContextCluster] = {}
        self.cameras: Dict[str, VirtualCamera] = {}
        self.graph: Dict[str, List[str]] = {}
        
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

    def handle_detection(self, camera_id: str, label: str, confidence: float, bbox=None):
        logger.info(f"Detection at {camera_id}: {label} ({confidence:.2f})")
        
        # 1. ReID / Feature Extraction (Simulated)
        # In a real system, we'd crop the bbox and run a ReID model.
        # Here, we generate a 'Suspect ID' based on the event.
        subject_id = f"SUSPECT-{int(time.time()) % 1000}"
        
        existing_event = self._find_event_for_camera(camera_id)
        
        if existing_event:
            logger.info(f"Updating existing event {existing_event.event_id}")
            # CONTINUOUS EXPANSION:
            # If a downstream camera confirms detection, we must check *its* neighbors too.
            # This enables the 2 -> 3 chain reaction.
            self._activate_spatial_neighbors(existing_event, camera_id)
        else:
            new_event = EventContextCluster(
                origin_sensor=camera_id,
                event_type=EventType.VIOLENCE if label == 'Violence' else EventType.SUSPICIOUS_OBJECT
            )
            new_event.add_sensor(camera_id)
            new_event.confidence_score = confidence
            # Store the ReID info in the event context
            new_event.metadata['subject_id'] = subject_id
            
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
