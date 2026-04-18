import logging
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
from .models import EventContextCluster, EventType
from .virtual_camera import VirtualCamera
from .yolo_orchestrator import YOLOOrchestrator
from .dart_scorer import DARTScorerBank       # NOVELTY 2: DART Score
from .vega import VEGAModule                  # NOVELTY 3: VEGA Module
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Orchestrator")

class OrchestrationEngine:
    def __init__(self, camera_graph_path: str):
        self.active_events: Dict[str, EventContextCluster] = {}
        self.cameras: Dict[str, VirtualCamera] = {}
        self.graph: Dict[str, List[str]] = {}

        # Cache for confirmed suspects (EventID -> Set of TrackIDs)
        self.confirmed_suspects: Dict[str, set] = {}

        # Pending Activations Queue: List of {event, cam_id, trigger_time}
        self.pending_activations: List[dict] = []

        # ReID Temporal Consistency Cache
        self.match_streaks: Dict[tuple, int] = {}  # (cam_id, track_id) -> streak_count

        # Initialize Intelligent Dispatcher (YOLOv26 + OSNet + GRU)
        self.dispatcher = YOLOOrchestrator()

        self._load_network(camera_graph_path)

        # NOVELTY 2: DART Score — per-camera adaptive violence threshold
        self.dart_bank = DARTScorerBank(
            decay_lambda=0.5,
            window_size=10,
            base_threshold=0.50,
            alpha=0.15
        )

        # NOVELTY 3: VEGA Module — graph-signal sensitivity propagation
        vega_graph = self.graph if self.graph else {cam_id: [] for cam_id in self.cameras}
        self.vega = VEGAModule(camera_graph=vega_graph, tau=30.0, max_boost=0.20)

    def _load_network(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)

        self.cameras = {}
        for cam_data in data['cameras']:
            cam = VirtualCamera(
                camera_id=cam_data['id'],
                zone_id=cam_data['zone'],
                source_path=cam_data['source'],
                location={'lat': cam_data['lat'], 'lng': cam_data['lng']}
            )
            if cam_data.get('status') == 'active':
                cam.start_stream()
            self.cameras[cam_data['id']] = cam

        # Load Travel Times
        self.travel_times = data.get('travel_times', {})
        logger.info(f"Loaded {len(self.travel_times)} travel constraints")

    def handle_feedback(self, camera_id: str, track_id: int, feedback_type: str):
        """
        Handle operator feedback from UI.
        feedback_type: 'false_positive'
        """
        logger.info(f"🛑 Feedback received: {feedback_type} for Cam {camera_id} Track {track_id}")

        if feedback_type == 'false_positive':
            for event_id, tracks in self.confirmed_suspects.items():
                if track_id in tracks:
                    tracks.remove(track_id)
                    logger.info(f"🔓 Unlocked/Removed Track {track_id} from Event {event_id}")

            key = (camera_id, track_id)
            if key in self.match_streaks:
                del self.match_streaks[key]

    def _update_suspect_summary(self, event: EventContextCluster, det: dict, camera_id: str):
        """
        Stage 3: Feature Aggregation
        Update running averages for Embedding, Height, and Speed.
        """
        if 'summary' not in event.metadata:
            event.metadata['summary'] = {
                'count': 0,
                'avg_height': 0.0,
                'avg_speed': 0.0,
                'last_centroid': None,
                'last_ts': 0.0,
                'posture': 'Unknown'
            }

        summ = event.metadata['summary']
        count = summ['count']

        new_h = det.get('height_rel', 0)
        if new_h > 0:
            summ['avg_height'] = (summ['avg_height'] * count + new_h) / (count + 1)

        new_centroid = det.get('centroid')
        curr_ts = time.time()

        if new_centroid and summ['last_centroid']:
            dx = new_centroid[0] - summ['last_centroid'][0]
            dy = new_centroid[1] - summ['last_centroid'][1]
            dist = (dx**2 + dy**2)**0.5

            dt = curr_ts - summ['last_ts']
            if 0 < dt < 1.0:
                current_speed = dist / dt
                summ['avg_speed'] = (summ['avg_speed'] * count + current_speed) / (count + 1)

                if summ['avg_speed'] < 20: summ['posture'] = 'Stationary'
                elif summ['avg_speed'] < 100: summ['posture'] = 'Walking'
                else: summ['posture'] = 'Running'

        if new_centroid: summ['last_centroid'] = new_centroid
        summ['last_ts'] = curr_ts
        summ['count'] += 1

        match_emb = det['embedding']
        current_avg = event.metadata.get('target_embedding')

        if current_avg is not None and count > 0:
            new_avg = (current_avg * count + match_emb) / (count + 1)
            new_avg = new_avg / np.linalg.norm(new_avg)
            event.metadata['target_embedding'] = new_avg
        else:
            event.metadata['target_embedding'] = match_emb

    def _is_spatially_feasible(self, source_cam: str, target_cam: str, time_delta: float) -> bool:
        if source_cam == target_cam:
            return True
        if not hasattr(self, 'travel_times'): return True

        key = f"{source_cam}_{target_cam}"
        if key in self.travel_times:
            constraints = self.travel_times[key]
            if time_delta < constraints['min']:
                logger.debug(f"⛔ Rejecting match {source_cam}->{target_cam}: Too fast ({time_delta:.1f}s < {constraints['min']}s)")
                return False
            return True
        return True

    def process_camera_stream(self, camera_id: str):
        """
        Main Loop for a single camera (called by thread/process)
        """
        cam = self.cameras.get(camera_id)
        if not cam: return

        while cam.streaming:
            ret, frame = cam.read_frame()
            if not ret: break

            analysis = self.dispatcher.process_frame(frame, camera_id=camera_id)
            self.process_analysis(camera_id, analysis, frame)

    def process_analysis(self, camera_id: str, analysis: dict, frame):
        """
        Centralized logic for handling dispatcher analysis results.
        Called by process_camera_stream (Real) and visual_simulation (Sim).
        Returns: matched_event_id (str) or None
        """
        if not analysis: return None

        v_score = analysis.get('violence_score', 0.0)
        c_score = analysis.get('crowd_score', 0.0)
        alert_type = analysis.get('alert')
        all_bboxes = analysis.get('all_bboxes', [])

        matched_event_id = None

        detections = analysis.get('detections', [])

        if not detections and 'embedding' in analysis:
            detections = [{'embedding': analysis['embedding'], 'track_id': analysis.get('track_id'), 'bbox': analysis.get('bbox')}]

        # 0. Check confirmed cache first (Fast Path)
        unconfirmed_detections = []
        for det in detections:
            tid = det.get('track_id')
            is_confirmed = False
            if tid is not None:
                for event_id, known_ids in self.confirmed_suspects.items():
                    event = self.active_events.get(event_id)
                    if event and tid in known_ids and event.status == 'active':
                        matched_event_id = event_id
                        is_confirmed = True

                        analysis['matched_details'] = {
                            'event_id': event_id,
                            'track_id': tid,
                            'bbox': det['bbox'],
                            'similarity': 1.0,
                            'streak': 999
                        }
                        break
                unconfirmed_detections.append(det)

        # 0.5 Deduplicate Detections (Spatial)
        unique_detections = []
        for det in unconfirmed_detections:
            is_duplicate = False
            for existing in unique_detections:
                c1 = det['bbox']
                c2 = existing['bbox']
                dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
                if dist < 10:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_detections.append(det)
        unconfirmed_detections = unique_detections

        # 1. ReID Check on Unconfirmed Detections
        if matched_event_id is None and unconfirmed_detections:

            is_crowded = len(detections) > 4
            candidacy_threshold = 0.80

            for event_id, event in self.active_events.items():
                if event.status != 'active': continue

                gallery = event.metadata.get('target_gallery', [])
                if not gallery:
                    if event.metadata.get('target_embedding') is not None:
                        gallery = [event.metadata['target_embedding']]
                if not gallery: continue

                frame_candidates = []

                for idx, det in enumerate(unconfirmed_detections):
                    embedding = det['embedding']
                    if embedding is None: continue

                    max_sim = 0.0
                    emb1 = embedding / np.linalg.norm(embedding)
                    for target_emb in gallery:
                        emb2 = target_emb / np.linalg.norm(target_emb)
                        sim = np.dot(emb1, emb2)
                        if sim > max_sim: max_sim = sim

                    if max_sim > candidacy_threshold:
                        frame_candidates.append((max_sim, det))
                    elif max_sim > 0.85:
                        logger.debug(f"📉 Candidate rejected by candidacy threshold: {max_sim:.2f} < {candidacy_threshold}")

                frame_candidates.sort(key=lambda x: x[0], reverse=True)

                if not frame_candidates:
                    continue

                best_sim = 0.0
                best_det = None
                margin = 0.0

                last_cam = event.metadata.get('last_cam_id', event.origin_sensor)
                last_ts = float(event.metadata.get('last_seen_ts', event.start_time.timestamp()))
                time_delta = time.time() - last_ts

                valid_candidates = []
                for sim, det in frame_candidates[:3]:
                    if self._is_spatially_feasible(last_cam, camera_id, time_delta):
                        valid_candidates.append((sim, det))
                    else:
                        logger.warning(f"⛔ Rejecting match {last_cam}->{camera_id}: Too fast ({time_delta:.1f}s) Sim={sim:.2f}")

                if not valid_candidates:
                    if len(frame_candidates) > 0:
                        logger.debug(f"⚠️ All {len(frame_candidates)} candidates rejected for {camera_id} from {last_cam}")
                    continue

                best_sim, best_det = valid_candidates[0]

                if len(valid_candidates) > 1:
                    margin = best_sim - valid_candidates[1][0]
                elif len(frame_candidates) > 1 and frame_candidates[1][0] < best_sim:
                    margin = best_sim - frame_candidates[1][0]
                else:
                    margin = 1.0

                final_threshold = 0.80

                if is_crowded:
                    if margin < 0.1:
                        final_threshold = 0.85
                        logger.debug(f"👥 Crowd + Low Margin ({margin:.2f}) -> Raising Threshold to {final_threshold}")
                    else:
                        final_threshold = 0.82
                        logger.debug(f"👥 Crowd + High Margin ({margin:.2f}) -> Trusting Distinct Match (Thresh {final_threshold})")

                if best_sim > 0.85 and best_sim < final_threshold:
                    logger.warning(f"📉 Match rejected by Final Threshold: {best_sim:.2f} < {final_threshold} (Margin: {margin:.2f})")

                if best_sim < 0.95 and margin < 0.05:
                    logger.warning(f"⚠️ Ambiguous ReID on {camera_id}: Top1={best_sim:.2f} Margin={margin:.2f}. Suppressing.")
                    continue

                if best_sim < final_threshold:
                    continue

                track_id = best_det.get('track_id')
                track_key = (camera_id, track_id) if track_id is not None else None

                if best_sim > 0.85:
                    event.metadata['last_cam_id'] = camera_id
                    event.metadata['last_seen_ts'] = str(time.time())

                if not hasattr(self, 'match_streaks'):
                    self.match_streaks = {}

                if track_key:
                    self.match_streaks[track_key] = self.match_streaks.get(track_key, 0) + 1
                    current_streak = self.match_streaks[track_key]
                else:
                    current_streak = 1

                logger.info(f"🔍 Best Candidate {camera_id}: Sim={best_sim:.2f} Streak={current_streak} (Event {event_id})")

                if camera_id == event.origin_sensor and best_sim > 0.90:
                    if len(gallery) < 5:
                        gallery.append(best_det['embedding'])
                        event.metadata['target_gallery'] = gallery
                        logger.info(f"📸 Added new view to Gallery {event_id} (Size={len(gallery)})")
                    matched_event_id = event_id

                if current_streak >= 3 or best_sim > 0.95:
                    logger.info(f"🧬 ReID MATCH on {camera_id}: Sim={best_sim:.2f} Streak={current_streak} (Event {event_id})")
                    self._activate_spatial_neighbors(event, camera_id)
                    matched_event_id = event_id

                    if event_id not in self.confirmed_suspects:
                        self.confirmed_suspects[event_id] = set()
                    if track_id is not None:
                        self.confirmed_suspects[event_id].add(track_id)
                        logger.info(f"📌 Locked Track ID {track_id} to Event {event_id}")

                    if best_sim > 0.95 and len(gallery) < 5:
                        gallery.append(best_det['embedding'])
                        event.metadata['target_gallery'] = gallery
                        logger.info(f"📸 Captured View for Gallery {event_id}")

                if matched_event_id == event_id:
                    self._update_suspect_summary(event, best_det, camera_id)
                    analysis['matched_details'] = {
                        'event_id': event_id,
                        'track_id': track_id,
                        'bbox': best_det['bbox'],
                        'similarity': best_sim,
                        'streak': current_streak
                    }

        # ---------------------------------------------------------------
        # NOVELTY 2: DART Score — Adaptive Per-Camera Violence Threshold
        # ---------------------------------------------------------------
        dart_score, dart_thresh, dart_triggered = self.dart_bank.update(camera_id, v_score)
        analysis['dart_score']     = dart_score
        analysis['dart_threshold'] = dart_thresh

        # ---------------------------------------------------------------
        # NOVELTY 3: VEGA — Graph-Boosted Effective Threshold
        # ---------------------------------------------------------------
        effective_threshold = self.vega.get_effective_threshold(camera_id, dart_thresh)
        analysis['vega_effective_threshold'] = effective_threshold
        analysis['vega_boost'] = round(dart_thresh - effective_threshold, 4)

        # DART + VEGA combined alert decision
        if dart_score >= effective_threshold and v_score > 0.30:
            if not alert_type:
                alert_type = "VIOLENCE_DART_VEGA"
                logger.info(
                    f"🎯 DART+VEGA triggered on {camera_id}: "
                    f"dart={dart_score:.3f} >= vega_thresh={effective_threshold:.3f} "
                    f"(boost={dart_thresh - effective_threshold:.3f})"
                )

        # 2. Check for Event Trigger (New or Update)
        if alert_type:
            logger.warning(f"🚨 ALERT on {camera_id}: {alert_type} (V:{v_score:.2f}, C:{c_score:.2f})")

            primary_embedding = detections[0]['embedding'] if detections else None
            self.handle_event_trigger(camera_id, alert_type, v_score, frame, primary_embedding)

        return matched_event_id

    def handle_event_trigger(self, camera_id: str, label: str, confidence: float, frame, embedding=None):
        existing_event = self._find_event_for_camera(camera_id)

        if existing_event:
            logger.info(f"Updating existing event {existing_event.event_id} with new confidence {confidence:.2f}")
            self._activate_spatial_neighbors(existing_event, camera_id)
        else:
            event_type = EventType.VIOLENCE if "VIOLENCE" in label else EventType.SUSPICIOUS_OBJECT

            new_event = EventContextCluster(
                origin_sensor=camera_id,
                event_type=event_type
            )
            new_event.add_sensor(camera_id)
            new_event.confidence_score = confidence

            new_event.metadata['action_plan'] = "Dispatch Police" if confidence > 0.8 else "Verify"
            if embedding is not None:
                new_event.metadata['target_embedding'] = embedding
                new_event.metadata['target_gallery'] = [embedding]

            self.active_events[new_event.event_id] = new_event
            logger.info(f"💥 NEW EVENT {new_event.event_id} | Type: {label}")

            # NOVELTY 3: Register alert in VEGA so neighbors become more sensitive
            self.vega.register_alert(camera_id, confidence)

            # TRIGGER SPATIAL ORCHESTRATION
            self._activate_spatial_neighbors(new_event, camera_id)

    def _find_event_for_camera(self, camera_id: str) -> Optional[EventContextCluster]:
        for event in self.active_events.values():
            if camera_id in event.active_sensors and event.status == "active":
                return event
        return None

    def _activate_spatial_neighbors(self, event: EventContextCluster, current_cam_id: str):
        """
        Dynamic Spatial Discovery with Delay.
        Find neighbors and schedule activation +0.5s later.
        """
        current_cam = self.cameras.get(current_cam_id)
        if not current_cam: return

        origin_lat = current_cam.location['lat']
        origin_lng = current_cam.location['lng']

        neighbors = []
        RADIUS_DEG = 0.004

        if current_cam_id in self.graph and self.graph[current_cam_id]:
            neighbors = self.graph[current_cam_id]
        else:
            for cam_id, cam in self.cameras.items():
                if cam_id == current_cam_id: continue

                d_lat = cam.location['lat'] - origin_lat
                d_lng = cam.location['lng'] - origin_lng
                dist = (d_lat**2 + d_lng**2)**0.5

                if dist < RADIUS_DEG:
                    neighbors.append(cam_id)

        new_neighbors = []
        for cam_id in neighbors:
            if cam_id in event.active_sensors: continue

            is_pending = any(p['cam_id'] == cam_id and p['event'] == event for p in self.pending_activations)
            if is_pending: continue

            new_neighbors.append(cam_id)

        if not new_neighbors: return

        trigger_time = time.time() + 0.5

        for neighbor_id in new_neighbors:
            logger.info(f"⏳ Scheduling activation for {neighbor_id} in 0.5s...")
            self.pending_activations.append({
                'event': event,
                'cam_id': neighbor_id,
                'trigger_time': trigger_time
            })

    def _check_pending_activations(self):
        now = time.time()
        for item in self.pending_activations[:]:
            if now >= item['trigger_time']:
                event = item['event']
                cam_id = item['cam_id']

                if cam_id not in event.active_sensors:
                    event.add_sensor(cam_id)
                    if cam_id in self.cameras:
                        self.cameras[cam_id].start_stream()
                    logger.info(f"🚀 Activating {cam_id} NOW (Delay Complete)")

                self.pending_activations.remove(item)

    def get_active_feeds(self):
        self._check_pending_activations()

        active_cams = []
        for event in self.active_events.values():
            active_cams.extend(event.active_sensors)
        return list(set(active_cams))
