"""
orchestration/vega.py

VEGA: Violence Escalation Graph Activator

A novel graph-signal propagation mechanism that raises the detection
sensitivity of cameras physically adjacent to a recently-alerted camera.

Core idea: if Camera 3 just detected violence, the probability that
Camera 2 and Camera 4 (its graph neighbors) contain the same suspect
RIGHT NOW is significantly higher than baseline. VEGA formalises this
intuition as a time-decaying, distance-weighted sensitivity boost that
temporarily lowers neighboring cameras' DART thresholds.

Boost formula:
    boost(cam_j, t) = sum_i [ alert_score(cam_i) x (1/hop_dist) x e^(-t/tau) ]

    where:
      cam_i    = any recently-alerted camera
      hop_dist = shortest graph distance from cam_i to cam_j
      t        = seconds since cam_i fired
      tau      = decay time constant (default 30s)
"""

import time
import numpy as np
from collections import defaultdict
from typing import Dict, List


class VEGAModule:
    """
    Violence Escalation Graph Activator.

    Propagates alert signals through the camera topology graph,
    temporarily boosting detection sensitivity in neighboring cameras.
    """

    def __init__(
        self,
        camera_graph: Dict[str, List[str]],
        tau: float = 30.0,
        max_boost: float = 0.20,
        hop_weight_base: float = 0.5
    ):
        self.graph = camera_graph
        self.tau = tau
        self.max_boost = max_boost
        self.hop_weight_base = hop_weight_base

        # Alert log: {cam_id: [(alert_score, timestamp), ...]}
        self.alert_log: Dict[str, list] = defaultdict(list)

        # Precomputed shortest path distances (hop counts)
        self._dist_cache: Dict[tuple, int] = {}
        self._precompute_distances()

    def _precompute_distances(self):
        """BFS to compute all-pairs shortest hop distances."""
        all_cams = list(self.graph.keys())
        for src in all_cams:
            visited = {src: 0}
            queue = [src]
            while queue:
                node = queue.pop(0)
                for neighbor in self.graph.get(node, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[node] + 1
                        queue.append(neighbor)
            for dst, dist in visited.items():
                self._dist_cache[(src, dst)] = dist

    def register_alert(self, camera_id: str, alert_score: float):
        """
        Call this whenever a violence alert fires on a camera.

        Args:
            camera_id   : the camera that fired the alert
            alert_score : confidence score of the violence detection (0-1)
        """
        self.alert_log[camera_id].append((alert_score, time.time()))
        if len(self.alert_log[camera_id]) > 5:
            self.alert_log[camera_id].pop(0)

    def compute_boost(self, target_cam: str) -> float:
        """
        Compute the VEGA sensitivity boost for a target camera.

        Returns a float in [0, max_boost] representing how much
        the target camera's detection threshold should be reduced.
        """
        now = time.time()
        total_boost = 0.0

        for src_cam, alerts in self.alert_log.items():
            if src_cam == target_cam:
                continue

            hop_dist = self._dist_cache.get((src_cam, target_cam), 999)
            if hop_dist == 0 or hop_dist >= 4:
                continue

            # Weight by hop distance: 1 hop=1.0, 2 hops=0.5, 3 hops=0.25
            hop_weight = self.hop_weight_base ** (hop_dist - 1)

            for alert_score, alert_time in alerts:
                t_elapsed = now - alert_time
                if t_elapsed > self.tau * 3:
                    continue

                time_decay = np.exp(-t_elapsed / self.tau)
                boost_contribution = alert_score * hop_weight * time_decay
                total_boost += boost_contribution

        return float(np.clip(total_boost, 0.0, self.max_boost))

    def get_effective_threshold(self, target_cam: str, base_threshold: float) -> float:
        """
        Returns the VEGA-adjusted threshold for a target camera.

        Args:
            target_cam     : camera whose threshold to adjust
            base_threshold : the current DART threshold for this camera

        Returns:
            effective_threshold: base_threshold minus the VEGA boost
        """
        boost = self.compute_boost(target_cam)
        effective = base_threshold - boost
        return float(np.clip(effective, 0.15, base_threshold))

    def get_boost_map(self) -> Dict[str, float]:
        """Return boost values for all cameras (for dashboard display)."""
        return {cam: self.compute_boost(cam) for cam in self.graph.keys()}

    def cleanup_old_alerts(self, max_age: float = 300.0):
        """Remove alerts older than max_age seconds to save memory."""
        now = time.time()
        for cam_id in list(self.alert_log.keys()):
            self.alert_log[cam_id] = [
                (s, t) for s, t in self.alert_log[cam_id]
                if now - t <= max_age
            ]
