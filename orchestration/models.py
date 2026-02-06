from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import uuid
from datetime import datetime

class EventType(Enum):
    VIOLENCE = "violence"
    SUSPICIOUS_OBJECT = "suspicious_object"
    TRAFFIC_INCIDENT = "traffic_incident"

@dataclass
class Location:
    lat: float
    lng: float
    zone_id: str

@dataclass
class SensorState:
    sensor_id: str
    status: str  # "active", "passive", "recording"
    last_trigger: datetime

@dataclass
class EventContextCluster:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.VIOLENCE
    origin_sensor: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    
    # The "Cluster" of relevant sensors
    active_sensors: List[str] = field(default_factory=list)
    
    # Tracking info
    suspect_signature: Optional[List[float]] = None  # ReID embedding
    confidence_score: float = 0.0
    status: str = "active" # active, resolved, escalated
    
    # Auxiliary data (ReID subject IDs, etc)
    metadata: Dict[str, str] = field(default_factory=dict)

    def escalate(self):
        """Increase priority and potentially request more resources (drones)"""
        self.status = "escalated"

    def add_sensor(self, sensor_id: str):
        if sensor_id not in self.active_sensors:
            self.active_sensors.append(sensor_id)
