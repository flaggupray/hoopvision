from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from ultralytics import YOLO
from engine.schema import Detection, BoundingBox


@dataclass
class DetectorConfig:
    model_name: str = "yolo11n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "auto"
    classes: dict[int, str] = field(default_factory=lambda: {
        0: "person",
        32: "ball",
    })


class Detector:
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.model = YOLO(self.config.model_name)
        self._next_track_id = 0
        self._track_cache: dict[int, int] = {}

    def detect(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> list[Detection]:
        results = self.model.track(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            persist=True,
            classes=list(self.config.classes.keys()),
            verbose=False,
        )
        detections: list[Detection] = []
        if results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()
            track_id_raw = int(boxes.id[i].item()) if boxes.id is not None else -1

            if track_id_raw not in self._track_cache:
                self._next_track_id += 1
                self._track_cache[track_id_raw] = self._next_track_id

            detections.append(Detection(
                frame_idx=frame_idx,
                timestamp=timestamp,
                bbox=BoundingBox(
                    x1=float(xyxy[0]), y1=float(xyxy[1]),
                    x2=float(xyxy[2]), y2=float(xyxy[3]),
                ),
                class_=self.config.classes.get(cls_id, "unknown"),
                track_id=self._track_cache[track_id_raw],
                confidence=conf,
            ))
        return detections

    def reset_tracking(self):
        self._next_track_id = 0
        self._track_cache.clear()
