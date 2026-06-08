from dataclasses import dataclass, field
import cv2
import numpy as np
from ultralytics import YOLO
from engine.schema import Detection, BoundingBox


@dataclass
class DetectorConfig:
    model_name: str = "yolo11m.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "auto"
    ball_confidence_threshold: float = 0.08
    classes: dict[int, str] = field(default_factory=lambda: {
        0: "person",
        32: "ball",
    })


class BallTracker:
    """Optical-flow based ball tracker for catching fast-moving small basketballs."""

    def __init__(self):
        self._prev_gray = None
        self._track_points = []
        self._track_ages = []
        self._next_id = 0

    def track(self, frame: np.ndarray, hoop_center: tuple[float, float],
              hoop_radius: float) -> list[dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = []

        if self._prev_gray is not None and self._track_points:
            pts = np.array(self._track_points, dtype=np.float32).reshape(-1, 1, 2)
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, pts, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            if new_pts is not None:
                new_track = []
                new_ages = []
                for i, (pt, st) in enumerate(zip(new_pts, status)):
                    if st[0] == 0:
                        continue
                    x, y = pt.ravel()
                    age = self._track_ages[i] + 1
                    new_track.append((float(x), float(y)))
                    new_ages.append(age)
                    if age > 3:
                        dist = np.sqrt((x - hoop_center[0])**2 + (y - hoop_center[1])**2)
                        results.append({
                            "x": float(x), "y": float(y),
                            "age": age,
                            "near_hoop": dist < hoop_radius * 1.8,
                        })
                self._track_points = new_track
                self._track_ages = new_ages

        # Find new candidate ball points: small bright moving blobs
        if len(self._track_points) < 30:
            candidates = self._find_ball_candidates(gray)
            for cx, cy in candidates:
                if len(self._track_points) >= 30:
                    break
                too_close = any(
                    np.sqrt((cx - px)**2 + (cy - py)**2) < 10
                    for px, py in self._track_points[-10:]
                ) if self._track_points else False
                if not too_close:
                    self._track_points.append((cx, cy))
                    self._track_ages.append(0)

        self._prev_gray = gray
        return results

    def _find_ball_candidates(self, gray: np.ndarray) -> list[tuple[float, float]]:
        h, w = gray.shape
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        candidates = []
        cells = [(0, 0), (w//2, 0), (0, h//2), (w//2, h//2)]
        for ox, oy in cells:
            roi = blur[oy:oy+h//2, ox:ox+w//2]
            if roi.size == 0:
                continue
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi)
            if max_val > 180:
                cx = ox + max_loc[0]
                cy = oy + max_loc[1]
                candidates.append((float(cx), float(cy)))
        return candidates[:5]

    def reset(self):
        self._prev_gray = None
        self._track_points = []
        self._track_ages = []


class Detector:
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        import os, sys
        model_path = self._resolve_model_path(self.config.model_name)
        print(f"[detector] Loading {model_path}...", file=sys.stderr, flush=True)
        self.model = YOLO(model_path)
        print(f"[detector] Model ready", file=sys.stderr, flush=True)
        self._next_track_id = 0
        self._track_cache: dict[int, int] = {}
        self._frame_count = 0
        self.ball_tracker = BallTracker()

    @staticmethod
    def _resolve_model_path(name: str) -> str:
        import os
        candidates = [os.path.join("/tmp", name), os.path.join("/Users/mac/hoopvision", name),
                      os.path.join(os.path.expanduser("~"), ".cache", "ultralytics", "weights", name),
                      os.path.join(os.getcwd(), name), name]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        return name

    def detect(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> list[Detection]:
        self._frame_count += 1
        try:
            results = self.model.track(
                frame,
                conf=min(self.config.confidence_threshold, self.config.ball_confidence_threshold),
                iou=self.config.iou_threshold,
                device="cpu",
                persist=True,
                classes=list(self.config.classes.keys()),
                verbose=False,
            )
        except Exception as e:
            import sys
            print(f"[detector error] {e}", file=sys.stderr, flush=True)
            return []

        detections: list[Detection] = []
        if results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        person_count = 0
        ball_count = 0
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            cls_name = self.config.classes.get(cls_id, "unknown")
            if cls_name == "ball" and conf < self.config.ball_confidence_threshold:
                continue
            if cls_name == "person" and conf < self.config.confidence_threshold:
                continue
            if cls_name == "person":
                person_count += 1
            elif cls_name == "ball":
                ball_count += 1
            xyxy = boxes.xyxy[i].cpu().numpy()
            track_id_raw = int(boxes.id[i].item()) if boxes.id is not None else -1
            if track_id_raw not in self._track_cache:
                self._next_track_id += 1
                self._track_cache[track_id_raw] = self._next_track_id
            detections.append(Detection(
                frame_idx=frame_idx, timestamp=timestamp,
                bbox=BoundingBox(x1=float(xyxy[0]), y1=float(xyxy[1]),
                                 x2=float(xyxy[2]), y2=float(xyxy[3])),
                class_=cls_name,
                track_id=self._track_cache[track_id_raw],
                confidence=conf,
            ))

        if self._frame_count % 30 == 0:
            import sys
            print(f"[detector] frame {self._frame_count}: {person_count} persons, "
                  f"{ball_count} balls", file=sys.stderr, flush=True)

        return detections

    def reset_tracking(self):
        self._next_track_id = 0
        self._track_cache.clear()
        self._frame_count = 0
        self.ball_tracker.reset()
