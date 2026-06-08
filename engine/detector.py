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
    """Optical-flow tracker for fast-moving basketballs."""

    def __init__(self, max_points: int = 20):
        self._prev_gray = None
        self._points: list[tuple[float, float]] = []
        self._ages: list[int] = []
        self._max_points = max_points
        self._frames_since_scan = 0

    def track(self, frame: np.ndarray, hoop_center: tuple[float, float],
              hoop_radius: float) -> list[dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frames_since_scan += 1
        results: list[dict] = []

        if self._prev_gray is not None and self._points:
            pts = np.array(self._points, dtype=np.float32).reshape(-1, 1, 2)
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, pts, None,
                winSize=(15, 15), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            if new_pts is not None:
                valid_pts: list[tuple[float, float]] = []
                valid_ages: list[int] = []
                for i, (pt, st) in enumerate(zip(new_pts, status)):
                    if st[0] == 0:
                        continue
                    x, y = pt.ravel()
                    if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                        continue
                    age = self._ages[i] + 1
                    valid_pts.append((float(x), float(y)))
                    valid_ages.append(age)
                    if age > 2:
                        dist = np.sqrt((x - hoop_center[0])**2 + (y - hoop_center[1])**2)
                        results.append({
                            "x": float(x), "y": float(y), "age": age,
                            "near_hoop": dist < hoop_radius * 1.6,
                        })
                self._points = valid_pts
                self._ages = valid_ages

        # Periodic scan for new candidates using frame difference
        if self._frames_since_scan >= 5:
            self._frames_since_scan = 0
            candidates = self._scan_candidates(self._prev_gray, gray, hoop_center)
            for cx, cy in candidates:
                if len(self._points) >= self._max_points:
                    break
                if not self._points or all(
                        np.sqrt((cx - px)**2 + (cy - py)**2) > 15
                        for px, py in self._points[-8:]):
                    self._points.append((cx, cy))
                    self._ages.append(0)

        # Prune stale points (age > 60 frames = 4s at 15fps)
        if len(self._points) > self._max_points * 1.5:
            keep = [(p, a) for p, a in zip(self._points, self._ages) if a < 60]
            if keep:
                self._points, self._ages = zip(*keep)
                self._points, self._ages = list(self._points), list(self._ages)
            else:
                self._points, self._ages = [], []

        self._prev_gray = gray
        return results

    def _scan_candidates(self, prev_gray, gray, hoop_center) -> list[tuple[float, float]]:
        """Find small fast-moving bright blobs using frame differencing."""
        if prev_gray is None:
            return []
        h, w = gray.shape
        diff = cv2.absdiff(gray, prev_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 200:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    dist = np.sqrt((cx - hoop_center[0])**2 + (cy - hoop_center[1])**2)
                    if dist < hoop_center[1] * 3:
                        candidates.append((float(cx), float(cy)))
        candidates.sort(key=lambda c: c[1])
        return candidates[:8]

    def reset(self):
        self._prev_gray = None
        self._points = []
        self._ages = []
        self._frames_since_scan = 0


class Detector:
    def __init__(self, config: DetectorConfig | None = None, model=None):
        self.config = config or DetectorConfig()
        if model is not None:
            self.model = model
        else:
            import os, sys
            path = self._resolve_model_path(self.config.model_name)
            print(f"[detector] Loading {os.path.basename(path)}...", file=sys.stderr, flush=True)
            self.model = YOLO(path)
            print(f"[detector] ready", file=sys.stderr, flush=True)
        self._next_track_id = 0
        self._track_cache: dict[int, int] = {}
        self._frame_count = 0
        self.ball_tracker = BallTracker()

    @staticmethod
    def _resolve_model_path(name: str) -> str:
        import os
        for d in ["/Users/mac/hoopvision", "/tmp",
                  os.path.expanduser("~/.cache/ultralytics/weights"),
                  os.getcwd()]:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return os.path.abspath(p)
        return name

    def detect(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> list[Detection]:
        self._frame_count += 1
        try:
            results = self.model.track(
                frame,
                conf=min(self.config.confidence_threshold, self.config.ball_confidence_threshold),
                iou=self.config.iou_threshold,
                device="cpu", persist=True,
                classes=list(self.config.classes.keys()),
                verbose=False,
            )
        except Exception:
            return []

        if results[0].boxes is None:
            return []

        boxes = results[0].boxes
        dets: list[Detection] = []
        pc = bc = 0
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            cls_name = self.config.classes.get(cls_id, "unknown")
            if cls_name == "ball" and conf < self.config.ball_confidence_threshold:
                continue
            if cls_name == "person" and conf < self.config.confidence_threshold:
                continue
            if cls_name == "person":
                pc += 1
            elif cls_name == "ball":
                bc += 1
            xyxy = boxes.xyxy[i].cpu().numpy()
            tid_raw = int(boxes.id[i].item()) if boxes.id is not None else -1
            if tid_raw not in self._track_cache:
                self._next_track_id += 1
                self._track_cache[tid_raw] = self._next_track_id
            dets.append(Detection(
                frame_idx=frame_idx, timestamp=timestamp,
                bbox=BoundingBox(x1=float(xyxy[0]), y1=float(xyxy[1]),
                                 x2=float(xyxy[2]), y2=float(xyxy[3])),
                class_=cls_name, track_id=self._track_cache[tid_raw],
                confidence=conf))

        if self._frame_count % 15 == 0:
            import sys
            print(f"[det] f{self._frame_count}: {pc}p {bc}b", file=sys.stderr, flush=True)
        return dets

    def reset_tracking(self):
        self._next_track_id = 0
        self._track_cache.clear()
        self._frame_count = 0
        self.ball_tracker.reset()
