"""Pose estimation for basketball action recognition.

Keypoint indices (COCO 17-point):
  0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
  5: left_shoulder, 6: right_shoulder
  7: left_elbow, 8: right_elbow
  9: left_wrist, 10: right_wrist
  11: left_hip, 12: right_hip
  13: left_knee, 14: right_knee
  15: left_ankle, 16: right_ankle
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from ultralytics import YOLO


@dataclass
class PoseConfig:
    model_name: str = "yolo11n-pose.pt"
    device: str = "cpu"
    confidence_threshold: float = 0.5


@dataclass
class PoseResult:
    wrists_above_shoulders: bool = False
    wrists_above_nose: bool = False
    shooting_pose: bool = False
    max_wrist_y: float = 0.0
    nose_y: float = 0.0
    shoulder_y: float = 0.0
    has_person: bool = False


class PoseEstimator:
    def __init__(self, config: PoseConfig | None = None):
        self.config = config or PoseConfig()
        import os, sys
        model_path = self._resolve_path(self.config.model_name)
        print(f"[pose] Loading {model_path}...", file=sys.stderr, flush=True)
        self.model = YOLO(model_path)
        print(f"[pose] Model ready", file=sys.stderr, flush=True)
        self._frame_count = 0

    @staticmethod
    def _resolve_path(name: str) -> str:
        import os
        for d in ["/tmp", "/Users/mac/hoopvision",
                  os.path.expanduser("~/.cache/ultralytics/weights"),
                  os.getcwd()]:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return os.path.abspath(p)
        return name

    def estimate(self, frame: np.ndarray, person_bboxes: list) -> dict[int, PoseResult]:
        """Return pose results keyed by track_id (matched to persons by bbox IoU)."""
        self._frame_count += 1
        results: dict[int, PoseResult] = {}

        if not person_bboxes:
            return results

        try:
            yolo_results = self.model(frame, device="cpu", classes=[0],
                                       conf=self.config.confidence_threshold, verbose=False)
        except Exception:
            return results

        if yolo_results[0].keypoints is None:
            return results

        kpts = yolo_results[0].keypoints
        for i in range(len(kpts)):
            xy = kpts.xy[i].cpu().numpy()
            confs = kpts.conf[i].cpu().numpy() if kpts.conf is not None else np.ones(17)

            if np.mean(confs) < 0.3:
                continue

            nose_y = xy[0][1] if confs[0] > 0.3 else float('inf')
            l_shoulder_y = xy[5][1] if confs[5] > 0.3 else float('inf')
            r_shoulder_y = xy[6][1] if confs[6] > 0.3 else float('inf')
            shoulder_y = min(l_shoulder_y, r_shoulder_y)

            l_wrist_y = xy[9][1] if confs[9] > 0.3 else float('inf')
            r_wrist_y = xy[10][1] if confs[10] > 0.3 else float('inf')
            max_wrist_y = min(l_wrist_y, r_wrist_y)  # smaller y = higher in image

            wrists_above_shoulders = max_wrist_y < shoulder_y - 20
            wrists_above_nose = max_wrist_y < nose_y - 15 if nose_y != float('inf') else False
            shooting_pose = wrists_above_shoulders and wrists_above_nose

            pr = PoseResult(
                wrists_above_shoulders=wrists_above_shoulders,
                wrists_above_nose=wrists_above_nose,
                shooting_pose=shooting_pose,
                max_wrist_y=max_wrist_y,
                nose_y=nose_y if nose_y != float('inf') else 0,
                shoulder_y=shoulder_y if shoulder_y != float('inf') else 0,
                has_person=True,
            )

            # Match to person bbox by closest center
            kx = np.mean(xy[:, 0])
            ky = np.mean(xy[:, 1])
            best_tid = -1
            best_dist = float('inf')
            for tid, bbox in person_bboxes:
                bx, by = bbox.center
                d = (kx - bx)**2 + (ky - by)**2
                if d < best_dist:
                    best_dist = d
                    best_tid = tid
            if best_tid >= 0:
                results[best_tid] = pr

        return results

    def reset(self):
        self._frame_count = 0
