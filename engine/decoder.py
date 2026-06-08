from dataclasses import dataclass, field
from typing import Iterator
import cv2
import numpy as np


@dataclass
class DecoderConfig:
    sample_rate: int = 6
    target_width: int = 1280
    skip_black_frames: bool = True
    black_threshold: int = 15


class VideoDecoder:
    def __init__(self, video_path: str, config: DecoderConfig | None = None):
        self.video_path = video_path
        self.config = config or DecoderConfig()
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

    def __iter__(self) -> Iterator[np.ndarray]:
        for frame, _ in self.iter_with_timestamps():
            yield frame

    def iter_with_timestamps(self) -> Iterator[tuple[np.ndarray, float]]:
        step = max(1, int(self.fps / self.config.sample_rate))
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                if self.config.skip_black_frames and self._is_mostly_black(frame):
                    frame_idx += 1
                    continue
                timestamp = frame_idx / self.fps
                frame = self._resize(frame)
                yield frame, timestamp
            frame_idx += 1
        self._cap.release()

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        target_w = self.config.target_width
        if w == target_w:
            return frame
        target_h = int(h * target_w / w)
        return cv2.resize(frame, (target_w, target_h))

    def _is_mostly_black(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) < self.config.black_threshold

    def close(self):
        if self._cap.isOpened():
            self._cap.release()

    @property
    def frame_count_estimate(self) -> int:
        step = max(1, int(self.fps / self.config.sample_rate))
        return self.total_frames // step

    def __del__(self):
        self.close()
