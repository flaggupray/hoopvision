import pytest
import numpy as np
import cv2
import os
import tempfile
from engine.decoder import VideoDecoder, DecoderConfig


def _create_test_video(path: str, fps: int = 30, frames: int = 90, width: int = 640, height: int = 480):
    """Create a test MP4 video with numbered frames on a gray background."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(frames):
        # Gray background (128) so skip_black_frames doesn't filter them out
        frame = np.full((height, width, 3), 128, dtype=np.uint8)
        cv2.putText(frame, str(i), (width // 2, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


class TestVideoDecoder:
    def test_decode_all_frames(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name
        try:
            _create_test_video(path, fps=30, frames=60)
            decoder = VideoDecoder(path, DecoderConfig(sample_rate=30))
            frames = list(decoder)
            assert len(frames) == 60
            assert all(isinstance(f, np.ndarray) for f in frames)
        finally:
            os.unlink(path)

    def test_sample_rate_subsampling(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name
        try:
            _create_test_video(path, fps=30, frames=90)
            config = DecoderConfig(sample_rate=6)
            decoder = VideoDecoder(path, config)
            frames = list(decoder)
            assert len(frames) == 18  # 90 frames @ 30fps, sample at 6fps = every 5th
        finally:
            os.unlink(path)

    def test_frame_shape(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name
        try:
            _create_test_video(path, fps=30, frames=10, width=1280, height=720)
            decoder = VideoDecoder(path, DecoderConfig(sample_rate=30))
            frames = list(decoder)
            assert frames[0].shape == (720, 1280, 3)
        finally:
            os.unlink(path)

    def test_timestamps(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name
        try:
            _create_test_video(path, fps=30, frames=60)
            decoder = VideoDecoder(path, DecoderConfig(sample_rate=6))
            frames_with_ts = list(decoder.iter_with_timestamps())
            assert len(frames_with_ts) == 12  # 60 frames @ 30fps, sample at 6fps
            assert abs(frames_with_ts[0][1]) < 0.01
        finally:
            os.unlink(path)

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            VideoDecoder("/nonexistent/path.mp4", DecoderConfig())

    def test_config_defaults(self):
        config = DecoderConfig()
        assert config.sample_rate == 6
        assert config.target_width == 1280
