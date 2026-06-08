import pytest
import tempfile
import os
import numpy as np
import cv2
from engine.pipeline import Pipeline, PipelineConfig
from engine.exporter import TimelineExporter
from engine.schema import GameMetadata


def _create_basketball_like_video(path: str, duration_seconds: int = 10, fps: int = 30):
    """Create a synthetic video with moving shapes that vaguely resemble basketball."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 1280, 720
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    n_frames = duration_seconds * fps

    ball_x, ball_y = 640.0, 360.0
    ball_vx, ball_vy = 5.0, -3.0

    for i in range(n_frames):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 240

        ball_x += ball_vx
        ball_y += ball_vy
        if ball_x < 0 or ball_x > w:
            ball_vx *= -1
        if ball_y < 0 or ball_y > h:
            ball_vy *= -1
        cv2.circle(frame, (int(ball_x), int(ball_y)), 15, (0, 100, 255), -1)

        cv2.rectangle(frame, (500, 100), (560, 200), (100, 100, 100), -1)

        for px in [200, 400, 600, 800]:
            cv2.rectangle(frame, (px - 30, 300), (px + 30, 600), (50, 50, 150), -1)

        writer.write(frame)
    writer.release()


class TestIntegration:
    def test_full_pipeline_end_to_end(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        try:
            _create_basketball_like_video(video_path, duration_seconds=5, fps=30)
            config = PipelineConfig(
                sample_rate=10,
                enable_ocr=False,
                enable_classifier=False,
                device="cpu",
                confidence_threshold=0.3,
            )
            pipeline = Pipeline(config)
            timeline = pipeline.run(
                video_path,
                GameMetadata(home_team="Lakers", away_team="Warriors", date="2024-12-25"),
            )

            assert timeline is not None
            assert timeline.metadata.home_team == "Lakers"

            exporter = TimelineExporter()
            html = exporter.export_html(timeline)
            assert "<!DOCTYPE html>" in html

            json_str = exporter.export_json(timeline)
            assert "Lakers" in json_str

            stats = pipeline.stats()
            assert stats["total_frames"] > 0
            assert stats["elapsed_seconds"] > 0

        finally:
            os.unlink(video_path)

    def test_cpu_config(self):
        config = PipelineConfig(device="cpu", enable_ocr=False, enable_classifier=False)
        assert config.device == "cpu"

    def test_gpu_config(self):
        config = PipelineConfig(device="mps", enable_ocr=False, enable_classifier=False)
        assert config.device == "mps"
