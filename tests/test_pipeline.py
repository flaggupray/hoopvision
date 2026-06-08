import pytest
import tempfile
import os
import numpy as np
import cv2
from engine.pipeline import Pipeline, PipelineConfig
from engine.schema import GameMetadata


def _create_test_video(path: str, fps: int = 30, frames: int = 60):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (640, 480))
    for i in range(frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


class TestPipeline:
    def test_create_pipeline(self):
        config = PipelineConfig(enable_ocr=False, enable_classifier=False)
        pipeline = Pipeline(config)
        assert pipeline is not None

    def test_run_on_test_video(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name
        try:
            _create_test_video(path, fps=30, frames=30)
            config = PipelineConfig(
                sample_rate=30,
                enable_ocr=False,
                enable_classifier=False,
                device="cpu",
            )
            pipeline = Pipeline(config)
            timeline = pipeline.run(path, GameMetadata(home_team="A", away_team="B"))
            assert timeline is not None
            assert timeline.metadata.home_team == "A"
        finally:
            os.unlink(path)

    def test_run_returns_timeline_with_metadata(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name
        try:
            _create_test_video(path, fps=30, frames=10)
            config = PipelineConfig(
                sample_rate=30,
                enable_ocr=False,
                enable_classifier=False,
                device="cpu",
            )
            pipeline = Pipeline(config)
            timeline = pipeline.run(
                path,
                GameMetadata(home_team="Lakers", away_team="Warriors", date="2024-12-25"),
            )
            assert timeline.metadata.home_team == "Lakers"
            assert timeline.metadata.away_team == "Warriors"
        finally:
            os.unlink(path)

    def test_pipeline_stats(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name
        try:
            _create_test_video(path, fps=30, frames=30)
            config = PipelineConfig(
                sample_rate=30,
                enable_ocr=False,
                enable_classifier=False,
                device="cpu",
            )
            pipeline = Pipeline(config)
            pipeline.run(path, GameMetadata())
            stats = pipeline.stats()
            assert "total_frames" in stats
            assert "total_events" in stats
            assert "elapsed_seconds" in stats
        finally:
            os.unlink(path)
