import pytest
import subprocess
import sys
import tempfile
import os
import numpy as np
import cv2


def _create_test_video(path: str):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, 30, (640, 480))
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


class TestCLI:
    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "engine.cli", "--help"],
            capture_output=True, text=True, cwd="/Users/mac/hoopvision",
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "hoopvision" in result.stdout.lower()

    def test_cli_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "engine.cli", "--version"],
            capture_output=True, text=True, cwd="/Users/mac/hoopvision",
        )
        assert result.returncode == 0
        assert "hoopvision" in result.stdout.lower()

    def test_cli_run_basic(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        output_path = tempfile.mktemp(suffix='.html')
        try:
            _create_test_video(video_path)
            result = subprocess.run([
                sys.executable, "-m", "engine.cli", "run",
                video_path, "--output", output_path,
                "--no-ocr", "--no-classify",
                "--device", "cpu",
                "--sample-rate", "30",
                "--home", "Lakers", "--away", "Warriors",
            ], capture_output=True, text=True, cwd="/Users/mac/hoopvision", timeout=60)
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.exists(output_path)
            with open(output_path) as f:
                content = f.read()
            assert "Lakers" in content
        finally:
            os.unlink(video_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_cli_run_json_output(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        output_path = tempfile.mktemp(suffix='.json')
        try:
            _create_test_video(video_path)
            result = subprocess.run([
                sys.executable, "-m", "engine.cli", "run",
                video_path, "--output", output_path,
                "--format", "json",
                "--no-ocr", "--no-classify",
                "--device", "cpu",
                "--sample-rate", "30",
            ], capture_output=True, text=True, cwd="/Users/mac/hoopvision", timeout=60)
            assert result.returncode == 0
            assert os.path.exists(output_path)
        finally:
            os.unlink(video_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
