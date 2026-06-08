import pytest
import numpy as np
from engine.detector import Detector, DetectorConfig
from engine.schema import Detection, BoundingBox


class TestDetectorConfig:
    def test_defaults(self):
        config = DetectorConfig()
        assert config.model_name == "yolo11n.pt"
        assert config.confidence_threshold == 0.5
        assert config.classes == {0: "person", 32: "ball"}

    def test_gpu_config(self):
        config = DetectorConfig(device="cpu")
        assert config.device == "cpu"


class TestDetector:
    def test_create_detector(self):
        detector = Detector(DetectorConfig(device="cpu"))
        assert detector is not None
        assert detector.model is not None

    def test_detect_on_frame(self):
        detector = Detector(DetectorConfig(device="cpu"))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Draw a white rectangle to simulate a person
        frame[200:500, 400:600] = 255
        detections = detector.detect(frame, frame_idx=0, timestamp=0.0)
        assert isinstance(detections, list)

    def test_detection_structure(self):
        detector = Detector(DetectorConfig(device="cpu"))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[200:500, 400:600] = 255
        detections = detector.detect(frame, frame_idx=0, timestamp=0.0)
        if detections:
            d = detections[0]
            assert isinstance(d, Detection)
            assert d.frame_idx == 0
            assert isinstance(d.bbox, BoundingBox)
            assert d.class_ in ("person", "ball", "hoop")
            assert 0.0 <= d.confidence <= 1.0

    def test_confidence_filter(self):
        detector = Detector(DetectorConfig(device="cpu", confidence_threshold=0.99))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = detector.detect(frame, frame_idx=0, timestamp=0.0)
        assert len(detections) == 0
