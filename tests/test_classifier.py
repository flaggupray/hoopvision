import pytest
from engine.classifier import ActionClassifier, ClassifierConfig
from engine.schema import Detection, BoundingBox, EventType


class TestClassifierConfig:
    def test_defaults(self):
        config = ClassifierConfig()
        assert config.window_size == 16
        assert config.window_stride == 8


class TestActionClassifier:
    def test_create_classifier(self):
        clf = ActionClassifier(ClassifierConfig())
        assert clf is not None
        assert clf.config.window_size == 16

    def test_rule_based_shot_detection(self):
        clf = ActionClassifier(ClassifierConfig())
        detections = [
            Detection(frame_idx=0, timestamp=0.0,
                      bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
                      class_="ball", track_id=1, confidence=0.9),
            Detection(frame_idx=0, timestamp=0.0,
                      bbox=BoundingBox(x1=500, y1=300, x2=550, y2=350),
                      class_="hoop", track_id=99, confidence=0.8),
        ]
        events = clf.classify_window(detections, [])
        assert isinstance(events, list)

    def test_empty_window(self):
        clf = ActionClassifier(ClassifierConfig())
        events = clf.classify_window([], [])
        assert events == []

    def test_no_ball_no_event(self):
        clf = ActionClassifier(ClassifierConfig())
        detections = [
            Detection(frame_idx=0, timestamp=0.0,
                      bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
                      class_="person", track_id=1, confidence=0.9),
        ]
        events = clf.classify_window(detections, [])
        assert len(events) == 0

    def test_event_type_is_valid(self):
        clf = ActionClassifier(ClassifierConfig())
        detections = [
            Detection(frame_idx=0, timestamp=0.0,
                      bbox=BoundingBox(x1=100, y1=100, x2=120, y2=120),
                      class_="ball", track_id=5, confidence=0.9),
            Detection(frame_idx=0, timestamp=0.0,
                      bbox=BoundingBox(x1=600, y1=200, x2=620, y2=220),
                      class_="hoop", track_id=99, confidence=0.8),
            Detection(frame_idx=0, timestamp=0.0,
                      bbox=BoundingBox(x1=0, y1=0, x2=50, y2=100),
                      class_="person", track_id=1, confidence=0.9),
        ]
        events = clf.classify_window(detections, [])
        for e in events:
            assert isinstance(e.event_type, EventType)
