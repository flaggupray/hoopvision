import pytest
from engine.classifier import ActionClassifier, ClassifierConfig
from engine.schema import Detection, BoundingBox, EventType, Player
from engine.pose import PoseResult


class TestClassifierConfig:
    def test_defaults(self):
        config = ClassifierConfig()
        assert config.hoop_zone_radius == 150.0
        assert config.event_cooldown_seconds == 3.0

    def test_hoop_center(self):
        config = ClassifierConfig(frame_width=1280, frame_height=720)
        clf = ActionClassifier(config)
        cx, cy = clf.hoop_center
        assert cx == 640.0
        assert cy == pytest.approx(158.4, rel=0.1)


class TestActionClassifier:
    def test_create(self):
        clf = ActionClassifier(ClassifierConfig())
        assert clf is not None

    def test_ball_near_hoop_triggers_shot(self):
        config = ClassifierConfig(
            frame_width=1280, frame_height=720,
            hoop_x_ratio=0.5, hoop_y_ratio=0.25,
            hoop_zone_radius=150.0,
            event_cooldown_seconds=0.0,
        )
        clf = ActionClassifier(config)
        d = [Detection(frame_idx=0, timestamp=0.0,
                       bbox=BoundingBox(x1=630, y1=150, x2=650, y2=170),
                       class_="ball", track_id=1, confidence=0.8),
             Detection(frame_idx=0, timestamp=0.0,
                       bbox=BoundingBox(x1=600, y1=100, x2=700, y2=300),
                       class_="person", track_id=2, confidence=0.9)]
        events = clf.classify_frame(d, [], {}, [])
        assert len(events) == 1
        assert events[0].event_type in (EventType.TWO_POINTER_MADE, EventType.THREE_POINTER_MADE)

    def test_shooting_pose_triggers_event(self):
        config = ClassifierConfig(
            frame_width=1280, frame_height=720,
            hoop_x_ratio=0.5, hoop_y_ratio=0.25,
            hoop_zone_radius=150.0,
            event_cooldown_seconds=0.0,
        )
        clf = ActionClassifier(config)
        d = [Detection(frame_idx=0, timestamp=0.0,
                       bbox=BoundingBox(x1=600, y1=150, x2=700, y2=350),
                       class_="person", track_id=1, confidence=0.9)]
        pr = {1: PoseResult(wrists_above_shoulders=True, wrists_above_nose=True,
                            shooting_pose=True, max_wrist_y=50, nose_y=120, shoulder_y=100,
                            has_person=True)}
        events = clf.classify_frame(d, [], pr, [])
        assert len(events) == 1

    def test_no_ball_no_pose_no_event(self):
        clf = ActionClassifier(ClassifierConfig(
            frame_width=1280, frame_height=720, event_cooldown_seconds=0.0))
        d = [Detection(frame_idx=0, timestamp=0.0,
                       bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
                       class_="person", track_id=1, confidence=0.9)]
        events = clf.classify_frame(d, [], {}, [])
        assert len(events) == 0

    def test_cooldown(self):
        config = ClassifierConfig(
            frame_width=1280, frame_height=720,
            hoop_zone_radius=150.0,
            event_cooldown_seconds=5.0,
        )
        clf = ActionClassifier(config)
        d1 = [Detection(frame_idx=0, timestamp=0.0,
                        bbox=BoundingBox(x1=630, y1=150, x2=650, y2=170),
                        class_="ball", track_id=1, confidence=0.8)]
        e1 = clf.classify_frame(d1, [], {}, [])
        assert len(e1) == 1
        d2 = [Detection(frame_idx=1, timestamp=1.0,
                        bbox=BoundingBox(x1=630, y1=150, x2=650, y2=170),
                        class_="ball", track_id=1, confidence=0.8)]
        clf._ball_was_near_hoop = False
        e2 = clf.classify_frame(d2, [], {}, [])
        assert len(e2) == 0

    def test_cooldown_expires(self):
        config = ClassifierConfig(
            frame_width=1280, frame_height=720,
            hoop_zone_radius=150.0,
            event_cooldown_seconds=5.0,
        )
        clf = ActionClassifier(config)
        d1 = [Detection(frame_idx=0, timestamp=0.0,
                        bbox=BoundingBox(x1=630, y1=150, x2=650, y2=170),
                        class_="ball", track_id=1, confidence=0.8)]
        clf.classify_frame(d1, [], {}, [])
        d2 = [Detection(frame_idx=50, timestamp=10.0,
                        bbox=BoundingBox(x1=630, y1=150, x2=650, y2=170),
                        class_="ball", track_id=1, confidence=0.8)]
        clf._ball_was_near_hoop = False
        e2 = clf.classify_frame(d2, [], {}, [])
        assert len(e2) == 1

    def test_empty(self):
        clf = ActionClassifier(ClassifierConfig())
        assert clf.classify_frame([], [], {}, []) == []

    def test_reset(self):
        config = ClassifierConfig(frame_width=1280, frame_height=720,
                                   hoop_zone_radius=150.0, event_cooldown_seconds=0.0)
        clf = ActionClassifier(config)
        d = [Detection(frame_idx=0, timestamp=0.0,
                       bbox=BoundingBox(x1=630, y1=150, x2=650, y2=170),
                       class_="ball", track_id=1, confidence=0.8)]
        clf.classify_frame(d, [], {}, [])
        assert len(clf.get_history()) == 1
        clf.reset()
        assert len(clf.get_history()) == 0
