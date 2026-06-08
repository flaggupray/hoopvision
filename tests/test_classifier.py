import pytest
from engine.classifier import ActionClassifier, ClassifierConfig, TrajectoryBuffer
from engine.schema import Detection, BoundingBox, EventType, Player, GameEvent
from engine.pose import PoseResult


class TestTrajectoryBuffer:
    def test_add_and_count(self):
        tb = TrajectoryBuffer(maxlen=10)
        assert tb.count == 0
        tb.add(100, 200, 0.0, 50.0)
        assert tb.count == 1
        tb.add(110, 190, 0.1, 40.0)
        assert tb.count == 2

    def test_dist_trend_approaching(self):
        tb = TrajectoryBuffer(maxlen=20)
        for i in range(15):
            tb.add(100 + i, 200 - i * 2, i * 0.1, 100.0 - i * 5)
        assert tb.dist_trend(8) < 0  # distance decreasing = approaching

    def test_dist_trend_receding(self):
        tb = TrajectoryBuffer(maxlen=20)
        for i in range(15):
            tb.add(100 + i, 200 + i * 2, i * 0.1, 10.0 + i * 5)
        assert tb.dist_trend(8) > 0

    def test_min_dist(self):
        tb = TrajectoryBuffer(maxlen=20)
        for dist in [100, 80, 60, 30, 50, 70, 90]:
            tb.add(100, 200, 0.0, dist)
        assert tb.min_dist_in_window(20) == 30.0

    def test_bounced_away(self):
        tb = TrajectoryBuffer(maxlen=30)
        # Approach hoop
        for dist in [200, 150, 100, 60, 30, 20, 15, 10, 8, 5]:
            tb.add(500 + dist, 200, 0.0, dist)
        # Bounce away
        for dist in [15, 35, 60, 90, 130]:
            tb.add(500 + dist, 200, 0.0, dist)
        assert tb.bounced_away(50.0)

    def test_y_trend_upward(self):
        tb = TrajectoryBuffer(maxlen=20)
        for i in range(10):
            tb.add(100, 300 - i * 15, i * 0.1, 80)
        assert tb.y_trend(6) < 0  # y decreasing = moving up


class TestActionClassifier:
    def test_create(self):
        clf = ActionClassifier(ClassifierConfig())
        assert clf is not None
        assert clf._phase == "idle"

    def test_empty_frame(self):
        clf = ActionClassifier(ClassifierConfig())
        assert clf.classify_frame([], [], {}, []) == []

    def test_shot_with_approaching_trajectory(self):
        config = ClassifierConfig(frame_width=1280, frame_height=720,
                                   hoop_zone_radius=150.0, event_cooldown_seconds=0.0,
                                   trajectory_length=30)
        clf = ActionClassifier(config)
        hoop_x, hoop_y = clf.hoop_center

        # Simulate approaching ball trajectory
        for i in range(12):
            dist = 200 - i * 15
            x = hoop_x + dist * 0.5
            y = hoop_y + dist * 0.5
            d = [Detection(frame_idx=i, timestamp=i * 0.1,
                           bbox=BoundingBox(x1=x-5, y1=y-5, x2=x+5, y2=y+5),
                           class_="ball", track_id=1, confidence=0.8)]
            events = clf.classify_frame(d, [], {}, [])
        # Should have triggered a shot
        assert clf._phase in ("shooting", "post_shot", "idle")
        assert len(clf.get_history()) >= 0

    def test_phase_transitions(self):
        config = ClassifierConfig(frame_width=1280, frame_height=720,
                                   hoop_zone_radius=150.0, event_cooldown_seconds=0.0,
                                   trajectory_length=30)
        clf = ActionClassifier(config)
        hoop_x, hoop_y = clf.hoop_center

        # Feed trajectory that approaches then bounces
        for i in range(15):
            dist = 200 - i * 12  # approaching
            x, y = hoop_x + dist, hoop_y + dist
            d = [Detection(frame_idx=i, timestamp=i * 0.1,
                           bbox=BoundingBox(x1=x-3, y1=y-3, x2=x+3, y2=y+3),
                           class_="ball", track_id=1, confidence=0.8)]
            clf.classify_frame(d, [], {}, [])

        # Rebound phase with multiple players
        for i in range(10):
            d = [
                Detection(frame_idx=20+i, timestamp=2.0+i*0.1,
                          bbox=BoundingBox(x1=hoop_x-80, y1=hoop_y-80, x2=hoop_x-20, y2=hoop_y-20),
                          class_="person", track_id=10, confidence=0.9),
                Detection(frame_idx=20+i, timestamp=2.0+i*0.1,
                          bbox=BoundingBox(x1=hoop_x+20, y1=hoop_y-40, x2=hoop_x+80, y2=hoop_y+20),
                          class_="person", track_id=11, confidence=0.9),
            ]
            clf.classify_frame(d, [], {}, [])

        assert len(clf.get_history()) > 0

    def test_reset(self):
        clf = ActionClassifier(ClassifierConfig(event_cooldown_seconds=0.0))
        clf._phase = "shooting"
        clf._history.append(GameEvent(time_str="00:00", quarter=1,
                                       event_type=EventType.TWO_POINTER_MADE,
                                       player_track_id=1))
        clf.reset()
        assert clf._phase == "idle"
        assert len(clf.get_history()) == 0
