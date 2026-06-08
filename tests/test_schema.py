import pytest
from engine.schema import (
    Detection, Keypoint, Player, GameEvent, EventType,
    GameMetadata, Timeline, BoundingBox
)


class TestDetection:
    def test_create_detection(self):
        d = Detection(
            frame_idx=42,
            timestamp=1.5,
            bbox=BoundingBox(x1=100, y1=200, x2=300, y2=500),
            class_="person",
            track_id=7,
            confidence=0.95
        )
        assert d.frame_idx == 42
        assert d.timestamp == 1.5
        assert d.class_ == "person"
        assert d.track_id == 7
        assert d.confidence == 0.95

    def test_bbox_width_height(self):
        b = BoundingBox(x1=100, y1=200, x2=400, y2=500)
        assert b.width == 300
        assert b.height == 300
        assert b.center == (250, 350)


class TestPlayer:
    def test_create_player(self):
        p = Player(
            track_id=3,
            jersey_number=23,
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=200)
        )
        assert p.track_id == 3
        assert p.jersey_number == 23
        assert p.name is None
        assert p.team is None

    def test_player_defaults(self):
        p = Player(track_id=1, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10))
        assert p.jersey_number is None
        assert p.pose == []


class TestGameEvent:
    def test_create_event(self):
        e = GameEvent(
            time_str="01:28",
            quarter=2,
            event_type=EventType.TWO_POINTER_MADE,
            player_track_id=3,
            player_number=23,
            score_before="52:51",
            score_after="54:51"
        )
        assert e.time_str == "01:28"
        assert e.quarter == 2
        assert e.event_type == EventType.TWO_POINTER_MADE
        assert e.time_seconds == 88.0

    def test_time_seconds_parsing(self):
        e = GameEvent(
            time_str="10:30",
            quarter=1,
            event_type=EventType.THREE_POINTER_MADE,
            player_track_id=5,
            player_number=30,
        )
        assert e.time_seconds == 630.0


class TestEventType:
    def test_all_types_present(self):
        types = list(EventType)
        assert len(types) == 12
        assert EventType.THREE_POINTER_MADE in types
        assert EventType.BLOCK in types

    def test_is_score_event(self):
        assert EventType.THREE_POINTER_MADE.is_score()
        assert EventType.TWO_POINTER_MADE.is_score()
        assert not EventType.DEFENSIVE_REBOUND.is_score()

    def test_event_category(self):
        assert EventType.THREE_POINTER_MADE.category() == "shot"
        assert EventType.DEFENSIVE_REBOUND.category() == "rebound"
        assert EventType.ASSIST.category() == "assist"
        assert EventType.BLOCK.category() == "defense"


class TestTimeline:
    def test_create_timeline(self):
        meta = GameMetadata(
            home_team="Lakers",
            away_team="Warriors",
            date="2024-12-25",
            total_duration_seconds=2880.0
        )
        events = [
            GameEvent(
                time_str="00:15", quarter=1,
                event_type=EventType.THREE_POINTER_MADE,
                player_track_id=1, player_number=23,
            )
        ]
        tl = Timeline(metadata=meta, events=events)
        assert len(tl.events) == 1
        assert tl.metadata.home_team == "Lakers"

    def test_timeline_to_dict(self):
        meta = GameMetadata(home_team="A", away_team="B")
        events = [
            GameEvent(
                time_str="00:01", quarter=1,
                event_type=EventType.TWO_POINTER_MADE,
                player_track_id=1, player_number=10,
            )
        ]
        tl = Timeline(metadata=meta, events=events)
        d = tl.model_dump()
        assert d["metadata"]["home_team"] == "A"
        assert len(d["events"]) == 1
