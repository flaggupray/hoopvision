import pytest
from engine.compiler import TimelineCompiler, CompilerConfig
from engine.schema import GameEvent, EventType, Timeline, GameMetadata


def _make_event(time_str: str, quarter: int, ev_type: EventType,
                player_id: int, player_num: int) -> GameEvent:
    return GameEvent(
        time_str=time_str, quarter=quarter, event_type=ev_type,
        player_track_id=player_id, player_number=player_num,
    )


class TestTimelineCompiler:
    def test_empty_events(self):
        compiler = TimelineCompiler(CompilerConfig())
        timeline = compiler.compile([], GameMetadata(home_team="A", away_team="B"))
        assert len(timeline.events) == 0
        assert timeline.metadata.home_team == "A"

    def test_dedup_consecutive_same_type(self):
        compiler = TimelineCompiler(CompilerConfig(dedup_window_seconds=5.0))
        events = [
            _make_event("01:00", 1, EventType.TWO_POINTER_MADE, 1, 23),
            _make_event("01:01", 1, EventType.TWO_POINTER_MADE, 1, 23),
            _make_event("01:02", 1, EventType.TWO_POINTER_MADE, 1, 23),
        ]
        timeline = compiler.compile(events, GameMetadata())
        assert len(timeline.events) == 1

    def test_no_dedup_different_types(self):
        compiler = TimelineCompiler(CompilerConfig(dedup_window_seconds=5.0))
        events = [
            _make_event("01:00", 1, EventType.TWO_POINTER_MADE, 1, 23),
            _make_event("01:01", 1, EventType.DEFENSIVE_REBOUND, 2, 11),
        ]
        timeline = compiler.compile(events, GameMetadata())
        assert len(timeline.events) == 2

    def test_sort_by_time(self):
        compiler = TimelineCompiler(CompilerConfig())
        events = [
            _make_event("02:00", 1, EventType.BLOCK, 3, 5),
            _make_event("01:00", 1, EventType.TWO_POINTER_MADE, 1, 23),
        ]
        timeline = compiler.compile(events, GameMetadata())
        assert timeline.events[0].time_str == "01:00"
        assert timeline.events[1].time_str == "02:00"

    def test_assign_quarter(self):
        compiler = TimelineCompiler(CompilerConfig())
        events = [
            _make_event("00:30", 1, EventType.STEAL, 1, 10),
            _make_event("05:00", 2, EventType.ASSIST, 2, 11),
        ]
        timeline = compiler.compile(events, GameMetadata())
        assert timeline.events[0].quarter == 1
        assert timeline.events[1].quarter == 2

    def test_max_events_limit(self):
        compiler = TimelineCompiler(CompilerConfig(max_events=5))
        events = [_make_event(f"{i:02d}:00", 1, EventType.ASSIST, i, i)
                  for i in range(10)]
        timeline = compiler.compile(events, GameMetadata())
        assert len(timeline.events) == 5
