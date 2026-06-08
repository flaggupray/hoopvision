import pytest
from engine.narrator import Narrator, NarratorConfig
from engine.schema import GameEvent, EventType


def _make_event(time_str: str, ev_type: EventType, player_id: int,
                player_num: int, player_name: str = "") -> GameEvent:
    return GameEvent(
        time_str=time_str, quarter=1, event_type=ev_type,
        player_track_id=player_id, player_number=player_num,
        player_name=player_name or None,
        distance=7.5,
        score_before="52:51", score_after="55:51",
        assist_track_id=3, assist_number=11,
    )


class TestNarrator:
    def test_narrate_three_pointer_made(self):
        narrator = Narrator(NarratorConfig())
        event = _make_event("01:28", EventType.THREE_POINTER_MADE, 1, 30, "Curry")
        text = narrator.narrate(event)
        assert "Curry" in text or "30" in text
        assert len(text) > 5

    def test_narrate_rebound(self):
        narrator = Narrator(NarratorConfig())
        event = _make_event("02:00", EventType.DEFENSIVE_REBOUND, 2, 11, "Davis")
        text = narrator.narrate(event)
        assert "Davis" in text or "11" in text
        assert ("篮板" in text)

    def test_narrate_block(self):
        narrator = Narrator(NarratorConfig())
        event = _make_event("03:00", EventType.BLOCK, 3, 5, "")
        text = narrator.narrate(event)
        assert "5" in text
        assert len(text) > 3

    def test_narrate_all_event_types(self):
        narrator = Narrator(NarratorConfig())
        for ev_type in EventType:
            event = _make_event("01:00", ev_type, 1, 23, "James")
            text = narrator.narrate(event)
            assert isinstance(text, str)
            assert len(text) > 2, f"Empty narrative for {ev_type}"

    def test_narrate_without_name_uses_number(self):
        narrator = Narrator(NarratorConfig())
        event = _make_event("01:00", EventType.TWO_POINTER_MADE, 1, 45, "")
        text = narrator.narrate(event)
        assert "45" in text

    def test_config_language(self):
        config = NarratorConfig(language="zh")
        narrator = Narrator(config)
        event = _make_event("01:00", EventType.STEAL, 1, 23, "James")
        text = narrator.narrate(event)
        assert isinstance(text, str)
