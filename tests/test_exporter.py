import pytest
import os
import tempfile
from engine.exporter import TimelineExporter, ExportConfig
from engine.schema import (
    Timeline, GameMetadata, GameEvent, EventType
)
from engine.narrator import Narrator


def _make_timeline() -> Timeline:
    narrator = Narrator()
    events = []
    for i, ev_type in enumerate([
        EventType.THREE_POINTER_MADE, EventType.DEFENSIVE_REBOUND,
        EventType.TWO_POINTER_MADE, EventType.ASSIST,
    ]):
        ev = GameEvent(
            time_str=f"0{i}:00", quarter=1, event_type=ev_type,
            player_track_id=i, player_number=10 + i,
            distance=7.0, score_before="0:0", score_after="3:0",
            assist_track_id=i + 1, assist_number=11 + i,
        )
        ev.narrative = narrator.narrate(ev)
        events.append(ev)
    return Timeline(
        metadata=GameMetadata(home_team="Lakers", away_team="Warriors", date="2024-12-25"),
        events=events,
    )


class TestTimelineExporter:
    def test_export_html(self):
        timeline = _make_timeline()
        exporter = TimelineExporter(ExportConfig())
        html = exporter.export_html(timeline)
        assert "<!DOCTYPE html>" in html
        assert "Lakers" in html
        assert "Warriors" in html
        assert "event-card" in html

    def test_export_html_file(self):
        timeline = _make_timeline()
        exporter = TimelineExporter(ExportConfig())
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = f.name
        try:
            exporter.export_html_file(timeline, path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content
        finally:
            os.unlink(path)

    def test_export_json(self):
        timeline = _make_timeline()
        exporter = TimelineExporter(ExportConfig())
        json_str = exporter.export_json(timeline)
        assert "Lakers" in json_str
        assert "events" in json_str

    def test_empty_timeline_html(self):
        timeline = Timeline(metadata=GameMetadata())
        exporter = TimelineExporter(ExportConfig())
        html = exporter.export_html(timeline)
        assert "<!DOCTYPE html>" in html
