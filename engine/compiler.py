from dataclasses import dataclass, field
from engine.schema import GameEvent, EventType, Timeline, GameMetadata


@dataclass
class CompilerConfig:
    dedup_window_seconds: float = 3.0
    max_events: int = 500
    filter_types: set[EventType] = field(default_factory=set)


class TimelineCompiler:
    def __init__(self, config: CompilerConfig | None = None):
        self.config = config or CompilerConfig()

    def compile(
        self,
        raw_events: list[GameEvent],
        metadata: GameMetadata,
    ) -> Timeline:
        if not raw_events:
            return Timeline(metadata=metadata, events=[])

        sorted_events = sorted(raw_events, key=lambda e: (e.quarter, e.time_seconds))

        filtered = [e for e in sorted_events
                    if not self.config.filter_types or e.event_type not in self.config.filter_types]

        deduped = self._deduplicate(filtered)

        if len(deduped) > self.config.max_events:
            deduped = deduped[:self.config.max_events]

        return Timeline(metadata=metadata, events=deduped)

    def _deduplicate(self, events: list[GameEvent]) -> list[GameEvent]:
        if not events:
            return []
        result = [events[0]]
        for ev in events[1:]:
            prev = result[-1]
            same_type = ev.event_type == prev.event_type
            same_player = ev.player_track_id == prev.player_track_id
            time_diff = abs(ev.time_seconds - prev.time_seconds)
            if same_type and same_player and time_diff < self.config.dedup_window_seconds:
                continue
            result.append(ev)
        return result
