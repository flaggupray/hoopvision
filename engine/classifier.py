from dataclasses import dataclass
import numpy as np
from engine.schema import Detection, Player, GameEvent, EventType, BoundingBox


@dataclass
class ClassifierConfig:
    window_size: int = 16
    window_stride: int = 8
    hoop_proximity_threshold: float = 150.0
    ball_near_hoop_threshold: float = 40.0


class ActionClassifier:
    def __init__(self, config: ClassifierConfig | None = None):
        self.config = config or ClassifierConfig()
        self._event_history: list[GameEvent] = []
        self._ball_above_rim = False
        self._last_ball_pos: tuple[float, float] | None = None

    def classify_window(
        self,
        detections: list[Detection],
        players: list[Player],
    ) -> list[GameEvent]:
        if not detections:
            return []

        events: list[GameEvent] = []
        balls = [d for d in detections if d.class_ == "ball"]
        hoops = [d for d in detections if d.class_ == "hoop"]
        persons = [d for d in detections if d.class_ == "person"]

        if not balls or not hoops:
            return events

        hoop = hoops[0]
        ball = balls[-1]

        ball_center = ball.bbox.center
        hoop_center = hoop.bbox.center
        dx = ball_center[0] - hoop_center[0]
        dy = ball_center[1] - hoop_center[1]
        dist = np.sqrt(dx * dx + dy * dy)

        near_player = self._find_nearest_person(ball, persons, players)

        if dist < self.config.ball_near_hoop_threshold:
            if not self._ball_above_rim:
                self._ball_above_rim = True
                ev_type = self._classify_shot_type(ball, hoop, near_player)
                if ev_type:
                    event = GameEvent(
                        time_str=self._format_time(ball.timestamp),
                        quarter=self._guess_quarter(ball.timestamp),
                        event_type=ev_type,
                        player_track_id=near_player.track_id if near_player else -1,
                        player_number=near_player.jersey_number if near_player else None,
                        distance=dist,
                    )
                    events.append(event)
                    self._event_history.append(event)
        else:
            self._ball_above_rim = False

        self._last_ball_pos = ball_center
        return events

    def _classify_shot_type(
        self, ball: Detection, hoop: Detection, player: Player | None
    ) -> EventType | None:
        ball_cy = ball.bbox.center[1]
        hoop_cy = hoop.bbox.center[1]
        if ball_cy < hoop_cy - 5:
            return EventType.THREE_POINTER_MADE
        else:
            return EventType.TWO_POINTER_MADE

    def _find_nearest_person(
        self, ball: Detection, persons: list[Detection], players: list[Player]
    ) -> Player | None:
        if not persons:
            return None
        ball_center = ball.bbox.center
        nearest = min(persons, key=lambda p: (
            (p.bbox.center[0] - ball_center[0]) ** 2 +
            (p.bbox.center[1] - ball_center[1]) ** 2
        ))
        for player in players:
            if player.track_id == nearest.track_id:
                return player
        return None

    def _format_time(self, seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def _guess_quarter(self, seconds: float) -> int:
        if seconds < 720:
            return 1
        elif seconds < 1440:
            return 2
        elif seconds < 2160:
            return 3
        return 4

    def get_history(self) -> list[GameEvent]:
        return list(self._event_history)

    def reset(self):
        self._event_history = []
        self._ball_above_rim = False
        self._last_ball_pos = None
