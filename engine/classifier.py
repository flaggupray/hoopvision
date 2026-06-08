"""Basketball action classifier.

Event detection fuses 3 signals with a state machine:
  1. YOLO ball detection proximity to estimated hoop zone
  2. Optical flow trajectory points near hoop zone
  3. Player shooting pose (wrists above head) near hoop zone

Shot tracking state machine:
  idle -> ball_approaches_hoop -> shot_attempt -> ball_enters_hoop(make) / ball_bounces(miss) -> rebound_possible -> idle
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from engine.schema import Detection, Player, GameEvent, EventType
from engine.pose import PoseResult


@dataclass
class ClassifierConfig:
    frame_width: int = 1280
    frame_height: int = 720
    hoop_x_ratio: float = 0.50
    hoop_y_ratio: float = 0.22
    hoop_zone_radius: float = 150.0
    three_point_y_ratio: float = 0.40
    event_cooldown_seconds: float = 1.5


class ScoreTracker:
    """Tracks game score from events."""

    def __init__(self):
        self.home = 0
        self.away = 0
        self._last_shooter_team: Optional[str] = None

    def update(self, event: GameEvent):
        pts = 0
        if event.event_type == EventType.THREE_POINTER_MADE:
            pts = 3
        elif event.event_type == EventType.TWO_POINTER_MADE:
            pts = 2
        elif event.event_type == EventType.FREE_THROW_MADE:
            pts = 1

        if pts == 0:
            return

        # Simple heuristic: alternating teams for scoring
        if self._last_shooter_team == "home":
            self.away += pts
            self._last_shooter_team = "away"
        else:
            self.home += pts
            self._last_shooter_team = "home"

        event.score_before = f"{self.home - pts if self._last_shooter_team == 'home' else self.home}:{self.away - pts if self._last_shooter_team == 'away' else self.away}"
        event.score_after = f"{self.home}:{self.away}"

    def reset(self):
        self.home = 0
        self.away = 0
        self._last_shooter_team = None


class ActionClassifier:
    def __init__(self, config: ClassifierConfig | None = None):
        self.config = config or ClassifierConfig()
        self._history: list[GameEvent] = []
        self._last_event_time: float = -999
        self._frame_count: int = 0
        self._ball_was_near_hoop: bool = False
        self._ball_entered_hoop: bool = False
        self._shot_fired: bool = False
        self._score = ScoreTracker()

    @property
    def hoop_center(self) -> tuple[float, float]:
        return (self.config.frame_width * self.config.hoop_x_ratio,
                self.config.frame_height * self.config.hoop_y_ratio)

    def classify_frame(
        self,
        detections: list[Detection],
        players: list[Player],
        pose_results: dict[int, PoseResult],
        flow_balls: list[dict],
    ) -> list[GameEvent]:
        self._frame_count += 1
        if not detections:
            return []

        events: list[GameEvent] = []
        balls = [d for d in detections if d.class_ == "ball"]
        persons = [d for d in detections if d.class_ == "person"]
        hoop_x, hoop_y = self.hoop_center
        has_ball = len(balls) > 0
        ts = detections[0].timestamp

        # --- Determine if ball is near hoop ---
        ball_near_hoop = False
        ball_in_hoop = False
        ball_y = hoop_y

        if has_ball:
            ball = balls[-1]
            bx, by = ball.bbox.center
            ball_y = by
            dist = np.sqrt((bx - hoop_x) ** 2 + (by - hoop_y) ** 2)
            ball_near_hoop = dist < self.config.hoop_zone_radius
            ball_in_hoop = dist < 30  # very close = went through

        # Flow ball check
        flow_near = any(fb.get("near_hoop", False) for fb in flow_balls)

        # Pose check: any player near hoop in shooting pose?
        pose_shot = False
        for p in persons:
            pcx, pcy = p.bbox.center
            if np.sqrt((pcx - hoop_x) ** 2 + (pcy - hoop_y) ** 2) < self.config.hoop_zone_radius:
                if pose_results.get(p.track_id, PoseResult()).shooting_pose:
                    pose_shot = True
                    break

        # --- State machine ---
        shot_signal = ball_near_hoop or flow_near or pose_shot

        # SHOT: signal rises when previously not near hoop
        if shot_signal and not self._ball_was_near_hoop and self._can_fire(ts):
            self._shot_fired = True
            ev = self._make_shot(ts, ball_y, balls, persons, players, pose_results)
            if ev:
                events.append(ev)

        # MADE/MISSED: ball enters tight hoop zone or bounces away
        if self._shot_fired and has_ball:
            if ball_in_hoop and not self._ball_entered_hoop:
                self._ball_entered_hoop = True
            if not ball_near_hoop and self._ball_was_near_hoop:
                # Ball left hoop zone without entering tight zone = miss
                if not self._ball_entered_hoop and self._can_fire(ts):
                    ev = self._make_miss(ts, ball_y, persons, players)
                    if ev:
                        events.append(ev)
                        self._shot_fired = False
                        self._ball_entered_hoop = False

        # REBOUND: after shot, multiple players near hoop
        if self._shot_fired and not shot_signal and len(persons) >= 2:
            near_count = sum(1 for p in persons if
                             np.sqrt((p.bbox.center[0] - hoop_x) ** 2 +
                                     (p.bbox.center[1] - hoop_y) ** 2) < self.config.hoop_zone_radius * 1.3)
            if near_count >= 2 and self._can_fire(ts):
                ev = self._make_rebound(ts, persons, players)
                if ev:
                    events.append(ev)
                    self._shot_fired = False
                    self._ball_entered_hoop = False

        self._ball_was_near_hoop = shot_signal
        return events

    def _make_shot(self, ts: float, ball_y: float, balls: list[Detection],
                   persons: list[Detection], players: list[Player],
                   pose_results: dict[int, PoseResult]) -> Optional[GameEvent]:
        three_line = self.config.frame_height * self.config.three_point_y_ratio
        ev_type = EventType.THREE_POINTER_MADE if ball_y < three_line else EventType.TWO_POINTER_MADE

        hoop_x, hoop_y = self.hoop_center
        pid = -1
        min_d = float('inf')
        for p in persons:
            d = np.sqrt((p.bbox.center[0] - hoop_x) ** 2 + (p.bbox.center[1] - hoop_y) ** 2)
            if d < min_d:
                min_d = d
                pid = p.track_id

        pl = next((p for p in players if p.track_id == pid), None)
        self._last_event_time = ts
        ev = GameEvent(
            time_str=self._fmt(ts), quarter=self._qtr(ts),
            event_type=ev_type, player_track_id=pid,
            player_number=pl.jersey_number if pl else None,
            distance=round(min_d / 10, 1),
        )
        self._score.update(ev)
        self._history.append(ev)
        return ev

    def _make_miss(self, ts: float, ball_y: float, persons: list[Detection],
                   players: list[Player]) -> Optional[GameEvent]:
        three_line = self.config.frame_height * self.config.three_point_y_ratio

        # Convert the made event to missed
        if self._history:
            last = self._history[-1]
            if last.event_type.is_score():
                if last.event_type == EventType.THREE_POINTER_MADE:
                    last.event_type = EventType.THREE_POINTER_MISSED
                elif last.event_type == EventType.TWO_POINTER_MADE:
                    last.event_type = EventType.TWO_POINTER_MISSED
                return None  # Modified existing, don't add new

        ev_type = EventType.THREE_POINTER_MISSED if ball_y < three_line else EventType.TWO_POINTER_MISSED
        self._last_event_time = ts
        ev = GameEvent(time_str=self._fmt(ts), quarter=self._qtr(ts), event_type=ev_type,
                       player_track_id=-1, distance=0.0)
        self._history.append(ev)
        return ev

    def _make_rebound(self, ts: float, persons: list[Detection],
                      players: list[Player]) -> Optional[GameEvent]:
        hoop_x, hoop_y = self.hoop_center
        nearest = min(persons, key=lambda p:
        (p.bbox.center[0] - hoop_x) ** 2 + (p.bbox.center[1] - hoop_y) ** 2)
        pl = next((p for p in players if p.track_id == nearest.track_id), None)

        # Check if last shot was missed → offensive rebound more likely
        last_missed = (self._history and self._history[-1].event_type.is_miss())
        ev_type = EventType.OFFENSIVE_REBOUND if last_missed else EventType.DEFENSIVE_REBOUND

        self._last_event_time = ts
        ev = GameEvent(time_str=self._fmt(ts), quarter=self._qtr(ts),
                       event_type=ev_type, player_track_id=nearest.track_id,
                       player_number=pl.jersey_number if pl else None,
                       distance=0.0)
        self._history.append(ev)
        return ev

    def _can_fire(self, ts: float) -> bool:
        return (ts - self._last_event_time) >= self.config.event_cooldown_seconds

    def _fmt(self, s: float) -> str:
        m, sec = divmod(int(s), 60)
        return f"{m:02d}:{sec:02d}"

    def _qtr(self, s: float) -> int:
        if s < 720: return 1
        if s < 1440: return 2
        if s < 2160: return 3
        return 4

    def get_history(self) -> list[GameEvent]:
        return list(self._history)

    def reset(self):
        self._history = []
        self._last_event_time = -999
        self._frame_count = 0
        self._ball_was_near_hoop = False
        self._ball_entered_hoop = False
        self._shot_fired = False
        self._score.reset()
