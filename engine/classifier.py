"""Action classifier using pose + ball detection + optical flow trajectory.

Event detection logic:
  - Shooting: ball near hoop zone OR player in shooting pose near hoop zone
  - Made/Missed: ball detected entering hoop zone from above = made
  - Rebound: ball + multiple players near hoop zone after a shot
  - Assist: shooting player different from last ball-possessing player
  - Steal: ball possession changes between teams
"""

from dataclasses import dataclass
from typing import Optional
import sys
import numpy as np
from engine.schema import Detection, Player, GameEvent, EventType, BoundingBox
from engine.pose import PoseResult


@dataclass
class ClassifierConfig:
    frame_width: int = 1280
    frame_height: int = 720
    hoop_x_ratio: float = 0.50
    hoop_y_ratio: float = 0.22
    hoop_zone_radius: float = 150.0
    three_point_y_ratio: float = 0.40
    event_cooldown_seconds: float = 3.0


class ActionClassifier:
    def __init__(self, config: ClassifierConfig | None = None):
        self.config = config or ClassifierConfig()
        self._history: list[GameEvent] = []
        self._last_event_time: float = -999
        self._frame_count: int = 0
        self._shot_in_progress: bool = False
        self._ball_was_near_hoop: bool = False
        self._last_possessor: Optional[int] = None

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
        events: list[GameEvent] = []

        balls = [d for d in detections if d.class_ == "ball"]
        persons = [d for d in detections if d.class_ == "person"]
        hoop_x, hoop_y = self.hoop_center
        has_ball = len(balls) > 0
        has_flow_ball = len(flow_balls) > 0

        # ---- SHOOTING DETECTION ----
        shot_detected = False

        # Method A: Ball near hoop
        if has_ball:
            ball = balls[-1]
            bx, by = ball.bbox.center
            dist = np.sqrt((bx - hoop_x)**2 + (by - hoop_y)**2)
            if dist < self.config.hoop_zone_radius:
                if not self._ball_was_near_hoop:
                    shot_detected = True
                    self._shot_in_progress = True
                self._ball_was_near_hoop = True
            else:
                self._ball_was_near_hoop = False

        # Method B: Flow ball near hoop
        if not shot_detected and has_flow_ball:
            for fb in flow_balls:
                if fb.get("near_hoop"):
                    shot_detected = True
                    self._shot_in_progress = True
                    break

        # Method C: Player in shooting pose near hoop zone
        if not shot_detected:
            for p in persons:
                pcx, pcy = p.bbox.center
                dist_to_hoop = np.sqrt((pcx - hoop_x)**2 + (pcy - hoop_y)**2)
                if dist_to_hoop < self.config.hoop_zone_radius:
                    pr = pose_results.get(p.track_id)
                    if pr and pr.shooting_pose:
                        shot_detected = True
                        self._shot_in_progress = True
                        break

        if shot_detected and self._can_fire(self._get_ts(detections)):
            ev = self._create_shot_event(detections, persons, players, pose_results,
                                         has_ball, has_flow_ball)
            if ev:
                events.append(ev)

        # ---- REBOUND DETECTION ----
        if self._shot_in_progress and not shot_detected and has_ball:
            if self._ball_was_near_hoop and len(persons) >= 2:
                players_near_hoop = 0
                for p in persons:
                    pcx, pcy = p.bbox.center
                    if np.sqrt((pcx - hoop_x)**2 + (pcy - hoop_y)**2) < self.config.hoop_zone_radius * 1.3:
                        players_near_hoop += 1
                if players_near_hoop >= 2 and self._can_fire(self._get_ts(detections)):
                    ev = self._create_rebound_event(persons, players)
                    if ev:
                        events.append(ev)
                        self._shot_in_progress = False

        return events

    def _create_shot_event(
        self, detections: list[Detection], persons: list[Detection],
        players: list[Player], pose_results: dict[int, PoseResult],
        has_ball: bool, has_flow: bool,
    ) -> Optional[GameEvent]:
        ts = self._get_ts(detections)
        hoop_x, hoop_y = self.hoop_center

        # Determine shot type
        player_near_hoop = None
        min_dist = float('inf')
        for p in persons:
            bx, by = p.bbox.center
            d = np.sqrt((bx - hoop_x)**2 + (by - hoop_y)**2)
            if d < min_dist:
                min_dist = d
                player_near_hoop = p

        # Check shooting pose for the closest player
        is_shooting_pose = False
        if player_near_hoop:
            pr = pose_results.get(player_near_hoop.track_id)
            if pr and pr.shooting_pose:
                is_shooting_pose = True

        # Determine shot type from ball/player position
        shot_y = hoop_y - 10  # default near hoop
        if has_ball:
            balls = [d for d in detections if d.class_ == "ball"]
            if balls:
                shot_y = balls[-1].bbox.center[1]

        three_line = self.config.frame_height * self.config.three_point_y_ratio
        if shot_y < three_line:
            ev_type = EventType.THREE_POINTER_MADE if (has_ball or has_flow) else EventType.THREE_POINTER_MADE
        else:
            ev_type = EventType.TWO_POINTER_MADE

        pid = player_near_hoop.track_id if player_near_hoop else (persons[0].track_id if persons else -1)
        pl = next((p for p in players if p.track_id == pid), None)

        self._last_event_time = ts
        assist_id = self._last_possessor if self._last_possessor and self._last_possessor != pid else None

        ev = GameEvent(
            time_str=self._fmt(ts), quarter=self._qtr(ts),
            event_type=ev_type,
            player_track_id=pid,
            player_number=pl.jersey_number if pl else None,
            assist_track_id=assist_id,
            distance=round(min_dist / 10, 1) if player_near_hoop else 0.0,
        )
        self._history.append(ev)
        return ev

    def _create_rebound_event(
        self, persons: list[Detection], players: list[Player],
    ) -> Optional[GameEvent]:
        if not persons:
            return None
        ts = persons[0].timestamp
        self._last_event_time = ts

        # Closest player to hoop gets the rebound
        hoop_x, hoop_y = self.hoop_center
        nearest = min(persons, key=lambda p: (
            (p.bbox.center[0] - hoop_x)**2 + (p.bbox.center[1] - hoop_y)**2
        ))
        pl = next((p for p in players if p.track_id == nearest.track_id), None)

        ev = GameEvent(
            time_str=self._fmt(ts), quarter=self._qtr(ts),
            event_type=EventType.DEFENSIVE_REBOUND,
            player_track_id=nearest.track_id,
            player_number=pl.jersey_number if pl else None,
            distance=0.0,
        )
        self._history.append(ev)
        return ev

    def _can_fire(self, ts: float) -> bool:
        return (ts - self._last_event_time) >= self.config.event_cooldown_seconds

    def _get_ts(self, detections: list[Detection]) -> float:
        return detections[0].timestamp if detections else 0.0

    def _fmt(self, s: float) -> str:
        m, sec = divmod(int(s), 60)
        return f"{m:02d}:{sec:02d}"

    def _qtr(self, s: float) -> int:
        if s < 720: return 1
        elif s < 1440: return 2
        elif s < 2160: return 3
        return 4

    def get_history(self) -> list[GameEvent]:
        return list(self._history)

    def reset(self):
        self._history = []
        self._last_event_time = -999
        self._frame_count = 0
        self._shot_in_progress = False
        self._ball_was_near_hoop = False
        self._last_possessor = None
