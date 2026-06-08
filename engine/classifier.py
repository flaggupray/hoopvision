"""Basketball action classifier with multi-frame trajectory analysis.

Events are detected by analyzing ball/player trajectories over time,
not single-frame snapshots.

Shot detection: ball approaches hoop from distance, passes through hoop zone.
Made/Missed: determined by whether ball passes through the hoop plane (y decreases
  then increases = went through; y decreases then bounces back up = miss).
Rebound: after a missed shot, players converge near hoop.
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import deque
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
    trajectory_length: int = 45


class TrajectoryBuffer:
    """Stores ball positions over time for multi-frame analysis."""

    def __init__(self, maxlen: int = 45):
        self._buf: deque[tuple[float, float, float, float]] = deque(maxlen=maxlen)

    def add(self, x: float, y: float, ts: float, dist_to_hoop: float):
        self._buf.append((x, y, ts, dist_to_hoop))

    @property
    def count(self) -> int: return len(self._buf)

    def clear(self): self._buf.clear()

    def dist_trend(self, n: int = 10) -> float:
        """Returns trend of distance to hoop: negative = approaching, positive = receding."""
        if len(self._buf) < n:
            return 0.0
        recent = [p[3] for p in list(self._buf)[-n:]]
        older = [p[3] for p in list(self._buf)[:n]]
        return np.mean(recent) - np.mean(older)

    def min_dist_in_window(self, n: int = 30) -> float:
        if not self._buf:
            return float('inf')
        return min(p[3] for p in list(self._buf)[-n:])

    def y_trend(self, n: int = 8) -> float:
        """Returns y trend: negative = moving upward, positive = moving downward."""
        if len(self._buf) < n:
            return 0.0
        recent = [p[1] for p in list(self._buf)[-n:]]
        older = [p[1] for p in list(self._buf)[:min(n, len(self._buf) // 2)]]
        return np.mean(recent) - np.mean(older)

    def y_range_in_window(self, n: int = 20) -> float:
        if len(self._buf) < n:
            return 0.0
        ys = [p[1] for p in list(self._buf)[-n:]]
        return max(ys) - min(ys)

    def entered_zone_from_above(self, hoop_y: float, radius: float, n: int = 10) -> bool:
        """Check if ball entered hoop zone from above (shot going through hoop)."""
        if len(self._buf) < n:
            return False
        recent = list(self._buf)[-n:]
        # Was it above the hoop zone, and now inside?
        above = any(p[1] < hoop_y - radius * 0.3 and p[3] > radius for p in recent[:n // 2])
        inside = any(p[1] >= hoop_y - radius * 0.5 and p[3] < radius * 0.5 for p in recent[-n // 2:])
        return above and inside

    def bounced_away(self, hoop_radius: float, n: int = 15) -> bool:
        """Check if ball bounced away from hoop after approaching close."""
        if len(self._buf) < n:
            return False
        recent = list(self._buf)[-n:]
        mid = list(self._buf)[-n // 2:-n // 4]
        end = list(self._buf)[-n // 4:]
        min_mid = min(p[3] for p in mid) if mid else float('inf')
        avg_end = np.mean([p[3] for p in end]) if end else 0
        return min_mid < hoop_radius * 0.8 and avg_end > min_mid + 30


class ActionClassifier:
    def __init__(self, config: ClassifierConfig | None = None):
        self.config = config or ClassifierConfig()
        self._history: list[GameEvent] = []
        self._last_event_time: float = -999
        self._frame_count: int = 0
        self._trajectory = TrajectoryBuffer(self.config.trajectory_length)
        self._phase: str = "idle"  # idle | approaching | shooting | post_shot
        self._phase_frames: int = 0
        self._shot_player_id: int = -1
        self._home_score: int = 0
        self._away_score: int = 0
        self._scoring_team: int = 0

    @property
    def hoop_center(self) -> tuple[float, float]:
        return (self.config.frame_width * self.config.hoop_x_ratio,
                self.config.frame_height * self.config.hoop_y_ratio)

    def classify_frame(
        self, detections: list[Detection], players: list[Player],
        pose_results: dict[int, PoseResult], flow_balls: list[dict],
    ) -> list[GameEvent]:
        self._frame_count += 1
        if not detections:
            self._trajectory.clear()
            return []

        events: list[GameEvent] = []
        balls = [d for d in detections if d.class_ == "ball"]
        persons = [d for d in detections if d.class_ == "person"]
        hoop_x, hoop_y = self.hoop_center
        ts = detections[0].timestamp

        # ---- Update trajectory ----
        if balls:
            b = balls[-1]
            bx, by = b.bbox.center
            dist = np.sqrt((bx - hoop_x)**2 + (by - hoop_y)**2)
            self._trajectory.add(bx, by, ts, dist)
        elif flow_balls:
            fb = flow_balls[-1]
            dist = np.sqrt((fb["x"] - hoop_x)**2 + (fb["y"] - hoop_y)**2)
            self._trajectory.add(fb["x"], fb["y"], ts, dist)
        self._phase_frames += 1

        # ---- Multi-frame trajectory analysis ----
        dist_trend = self._trajectory.dist_trend(12)
        min_dist = self._trajectory.min_dist_in_window(30)
        y_trend = self._trajectory.y_trend(8)
        approaching = dist_trend < -10 and min_dist < self.config.hoop_zone_radius * 2
        very_close = min_dist < self.config.hoop_zone_radius * 0.5
        near_hoop = min_dist < self.config.hoop_zone_radius
        bounced = self._trajectory.bounced_away(self.config.hoop_zone_radius)
        from_above = self._trajectory.entered_zone_from_above(hoop_y, self.config.hoop_zone_radius)
        trajectory_ready = self._trajectory.count >= 8

        # Pose-assisted: player near hoop with shooting form
        pose_near = False
        pose_pid = -1
        for p in persons:
            d = np.sqrt((p.bbox.center[0] - hoop_x)**2 + (p.bbox.center[1] - hoop_y)**2)
            if d < self.config.hoop_zone_radius:
                pr = pose_results.get(p.track_id)
                if pr and pr.shooting_pose:
                    pose_near = True
                    pose_pid = p.track_id
                    break

        # ---- Phase state machine ----
        if self._phase == "idle":
            if trajectory_ready and (approaching or pose_near):
                self._phase = "approaching"
                self._phase_frames = 0
                self._shot_player_id = pose_pid if pose_near else (
                    self._find_nearest_player(persons, hoop_x, hoop_y))

        elif self._phase == "approaching":
            if very_close or from_above:
                self._phase = "shooting"
                self._phase_frames = 0
                if self._can_fire(ts):
                    ev = self._emit_shot(ts, persons, players, hoop_y)
                    if ev:
                        events.append(ev)
            elif self._phase_frames > 60:
                self._phase = "idle"

        elif self._phase == "shooting":
            if bounced:
                if self._can_fire(ts):
                    ev = self._emit_miss(ts, persons, players, hoop_y)
                    if ev:
                        events.append(ev)
                self._phase = "post_shot"
                self._phase_frames = 0
            elif self._phase_frames > 30:
                # No bounce detected - assume made
                self._phase = "idle"
                self._phase_frames = 0

        elif self._phase == "post_shot":
            # Rebound detection: players converge near hoop after miss
            near_count = sum(1 for p in persons if
                             np.sqrt((p.bbox.center[0] - hoop_x)**2 +
                                     (p.bbox.center[1] - hoop_y)**2) < self.config.hoop_zone_radius * 1.2)
            if near_count >= 2 and self._can_fire(ts):
                ev = self._emit_rebound(ts, persons, players, hoop_x, hoop_y)
                if ev:
                    events.append(ev)
                self._phase = "idle"
            elif self._phase_frames > 45:
                self._phase = "idle"

        return events

    def _find_nearest_player(self, persons: list[Detection], hx: float, hy: float) -> int:
        if not persons:
            return -1
        return min(persons, key=lambda p: (p.bbox.center[0] - hx)**2 + (p.bbox.center[1] - hy)**2).track_id

    def _emit_shot(self, ts: float, persons: list[Detection], players: list[Player],
                   hoop_y: float) -> Optional[GameEvent]:
        hoop_x, hoop_y2 = self.hoop_center
        pid = self._shot_player_id if self._shot_player_id > 0 else self._find_nearest_player(persons, hoop_x, hoop_y2)
        pl = next((p for p in players if p.track_id == pid), None)

        min_dist = self._trajectory.min_dist_in_window(20)
        ball_y = hoop_y - 50
        if self._trajectory.count > 0:
            ball_y = list(self._trajectory._buf)[-1][1]

        three_line = self.config.frame_height * self.config.three_point_y_ratio
        ev_type = EventType.THREE_POINTER_MADE if ball_y < three_line else EventType.TWO_POINTER_MADE

        self._last_event_time = ts
        ev = GameEvent(time_str=self._fmt(ts), quarter=self._qtr(ts),
                       event_type=ev_type, player_track_id=pid,
                       player_number=pl.jersey_number if pl else None,
                       distance=round(min_dist / 10, 1) if min_dist < 9999 else 0.0)
        self._update_score(ev)
        self._history.append(ev)
        return ev

    def _emit_miss(self, ts: float, persons: list[Detection], players: list[Player],
                   hoop_y: float) -> Optional[GameEvent]:
        hoop_x, hoop_y2 = self.hoop_center
        pid = self._shot_player_id if self._shot_player_id > 0 else self._find_nearest_player(persons, hoop_x, hoop_y2)
        pl = next((p for p in players if p.track_id == pid), None)

        three_line = self.config.frame_height * self.config.three_point_y_ratio
        ball_y = hoop_y - 50
        if self._trajectory.count > 0:
            ball_y = list(self._trajectory._buf)[-1][1]
        ev_type = EventType.THREE_POINTER_MISSED if ball_y < three_line else EventType.TWO_POINTER_MISSED

        self._last_event_time = ts
        ev = GameEvent(time_str=self._fmt(ts), quarter=self._qtr(ts),
                       event_type=ev_type, player_track_id=pid,
                       player_number=pl.jersey_number if pl else None,
                       distance=0.0)
        self._history.append(ev)
        return ev

    def _emit_rebound(self, ts: float, persons: list[Detection], players: list[Player],
                      hx: float, hy: float) -> Optional[GameEvent]:
        nearest = min(persons, key=lambda p: (p.bbox.center[0] - hx)**2 + (p.bbox.center[1] - hy)**2)
        pl = next((p for p in players if p.track_id == nearest.track_id), None)

        last_missed = bool(self._history and self._history[-1].event_type.is_miss())
        ev_type = EventType.OFFENSIVE_REBOUND if last_missed else EventType.DEFENSIVE_REBOUND

        self._last_event_time = ts
        ev = GameEvent(time_str=self._fmt(ts), quarter=self._qtr(ts),
                       event_type=ev_type, player_track_id=nearest.track_id,
                       player_number=pl.jersey_number if pl else None, distance=0.0)
        self._history.append(ev)
        return ev

    def _update_score(self, ev: GameEvent):
        pts = 3 if ev.event_type == EventType.THREE_POINTER_MADE else 2
        if self._scoring_team == 0:
            self._home_score += pts
            self._scoring_team = 1
        else:
            self._away_score += pts
            self._scoring_team = 0
        ev.score_before = f"{self._home_score - pts if self._scoring_team == 0 else self._home_score}:{self._away_score - pts if self._scoring_team == 1 else self._away_score}"
        ev.score_after = f"{self._home_score}:{self._away_score}"

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
        self._trajectory.clear()
        self._phase = "idle"
        self._phase_frames = 0
        self._shot_player_id = -1
        self._home_score = 0
        self._away_score = 0
        self._scoring_team = 0
