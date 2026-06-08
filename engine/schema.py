from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class EventType(str, Enum):
    THREE_POINTER_MADE = "three_pointer_made"
    THREE_POINTER_MISSED = "three_pointer_missed"
    TWO_POINTER_MADE = "two_pointer_made"
    TWO_POINTER_MISSED = "two_pointer_missed"
    FREE_THROW_MADE = "free_throw_made"
    FREE_THROW_MISSED = "free_throw_missed"
    OFFENSIVE_REBOUND = "offensive_rebound"
    DEFENSIVE_REBOUND = "defensive_rebound"
    ASSIST = "assist"
    STEAL = "steal"
    BLOCK = "block"
    TIMEOUT = "timeout"

    def is_score(self) -> bool:
        return self in {
            EventType.THREE_POINTER_MADE, EventType.TWO_POINTER_MADE,
            EventType.FREE_THROW_MADE
        }

    def is_miss(self) -> bool:
        return self in {
            EventType.THREE_POINTER_MISSED, EventType.TWO_POINTER_MISSED,
            EventType.FREE_THROW_MISSED
        }

    def is_rebound(self) -> bool:
        return self in {
            EventType.OFFENSIVE_REBOUND, EventType.DEFENSIVE_REBOUND
        }

    def category(self) -> str:
        if self in {EventType.THREE_POINTER_MADE, EventType.THREE_POINTER_MISSED,
                     EventType.TWO_POINTER_MADE, EventType.TWO_POINTER_MISSED,
                     EventType.FREE_THROW_MADE, EventType.FREE_THROW_MISSED}:
            return "shot"
        if self.is_rebound():
            return "rebound"
        if self == EventType.ASSIST:
            return "assist"
        if self in {EventType.STEAL, EventType.BLOCK}:
            return "defense"
        return "other"


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class Keypoint(BaseModel):
    x: float
    y: float
    confidence: float = 1.0


class Detection(BaseModel):
    frame_idx: int
    timestamp: float
    bbox: BoundingBox
    class_: str = Field(alias="class")
    track_id: int
    confidence: float = Field(ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class Player(BaseModel):
    track_id: int
    jersey_number: Optional[int] = None
    name: Optional[str] = None
    team: Optional[str] = None
    pose: list[Keypoint] = Field(default_factory=list)
    bbox: BoundingBox


class GameEvent(BaseModel):
    time_str: str
    quarter: int = Field(ge=1, le=4)
    event_type: EventType
    player_track_id: int
    player_number: Optional[int] = None
    player_name: Optional[str] = None
    assist_track_id: Optional[int] = None
    assist_number: Optional[int] = None
    assist_name: Optional[str] = None
    distance: Optional[float] = None
    score_before: Optional[str] = None
    score_after: Optional[str] = None
    detail: Optional[str] = None

    @property
    def time_seconds(self) -> float:
        parts = self.time_str.split(":")
        return float(parts[0]) * 60 + float(parts[1])


class GameMetadata(BaseModel):
    home_team: str = ""
    away_team: str = ""
    date: str = ""
    arena: Optional[str] = None
    total_duration_seconds: float = 2880.0


class Timeline(BaseModel):
    metadata: GameMetadata = Field(default_factory=GameMetadata)
    events: list[GameEvent] = Field(default_factory=list)
