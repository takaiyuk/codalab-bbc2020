from dataclasses import dataclass

from src.config.config import Config
from src.features.aggregate import (
    HoopDist,
    PlayerArea,
    PlayerDirection,
    PlayerDist,
)


@dataclass
class Basic:
    name: str = "fe003"


@dataclass
class Column:
    target: str = "is_screen_play"


@dataclass
class Feature:
    hoop_dist: HoopDist = HoopDist
    player_area: PlayerArea = PlayerArea
    player_dict: PlayerDist = PlayerDist
    player_direction: PlayerDirection = PlayerDirection


@dataclass
class Frame:
    start: float = 0.125
    end: float = 0.875
    assert start >= 0.0 and start < 1.0, "Frame.start must be between 0.0 and 1.0"
    assert end > 0.0 and end <= 1.0, "Frame.end must be between 0.0 and 1.0"
    assert start < end, "Frame.srart must be smaller than Frame.end"


@dataclass
class FeConfig(Config):
    basic: Basic = Basic()
    column: Column = Column()
    feature: Feature = Feature()
    frame: Frame = Frame()
