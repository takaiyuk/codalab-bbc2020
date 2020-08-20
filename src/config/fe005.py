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
    name: str = "fe005"


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
    column: str = "dist_scr_uDF_min"
    window: float = 0.5
    assert window >= 0.0 and window <= 1.0, "Frame.window must be between 0.0 and 1.0"


@dataclass
class FeConfig(Config):
    basic: Basic = Basic()
    column: Column = Column()
    feature: Feature = Feature()
    frame: Frame = Frame()
