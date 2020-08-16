from dataclasses import dataclass

from src.config.config import Config
from src.features.aggregate import HoopDist, PlayerArea, PlayerDist


@dataclass
class Basic:
    name: str = "fe000"


@dataclass
class Column:
    target: str = "is_screen_play"


@dataclass
class Feature:
    hoop_dist: HoopDist = HoopDist
    player_area: PlayerArea = PlayerArea
    player_dict: PlayerDist = PlayerDist


@dataclass
class FeConfig(Config):
    basic: Basic = Basic()
    column: Column = Column()
    feature: Feature = Feature()
