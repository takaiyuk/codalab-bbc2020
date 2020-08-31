from dataclasses import dataclass, field
from typing import List

import pandas as pd

from src.config.config import Config
from src.features.aggregate import (
    HoopDist,
    PlayerArea,
    PlayerDirection,
    PlayerDist,
    PlayerDistDiff,
)


def kurtosis(ser: pd.Series):
    return ser.kurtosis()


def skewness(ser: pd.Series):
    return ser.skew()


@dataclass
class Aggregate:
    methods: List[str] = field(
        default_factory=lambda: [
            "mean",
            "min",
            "max",
            "std",
            "median",
            kurtosis,
            skewness,
        ]
    )


@dataclass
class Basic:
    name: str = "fe009"


@dataclass
class Column:
    target: str = "is_screen_play"


@dataclass
class Feature:
    hoop_dist: HoopDist = HoopDist
    player_area: PlayerArea = PlayerArea
    player_dist: PlayerDist = PlayerDist
    player_direction: PlayerDirection = PlayerDirection
    player_dist_diff: PlayerDistDiff = PlayerDistDiff


@dataclass
class Frame:
    column: str = "dist_scr_uDF_min"
    window: float = 0.5
    assert window >= 0.0 and window <= 1.0, "Frame.window must be between 0.0 and 1.0"


@dataclass
class FeConfig(Config):
    aggregate: Aggregate = Aggregate()
    basic: Basic = Basic()
    column: Column = Column()
    feature: Feature = Feature()
    frame: Frame = Frame()
