from dataclasses import dataclass

from src.config.config import Config


@dataclass
class Basic:
    name: str = "fe000"


@dataclass
class Column:
    target: str = "is_screen_play"


@dataclass
class FeConfig(Config):
    basic: Basic = Basic()
    column: Column = Column()
