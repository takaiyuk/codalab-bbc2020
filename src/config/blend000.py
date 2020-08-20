from dataclasses import dataclass

from src.config.config import Config


@dataclass
class Basic:
    description: str = "Ridge Blender"
    exp_name: str = "codalab-bbc2020"
    fe_name: str = "fe001"
    name: str = "blend000"
    mode: str = "training"
    is_debug: bool = False
    seed: int = 42


@dataclass
class Column:
    target: str = "is_screen_play"
    # categorical: List[str] = field(default_factory=lambda: [""])


@dataclass
class Kfold:
    number: int = 5
    method: str = "stratified"
    str_col: str = ""
    grp_col: str = ""


@dataclass
class Model:
    eval_metric: str = "auc"
    name: str = "ModelRidge"


@dataclass
class RidgeParams:
    random_state: int = 42


@dataclass
class Result:
    run002: bool = True
    run005: bool = True
    run006: bool = True


@dataclass
class BlendConfig(Config):
    basic: Basic = Basic()
    column: Column = Column()
    kfold: Kfold = Kfold()
    model: Model = Model()
    params: RidgeParams = RidgeParams()
    result: Result = Result()
