from dataclasses import dataclass, field
from typing import List

from src.config.config import Config


@dataclass
class Basic:
    description: str = "Lightgbm Optuna Classfifier"
    exp_name: str = "codalab-bbc2020"
    fe_name: str = "fe006"
    name: str = "run010"
    mode: str = "training"
    is_debug: bool = False
    seed: int = 42


@dataclass
class Column:
    target: str = "is_screen_play"
    # categorical: List[str] = field(default_factory=lambda: [""])


@dataclass
class Kfold:
    number: int = 10
    method: str = "stratified"
    str_col: str = ""
    grp_col: str = ""


@dataclass
class Model:
    eval_metric: str = "auc"
    name: str = "ModelOptunaLGBM"


@dataclass
class LGBMParams:
    objective: str = "binary"
    boosting_type: str = "gbdt"
    num_boost_round: int = 4000
    learning_rate: float = 0.05
    num_leaves: int = 47
    tree_learner: str = "serial"
    n_jobs: int = 8
    seed: int = 42
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.9
    subsample_freq: int = 1
    colsample_bytree: float = 0.9
    early_stopping_rounds: int = 200
    reg_alpha: float = 0.01
    reg_lambda: float = 0.01
    min_data_per_group: int = 100
    max_cat_threshold: int = 32
    cat_l2: float = 10.0
    cat_smooth: float = 10.0
    metric: str = "auc"
    verbose: int = -1
    device: str = "cpu"
    gpu_platform_id: int = -1
    gpu_device_id: int = -1


@dataclass
class LGBMOptunaParams:
    objective: str = "binary"
    boosting_type: str = "gbdt"
    num_boost_round: int = 4000
    learning_rate: float = 0.05
    num_leaves: int = 47
    tree_learner: str = "serial"
    n_jobs: int = 8
    seed: int = 42
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.9
    subsample_freq: int = 1
    colsample_bytree: float = 0.9
    early_stopping_rounds: int = 200
    reg_alpha: float = 0.01
    reg_lambda: float = 0.01
    min_data_per_group: int = 100
    max_cat_threshold: int = 32
    cat_l2: float = 10.0
    cat_smooth: float = 10.0
    metric: str = "auc"
    verbose: int = -1
    device: str = "cpu"
    gpu_platform_id: int = -1
    gpu_device_id: int = -1


@dataclass
class Selector:
    name: str = "GBDTFeatureSelector"
    threshold: float = 0.5
    lgbm_params: dict = field(default_factory=dict)
    lgbm_fit_kwargs: dict = field(default_factory=dict)

    assert threshold >= 0 or threshold <= 1


@dataclass
class RunConfig(Config):
    basic: Basic = Basic()
    column: Column = Column()
    kfold: Kfold = Kfold()
    model: Model = Model()
    params: LGBMOptunaParams = LGBMOptunaParams()
    selector: Selector = Selector(lgbm_params={"objective": "binary"})
