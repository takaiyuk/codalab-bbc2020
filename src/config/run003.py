from dataclasses import dataclass

from src.config.config import Config


@dataclass
class Basic:
    description: str = "Conv1D Classifier"
    exp_name: str = "codalab-bbc2020"
    name: str = "run003"
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
    name: str = "ModelConv1D"


@dataclass
class NNParams:
    num_classes: int = 1
    nb_epoch: int = 10
    batch_size: int = 32


@dataclass
class Loss:
    name: str = "binary_crossentropy"


@dataclass
class Optimizer:
    name: str = "Adam"
    learning_rate: float = 0.001


@dataclass
class Scheduler:
    name: str = "ReduceLROnPlateau"
    factor: float = 0.1
    patience: int = 10


@dataclass
class Metrics:
    name: str = "AUC"
    num_thresholds: int = 1000
    curve: str = "ROC"


@dataclass
class Advanced:
    x: int = 0


@dataclass
class RunConfig(Config):
    basic: Basic = Basic()
    column: Column = Column()
    kfold: Kfold = Kfold()
    model: Model = Model()
    params: NNParams = NNParams()
    loss: Loss = Loss()
    optimizer: Optimizer = Optimizer()
    scheduler: Scheduler = Scheduler()
    metrics: Metrics = Metrics()
    advanced: Advanced = Advanced()
