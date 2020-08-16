from dataclasses import dataclass

from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from src.config.config import Config


@dataclass
class Basic:
    description: str = "DNN Classifier"
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
    name: str = "ModelNN"


@dataclass
class Loss:
    name: str = "binary_crossentropy"


@dataclass
class Optimizer:
    name: Adam = Adam(learning_rate=0.001)


@dataclass
class Scheduler:
    name: ReduceLROnPlateau = ReduceLROnPlateau(factor=0.1, patience=10)


@dataclass
class Metrics:
    name: AUC = AUC(num_thresholds=200, curve="ROC")


@dataclass
class NNParams:
    num_classes: int = 2
    nb_epoch: int = 2
    batch_size: int = 32
    loss: Loss = Loss().name
    optimizer: Optimizer = Optimizer().name
    scheduler: Scheduler = Scheduler().name
    metrics: Metrics = Metrics().name


@dataclass
class Advanced:
    pass


@dataclass
class RunConfig(Config):
    basic: Basic = Basic()
    column: Column = Column()
    kfold: Kfold = Kfold()
    model: Model = Model()
    params: NNParams = NNParams()
    advanced: Advanced = Advanced()
