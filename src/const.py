from dataclasses import dataclass


@dataclass
class ExternalPath:
    prefix: str = "data/external"


@dataclass
class InterimPath:
    prefix: str = "data/interim"
    train: str = f"{prefix}/train"
    test: str = f"{prefix}/test"


@dataclass
class ProcessedPath:
    prefix: str = "data/processed"
    X_train: str = f"{prefix}/X_train"
    y_train: str = f"{prefix}/y_train"
    X_test: str = f"{prefix}/X_test"


@dataclass
class RawPath:
    prefix: str = "data/raw"
    train_dir: str = f"{prefix}/train"
    test_dir: str = f"{prefix}/test"


@dataclass
class DataPath:
    external: ExternalPath = ExternalPath
    interim: InterimPath = InterimPath
    processed: ProcessedPath = ProcessedPath
    raw: RawPath = RawPath


@dataclass
class ModelPath:
    prefix: str = "models"
    importance: str = f"{prefix}/importance"
    model: str = f"{prefix}/model"
    optuna: str = f"{prefix}/optuna"
    prediction: str = f"{prefix}/prediction"
    submission: str = "submissions"


@dataclass
class WorkPath:
    data: DataPath = DataPath
    model: ModelPath = ModelPath
