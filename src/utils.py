import joblib
from typing import Any


def load_joblib(path: str) -> Any:
    return joblib.load(path)


def save_joblib(obj_: Any, path: str) -> None:
    joblib.dump(obj_, path, compress=3)
