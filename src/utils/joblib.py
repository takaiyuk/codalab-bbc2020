from typing import Any

import joblib


class Jbl:
    @staticmethod
    def load(path: str) -> Any:
        return joblib.load(path)

    @staticmethod
    def save(obj: Any, path: str) -> None:
        joblib.dump(obj, path, compress=3)
