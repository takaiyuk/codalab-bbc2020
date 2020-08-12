from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score


def optimize_threshold(
    y_true: np.array, y_pred: np.array, type: str = "acc"
) -> Tuple[float, float]:
    best_score = 0.0
    best_threshold = 0.0
    thresholds = [i * 0.01 for i in range(100)]
    if type == "acc":
        for t in thresholds:
            y_pred_binary = np.where(y_pred > t, 1, 0)
            score = accuracy_score(y_true, y_pred_binary)
            if score > best_score:
                best_score = score
                best_threshold = t
    return best_threshold, best_score
