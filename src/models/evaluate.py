import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def evaluate(y_true: np.array, y_pred: np.array, eval_metric: str) -> float:
    if eval_metric == "log_loss":
        score = log_loss(y_true, y_pred, eps=1e-15, normalize=True)
    elif eval_metric == "mean_absolute_error":
        score = mean_absolute_error(y_true, y_pred)
    elif eval_metric == "rmse":
        score = np.sqrt(mean_squared_error(y_true, y_pred))
    elif eval_metric == "rmsle":
        score = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    elif eval_metric == "auc":
        score = roc_auc_score(y_true, y_pred)
    elif eval_metric == "prauc":
        score = average_precision_score(y_true, y_pred)
    elif eval_metric == "accuracy":
        score = accuracy_score(y_true, y_pred)
    else:
        raise Exception(f"Unknown evaluation metric: {eval_metric}")
    return score
