from src.models.optimize import optimize_threshold


def accuracy(preds, data):
    """精度 (Accuracy) を計算する関数"""
    y_true = data.get_label()
    _, best_score = optimize_threshold(y_true, preds)
    # name, result, is_higher_better
    return "accuracy", best_score, True
