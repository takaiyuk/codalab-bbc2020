import random
from collections import Counter, defaultdict
from typing import Any, Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


# https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
class StratifiedGroupKFold:
    def __init__(self, n_splits: int, random_state: int = None) -> None:
        self.n_splits = n_splits
        self.seed = random_state

    def _float_to_bins(self, y: pd.Series) -> pd.Series:
        """For the case when y is continuous"""
        y_bin = pd.cut(y, self.n_splits).cat.codes
        return y_bin

    def split(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Generator:
        """
        :params X: 学習データの特徴量
        :params y: 学習データの目的変数
        :params groups: Group labels for the samples used while splitting the dataset into train/test set.
        :yields train: The training set indices for that split.
        :yields test: The testing set indices for that split.
        example
        -------
        > sgkf = StratifiedGroupKFold(n_splits=num_folds, random_state=random_state)
        > for tr_idx, val_idx in sgkf.split(X, y, groups=X[group_col]):
           ...
        """
        if y.nunique() != (np.max(y) + 1):
            y = self._float_to_bins(y)
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def _eval_y_counts_per_fold(y_counts: float, fold: int) -> float:
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std(
                    [
                        y_counts_per_fold[i][label] / y_distr[label]
                        for i in range(self.n_splits)
                    ]
                )
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(self.seed).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                fold_eval = _eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(self.n_splits):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices


def generate_cv(run_config: dict) -> Any:
    if run_config.kfold.method == "normal":
        cv = KFold(
            n_splits=run_config.kfold.number,
            shuffle=True,
            random_state=run_config.basic.seed,
        )
    elif run_config.kfold.method == "stratified":
        cv = StratifiedKFold(
            n_splits=run_config.kfold.number,
            shuffle=True,
            random_state=run_config.basic.seed,
        )
    elif run_config.kfold.method == "group":
        cv = GroupKFold(n_splits=run_config.kfold.number)
    elif run_config.kfold.method == "stratified_group":
        cv = StratifiedGroupKFold(
            n_splits=run_config.kfold.number, random_state=run_config.basic.seed,
        )
    else:
        raise ValueError(f"Unknown kfold method: {run_config.kfold.method}")
    return cv
