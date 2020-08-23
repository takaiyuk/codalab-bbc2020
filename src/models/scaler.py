from typing import Any, Tuple

import pandas as pd


class SklearnScaler:
    @staticmethod
    def run(
        X: pd.DataFrame, sklearn_scaler: Any, is_train: bool
    ) -> Tuple[pd.DataFrame, Any]:
        scaler = sklearn_scaler
        if not is_train and scaler is None:
            raise ValueError("sklearn_scaler must not be None when is_train is False")
        if is_train:
            scaler.fit(X.values)
        X_trans = scaler.transform(X.values)
        X_trans = pd.DataFrame(X_trans, index=X.index, columns=X.columns)
        return X_trans, scaler
