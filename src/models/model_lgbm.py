import dataclasses
import json
import os

import lightgbm as lgb
import optuna.integration.lightgbm as optuna_lgb
import pandas as pd

from src.models.model import Model
from src.utils.joblib import Jbl


class ModelLGBM(Model):
    def train(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None):
        # データのセット
        validation = va_x is not None
        lgb_train = lgb.Dataset(
            tr_x, tr_y, categorical_feature=self.categorical_features
        )
        lgb_eval = None
        if validation:
            lgb_eval = lgb.Dataset(
                va_x,
                va_y,
                reference=lgb_train,
                categorical_feature=self.categorical_features,
            )
        # ハイパーパラメータの設定
        params = dataclasses.asdict(self.params)
        num_round = params.pop("num_boost_round")
        # 学習
        if validation:
            early_stopping_rounds = params.pop("early_stopping_rounds")
            self.model = lgb.train(
                params,
                lgb_train,
                num_round,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=500,
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            self.model = lgb.train(
                params, lgb_train, num_round, valid_sets=[lgb_train], verbose_eval=500
            )

    def predict(self, te_x):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)

    def feature_importance(self, te_x):
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = te_x.columns.values
        fold_importance_df["importance"] = self.model.feature_importance(
            importance_type="gain"
        )
        return fold_importance_df

    def save_model(self, path: str = "models/model"):
        model_path = os.path.join(path, f"{self.run_fold_name}.model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Jbl.save(self.model, model_path)

    def load_model(self, path: str = "models/model"):
        model_path = os.path.join(path, f"{self.run_fold_name}.model")
        self.model = Jbl.load(model_path)


class ModelOptunaLGBM(ModelLGBM):
    def __init__(
        self,
        run_fold_name: str,
        params: dict,
        categorical_features=None,
        optuna_path: str = "models/optuna",
    ):
        super().__init__(run_fold_name, params, categorical_features)
        self.optuna_path = optuna_path

    def train(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None):
        # データのセット
        validation = va_x is not None
        lgb_train = optuna_lgb.Dataset(
            tr_x,
            tr_y,
            categorical_feature=self.categorical_features,
            free_raw_data=False,
        )
        if validation:
            lgb_eval = optuna_lgb.Dataset(
                va_x,
                va_y,
                reference=lgb_train,
                categorical_feature=self.categorical_features,
                free_raw_data=False,
            )
        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop("num_boost_round")
        best_params, tuning_history = dict(), list()
        # 学習
        lgb_eval = None
        if validation:
            early_stopping_rounds = params.pop("early_stopping_rounds")
            self.model = optuna_lgb.train(
                params,
                lgb_train,
                num_round,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=1000,
                early_stopping_rounds=early_stopping_rounds,
                best_params=best_params,
                tuning_history=tuning_history,
            )
        else:
            self.model = optuna_lgb.train(
                params,
                lgb_train,
                num_round,
                valid_sets=[lgb_train],
                verbose_eval=1000,
                best_params=best_params,
                tuning_history=tuning_history,
            )
        print(f"Best Params: {best_params}")
        with open(
            f"{self.optuna_path}/{self.run_fold_name}_best_params.json", "w"
        ) as f:
            json.dump(best_params, f, indent=4, separators=(",", ": "))
