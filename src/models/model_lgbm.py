import dataclasses
import json
import os

import lightgbm as lgb
import optuna.integration.lightgbm as optuna_lgb
import pandas as pd

from src.const import ModelPath
from src.models.model import Model
from src.utils.joblib import Jbl


class ModelLGBM(Model):
    def train(self, tr_x, tr_y, va_x=None, va_y=None):
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
        print(f"{model_path} is saved")

    def load_model(self, path: str = "models/model"):
        model_path = os.path.join(path, f"{self.run_fold_name}.model")
        self.model = Jbl.load(model_path)
        print(f"{model_path} is loaded")


class ModelOptunaLGBM(ModelLGBM):
    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # データのセット
        validation = va_x is not None
        lgb_train = optuna_lgb.Dataset(
            tr_x,
            tr_y,
            categorical_feature=self.categorical_features,
            free_raw_data=False,
        )
        lgb_eval = None
        if validation:
            lgb_eval = optuna_lgb.Dataset(
                va_x,
                va_y,
                reference=lgb_train,
                categorical_feature=self.categorical_features,
                free_raw_data=False,
            )
        # ハイパーパラメータの設定
        params = dataclasses.asdict(self.params)
        num_round = params.pop("num_boost_round")
        # 学習
        if validation:
            early_stopping_rounds = params.pop("early_stopping_rounds")
            self.model = optuna_lgb.train(
                params,
                lgb_train,
                num_round,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=500,
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            self.model = optuna_lgb.train(
                params, lgb_train, num_round, valid_sets=[lgb_train], verbose_eval=500,
            )
        best_params = self.model.params
        print(f"Optuna Best Params: {best_params}")
        with open(
            f"{ModelPath.optuna}/{self.run_fold_name}_best_params.json", "w"
        ) as f:
            json.dump(best_params, f, indent=4, separators=(",", ": "))
