import gc
import lightgbm as lgb
from logging import Logger
import mlflow
from mlflow.tracking import MlflowClient
import os
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, ListConfig
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from typing import Generator, Tuple

from src.utils import get_original_cwd, get_hydra_session_id, save_joblib, mkdir


def preprocess(
    df: pd.DataFrame, config: DictConfig = None, is_train: bool = False
) -> pd.DataFrame:
    df.reset_index(drop=True, inplace=True)
    return df


def postprocess(pred: np.array, X: pd.DataFrame) -> np.array:
    return pred


def ACC(y_true: np.array, y_pred: np.array) -> float:
    return round(accuracy_score(y_true, y_pred), 6)


def AUC(y_true: np.array, y_pred: np.array) -> float:
    return round(roc_auc_score(y_true, y_pred), 6)


def LOGLOSS(y_true: np.array, y_pred: np.array) -> float:
    return round(log_loss(y_true, y_pred), 6)


class BaseModel:
    def __init__(self, config: DictConfig, logger: Logger) -> None:
        self.config = config
        self.params = config["params"]
        self.seed = config["mode"]["seed"]
        self.logger = logger
        self.cwd = get_original_cwd()
        self.session_id = get_hydra_session_id()

        self.target_column = config["column"]["target"]
        self.cat_columns = None
        self.model = None
        self.models = []
        self.features = []
        self.pred_valid = np.array([])
        self.pred_test = np.array([])
        self.valid_scores = {}

        self.lgb_params = config["params"]["lgb"]
        self.kfold_method = config["params"]["kfold"]["method"]
        self.kfold_number = config["params"]["kfold"]["number"]
        self.str_column = config["params"]["kfold"]["stratified_column"]

        self.writer = MlflowWriter(self.session_id)

    def _fit(self):
        raise NotImplementedError

    def _predict(self):
        raise NotImplementedError

    def _evaluate(self):
        raise NotImplementedError

    def _generate_kfold(
        self, X: pd.DataFrame, X_test: pd.DataFrame, y: np.array, shfl: bool = True
    ) -> Tuple[Generator, pd.DataFrame, pd.DataFrame]:
        if self.kfold_method == "normal":
            kfolds = KFold(self.kfold_number, shuffle=shfl, random_state=self.seed)
            kfold_generator = kfolds.split(X, y)
        elif self.kfold_method == "stratified":
            kfolds = StratifiedKFold(
                self.kfold_number, shuffle=shfl, random_state=self.seed
            )
            kfold_generator = kfolds.split(X, y)
            if self.str_column in X.columns:
                X.drop(self.str_column, axis=1, inplace=True)
                X_test.drop(self.str_column, axis=1, inplace=True)
        elif self.kfold_method == "group":
            kfolds = GroupKFold(self.kfold_number)
            kfold_generator = kfolds.split(X, y, groups=X[self.grp_column])
            if self.grp_column in X.columns:
                X.drop(self.grp_column, axis=1, inplace=True)
                X_test.drop(self.grp_column, axis=1, inplace=True)
        return kfold_generator, X, X_test

    def kfold_fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        X, y = train.drop(self.target_column, axis=1), train[self.target_column]

        kfold_generator, X, test = self._generate_kfold(X, test, y, shfl=True)
        self.features = X.columns.tolist()
        self.pred_valid = np.zeros(len(X))
        self.pred_test = np.zeros(len(test))
        valid_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(kfold_generator):
            X_train, X_valid = (
                X.loc[tr_idx, self.features],
                X.loc[val_idx, self.features],
            )
            y_train, y_valid = y[tr_idx], y[val_idx]
            X_test = test.loc[:, self.features]
            if fold_idx == 0:
                self.logger.info(
                    f"{X_train.shape}, {y_train.shape}, {X_valid.shape}, {y_valid.shape}"
                )
                self.logger.info(f"{X_train.head(1)}")
            if not self.config["skip"]["feature_select"]:
                self.logger.info("Process feature selection")
                self.cat_columns = self.config["params"]["categorical"]
                if self.featureselector.features is None:
                    self.selected_features = self.featureselector.select_features(
                        X_train, y_train, threshold=self.ni_threshold
                    )
                else:
                    self.selected_features = self.featureselector.features
                X_train = X_train.loc[:, self.selected_features]
                X_valid = X_valid.loc[:, self.selected_features]
                X_test = X_test.loc[:, self.selected_features]
                self.cat_columns = [
                    col for col in self.cat_columns if col in self.selected_features
                ]
                self.logger.info(
                    f"num_features: {len(self.features)} -> {len(self.selected_features)}"
                )
                self.logger.info(f"selected_features: {self.selected_features}")
            else:
                self.logger.info("Skip feature selection")
            self._fit(X_train, y_train, X_valid, y_valid)
            self.models.append(self.model)
            pred_val = self._predict(X_valid)
            pred_te = self._predict(X_test)
            self.pred_valid[val_idx] = pred_val
            self.pred_test += pred_te / self.kfold_number
            valid_score = self._evaluate(y_valid, pred_val)
            valid_scores.append(valid_score)
            self.logger.info(f"fold_idx: {fold_idx+1}\tvalid_score: {valid_score}")
            del X_train, y_train, X_valid, y_valid, self.model
            gc.collect()
        valid_score_all = self._evaluate(y, self.pred_valid)
        self.logger.info(f"Best Threshold: {self.best_threshold}")
        self.valid_score_acc = valid_score_all["acc"]
        self.logger.info(
            f"cv score: {self.valid_score_acc}\t(std: {np.std([d['acc'] for d in valid_scores])})"
        )

    def _kfold_feature_importance(
        self, model_type: str, top_features: int = 60
    ) -> pd.DataFrame:
        raise NotImplementedError

    def plot_feature_importance(
        self, model_type: str, save: bool = True, suffix: str = ""
    ) -> None:
        cwd = get_original_cwd()
        importance_prefix = self.config["path"]["prefix"]["importance"]
        mkdir(f"{cwd}/{importance_prefix}")

        df_fi = self._kfold_feature_importance(model_type)
        sns.set()
        plt.figure(figsize=(6, 10))
        sns.barplot(y=df_fi["feature"], x=df_fi["importance"])
        plt.tight_layout()
        if save is True:
            plt.savefig(
                f"{cwd}/{importance_prefix}/importance_{model_type}_{suffix}.png",
                dpi=100,
            )
        else:
            plt.show()
        plt.close()
        return df_fi

    def save_model(self, suffix: str = "") -> None:
        session_id = self.session_id
        cwd = get_original_cwd()
        model_prefix = self.config["path"]["prefix"]["model"]
        mkdir(f"{cwd}/{model_prefix}")

        pred_test_binary = np.where(self.pred_test > self.best_threshold, 1, 0)

        save_joblib(
            self.features, f"{cwd}/{model_prefix}/lgb_model_features_{session_id}.jbl",
        )
        save_joblib(
            self.params, f"{cwd}/{model_prefix}/lgb_model_params_{session_id}.jbl",
        )
        save_joblib(
            self.pred_valid,
            f"{cwd}/{model_prefix}/lgb_model_pred_valid_{session_id}.jbl",
        )
        save_joblib(
            self.pred_test,
            f"{cwd}/{model_prefix}/lgb_model_pred_test_{session_id}.jbl",
        )
        save_joblib(
            pred_test_binary,
            f"{cwd}/{model_prefix}/lgb_model_pred_test_binary_{session_id}.jbl",
        )
        save_joblib(
            self.session_id,
            f"{cwd}/{model_prefix}/lgb_model_session_id_{session_id}.jbl",
        )

    def mlflow(self) -> None:
        try:
            self.writer.log_params_from_omegaconf_dict(self.config)

            # self.writer.log_lgb_model(self.model)
            self.writer.log_metric("RMSE", self.valid_scores["RMSE"])
            self.writer.log_metric("WRMSSE", self.valid_scores["WRMSSE"])

            self.writer.log_artifact(os.path.join(os.getcwd(), ".hydra/config.yaml"))
            self.writer.log_artifact(os.path.join(os.getcwd(), ".hydra/hydra.yaml"))
            self.writer.log_artifact(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))
            self.writer.log_artifact(os.path.join(os.getcwd(), "main.log"))
        except Exception as e:
            print(f"error occurred: {e}")


class LGBModel(BaseModel):
    def __init__(self, config: DictConfig, logger: Logger) -> None:
        super().__init__(config, logger)
        self.best_threshold = 0.0

    def _fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.array,
        X_valid: pd.DataFrame = None,
        y_valid: np.array = None,
        w: np.array = None,
        eval_w: np.array = None,
    ) -> None:
        self.model = lgb.LGBMClassifier(**self.params["lgb"])
        self.model.fit(
            X_train,
            y_train,
            # sample_weight=w,
            categorical_feature=self.cat_columns,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_names=["train", "valid"],
            # eval_sample_weight=[w, eval_w],
            eval_metric=self.params["lgb"]["metric"],
            early_stopping_rounds=self.params["lgb"]["early_stopping_rounds"],
            verbose=self.params["lgb"]["verbose"],
        )

    def _predict(self, X: pd.DataFrame, is_postprocess: bool = False) -> np.array:
        pred = self.model.predict_proba(X)[:, 1]
        if is_postprocess:
            self.logger.info("Postprocess")
            pred = postprocess(pred, X)
        return pred

    def _evaluate(self, y_true: np.array, y_pred: np.array) -> dict:
        scores = {}
        self._optimize_threshold(y_true, y_pred)
        y_pred_binary = np.where(y_pred > self.best_threshold, 1, 0)
        scores["acc"] = ACC(y_true, y_pred_binary)
        scores["auc"] = AUC(y_true, y_pred_binary)
        scores["log_loss"] = LOGLOSS(y_true, y_pred)
        return scores

    def _optimize_threshold(
        self, y_true: np.array, y_pred: np.array, type: str = "acc"
    ):
        best_score = 0.0
        thresholds = [i * 0.01 for i in range(100)]
        if type == "acc":
            for t in thresholds:
                y_pred_binary = np.where(y_pred > t, 1, 0)
                score = accuracy_score(y_true, y_pred_binary)
                if score > best_score:
                    best_score = score
                    self.best_threshold = t

    def _kfold_feature_importance(
        self, model_type: str = "lgb", top_features: int = 60
    ) -> pd.DataFrame:
        df_fi = pd.DataFrame()
        features = self.features
        for i, model in enumerate(self.models):
            importances = model.booster_.feature_importance(importance_type="gain")
            df_tmp = pd.DataFrame(
                {"feature": features, f"importance_{i}": importances}
            ).set_index("feature")
            if i == 0:
                df_fi = df_tmp.copy()
            else:
                df_fi = df_fi.join(df_tmp, how="left", on="feature")
            del df_tmp
            gc.collect()
        df_fi["importance"] = df_fi.values.mean(axis=1)
        df_fi.sort_values("importance", ascending=False, inplace=True)
        df_fi.reset_index(inplace=True)
        if top_features > 0 and top_features < len(df_fi):
            df_fi = df_fi.iloc[:top_features, :]
        return df_fi


class MlflowWriter:
    def __init__(self, experiment_name, **kwargs):
        """https://ymym3412.hatenablog.com/entry/2020/02/09/034644"""

        self.client = MlflowClient(**kwargs)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            self.experiment_id = self.client.get_experiment_by_name(
                experiment_name
            ).experiment_id

        self.run_id = self.client.create_run(self.experiment_id).info.run_id

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f"{parent_name}.{k}", v)
                else:
                    self.client.log_param(self.run_id, f"{parent_name}.{k}", v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f"{parent_name}.{i}", v)

    def log_lgb_model(self, model):
        with mlflow.start_run(self.run_id):
            mlflow.lightgbm.log_model(model, "model")

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)
