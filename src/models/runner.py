from typing import Dict, List, Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow import log_artifact, log_metric, log_param

from src.config.config import Config
from src.const import DataPath, ModelPath
from src.models import Model, ModelLGBM, ModelOptunaLGBM
from src.models.evaluate import evaluate
from src.models.kfold import generate_cv
from src.models.optimize import optimize_threshold
from src.utils.joblib import Jbl

models_map = {
    "ModelLGBM": ModelLGBM,
    "ModelOptunaLGBM": ModelOptunaLGBM,
}


class AbstractRunner:
    def __init__(self, cfgs: Dict[str, Config], logger):
        fe_cfg = cfgs["fe"]
        run_cfg = cfgs["run"]
        self.description = run_cfg.basic.description
        self.exp_name = run_cfg.basic.exp_name
        self.run_name = run_cfg.basic.name
        self.run_id = None
        self.fe_name = fe_cfg.basic.name
        self.cv = generate_cv(run_cfg)
        self.column = run_cfg.column
        self.cat_cols = (
            run_cfg.column.categorical
            if "categorical" in run_cfg.column.__annotations__
            else None
        )
        self.kfold = run_cfg.kfold
        self.params = run_cfg.params
        self.evaluation_metric = run_cfg.model.eval_metric
        self.advanced = (
            run_cfg.advanced if "advanced" in run_cfg.__annotations__ else None
        )
        self.logger = logger

        if run_cfg.model.name in models_map.keys():
            self.model_cls = models_map[run_cfg.model.name]
        else:
            raise ValueError(f"model_name {self.model_cls} not found")

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う
        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f"{self.run_name}-{i_fold}"
        return self.model_cls(run_fold_name, self.params, self.cat_cols)

    def load_index_fold(self, i_fold: int) -> list:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        # ここでは乱数を固定して毎回作成しているが、ファイルに保存する方法もある
        if self.kfold.method == "normal":
            return list(self.cv.split(self.X_train, self.y_train))[i_fold]
        elif self.kfold.method == "stratified":
            return list(self.cv.split(self.X_train, self.y_train))[i_fold]
        elif self.kfold.method == "group":
            return list(
                self.cv.split(
                    self.X_train, self.y_train, groups=self.X_train[self.kfold.grp_col],
                )
            )[i_fold]
        elif self.kfold.method == "stratified_group":
            return list(
                self.cv.split(
                    self.X_train, self.y_train, groups=self.X_train[self.kfold.grp_col],
                )
            )[i_fold]
        else:
            raise Exception("Invalid kfold method")

    def reset_mlflow(self):
        mlflow.end_run()


class TrainRunner(AbstractRunner):
    def __init__(self, cfgs: Dict[str, Config], logger):
        super().__init__(cfgs, logger)
        self.X_train = Jbl.load(f"{DataPath.processed.X_train}_{self.fe_name}.jbl")
        self.y_train = Jbl.load(f"{DataPath.processed.y_train}_{self.fe_name}.jbl")
        self.X_test = Jbl.load(f"{DataPath.processed.X_test}_{self.fe_name}.jbl")

        self.best_threshold = 0.0

    def train_fold(self, i_fold: int):
        """クロスバリデーションでのfoldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()

        # 残差の設定
        if self.advanced and "ResRunner" in self.advanced.__annotations__:
            oof = Jbl.load(self.advanced["ResRunner"]["oof"])
            X_train["res"] = (y_train - oof).abs()

        # 学習データ・バリデーションデータをセットする
        tr_idx, va_idx = self.load_index_fold(i_fold)
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[va_idx], y_train.iloc[va_idx]

        # 残差でダウンサンプリング
        if self.advanced and "ResRunner" in self.advanced.__annotations__:
            X_tr = X_tr.loc[
                (X_tr["res"] < self.advanced.ResRunner.res_threshold).values
            ]
            y_tr = y_tr.loc[
                (X_tr["res"] < self.advanced.ResRunner.res_threshold).values
            ]
            print(X_tr.shape)
            X_tr.drop("res", axis=1, inplace=True)
            X_val.drop("res", axis=1, inplace=True)

        # Pseudo Lebeling
        if self.advanced and "PseudoRunner" in self.advanced.__annotations__:
            y_test_pred = Jbl.load(self.advanced.PseudoRunner.y_test_pred)
            if "pl_threshold" in self.advanced.PseudoRunner:
                X_add = self.X_test.loc[
                    (y_test_pred < self.advanced.PseudoRunner.pl_threshold)
                    | (y_test_pred > 1 - self.advanced.PseudoRunner.pl_threshold)
                ]
                y_add = pd.DataFrame(y_test_pred).loc[
                    (y_test_pred < self.advanced.PseudoRunner.pl_threshold)
                    | (y_test_pred > 1 - self.advanced.PseudoRunner.pl_threshold)
                ]
                y_add = pd.DataFrame(([1 if ya > 0.5 else 0 for ya in y_add[0]]))
            elif "pl_threshold_neg" in self.advanced.PseudoRunner:
                X_add = self.X_test.loc[
                    (y_test_pred < self.advanced.PseudoRunner.pl_threshold_neg)
                    | (y_test_pred > self.advanced.PseudoRunner.pl_threshold_pos)
                ]
                y_add = pd.DataFrame(y_test_pred).loc[
                    (y_test_pred < self.advanced.PseudoRunner.pl_threshold_neg)
                    | (y_test_pred > self.advanced.PseudoRunner.pl_threshold_pos)
                ]
                y_add = pd.DataFrame(([1 if ya > 0.5 else 0 for ya in y_add[0]]))
            else:
                X_add = self.X_test
                y_add = pd.DataFrame(y_test_pred)
            print(f"added X_test: {len(X_add)}")
            X_tr = pd.concat([X_tr, X_add])
            y_tr = pd.concat([y_tr, y_add])

        # 前処理
        y_tr = preprocess_target(y_tr)
        y_val = preprocess_target(y_val)

        # 学習を行う
        model = self.build_model(i_fold)
        model.train(X_tr, y_tr, X_val, y_val, self.X_test)

        # バリデーションデータへの予測・評価を行う
        pred_val = model.predict(X_val)

        # 後処理
        y_val = postprocess_prediction(y_val.values)
        pred_val = postprocess_prediction(pred_val)

        score = self.evaluate(y_val, pred_val)

        # モデル、インデックス、予測値、評価を返す
        return model, va_idx, pred_val, score

    def run_train_cv(self) -> None:
        """クロスバリデーションでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        # mlflow
        mlflow.set_experiment(self.exp_name)
        mlflow.start_run(run_name=self.run_name)
        self.logger.info(f"{self.run_name} - start training cv")

        scores = []
        va_idxes = []
        preds = []

        # Adversarial validation
        if self.advanced and "adversarial_validation" in self.advanced.__annotations__:
            X_train = self.X_train.copy()
            X_test = self.X_test.copy()
            X_train["target"] = 0
            X_test["target"] = 1
            X_train = pd.concat([X_train, X_test], sort=False).reset_index(drop=True)
            y_train = X_train["target"]
            X_train.drop("target", axis=1, inplace=True)
            X_test.drop("target", axis=1, inplace=True)
            self.X_train = X_train
            self.y_train = y_train

        # 各foldで学習を行う
        for i_fold in range(self.cv.n_splits):
            # 学習を行う
            self.logger.info(f"{self.run_name} fold {i_fold} - start training")
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            self.logger.info(
                f"{self.run_name} fold {i_fold} - end training - score {score}\tbest_iteration: {model.model.best_iteration}"
            )

            # モデルを保存する
            model.save_model()

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        cv_score = self.evaluate(self.y_train.values, preds)
        preds_binarized = np.where(preds > self.best_threshold, 1, 0)

        self.logger.info(f"{self.run_name} - end training cv - score {cv_score}")
        self.logger.info(f"{self.run_name} - best threshold - {self.best_threshold}")

        # 予測結果の保存
        Jbl.save(preds, f"{ModelPath.prediction}/{self.run_name}-train.jbl")
        Jbl.save(
            preds_binarized,
            f"{ModelPath.prediction}/{self.run_name}-train-binarized.jbl",
        )
        Jbl.save(
            self.best_threshold,
            f"{ModelPath.prediction}/{self.run_name}-best-threshold.jbl",
        )

        # mlflow
        self.mlflow(cv_score, scores)

    def evaluate(self, y_true: np.array, y_pred: np.array) -> dict:
        """指定の評価指標をもとにしたスコアを計算して返す
        :param y_true: 真値
        :param y_pred: 予測値
        :return: スコア
        """
        scores = {}
        best_threshold, _ = optimize_threshold(y_true, y_pred)
        self.best_threshold = best_threshold
        y_pred_binary = np.where(y_pred > best_threshold, 1, 0)
        scores["acc"] = evaluate(y_true, y_pred_binary, "accuracy")
        scores["auc"] = evaluate(y_true, y_pred_binary, "auc")
        scores["log_loss"] = evaluate(y_true, y_pred, "log_loss")
        return scores

    def mlflow(self, cv_score: Dict[str, float], scores: List[Dict[str, float]]):
        self.run_id = mlflow.active_run().info.run_id
        log_param("model_name", self.model_cls.__class__.__name__)
        log_param("fe_name", self.fe_name)
        log_param("train_params", self.params)
        log_param("cv_strategy", str(self.cv))
        log_param("evaluation_metric", self.evaluation_metric)
        for metric, score in cv_score.items():
            log_metric(f"cv_score_{metric}", score)
        log_param(
            "fold_scores_accuracy",
            dict(
                zip(
                    [f"fold_{i}" for i in range(len(scores))],
                    [round(s["acc"], 4) for s in scores],
                )
            ),
        )
        log_param("cols_definition", self.column)
        log_param("description", self.description)
        mlflow.end_run()


class PredictRunner(AbstractRunner):
    def __init__(self, cfgs: Dict[str, Config], logger):
        super().__init__(cfgs, logger)
        self.X_test = Jbl.load(f"{DataPath.processed.X_test}_{self.fe_name}.jbl")

    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う
        あらかじめrun_train_cvを実行しておく必要がある
        """

        self.logger.info(f"{self.run_name} - start prediction cv")
        X_test = self.X_test
        preds = []

        show_feature_importance = "LGBM" in str(self.model_cls)
        feature_importances = pd.DataFrame()

        # 各foldのモデルで予測を行う
        for i_fold in range(self.cv.n_splits):
            self.logger.info(f"{self.run_name} - start prediction fold:{i_fold}")
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(X_test)
            # 後処理
            pred = postprocess_prediction(pred)
            preds.append(pred)
            self.logger.info(f"{self.run_name} - end prediction fold:{i_fold}")
            if show_feature_importance:
                feature_importances = pd.concat(
                    [feature_importances, model.feature_importance(X_test)], axis=0
                )

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 閾値で2値化
        best_threshold = Jbl.load(
            f"{ModelPath.prediction}/{self.run_name}-best-threshold.jbl"
        )
        pred_avg_binarized = np.where(pred_avg > best_threshold, 1, 0)

        # 予測結果の保存
        Jbl.save(pred_avg, f"{ModelPath.prediction}/{self.run_name}-test.jbl")
        Jbl.save(
            pred_avg_binarized,
            f"{ModelPath.prediction}/{self.run_name}-test-binarized.jbl",
        )

        self.logger.info(f"{self.run_name} - end prediction cv")

        # 特徴量の重要度
        if show_feature_importance:
            aggs = (
                feature_importances.groupby("Feature")
                .mean()
                .sort_values(by="importance", ascending=False)
            )
            cols = aggs[:200].index
            pd.DataFrame(aggs.index).to_csv(
                f"{ModelPath.importance}/{self.run_name}-fi.csv", index=False
            )

            best_features = feature_importances.loc[
                feature_importances.Feature.isin(cols)
            ]
            plt.figure(figsize=(14, 26))
            sns.barplot(
                x="importance",
                y="Feature",
                data=best_features.sort_values(by="importance", ascending=False),
            )
            plt.title("LightGBM Features (averaged over folds)")
            plt.tight_layout()
            plt.savefig(f"{ModelPath.importance}/{self.run_name}-fi.png")
            plt.show()

            # mlflow
            mlflow.start_run(run_id=self.run_id)
            log_artifact(f"{ModelPath.importance}/{self.run_name}-fi.png")
            mlflow.end_run()

    def submission(self):
        if self.advanced and "separate" in self.advanced.__annotations__:
            sub = Jbl.load(
                f"{DataPath.processed.prefix}/X_test_{self.fe_name}.jbl"
            ).loc[:, [self.separate_col]]
            separate_col_uniques = sub[self.separate_col].unique()
            results = {}
            for separate_col_val in separate_col_uniques:
                pred = Jbl.load(
                    f"{ModelPath.prediction}/{self.run_name}-{separate_col_val}-test.jbl"
                )
                sub_separate_idx = sub[sub[self.separate_col] == separate_col_val].index
                result = {idx_: [p_] for idx_, p_ in zip(sub_separate_idx, pred)}
                results.update(result)
            sub = (
                pd.DataFrame(results)
                .T.reset_index()
                .rename(columns={"index": "id", 0: self.column.target})
                .sort_values("id")
                .reset_index(drop=True)
            )
            sub.loc[:, "id"] = (
                Jbl.load(f"{DataPath.interim.test}").loc[:, ["id"]].values
            )
            pred = sub[self.column.target].values
        else:
            # sub = Jbl.load(f"{DataPath.interim.test}").loc[:, ["id"]]
            # pred = Jbl.load(f"{ModelPath.prediction}/{self.run_name}-test.jbl")
            sub = pd.DataFrame()
            pred = Jbl.load(
                f"{ModelPath.prediction}/{self.run_name}-test-binarized.jbl"
            )
        if self.advanced and "predict_exp" in self.advanced.__annotations__:
            sub[self.column.target] = np.exp(pred)
        else:
            sub[self.column.target] = pred
        # sub.to_csv(
        #     f"{DataPath.submission}/submission_{self.run_name}.csv", index=False,
        # )
        sub.to_csv(
            f"{ModelPath.submission}/submission_{self.run_name}.csv",
            index=False,
            header=None,
        )


def preprocess_target(y):
    return y


def postprocess_prediction(p):
    return p