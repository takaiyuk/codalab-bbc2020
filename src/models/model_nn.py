import dataclasses
import os

import keras
import keras.layers as layers
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import AUC

from src.config.config import Config
from src.const import ModelPath
from src.models.model import Model
from src.models.scaler import SklearnScaler
from src.models.seed import fix_seeds
from src.utils.joblib import Jbl


class ModelNN(Model):
    def __init__(self, run_fold_name: str, run_cfg: Config, categorical_features=None):
        super().__init__(run_fold_name, run_cfg, categorical_features)
        self.loss = run_cfg.loss
        self.optimizer = run_cfg.optimizer
        self.scheduler = run_cfg.scheduler
        self.metrics = run_cfg.metrics
        self.scaler = None

        fix_seeds()

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # データのセット
        validation = va_x is not None
        tr_x_, self.scaler = SklearnScaler.run(tr_x, MinMaxScaler(), is_train=True)
        tr_x_ = self._reshape(tr_x_.values)
        va_x_ = None
        if validation:
            va_x_, _ = SklearnScaler.run(va_x, self.scaler, is_train=False)
            va_x_ = self._reshape(va_x_.values)
        os.makedirs(ModelPath.scaler, exist_ok=True)
        Jbl.save(
            self.scaler, os.path.join(ModelPath.scaler, f"{self.run_fold_name}.scaler")
        )
        # モデルの構築
        self.model = self._build_model(tr_x_, is_show=False)
        # 学習
        if validation:
            self.history = self.model.fit(
                tr_x_,
                tr_y,
                nb_epoch=self.params.nb_epoch,
                validation_data=(va_x_, va_y),
                batch_size=self.params.batch_size,
                verbose=1,
            )
        else:
            self.history = self.model.fit(
                tr_x_,
                tr_y,
                nb_epoch=self.params.nb_epoch,
                batch_size=self.params.batch_size,
                verbose=1,
            )

    def predict(self, te_x):
        self.scaler = Jbl.load(
            os.path.join(ModelPath.scaler, f"{self.run_fold_name}.scaler")
        )
        te_x_, _ = SklearnScaler.run(te_x, self.scaler, is_train=False)
        te_x_ = self._reshape(te_x_.values)
        return self.model.predict(te_x_)

    def save_model(self, path: str = ModelPath.model):
        model_path = os.path.join(path, f"{self.run_fold_name}.model")
        self.model.save(model_path)
        print(f"{model_path} is saved")

    def load_model(self, path: str = ModelPath.model):
        model_path = os.path.join(path, f"{self.run_fold_name}.model")
        self.model = keras.models.load_model(model_path)
        print(f"{model_path} is loaded")

    def _set_params(self):
        raise NotImplementedError


class ModelConv1D(ModelNN):
    def __init__(self, run_fold_name: str, run_cfg: Config, categorical_features=None):
        super().__init__(run_fold_name, run_cfg, categorical_features)

    def _reshape(self, X: np.array) -> np.array:
        return X.reshape(X.shape[0], X.shape[1], 1)

    def _set_params(self):
        self.loss_func = self.loss.name

        optimizer = dataclasses.asdict(self.optimizer)
        optimizer_name = optimizer.pop("name")
        if optimizer_name == "Adam":
            self.optimizer_func = Adam(**optimizer)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        scheduler = dataclasses.asdict(self.scheduler)
        scheduler_name = scheduler.pop("name")
        if scheduler_name == "ReduceLROnPlateau":
            self.scheduler_func = ReduceLROnPlateau(**scheduler)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        # metrics = dataclasses.asdict(self.metrics)
        # metrics_name = metrics.pop("name")
        # if metrics_name == "AUC":
        #     self.metrics_func = AUC(**metrics)
        # else:
        #     raise ValueError(f"Unknown metrics: {metrics_name}")
        self.metrics_func = "accuracy"

    def _build_model(self, X: np.array, is_show: bool = False):
        self._set_params()
        # os.makedirs("models/metrics", exist_ok=True)
        # Jbl.save(self.metrics_func, f"models/metrics/{self.run_fold_name}.metrics")
        num_features = X.shape[1]

        inputs = keras.Input(shape=(num_features, 1))
        x = layers.Conv1D(256, kernel_size=8, strides=2, padding="same")(inputs)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, kernel_size=8, strides=2, padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, kernel_size=8, strides=2, padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Bidirectional(layers.LSTM(32))(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(8)(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(self.params.num_classes, activation="sigmoid")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss=self.loss_func,
            optimizer=self.optimizer_func,
            metrics=[self.metrics_func],
        )
        if is_show:
            print(model.summary())
        return model

    # def load_model(self, path: str = ModelPath.model):
    #     self.metrics_func = Jbl.load(f"models/metrics/{self.run_fold_name}.metrics")
    #     fold_name = self.run_fold_name.split("-")[1]
    #     custom_objects_key = "auc" if fold_name == "0" else f"auc_{fold_name}"

    #     model_path = os.path.join(path, f"{self.run_fold_name}.model")
    #     self.model = keras.models.load_model(
    #         model_path, custom_objects={custom_objects_key: self.metrics_func}
    #     )
    #     print(f"{model_path} is loaded")
