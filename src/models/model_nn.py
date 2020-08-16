import os

import keras
import keras.layers as layers
import numpy as np

from src.models.model import Model


class ModelNN(Model):
    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # データのセット
        validation = va_x is not None
        tr_x_ = None
        va_x_ = None
        if validation:
            tr_x_ = self._reshape(tr_x.values)
            va_x_ = self._reshape(va_x.values)
        # モデルの構築
        self.model = self._build_model(tr_x_)
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
        return self.model.predict(te_x)

    def save_model(self, path: str = "models/model"):
        model_path = os.path.join(path, f"{self.run_fold_name}.model")
        self.model.save(model_path)
        print(f"{model_path} is saved")

    def load_model(self, path: str = "models/model"):
        model_path = os.path.join(path, f"{self.run_fold_name}.model")
        self.model = keras.models.load_model(model_path)
        print(f"{model_path} is loaded")


class ModelConv2D(ModelNN):
    def _reshape(self, X: np.array) -> np.array:
        return X.reshape(X.shape[0], X.shape[1], 1)

    def _build_model(self, X: np.array, is_show: bool = False):
        num_features = X.shape[1]

        inputs = keras.Input(shape=(num_features, 1))
        x = layers.Conv2D(256, kernel_size=8, strides=2, padding="same")(inputs)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, kernel_size=8, strides=2, padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, kernel_size=8, strides=2, padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Bidirectional(layers.LSTM(32))(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(8)(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(self.params.num_classes, activation="sigmoid")(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss=self.params.loss,
            optimizer=self.params.optimizer,
            metrics=self.params.metrics,
        )
        if is_show:
            print(model.summary())
        return model
