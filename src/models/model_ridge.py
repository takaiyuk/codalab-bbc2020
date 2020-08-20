import dataclasses
import os

from sklearn.linear_model import Ridge

from src.const import ModelPath
from src.models.model import Model
from src.utils.joblib import Jbl


class ModelRidge(Model):
    def train(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None):

        # ハイパーパラメータの設定
        params = dataclasses.asdict(self.params)
        self.model = Ridge(**params)
        self.model.fit(tr_x, tr_y)
        print(self.model.coef_)

    def predict(self, te_x):
        return self.model.predict(te_x)

    def save_model(self):
        model_path = os.path.join(f"{ModelPath.model}", f"{self.run_fold_name}.model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Jbl.save(self.model, model_path)

    def load_model(self):
        model_path = os.path.join(f"{ModelPath.model}", f"{self.run_fold_name}.model")
        self.model = Jbl.load(model_path)
