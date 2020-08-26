import logging
from dataclasses import dataclass
from typing import Dict

import pandas as pd

from src.config.config import Config
from src.const import ModelPath
from src.models.kfold import generate_cv
from src.models.runner import PredictRunner, TrainRunner, models_map
from src.utils.joblib import Jbl


class BlendTrainRunner(TrainRunner):
    def __init__(self, cfgs: Dict[str, Config], logger: logging.Logger):
        blend_cfg = cfgs["blend"]

        self.description = blend_cfg.basic.description
        self.exp_name = blend_cfg.basic.exp_name
        self.run_name = blend_cfg.basic.name
        self.run_id = None
        self.fe_name = blend_cfg.basic.fe_name
        self.run_cfg = blend_cfg
        self.params = blend_cfg.params
        self.cv = generate_cv(blend_cfg)
        self.column = blend_cfg.column
        self.cat_cols = (
            blend_cfg.column.categorical
            if "categorical" in blend_cfg.column.__annotations__
            else None
        )
        self.kfold = blend_cfg.kfold
        self.evaluation_metric = blend_cfg.model.eval_metric
        self.logger = logger

        @dataclass
        class advanced:
            PseudoRunner: PseudoRunner = None
            ResRunner: ResRunner = None
            AdversarialValidation: AdversarialValidation = None
            Selector: Selector = None

        self.advanced = advanced

        if blend_cfg.model.name in models_map.keys():
            self.model_cls = models_map[blend_cfg.model.name]
        else:
            raise ValueError(f"model_name {self.model_cls} not found")

        trs = []
        tes = []
        for run_name, _ in blend_cfg.result.__annotations__.items():
            tr = Jbl.load(f"{ModelPath.prediction}/{run_name}-train.jbl")
            te = Jbl.load(f"{ModelPath.prediction}/{run_name}-test.jbl")
            trs.append(tr)
            tes.append(te)
        train = pd.DataFrame(trs).T
        train.columns = list(blend_cfg.result.__annotations__.keys())
        test = pd.DataFrame(tes).T
        test.columns = list(blend_cfg.result.__annotations__.keys())
        target = [1] * 400 + [0] * (1528 - 400)
        train["y"] = target
        self.X_train = train.drop("y", axis=1)
        self.y_train = train["y"]
        self.X_test = test.copy()

        self.best_threshold = 0.0


class BlendPredictRunner(PredictRunner):
    def __init__(self, cfgs: Dict[str, Config], logger: logging.Logger):
        blend_cfg = cfgs["blend"]

        self.description = blend_cfg.basic.description
        self.exp_name = blend_cfg.basic.exp_name
        self.run_name = blend_cfg.basic.name
        self.run_id = None
        self.fe_name = blend_cfg.basic.fe_name
        self.run_cfg = blend_cfg
        self.params = blend_cfg.params
        self.cv = generate_cv(blend_cfg)
        self.column = blend_cfg.column
        self.cat_cols = (
            blend_cfg.column.categorical
            if "categorical" in blend_cfg.column.__annotations__
            else None
        )
        self.kfold = blend_cfg.kfold
        self.logger = logger

        @dataclass
        class advanced:
            PseudoRunner: PseudoRunner = None
            ResRunner: ResRunner = None
            AdversarialValidation: AdversarialValidation = None
            Selector: Selector = None

        self.advanced = advanced

        if blend_cfg.model.name in models_map.keys():
            self.model_cls = models_map[blend_cfg.model.name]
        else:
            raise ValueError(f"model_name {self.model_cls} not found")

        tes = []
        for run_name, _ in blend_cfg.result.__annotations__.items():
            te = Jbl.load(f"{ModelPath.prediction}/{run_name}-test.jbl")
            tes.append(te)
        test = pd.DataFrame(tes).T
        test.columns = list(blend_cfg.result.__annotations__.keys())
        self.X_test = test.copy()
