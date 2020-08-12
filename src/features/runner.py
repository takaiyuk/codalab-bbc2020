from typing import Dict

from src.config.config import Config
from src.data.load import join_data
from src.features.preprocess import preprocess


class FeatureRunner:
    def __init__(self, cfgs: Dict[str, Config]):
        self.fe_cfg = cfgs["fe"]

    def run(self):
        join_data()
        preprocess(self.fe_cfg)
