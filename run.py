import argparse
import os
import warnings
from typing import Dict

from omegaconf import OmegaConf

from src import config
from src.config.config import Config
from src.const import DataPath
from src.features.runner import FeatureRunner
from src.models.blend import BlendPredictRunner, BlendTrainRunner
from src.models.runner import PredictRunner, TrainRunner
from src.utils.logger import Logger


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend")
    parser.add_argument("--fe")
    parser.add_argument("--run")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return args


def load_config(args: argparse.Namespace) -> Dict[str, Config]:
    blend_name = args.blend
    if blend_name is not None:
        if "." in blend_name:
            blend_name = os.path.splitext(blend_name)[0]
    fe_name = args.fe
    if fe_name is not None:
        if "." in fe_name:
            fe_name = os.path.splitext(fe_name)[0]
    run_name = args.run
    if run_name is not None:
        if "." in run_name:
            run_name = os.path.splitext(run_name)[0]
    blend_dict = {
        "blend000": config.blend000,
    }
    fe_dict = {
        "fe000": config.fe000,
        "fe001": config.fe001,
        "fe002": config.fe002,
        "fe003": config.fe003,
        "fe004": config.fe004,
        "fe005": config.fe005,
    }
    run_dict = {
        "run000": config.run000,
        "run001": config.run001,
        "run002": config.run002,
        "run003": config.run003,
        "run004": config.run004,
        "run005": config.run005,
        "run006": config.run006,
        "run007": config.run007,
        "run008": config.run008,
    }
    return {
        "blend": blend_dict[blend_name].BlendConfig if blend_name is not None else None,
        "fe": fe_dict[fe_name].FeConfig if fe_name is not None else None,
        "run": run_dict[run_name].RunConfig if run_name is not None else None,
    }


def check_exists(fe_name: str) -> bool:
    is_x_train_exists = os.path.exists(f"{DataPath.processed.X_train}_{fe_name}.jbl")
    is_y_train_exists = os.path.exists(f"{DataPath.processed.y_train}_{fe_name}.jbl")
    is_x_test_exists = os.path.exists(f"{DataPath.processed.X_test}_{fe_name}.jbl")
    if is_x_train_exists and is_y_train_exists and is_x_test_exists:
        return True
    else:
        return False


def main(cfgs: Dict[str, Config], is_overwrite: bool):
    logger = Logger()
    warnings.filterwarnings("ignore")
    # Blending時は別ランナーを用いる
    if cfgs["blend"] is not None:
        BlendTrainRunner(cfgs, logger).run_train_cv()
        BlendPredictRunner(cfgs, logger).run_predict_cv()
        BlendPredictRunner(cfgs, logger).submission()
    else:
        # 前処理済みデータが存在しないか上書きオプションが有効のときだけ実行する
        if not check_exists(cfgs["fe"].basic.name) or is_overwrite:
            logger.info(f'{cfgs["fe"].basic.name} - Process features')
            FeatureRunner(cfgs).run()
        else:
            logger.info(f'{cfgs["fe"].basic.name} - Skip processing features')
        logger.info(f'{cfgs["run"].basic.name} - Process training')
        TrainRunner(cfgs, logger).run_train_cv()
        logger.info(f'{cfgs["run"].basic.name} - Process prediction')
        PredictRunner(cfgs, logger).run_predict_cv()
        logger.info(f'{cfgs["run"].basic.name} - Process submission')
        PredictRunner(cfgs, logger).submission()


if __name__ == "__main__":
    args = parse_arg()
    cfgs = load_config(args)
    for _, cfg in cfgs.items():
        print(OmegaConf.structured(cfg).pretty())
    main(cfgs, args.overwrite)
