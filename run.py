import argparse
import os
import warnings
from typing import Dict

from omegaconf import OmegaConf

from src import config
from src.config.config import Config
from src.features.runner import FeatureRunner
from src.models.runner import PredictRunner, TrainRunner
from src.utils.logger import Logger


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fe")
    parser.add_argument("--run")
    args = parser.parse_args()
    return args


def load_config(args: argparse.Namespace) -> Dict[str, Config]:
    fe_name = args.fe
    if "." in fe_name:
        fe_name = os.path.splitext(fe_name)[0]
    run_name = args.run
    if "." in run_name:
        run_name = os.path.splitext(run_name)[0]
    fe_dict = {"fe000": config.fe000, "fe001": config.fe001}
    run_dict = {
        "run000": config.run000,
        "run001": config.run001,
    }
    return {"fe": fe_dict[fe_name].FeConfig, "run": run_dict[run_name].RunConfig}


def main(cfgs: Dict[str, Config]):
    logger = Logger()
    warnings.filterwarnings("ignore")
    logger.info(f'{cfgs["fe"].basic.name} - Process features')
    FeatureRunner(cfgs).run()
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
    main(cfgs)
