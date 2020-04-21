# -*- coding: utf-8 -*-
import hydra
import logging
from omegaconf import DictConfig
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.models.train_model import LGBModel, preprocess
from src.utils import load_joblib, get_original_cwd


@hydra.main(config_path="../../config.yml")
def main(config: DictConfig) -> None:
    """ Runs modeling scripts to train and predict from processed data from (../processed). """
    logger = logging.getLogger(__name__)
    logger.info("making processed data set from interim data")

    cwd = get_original_cwd()
    input_prefix = Path(cwd) / config["path"]["prefix"]["processed"]
    train_input_filepath = Path(input_prefix) / "train.jbl"
    test_input_filepath = Path(input_prefix) / "test.jbl"
    train = load_joblib(train_input_filepath)
    test = load_joblib(test_input_filepath)

    train = preprocess(train)
    test = preprocess(test)
    lgb_model = LGBModel(config, logger)
    lgb_model.kfold_fit_predict(train, test)
    lgb_model.plot_feature_importance(model_type="lgb")
    lgb_model.save_model()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
