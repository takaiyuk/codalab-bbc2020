# -*- coding: utf-8 -*-
import logging
import os
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.utils import (
    get_original_cwd,
    load_joblib,
    load_yaml,
    mkdir,
)


def main() -> None:
    """ Runs modeling scripts to train and predict from processed data from (../processed). """
    logger = logging.getLogger(__name__)
    logger.info("making submission data set from predicted data")
    config = load_yaml()

    cwd = get_original_cwd()
    model_prefix = config["path"]["prefix"]["model"]
    model_filepath = sorted(os.listdir(f"{cwd}/{model_prefix}"), reverse=True)[0]
    session_id = model_filepath.split("_")[-1].replace(".jbl", "")
    logger.info(f"session_id: {session_id}")
    pred_test = load_joblib(
        f"{cwd}/{model_prefix}/lgb_model_pred_test_binary_{session_id}.jbl"
    )

    submit = pd.DataFrame({"preds": pred_test})
    submit_prefix = config["path"]["prefix"]["submit"]
    mkdir(f"{cwd}/{submit_prefix}")
    submit.to_csv(f"{cwd}/{submit_prefix}/{session_id}.csv", index=False, header=None)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
