# -*- coding: utf-8 -*-
import click
import gc
import logging
from pathlib import Path
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.utils import save_joblib


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.argument("is_train", type=bool)
def main(input_filepath: str, output_filepath: str, is_train: str) -> None:
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making interim data set from raw data")

    if is_train:
        input_filepath = Path(input_filepath) / "train"
    else:
        input_filepath = Path(input_filepath) / "test"
    input_files = [*input_filepath.glob("*.csv")]
    dfs = []
    for f in input_files:
        df_tmp = pd.read_csv(f)
        if int(f.stem) < 400:
            df_tmp["is_screen_play"] = 1
        else:
            df_tmp["is_screen_play"] = 0
        df_tmp["filename"] = f.stem
        dfs.append(df_tmp)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.sort_values(["filename", "frame"]).reset_index(drop=True)
    del dfs
    gc.collect()

    if is_train:
        output_filepath = Path(output_filepath) / "train.jbl"
    else:
        output_filepath = Path(output_filepath) / "test.jbl"
    save_joblib(df, output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
