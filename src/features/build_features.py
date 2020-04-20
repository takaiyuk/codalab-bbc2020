# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.utils import load_joblib, save_joblib, calc_dists


scr_cols = ("scr_x", "scr_y")
usr_cols = ("usr_x", "usr_y")
uDF_cols = ("uDF_x", "uDF_y")
ball_cols = ("bal_x", "bal_y")


def calc_players_dists(df: pd.DataFrame) -> pd.DataFrame:
    df = calc_dists(df, usr_cols, scr_cols)
    df = calc_dists(df, usr_cols, uDF_cols)
    df = calc_dists(df, scr_cols, uDF_cols)
    df = calc_dists(df, usr_cols, ball_cols)
    df = calc_dists(df, scr_cols, ball_cols)
    df = calc_dists(df, uDF_cols, ball_cols)
    return df


def aggregate_players_dists(df: pd.DataFrame) -> pd.DataFrame:
    agg_methods = ["mean", "min", "max", "std"]
    df_agg = df.groupby("filename").agg(
        {
            "is_screen_play": ["mean"],
            "dist_usr_scr": agg_methods,
            "dist_usr_uDF": agg_methods,
            "dist_scr_uDF": agg_methods,
            "dist_usr_bal": agg_methods,
            "dist_scr_bal": agg_methods,
            "dist_uDF_bal": agg_methods,
        }
    )
    df_agg.columns = [f"{col[0]}_{col[1]}" for col in df_agg.columns]
    return df_agg


def preprocess(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = calc_players_dists(df)
    df_agg = aggregate_players_dists(df)
    return df_agg


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.argument("is_train", type=bool)
def main(input_filepath: str, output_filepath: str, is_train: str) -> None:
    """ Runs data processing scripts to turn interim data from (../interim) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making processed data set from interim data")

    if is_train:
        input_filepath = Path(input_filepath) / "train.jbl"
    else:
        input_filepath = Path(input_filepath) / "test.jbl"
    df = load_joblib(input_filepath)

    df = preprocess(df, is_train)

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
