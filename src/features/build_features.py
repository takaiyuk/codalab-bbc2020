# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.visualization.visualize import hoop_xy
from src.utils import load_joblib, save_joblib, calc_dists


scr_cols = ("scr_x", "scr_y")
usr_cols = ("usr_x", "usr_y")
uDF_cols = ("uDF_x", "uDF_y")
ball_cols = ("bal_x", "bal_y")
hoop_cols = ("hoop_x", "hoop_y")


def aggregate_target(df: pd.DataFrame) -> pd.DataFrame:
    df_agg = df.groupby("filename").agg({"is_screen_play": ["mean"]})
    df_agg.columns = [f"{col[0]}" for col in df_agg.columns]
    return df_agg


class PlayerDist:
    def __init__(self) -> None:
        pass

    def calc(self, df: pd.DataFrame) -> pd.DataFrame:
        df = calc_dists(df, usr_cols, scr_cols)
        df = calc_dists(df, usr_cols, uDF_cols)
        df = calc_dists(df, scr_cols, uDF_cols)
        df = calc_dists(df, usr_cols, ball_cols)
        df = calc_dists(df, scr_cols, ball_cols)
        df = calc_dists(df, uDF_cols, ball_cols)
        return df

    def shift(self, df: pd.DataFrame) -> pd.DataFrame:
        dist_cols = [
            "dist_usr_scr",
            "dist_usr_uDF",
            "dist_scr_uDF",
            "dist_usr_bal",
            "dist_scr_bal",
            "dist_uDF_bal",
        ]
        df_shift = df.loc[:, dist_cols].shift(fill_value=0)
        df_shift.rename(
            columns={col: f"{col}_shift" for col in df_shift.columns}, inplace=True
        )
        df = pd.concat((df, df_shift), axis=1)
        for col in dist_cols:
            df[f"{col}_shift_diff"] = df[f"{col}"] - df[f"{col}_shift"]
        return df

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        dist_cols = [
            "dist_usr_scr",
            "dist_usr_uDF",
            "dist_scr_uDF",
            "dist_usr_bal",
            "dist_scr_bal",
            "dist_uDF_bal",
        ]
        agg_methods = ["mean", "min", "max", "std", "median"]

        df_agg = df.groupby("filename").agg({col: agg_methods for col in dist_cols})
        df_agg.columns = [f"{col[0]}_{col[1]}" for col in df_agg.columns]
        df_agg_shift = df.groupby("filename").agg(
            {f"{col}_shift": agg_methods for col in dist_cols}
        )
        df_agg_shift.columns = [f"{col[0]}_{col[1]}" for col in df_agg_shift.columns]
        df_agg_shift_diff = df.groupby("filename").agg(
            {f"{col}_shift_diff": agg_methods for col in dist_cols}
        )
        df_agg_shift_diff.columns = [
            f"{col[0]}_{col[1]}" for col in df_agg_shift_diff.columns
        ]

        df_agg = pd.concat((df_agg, df_agg_shift, df_agg_shift_diff), axis=1)
        return df_agg

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calc(df)
        df = self.shift(df)
        df_agg_player = self.aggregate(df)
        return df_agg_player


class HoopDist:
    def __init__(self) -> None:
        pass

    def calc(self, df: pd.DataFrame) -> pd.DataFrame:
        df["hoop_x"] = hoop_xy[0]
        df["hoop_y"] = hoop_xy[1]

        df = calc_dists(df, usr_cols, hoop_cols)
        df = calc_dists(df, scr_cols, hoop_cols)
        df = calc_dists(df, uDF_cols, hoop_cols)
        df["dist_diff_usr_scr_hoop"] = df["dist_usr_hoop"] - df["dist_scr_hoop"]
        df["dist_diff_usr_uDF_hoop"] = df["dist_usr_hoop"] - df["dist_uDF_hoop"]
        return df

    def shift(self, df: pd.DataFrame) -> pd.DataFrame:
        dist_cols = [
            "dist_usr_hoop",
            "dist_scr_hoop",
            "dist_uDF_hoop",
            "dist_diff_usr_scr_hoop",
            "dist_diff_usr_uDF_hoop",
        ]
        df_shift = df.loc[:, dist_cols].shift(fill_value=0)
        df_shift.rename(
            columns={col: f"{col}_shift" for col in df_shift.columns}, inplace=True
        )
        df = pd.concat((df, df_shift), axis=1)
        for col in dist_cols:
            df[f"{col}_shift_diff"] = df[f"{col}"] - df[f"{col}_shift"]
        return df

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        dist_cols = [
            "dist_usr_hoop",
            "dist_scr_hoop",
            "dist_uDF_hoop",
            "dist_diff_usr_scr_hoop",
            "dist_diff_usr_uDF_hoop",
        ]
        agg_methods = ["mean", "min", "max", "std", "median"]

        df_agg = df.groupby("filename").agg({col: agg_methods for col in dist_cols})
        df_agg.columns = [f"{col[0]}_{col[1]}" for col in df_agg.columns]
        df_agg_shift = df.groupby("filename").agg(
            {f"{col}_shift": agg_methods for col in dist_cols}
        )
        df_agg_shift.columns = [f"{col[0]}_{col[1]}" for col in df_agg_shift.columns]
        df_agg_shift_diff = df.groupby("filename").agg(
            {f"{col}_shift_diff": agg_methods for col in dist_cols}
        )
        df_agg_shift_diff.columns = [
            f"{col[0]}_{col[1]}" for col in df_agg_shift_diff.columns
        ]

        df_agg = pd.concat((df_agg, df_agg_shift, df_agg_shift_diff), axis=1)
        return df_agg

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calc(df)
        df = self.shift(df)
        df_agg_hoop = self.aggregate(df)
        return df_agg_hoop


def preprocess(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df_agg_target = aggregate_target(df)
    df_agg_player = PlayerDist().run(df)
    df_agg_hoop = HoopDist().run(df)

    df_agg = pd.concat((df_agg_target, df_agg_player, df_agg_hoop), axis=1)
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
