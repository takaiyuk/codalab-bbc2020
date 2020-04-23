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


class BaseAggregator:
    def __init__(self, calc_columns: list):
        self.calc_cols = calc_columns

    def calc(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def shift(self, df: pd.DataFrame) -> pd.DataFrame:
        df_shift = df.loc[:, self.calc_cols].shift(fill_value=0)
        df_shift.rename(
            columns={col: f"{col}_shift" for col in df_shift.columns}, inplace=True
        )
        df = pd.concat((df, df_shift), axis=1)
        for col in self.calc_cols:
            df[f"{col}_shift_diff"] = df[f"{col}"] - df[f"{col}_shift"]
        return df

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        agg_methods = ["mean", "min", "max", "std", "median"]

        df_agg = df.groupby("filename").agg(
            {col: agg_methods for col in self.calc_cols}
        )
        df_agg.columns = [f"{col[0]}_{col[1]}" for col in df_agg.columns]
        df_agg_shift = df.groupby("filename").agg(
            {f"{col}_shift": agg_methods for col in self.calc_cols}
        )
        df_agg_shift.columns = [f"{col[0]}_{col[1]}" for col in df_agg_shift.columns]
        df_agg_shift_diff = df.groupby("filename").agg(
            {f"{col}_shift_diff": agg_methods for col in self.calc_cols}
        )
        df_agg_shift_diff.columns = [
            f"{col[0]}_{col[1]}" for col in df_agg_shift_diff.columns
        ]

        df_agg = pd.concat((df_agg, df_agg_shift, df_agg_shift_diff), axis=1)
        return df_agg

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calc(df)
        df = self.shift(df)
        df_agg = self.aggregate(df)
        return df_agg


class PlayerDist(BaseAggregator):
    def __init__(self) -> None:
        dist_cols = [
            "dist_usr_scr",
            "dist_usr_uDF",
            "dist_scr_uDF",
            "dist_usr_bal",
            "dist_scr_bal",
            "dist_uDF_bal",
        ]
        super().__init__(dist_cols)

    def calc(self, df: pd.DataFrame) -> pd.DataFrame:
        df = calc_dists(df, usr_cols, scr_cols)
        df = calc_dists(df, usr_cols, uDF_cols)
        df = calc_dists(df, scr_cols, uDF_cols)
        df = calc_dists(df, usr_cols, ball_cols)
        df = calc_dists(df, scr_cols, ball_cols)
        df = calc_dists(df, uDF_cols, ball_cols)
        return df


class HoopDist(BaseAggregator):
    def __init__(self) -> None:
        dist_cols = [
            "dist_usr_hoop",
            "dist_scr_hoop",
            "dist_uDF_hoop",
            "dist_diff_usr_scr_hoop",
            "dist_diff_usr_uDF_hoop",
        ]
        super().__init__(dist_cols)

    def calc(self, df: pd.DataFrame) -> pd.DataFrame:
        df["hoop_x"] = hoop_xy[0]
        df["hoop_y"] = hoop_xy[1]

        df = calc_dists(df, usr_cols, hoop_cols)
        df = calc_dists(df, scr_cols, hoop_cols)
        df = calc_dists(df, uDF_cols, hoop_cols)
        df["dist_diff_usr_scr_hoop"] = df["dist_usr_hoop"] - df["dist_scr_hoop"]
        df["dist_diff_usr_uDF_hoop"] = df["dist_usr_hoop"] - df["dist_uDF_hoop"]
        return df


class PlayerArea(BaseAggregator):
    def __init__(self):
        area_cols = [
            "player_area",
        ]
        super().__init__(area_cols)

    def _calc_triangle_area(self, p0: tuple, p1: tuple, p2: tuple) -> float:
        """http://blog.livedoor.jp/portal8/archives/1619626.html"""
        area = (
            p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1])
        ) * 0.5
        return area

    def calc(self, df: pd.DataFrame) -> pd.DataFrame:
        pos_cols = list(scr_cols) + list(usr_cols) + list(uDF_cols)
        pos_dict_list = df.loc[:, pos_cols].to_dict(orient="records")
        areas = []
        for pos_dict in pos_dict_list:
            p0 = (pos_dict["scr_x"], pos_dict["scr_y"])
            p1 = (pos_dict["usr_x"], pos_dict["usr_y"])
            p2 = (pos_dict["uDF_x"], pos_dict["uDF_y"])
            area = self._calc_triangle_area(p0, p1, p2)
            areas.append(area)
        df["player_area"] = areas
        return df


def preprocess(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df_agg_target = aggregate_target(df)
    df_agg_player = PlayerDist().run(df)
    df_agg_hoop = HoopDist().run(df)
    df_agg_area = PlayerArea().run(df)

    df_agg = pd.concat((df_agg_target, df_agg_player, df_agg_hoop, df_agg_area), axis=1)
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
