from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from src.utils.utils import calc_dists
from src.utils.visualize import BasketCourt


@dataclass
class BasketColumns:
    scr: Tuple[str, str] = ("scr_x", "scr_y")
    usr: Tuple[str, str] = ("usr_x", "usr_y")
    uDF: Tuple[str, str] = ("uDF_x", "uDF_y")
    ball: Tuple[str, str] = ("bal_x", "bal_y")
    hoop: Tuple[str, str] = ("hoop_x", "hoop_y")


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
        df = calc_dists(df, BasketColumns.usr, BasketColumns.scr)
        df = calc_dists(df, BasketColumns.usr, BasketColumns.uDF)
        df = calc_dists(df, BasketColumns.scr, BasketColumns.uDF)
        df = calc_dists(df, BasketColumns.usr, BasketColumns.ball)
        df = calc_dists(df, BasketColumns.scr, BasketColumns.ball)
        df = calc_dists(df, BasketColumns.uDF, BasketColumns.ball)
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
        df["hoop_x"] = BasketCourt.hoop_xy[0]
        df["hoop_y"] = BasketCourt.hoop_xy[1]

        df = calc_dists(df, BasketColumns.usr, BasketColumns.hoop)
        df = calc_dists(df, BasketColumns.scr, BasketColumns.hoop)
        df = calc_dists(df, BasketColumns.uDF, BasketColumns.hoop)
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
        pos_cols = (
            list(BasketColumns.scr) + list(BasketColumns.usr) + list(BasketColumns.uDF)
        )
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
