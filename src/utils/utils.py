from typing import Tuple

import pandas as pd


def calc_dist(pos0: Tuple[float, float], pos1: Tuple[float, float]) -> float:
    dist = ((pos0[0] - pos1[0]) ** 2 + (pos0[1] - pos1[1]) ** 2) ** 0.5
    return dist


def calc_dists(
    df: pd.DataFrame, pos0_col: Tuple[str, str], pos1_col: Tuple[str, str]
) -> pd.DataFrame:
    dists = [
        calc_dist((pos0_x, pos0_y), (pos1_x, pos1_y))
        for pos0_x, pos0_y, pos1_x, pos1_y in zip(
            df[pos0_col[0]], df[pos0_col[1]], df[pos1_col[0]], df[pos1_col[1]]
        )
    ]
    new_col = "dist_" + pos0_col[0].split("_")[0] + "_" + pos1_col[0].split("_")[0]
    df[new_col] = dists
    return df
