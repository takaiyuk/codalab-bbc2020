import hydra
import joblib
import os
import pandas as pd
from typing import Any, Tuple
import yaml
import warnings

warnings.filterwarnings("ignore")


def get_hydra_session_id() -> str:
    """hydra cwd: ${project_path}/outputs/YYYY-mm-dd/HH-MM-SS"""
    hydra_cwd = os.getcwd()
    session_id = "-".join(hydra_cwd.split("/")[-2:])
    return session_id


def get_original_cwd() -> str:
    try:
        return hydra.utils.get_original_cwd()
    except AttributeError:
        return "."


def load_yaml(path: str = "./config.yml") -> dict:
    with open(path) as f:
        config = yaml.load(f)
    k_method = config["params"]["kfold"]["method"]
    assert k_method in ["normal", "stratified", "group"]
    return config


def load_joblib(path: str) -> Any:
    return joblib.load(path)


def save_joblib(obj_: Any, path: str) -> None:
    joblib.dump(obj_, path, compress=3)


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


def mkdir(path: str) -> None:
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
