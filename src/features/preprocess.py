import pandas as pd

from src.config.config import Config
from src.const import DataPath
from src.features.aggregate import aggregate_target
from src.utils.joblib import Jbl


def _build_features(df: pd.DataFrame, is_train: bool, fe_cfg: Config) -> pd.DataFrame:
    feature = fe_cfg.feature
    df_agg_target = pd.DataFrame()
    if is_train:
        df_agg_target = aggregate_target(df)
    df_aggs = []
    for _, f in feature.__annotations__.items():
        df_agg_ = f().run(df)
        df_aggs.append(df_agg_)
    if is_train:
        df_aggs.append(df_agg_target)
    df_agg = pd.concat(df_aggs, axis=1)
    return df_agg


def _filter_frame(
    df: pd.DataFrame, frame_start_q: float, frame_end_q: float
) -> pd.DataFrame:
    dfs = []
    for filename in df.filename.unique():
        df_file = df[df["filename"] == filename]
        frame_start = df_file.frame.quantile(frame_start_q)
        frame_end = df_file.frame.quantile(frame_end_q)
        df_clipped = df_file[
            (df_file["frame"] >= frame_start) & (df_file["frame"] <= frame_end)
        ]
        dfs.append(df_clipped)
    df_ = pd.concat(dfs, axis=0, ignore_index=True)
    return df_


def preprocess(fe_cfg: Config):
    fe_name = fe_cfg.basic.name
    target_col = fe_cfg.column.target
    train_path = f"{DataPath.interim.train}.jbl"
    test_path = f"{DataPath.interim.test}.jbl"

    for path, is_train in zip([train_path, test_path], [True, False]):
        df = Jbl.load(path)
        if "frame" in fe_cfg.__annotations__:
            frame_start_q = fe_cfg.frame.start
            frame_end_q = fe_cfg.frame.end
            df = _filter_frame(df, frame_start_q, frame_end_q)
        df_processed = _build_features(df, is_train, fe_cfg)
        if is_train:
            X = df_processed.drop(target_col, axis=1)
            y = df_processed[target_col]
        else:
            X = df_processed.copy()
            y = None
        X_save_path = (
            f"{DataPath.processed.X_train}_{fe_name}.jbl"
            if is_train
            else f"{DataPath.processed.X_test}_{fe_name}.jbl"
        )
        Jbl.save(X, X_save_path)
        if is_train:
            y_save_path = f"{DataPath.processed.y_train}_{fe_name}.jbl"
            Jbl.save(y, y_save_path)
