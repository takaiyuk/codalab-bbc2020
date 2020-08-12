import gc
import os

import pandas as pd

from src.const import DataPath
from src.utils.joblib import Jbl


def join_data():
    train_files = os.listdir(DataPath.raw.train_dir)
    test_files = os.listdir(DataPath.raw.test_dir)

    for files, is_train in zip([train_files, test_files], [True, False]):
        dfs = []
        for f in files:
            path = f"{DataPath.raw.train_dir}/{f}"
            df_tmp = pd.read_csv(path)
            stem = os.path.splitext(f)[0]
            if int(stem) < 400:
                df_tmp["is_screen_play"] = 1
            else:
                df_tmp["is_screen_play"] = 0
            df_tmp["filename"] = stem
            dfs.append(df_tmp)
        df = pd.concat(dfs, axis=0, ignore_index=True)
        df = df.sort_values(["filename", "frame"]).reset_index(drop=True)
        del dfs
        gc.collect()

        save_path = (
            f"{DataPath.interim.train}.jbl"
            if is_train
            else f"{DataPath.interim.test}.jbl"
        )
        Jbl.save(df, save_path)
