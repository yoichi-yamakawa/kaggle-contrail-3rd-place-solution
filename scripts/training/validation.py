import random
from collections import Counter, defaultdict
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def get_kfolds(
    df: pd.DataFrame, n_folds: int, is_stratified: bool = True, shuffle: bool = True, seed: int = 773
) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    if is_stratified:
        print("--- stratified KFold ---")
        kf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
        return list(kf.split(df.index.values, df.target.values))
    else:
        kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
        return list(kf.split(df.index.values, df.index.values))


def get_stratified_group_kolds(
    df: pd.DataFrame,
    n_folds: int,
    target_name: str,
    group_name: str,
    shuffle: bool = True,
    seed: int = 773,
) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    kf = StratifiedGroupKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)

    return list(kf.split(df, df[target_name], df[group_name]))


if __name__ == "__main__":
    _test_get_folds()
