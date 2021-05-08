from typing import Tuple, List

import pandas as pd

from src.enities import TrainTestSplitParams, KFoldParams
from sklearn.model_selection import train_test_split, KFold


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def train_test_split_(data: pd.DataFrame, params: TrainTestSplitParams
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(data,
                            test_size=params.test_size,
                            random_state=params.random_state,
                            )


def k_fold_cross_validation(data: pd.DataFrame, params: KFoldParams
                            ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    k_fold = KFold(random_state=params.random_state,
                   n_splits=params.n_splits,
                   shuffle=params.shuffle,
                   )

    n_splits_list = []

    for train_index, test_index in k_fold.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        n_splits_list.append((train_data, test_data))

    return n_splits_list
