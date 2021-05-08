import pandas as pd

from src.data.make_dataset import read_data, train_test_split_, k_fold_cross_validation
from src.enities import TrainTestSplitParams, KFoldParams


def test_make_dataset(sample_dataset_path: str):
    dataset = read_data(sample_dataset_path)
    expected_dataset_shape = (11, 3)
    assert expected_dataset_shape == dataset.shape, (
        f"Dataset was loaded incorrectly. Shape is {dataset.shape}, "
        f"while expected shape is  {expected_dataset_shape}"
    )


def test_train_test_split(sample_dataset: pd.DataFrame):
    split_params = TrainTestSplitParams(test_size=0.2, random_state=52)
    train_data, test_data = train_test_split_(data=sample_dataset, params=split_params)
    expected_train_data_shape = (8, 3)
    assert expected_train_data_shape == train_data.shape, (
        f"Dataset was split incorrectly. Shape of train dataset is {train_data.shape}, "
        f"while expected shape is  {expected_train_data_shape}"
    )
    expected_test_data_shape = (3, 3)
    assert expected_test_data_shape == test_data.shape, (
        f"Dataset was split incorrectly. Shape of test dataset is {test_data.shape}, "
        f"while expected shape is  {expected_test_data_shape}"
    )


def test_k_fold_cross_validation(sample_dataset: pd.DataFrame):
    k_fold_params = KFoldParams(n_splits=2, random_state=None, shuffle=False)
    n_splits_list = k_fold_cross_validation(data=sample_dataset, params=k_fold_params)
    assert isinstance(n_splits_list, list), (
        f"Dataset was split into folds incorrectly. Type is {type(n_splits_list)}, "
        f"while expected type is list"
    )
    expected_n_splits_list_len = 2
    assert expected_n_splits_list_len == len(n_splits_list), (
        f"Dataset was split into folds incorrectly. Length is {len(n_splits_list)}, "
        f"while expected length is {expected_n_splits_list_len}"
    )
    assert isinstance(n_splits_list[0], tuple), (
        f"Dataset was split into folds incorrectly. Type of the fold is {type(n_splits_list[0])}, "
        f"while expected type is tuple"
    )
