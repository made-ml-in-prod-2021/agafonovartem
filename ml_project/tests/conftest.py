from typing import List
import os

import pytest
import pandas as pd

from src.data.make_dataset import read_data

CURDIR = os.path.dirname(__file__)
SAMPLE_DATASET_FILENAME = "datasets/generated_dataset_sample.csv"
ORIGINAL_DATASET_SAMPLE_FILENAME = "datasets/original_dataset_sample.csv"
CONFIG_TRAIN_TEST_SPLIT_FILENAME = \
    "configs_data/config_val_params_train_test_split.yaml"
CONFIG_K_FOLD_CROSS_VALIDATION_SHUFFLE_FILENAME = \
    "configs_data/config_val_params_k_fold_cross_validation_shuffle.yaml"
CONFIG_K_FOLD_CROSS_VALIDATION_NO_SHUFFLE_FILENAME = \
    "configs_data/config_val_params_k_fold_cross_validation_no_shuffle.yaml"
CONFIG_FEATURE_PARAMS_FILENAME = \
    "configs_data/config_feature_params.yaml"
CONFIG_MODEL_PARAMS_FILENAME = "configs_data/config_model_params.yaml"
CONFIG_ALL_FILENAME = "configs_data/config_all.yaml"
CONFIG_TEST_FILENAME = "configs_data/config_test.yaml"


@pytest.fixture()
def sample_dataset_path():
    return os.path.join(CURDIR, SAMPLE_DATASET_FILENAME)


@pytest.fixture
def sample_dataset(sample_dataset_path: str) -> pd.DataFrame:
    return read_data(sample_dataset_path)


@pytest.fixture()
def config_val_train_test_split_params_path():
    return os.path.join(CURDIR, CONFIG_TRAIN_TEST_SPLIT_FILENAME)


@pytest.fixture()
def config_k_fold_cross_validation_shuffle_params_path():
    return os.path.join(CURDIR, CONFIG_K_FOLD_CROSS_VALIDATION_SHUFFLE_FILENAME)


@pytest.fixture()
def config_k_fold_cross_validation_no_shuffle_params_path():
    return os.path.join(CURDIR, CONFIG_K_FOLD_CROSS_VALIDATION_NO_SHUFFLE_FILENAME)


@pytest.fixture()
def config_feature_params_path():
    return os.path.join(CURDIR, CONFIG_FEATURE_PARAMS_FILENAME)


@pytest.fixture()
def config_model_params_path():
    return os.path.join(CURDIR, CONFIG_MODEL_PARAMS_FILENAME)


@pytest.fixture()
def config_all_path():
    return os.path.join(CURDIR, CONFIG_ALL_FILENAME)


@pytest.fixture()
def config_test_path():
    return os.path.join(CURDIR, CONFIG_TEST_FILENAME)


@pytest.fixture()
def original_dataset_sample_path():
    return os.path.join(CURDIR, ORIGINAL_DATASET_SAMPLE_FILENAME)


@pytest.fixture()
def original_dataset_sample(original_dataset_sample_path: str) -> pd.DataFrame:
    return read_data(original_dataset_sample_path)


@pytest.fixture()
def target_col():
    return "target"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope"
    ]


@pytest.fixture()
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
        "ca"
    ]


@pytest.fixture()
def features_to_drop() -> List[str]:
    return ["thal"]

