from typing import List

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import process_categorical_features


@pytest.fixture()
def sample_dataset_categorical_feature() -> str:
    return "Sex"


@pytest.fixture
def sample_dataset_categorical_data(
    sample_dataset: pd.DataFrame, sample_dataset_categorical_feature: str
) -> pd.DataFrame:
    return sample_dataset[[sample_dataset_categorical_feature]]


def test_process_sample_dataset_categorical_feature(sample_dataset_categorical_data: pd.DataFrame):
    transformed: pd.DataFrame = process_categorical_features(sample_dataset_categorical_data)
    assert transformed.shape[1] == 2
    assert transformed.sum().sum() == sample_dataset_categorical_data.shape[0]


@pytest.fixture()
def categorical_feature() -> str:
    return "gender"


@pytest.fixture()
def categorical_values() -> List[str]:
    return ["agender", "androgyne", "bigender", "butch", "cisgender", "gender_expansive",
            "genderfluid", "gender_outlaw", "genderqueer", "masculine_of_center",
            "nonbinary", "omnigender", "polygender", "transgender", "trans", "two_spirit"]


@pytest.fixture()
def categorical_values_with_nan(categorical_values: List[str]) -> List[str]:
    return categorical_values + [np.nan]


@pytest.fixture
def fake_categorical_data(
    categorical_feature: str, categorical_values_with_nan: List[str]
) -> pd.DataFrame:
    return pd.DataFrame({categorical_feature: categorical_values_with_nan})


def test_process_fake_categorical_features(
    fake_categorical_data: pd.DataFrame,
):
    transformed: pd.DataFrame = process_categorical_features(fake_categorical_data)
    assert transformed.shape[1] == 16
    assert transformed.sum().sum() == 17
