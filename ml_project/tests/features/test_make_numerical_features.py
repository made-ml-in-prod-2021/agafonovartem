from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (process_numerical_features, NumericalTransformer,
                                         StandardScalerTransformer, PCATransformer)
from src.custom_transformers.squared_features import SquaredFeatures


@pytest.fixture()
def sample_dataset_numerical_feature() -> str:
    return "Age"


@pytest.fixture
def sample_dataset_numerical_data(
    sample_dataset: pd.DataFrame, sample_dataset_numerical_feature: str
) -> pd.DataFrame:
    return sample_dataset[[sample_dataset_numerical_feature]]


@pytest.fixture
def simple_imputer_transform() -> Optional:
    return None


def test_process_sample_dataset_numerical_feature_simple_imputer_transform(
        sample_dataset_numerical_data: pd.DataFrame,
        simple_imputer_transform: Optional):
    transformed: pd.DataFrame = process_numerical_features(numerical_df=sample_dataset_numerical_data,
                                                           numerical_transformers=simple_imputer_transform)
    comparison = transformed.values == sample_dataset_numerical_data.values
    assert comparison.all(), (
        f"Identity transform done wrong."
    )


@pytest.fixture
def scale_transform() -> List[StandardScalerTransformer]:
    return [StandardScalerTransformer()]


def test_process_sample_dataset_numerical_feature_scale_transform(
        sample_dataset_numerical_data: pd.DataFrame,
        scale_transform: List[StandardScalerTransformer]):
    transformed: pd.DataFrame = process_numerical_features(numerical_df=sample_dataset_numerical_data,
                                                           numerical_transformers=scale_transform)
    comparison = transformed.values != sample_dataset_numerical_data.values
    assert comparison.all(), (
        f"Normalize transform done wrong."
    )


@pytest.fixture()
def numerical_features() -> List[str]:
    return ["1", "2"]


@pytest.fixture()
def numerical_values() -> List[List[float]]:
    return [[1, 0], [0, 1], [0, 0.5], [1, 1], [0.5, 0]]


@pytest.fixture
def fake_numerical_data(
    numerical_features: str, numerical_values: List[str]
) -> pd.DataFrame:
    return pd.DataFrame(np.array(numerical_values), columns=numerical_features)


@pytest.fixture()
def numerical_values_nan() -> List[List[float]]:
    return [[1, 0], [0, 1], [0, np.nan], [1, 1], [np.nan, 0]]


@pytest.fixture
def fake_numerical_data_nan(
    numerical_features: str, numerical_values_nan: List[str]
) -> pd.DataFrame:
    return pd.DataFrame(np.array(numerical_values_nan), columns=numerical_features)


def test_process_fake_dataset_numerical_feature_simple_imputer_transform(
        fake_numerical_data: pd.DataFrame,
        simple_imputer_transform: Optional):
    transformed: pd.DataFrame = process_numerical_features(numerical_df=fake_numerical_data,
                                                           numerical_transformers=simple_imputer_transform)
    comparison = transformed.values == fake_numerical_data.values
    assert comparison.all(), (
        f"Simple impute done wrong."
    )


def test_process_fake_dataset_numerical_feature_scale_transform(
        fake_numerical_data: pd.DataFrame,
        fake_numerical_data_nan: pd.DataFrame,
        scale_transform: List[StandardScalerTransformer]):
    transformed: pd.DataFrame = process_numerical_features(numerical_df=fake_numerical_data,
                                                           numerical_transformers=scale_transform)
    transformed_nan: pd.DataFrame = process_numerical_features(numerical_df=fake_numerical_data_nan,
                                                               numerical_transformers=scale_transform)
    comparison = transformed.values == transformed_nan.values
    assert comparison.all(), (
        "Standard scaling done wrong."
    )


@pytest.fixture
def scale_pca_transform() -> List[NumericalTransformer]:
    return [StandardScalerTransformer(), PCATransformer(n_components=1)]


@pytest.fixture()
def numerical_values_dependent() -> List[List[float]]:
    return [[1, 0], [1, 0], [0.5, -0.5], [0.5, -0.5], [0, -1]]


@pytest.fixture
def fake_numerical_data_dependent(
    numerical_features: str, numerical_values_dependent: List[str]
) -> pd.DataFrame:
    return pd.DataFrame(np.array(numerical_values_dependent), columns=numerical_features)


def check_collinearity(vector1, vector2):
    return len(set(np.around(vector1 / vector2, decimals=5))) == 1


def test_process_fake_dataset_numerical_feature_scale_pca_transform(
        fake_numerical_data_dependent: pd.DataFrame,
        scale_pca_transform: List[StandardScalerTransformer],
        scale_transform: List[NumericalTransformer]):
    pca_transformed: pd.DataFrame = process_numerical_features(numerical_df=fake_numerical_data_dependent,
                                                               numerical_transformers=scale_pca_transform)
    scale_transformed: pd.DataFrame = process_numerical_features(numerical_df=fake_numerical_data_dependent,
                                                                 numerical_transformers=scale_transform)
    assert check_collinearity(pca_transformed.values.reshape(5), scale_transformed[0].values), (
        "PCA performed on dependent (collinear) data with 2 features and one principal component,"
        "did not output collinear (dependent) vector"
    )


def test_make_squared_features(fake_numerical_data_dependent: pd.DataFrame):
    squared_features = SquaredFeatures()
    squared_features.fit(data=fake_numerical_data_dependent.values)
    data_transformed = squared_features.transform(data=fake_numerical_data_dependent.values)
    assert len(data_transformed.T) == 5
