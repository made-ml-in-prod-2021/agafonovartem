from typing import List

from py._path.local import LocalPath
import pandas as pd
import pytest

from src.enities.feature_params import FeatureParams, PCATransformer, StandardScalerTransformer, SquaredFeaturesTransformer
from src.features.build_features import (ColumnTransformer, make_features,
                                         extract_target, build_transformer,
                                         load_transformer, dump_transformer)


@pytest.fixture
def feature_params(
    categorical_features: List[str],
    features_to_drop: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
        numerical_transformers=[StandardScalerTransformer(),
                                PCATransformer(n_components=3)
                                ],
    )
    return params


def test_make_features(
    feature_params: FeatureParams, original_dataset_sample: pd.DataFrame,
):
    transformer = build_transformer(feature_params)
    transformer.fit(original_dataset_sample)
    features = make_features(transformer, original_dataset_sample)
    assert not pd.isnull(features).any().any()
    assert all(x not in features.columns for x in feature_params.features_to_drop)


def test_extract_features(feature_params: FeatureParams, original_dataset_sample: pd.DataFrame):
    target = extract_target(original_dataset_sample, feature_params)
    print(target)
    comparison = original_dataset_sample[feature_params.target_col].values == target.to_numpy()
    assert comparison.all(), (
        "Target was incorrectly extracted."
    )


@pytest.fixture()
def simple_transformer(feature_params) -> ColumnTransformer:
    return build_transformer(feature_params)


def test_dump_load_transformer(tmpdir: LocalPath,
                               simple_transformer: ColumnTransformer,
                               original_dataset_sample: pd.DataFrame):
    expected_output = tmpdir.join("pipeline.joblib")
    simple_transformer.fit(original_dataset_sample)
    real_output = dump_transformer(simple_transformer, expected_output)
    assert real_output == expected_output, (
        "Transformer was dumped incorrectly."
    )
    real_transformer = load_transformer(real_output)
    real_features = make_features(real_transformer, original_dataset_sample)
    expected_features = make_features(simple_transformer, original_dataset_sample)
    assert real_features.equals(expected_features), (
        "Transformer was loaded incorrectly."
    )


@pytest.fixture()
def feature_params_squared(feature_params: FeatureParams):
    feature_params.numerical_transformers.append(SquaredFeaturesTransformer())
    return feature_params


def test_make_features_squared(
    feature_params_squared: FeatureParams, original_dataset_sample: pd.DataFrame,
):
    print(feature_params_squared.numerical_transformers)
    transformer = build_transformer(feature_params_squared)
    transformer.fit(original_dataset_sample)
    features = make_features(transformer, original_dataset_sample)
    assert not pd.isnull(features).any().any()
    assert all(x not in features.columns for x in feature_params_squared.features_to_drop)
    assert features.shape == (20, 24)



