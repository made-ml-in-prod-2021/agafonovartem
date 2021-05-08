import os
from typing import Tuple, List

import pickle
import numpy
import pytest
from py._path.local import LocalPath
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.enities.feature_params import FeatureParams
from src.data.make_dataset import read_data
from src.features.build_features import build_transformer, make_features, extract_target
from src.models.model_fit_predict import train_model, predict_model, serialize_model
from src.enities.model_params import ModelParams


@pytest.fixture()
def features_and_target(
    original_dataset_sample_path: str, categorical_features: List[str],
    numerical_features: List[str], features_to_drop: List[str],
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
        numerical_transformers=None
    )
    data = read_data(original_dataset_sample_path)
    transformer = build_transformer(params)
    transformer.fit(data)
    features = make_features(transformer, data)
    target = extract_target(data, params)
    return features, target


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target,
                        model_params=ModelParams(name="LogisticRegression")
                        )
    assert isinstance(model, LogisticRegression)
    assert model.predict(features).shape[0] == target.shape[0]


@pytest.fixture()
def logistic_regression_params() -> dict:
    return {"penalty": "l2",
            "C": 3.5,
            "random_state": 52,
            "solver": "liblinear"
            }


def test_train_model_with_params(
        features_and_target: Tuple[pd.DataFrame, pd.Series],
        logistic_regression_params: dict):
    features, target = features_and_target
    model_params = ModelParams(name="LogisticRegression",
                               params=logistic_regression_params)
    model = train_model(features, target, model_params=model_params)
    print(model)
    assert isinstance(model, LogisticRegression)
    assert model.predict(features).shape[0] == target.shape[0]
    assert model.C == 3.5
    assert model.random_state == 52
    assert model.solver == "liblinear"


def test_predict_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target,
                        model_params=ModelParams(name="LogisticRegression")
                        )
    predicts = predict_model(model, features)
    assert isinstance(predicts, numpy.ndarray)
    comparison = model.predict(features) == predicts
    assert comparison.all()


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    model = LogisticRegression()
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, LogisticRegression)
