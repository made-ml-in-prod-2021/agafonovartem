import pickle
from typing import Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier

from src.enities.model_params import ModelParams

SklearnClassificationModel = Union[LogisticRegression, DecisionTreeClassifier]


def train_model(
        features: pd.DataFrame, target: pd.Series, model_params: ModelParams
) -> SklearnClassificationModel:
    if model_params.name == "LogisticRegression":
        if model_params.params:
            model = LogisticRegression(**model_params.params)
        else:
            model = LogisticRegression()
    elif model_params.name == "DecisionTree":
        if model_params.params:
            model = DecisionTreeClassifier(**model_params.params)
        else:
            model = DecisionTreeClassifier()
    else:
        raise NotImplementedError("No such model.")
    model.fit(features, target)
    return model


def predict_model(
        model: SklearnClassificationModel, features: pd.DataFrame
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
        predicts: np.ndarray, target: pd.Series
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(predicts, target),
        "precision": precision_score(predicts, target),
        "recall": recall_score(predicts, target),
        "roc_auc": roc_auc_score(predicts, target)
    }


def serialize_model(model: SklearnClassificationModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(input_: str) -> SklearnClassificationModel:
    with open(input_, "rb") as f:
        model = pickle.load(f)
    return model
