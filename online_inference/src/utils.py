from typing import Union, List
import pickle

import pandas as pd
from dataclasses import dataclass, field
from pydantic import BaseModel, conlist
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer


FEATURE_TO_TYPE = {
    "age": int,
    "sex": int,
    "cp": int,
    "trestbps": float,
    "chol": float,
    "fbs": int,
    "restecg": int,
    "thalach": float,
    "exang": int,
    "oldpeak": float,
    "slope": int,
    "ca": int,
    "thal": int,
}


CATEGORICAL_FEATURE_TO_VALUES = {
    "sex": {0, 1},
    "cp": {0, 1, 2, 3},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {0, 1, 2},
    "ca": {0, 1, 2, 3},
    "thal": {1, 2, 3},
}

FEATURES_RIGHT_ORDER = list(FEATURE_TO_TYPE.keys())


SklearnClassificationModel = Union[LogisticRegression, DecisionTreeClassifier]


def load_object(input_: str) -> Union[SklearnClassificationModel, ColumnTransformer]:
    with open(input_, "rb") as f:
        model = pickle.load(f)
    return model


class HeartDiseaseModel(BaseModel):
    data: List[conlist(Union[int, str, None], min_items=13, max_items=13)]
    features: conlist(str, min_items=13, max_items=13)


class HeartDiseaseResponse(BaseModel):
    disease: int


def make_predict(
        data: List, features: List[str],
        model: SklearnClassificationModel,
        transformer: ColumnTransformer,
) -> List[HeartDiseaseResponse]:
    data = pd.DataFrame(data, columns=features)
    data_transformed = pd.DataFrame(transformer.transform(data))
    predictions = model.predict(data_transformed)

    return [
        HeartDiseaseResponse(disease=prediction) for prediction in predictions
    ]


def validate_features_same_order(input_features: List[str]):
    return FEATURES_RIGHT_ORDER == input_features


def validate_features_right_type(request: HeartDiseaseModel):
    for object_ in request.data:
        for feature_val, feature_name in zip(object_, FEATURES_RIGHT_ORDER):
            if CATEGORICAL_FEATURE_TO_VALUES.get(feature_name):
                if feature_val not in CATEGORICAL_FEATURE_TO_VALUES.get(feature_name):
                    return False
    return True
