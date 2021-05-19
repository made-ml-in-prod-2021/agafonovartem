import os
from typing import List

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app import app
from src.utils import make_predict, load_object
from random import shuffle


client = TestClient(app)

CURDIR = os.path.dirname(__file__)
MODEL_PATH = "data/model.pkl"
TRANSFORMER_PATH = "data/transformer.pkl"
DATASET_PATH = "data/data_test.csv"


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "it is entry point of our predictor"


def pytest_generate_tests():
    os.environ['PATH_TO_MODEL'] = os.path.join(CURDIR, MODEL_PATH)
    os.environ['PATH_TO_TRANSFORMER'] = os.path.join(CURDIR, TRANSFORMER_PATH)


def test_model_health():
    with client:
        response = client.get("/healz/model")
        assert response.status_code == 200
        assert response.json()


def test_transformer_health():
    with client:
        response = client.get("/healz/transformer")
        assert response.status_code == 200
        assert response.json()


@pytest.fixture()
def request_data():
    data_path = os.path.join(CURDIR, DATASET_PATH)
    data = pd.read_csv(data_path)
    return data.values[:, :-1].tolist()


@pytest.fixture()
def request_features():
    data_path = os.path.join(CURDIR, DATASET_PATH)
    data = pd.read_csv(data_path)
    return data.columns[:-1].tolist()


@pytest.fixture()
def model():
    return load_object(os.path.join(CURDIR, MODEL_PATH))


@pytest.fixture()
def transformer():
    return load_object(os.path.join(CURDIR, TRANSFORMER_PATH))


def test_make_predict(request_data, request_features, model, transformer):
    with client:
        response = client.get("/predict/",
                              json={"data": request_data,
                                    "features": request_features})
        assert response.status_code == 200
        assert isinstance(response.json(), List)
        assert response.json() == make_predict(request_data,
                                               request_features,
                                               model,
                                               transformer)


def test_make_predict_incorrect_request():
    with client:
        response = client.get("/predict/", json={"Hot": "Rats"})
        assert response.status_code == 422


def test_make_predict_wrong_feature_order(request_data, request_features):
    with client:
        shuffle(request_features)
        response = client.get("/predict/",
                              json={"data": request_data,
                                    "features": request_features})
        assert response.status_code == 400
        assert response.json() == {"detail": "Wrong feature order!"}


@pytest.fixture()
def request_data_wrong_value():
    return [[63, "boris", 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]


def test_make_predict_data_wrong_value(request_data_wrong_value, request_features):
    with client:
        response = client.get("/predict/",
                              json={"data": request_data_wrong_value,
                                    "features": request_features})
        assert response.status_code == 400
        print(response.json())
        assert response.json() == {"detail": "Wrong data values!"}


@pytest.fixture()
def request_data_wrong_value_v2():
    return [[63, -2.5, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]


def test_make_predict_data_wrong_value_v2(request_data_wrong_value_v2, request_features):
    with client:
        response = client.get("/predict/",
                              json={"data": request_data_wrong_value_v2,
                                    "features": request_features})
        assert response.status_code == 400
        assert response.json() == {"detail": "Wrong data values!"}
