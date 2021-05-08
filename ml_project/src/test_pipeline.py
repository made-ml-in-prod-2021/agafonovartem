import logging
import sys

import click
import numpy as np
import pandas as pd

from src.enities._test_pipeline_params import _TestingPipelineParams
from src.data.make_dataset import read_data
from src.features.build_features import load_transformer
from src.models.model_fit_predict import load_model, predict_model
from src.features.build_features import make_features
from src.enities.pipeline_params import read_pipeline_params


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def _test_pipeline(test_pipeline_params: _TestingPipelineParams) -> np.ndarray:
    logger.info(f"Test pipeline is starting with params {test_pipeline_params}")
    data = read_data(test_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    transformer = load_transformer(test_pipeline_params.input_transformer_path)
    data_features = make_features(transformer, data)

    logger.info(f"data_features.shape is {data_features.shape}")

    model = load_model(test_pipeline_params.input_model_path)
    predicts = predict_model(model, data_features)
    pd.DataFrame(predicts).to_csv(test_pipeline_params.output_predictions_path)
    return predicts


@click.command(name="test_pipeline")
@click.argument("config_path")
def test_pipeline_command(config_path: str):
    params = read_pipeline_params(config_path)
    _test_pipeline(params)

if __name__ == "__main__":
    test_pipeline_command()