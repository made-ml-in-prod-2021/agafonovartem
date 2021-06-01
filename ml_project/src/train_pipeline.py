import os
import json
import logging
import sys

import click
import pandas as pd
import numpy as np

from src.enities.train_pipeline_params import TrainingPipelineParams
from src.enities.pipeline_params import read_pipeline_params
from src.data.make_dataset import read_data, k_fold_cross_validation, train_test_split_
from src.features.build_features import build_transformer
from src.models.model_fit_predict import train_model, predict_model, evaluate_model, serialize_model, SklearnClassificationModel
from src.features.build_features import make_features, extract_target, dump_transformer
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def apply_transformer(train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      training_pipeline_params: TrainingPipelineParams)\
        -> ((pd.DataFrame, pd.DataFrame), (pd.DataFrame, pd.DataFrame), Pipeline):
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    train_features = make_features(transformer, train_df)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    logger.info(f"train_features.shape is {train_features.shape}")

    test_features = make_features(transformer, test_df)
    test_target = extract_target(test_df, training_pipeline_params.feature_params)

    logger.info(f"test_features.shape is {test_features.shape}")

    return (train_features, train_target), (test_features, test_target), transformer


def apply_model(train_features_target: (pd.DataFrame, pd.DataFrame),
                test_features_target: (pd.DataFrame, pd.DataFrame),
                training_pipeline_params: TrainingPipelineParams
                ) -> (SklearnClassificationModel, np.ndarray, dict):
    train_features, train_target = train_features_target
    test_features, test_target = test_features_target
    model = train_model(train_features, train_target, training_pipeline_params.model_params)
    predicts = predict_model(model, test_features)
    metrics = evaluate_model(predicts, test_target)
    return model, predicts, metrics


def train_k_fold_cross_validation_pipeline(data: pd.DataFrame,
                                           training_pipeline_params: TrainingPipelineParams
                                           ) -> (dict, dict):
    logger.info(f"{training_pipeline_params.validation_params.n_splits} fold cross validation")
    train_test_data = k_fold_cross_validation(data, training_pipeline_params.validation_params)
    metrics_dict = {}
    paths_to_model_dict = {}
    paths_to_transformer_dict = {}
    for i, (train_df, test_df) in enumerate(train_test_data):
        logger.info(f"{i} split out of {training_pipeline_params.validation_params.n_splits}")
        logger.info(f"train.shape {train_df.shape} test shape {test_df.shape}")
        train_features_target, test_features_target, transformer = apply_transformer(train_df,
                                                                                     test_df,
                                                                                     training_pipeline_params)

        output_transformer_filename, extension = os.path.splitext(training_pipeline_params.output_transformer_path)
        output_transformer_path = output_transformer_filename + str(i) + extension
        path_to_transformer = dump_transformer(transformer, output_transformer_path)
        paths_to_transformer_dict[i] = path_to_transformer

        model, predicts, metrics = apply_model(train_features_target,
                                               test_features_target,
                                               training_pipeline_params)
        metrics_dict[i] = metrics
        logger.info(f"{i} split metrics are {metrics}")

        filename, extension = os.path.splitext(training_pipeline_params.output_model_path)
        path = filename + str(i) + extension
        path_to_model = serialize_model(model, path)
        paths_to_model_dict[i] = path_to_model
        logger.info(f"{i} split model is dumped to the file {os.path.split(path_to_model)[1]}")

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics_dict, metric_file)

    logger.info(f"metrics is {metrics_dict}")
    return metrics_dict, paths_to_model_dict, paths_to_transformer_dict


def train_test_split_pipeline(data: pd.DataFrame,
                              training_pipeline_params: TrainingPipelineParams) -> (dict, str):
    train_df, test_df = train_test_split_(data, training_pipeline_params.validation_params)
    train_features_target, test_features_target, transformer = apply_transformer(train_df,
                                                                                 test_df,
                                                                                 training_pipeline_params)
    transformer_path = dump_transformer(transformer, training_pipeline_params.output_transformer_path)
    model, predicts, metrics = apply_model(train_features_target, test_features_target, training_pipeline_params)
    model_path = serialize_model(model, training_pipeline_params.output_model_path)
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")
    return metrics, model_path, transformer_path


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"Train pipeline is starting with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    if training_pipeline_params.validation_params.val_name == "KFoldCrossValidation":
        metrics_dict, paths_to_model_dict, paths_to_transformer_dict = \
            train_k_fold_cross_validation_pipeline(data,
                                                   training_pipeline_params)
        return metrics_dict, paths_to_model_dict, paths_to_transformer_dict
    elif training_pipeline_params.validation_params.val_name == "TrainTestSplit":
        metrics, model_path, transformer_path = train_test_split_pipeline(data, training_pipeline_params)
        return metrics, model_path, transformer_path
    else:
        raise NotImplementedError()


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
