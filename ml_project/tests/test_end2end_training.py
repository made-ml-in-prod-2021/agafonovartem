import os
from typing import List

from py._path.local import LocalPath

from src.train_pipeline import train_pipeline
from src.enities import (
    TrainingPipelineParams,
    KFoldParams,
    FeatureParams,
    TrainTestSplitParams,
    ModelParams,
)


def test_train_e2e_train_test_split(
    tmpdir: LocalPath,
    original_dataset_sample_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
    features_to_drop: List[str],
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_output_transformer_path = tmpdir.join("transformer.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        input_data_path=original_dataset_sample_path,
        output_model_path=expected_output_model_path,
        output_transformer_path=expected_output_transformer_path,
        metric_path=expected_metric_path,
        validation_params=TrainTestSplitParams(),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop,
            numerical_transformers=None,
        ),
        model_params=ModelParams(name="LogisticRegression"),
    )
    metrics, real_model_path, real_transformer_path = train_pipeline(params)
    assert metrics["roc_auc"] > 0
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)
    assert os.path.exists(real_transformer_path)


def test_train_e2e_k_fold_cross_validation(
    tmpdir: LocalPath,
    original_dataset_sample_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
    features_to_drop: List[str],
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_output_transformer_path = tmpdir.join("transformer.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        input_data_path=original_dataset_sample_path,
        output_model_path=expected_output_model_path,
        output_transformer_path=expected_output_transformer_path,
        metric_path=expected_metric_path,
        validation_params=KFoldParams(n_splits=2, random_state=52, shuffle=True),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop,
            numerical_transformers=None,
        ),
        model_params=ModelParams(name="LogisticRegression"),
    )
    metrics_dict, model_paths_dict, transformer_paths_dict = train_pipeline(params)
    assert metrics_dict[1]["roc_auc"] > 0
    assert os.path.exists(model_paths_dict[0])
    assert os.path.exists(transformer_paths_dict[1])
