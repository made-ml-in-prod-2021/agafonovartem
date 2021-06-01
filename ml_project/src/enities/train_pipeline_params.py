from typing import Union

from dataclasses import dataclass


from .validation_params import ValParams, TrainTestSplitParams, KFoldParams, read_val_params
from .feature_params import FeatureParams, read_feature_params
from .model_params import ModelParams, read_model_params


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    output_transformer_path: str
    metric_path: str
    validation_params: Union[ValParams, TrainTestSplitParams, KFoldParams]
    feature_params: FeatureParams
    model_params: ModelParams


def read_training_pipeline_params(yaml_dict: dict) -> TrainingPipelineParams:
    input_data_path = yaml_dict.get("input_data_path")
    output_model_path = yaml_dict.get("output_model_path")
    output_transformer_path = yaml_dict.get("output_transformer_path")
    metric_path = yaml_dict.get("metric_path")
    validation_params = read_val_params(yaml_dict)
    feature_params = read_feature_params(yaml_dict)
    model_params = read_model_params(yaml_dict)
    return TrainingPipelineParams(
        input_data_path=input_data_path,
        output_model_path=output_model_path,
        output_transformer_path=output_transformer_path,
        metric_path=metric_path,
        validation_params=validation_params,
        feature_params=feature_params,
        model_params=model_params,
    )
