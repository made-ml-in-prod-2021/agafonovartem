from typing import Union

from .train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params
from ._test_pipeline_params import _TestingPipelineParams, read_test_pipeline_params
import yaml


def read_pipeline_params_to_dict(path: str) -> dict:
    with open(path, "r") as input_stream:
        yaml_dict = yaml.safe_load(input_stream)
    return yaml_dict


def read_pipeline_params(path: str) -> Union[TrainingPipelineParams, _TestingPipelineParams]:
    yaml_dict = read_pipeline_params_to_dict(path)
    mode = yaml_dict.pop("mode", None)
    if mode == "Train" or mode is None:
        return read_training_pipeline_params(yaml_dict)
    elif mode == "Test":
        return read_test_pipeline_params(yaml_dict)
    else:
        raise NotImplementedError
