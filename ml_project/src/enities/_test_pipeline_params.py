from dataclasses import dataclass


@dataclass()
class _TestingPipelineParams:
    input_data_path: str
    input_model_path: str
    input_transformer_path: str
    output_predictions_path: str


def read_test_pipeline_params(yaml_dict: dict) -> _TestingPipelineParams:
    input_data_path = yaml_dict.get("input_data_path")
    input_model_path = yaml_dict.get("input_model_path")
    input_transformer_path = yaml_dict.get("input_transformer_path")
    output_predictions_path = yaml_dict.get("output_predictions_path")
    return _TestingPipelineParams(
        input_data_path=input_data_path,
        input_model_path=input_model_path,
        input_transformer_path=input_transformer_path,
        output_predictions_path=output_predictions_path
    )
