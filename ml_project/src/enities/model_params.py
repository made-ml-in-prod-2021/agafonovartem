from typing import Optional
from dataclasses import dataclass, field


@dataclass()
class ModelParams:
    name: str
    params: Optional[dict] = field(default=None)


def read_model_params(yaml_dict: dict) -> ModelParams:
    model_params_dict = yaml_dict.get("model_params")
    print(model_params_dict)
    return ModelParams(model_params_dict.get("name"),
                       model_params_dict.get("params"))
