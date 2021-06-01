from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class NumericalTransformer:
    name: str


@dataclass()
class StandardScalerTransformer(NumericalTransformer):
    name: str = field(default="StandardScaler")


@dataclass()
class PCATransformer(NumericalTransformer):
    name: str = field(default="PCA")
    n_components: Optional[int] = field(default=None)


@dataclass()
class SquaredFeaturesTransformer(NumericalTransformer):
    name: str = field(default="SquaredFeatures")


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    features_to_drop: List[str]
    target_col: str
    numerical_transformers: Optional[List[NumericalTransformer]]


def read_feature_params(yaml_dict: dict) -> FeatureParams:
    validation_params_dict = yaml_dict.get("feature_params")
    numerical_transformers_list_of_dicts = validation_params_dict.get("numerical_transformers")
    numerical_transformers_list = read_numerical_transformers_params(numerical_transformers_list_of_dicts)
    feature_params = FeatureParams(
        categorical_features=validation_params_dict.get("categorical_features"),
        numerical_features=validation_params_dict.get("numerical_features"),
        features_to_drop=validation_params_dict.get("features_to_drop"),
        target_col=validation_params_dict.get("target_col"),
        numerical_transformers=numerical_transformers_list,
    )
    return feature_params


def read_numerical_transformers_params(numerical_transformers_list_of_dicts: Optional[List[dict]]
                                       ) -> Optional[List[NumericalTransformer]]:
    if numerical_transformers_list_of_dicts is None:
        return None
    numerical_transformers_list = []
    for numerical_transformer in numerical_transformers_list_of_dicts:
        if numerical_transformer.get("name") == "StandardScaler":
            normalize_transformer = StandardScalerTransformer()
            numerical_transformers_list.append(normalize_transformer)
        elif numerical_transformer.get("name") == "PCA":
            n_components = numerical_transformer.get("n_components")
            pca_transformer = PCATransformer(n_components=n_components)
            numerical_transformers_list.append(pca_transformer)
        elif numerical_transformer.get("name") == "SquaredFeatures":
            squared_features_transformer = SquaredFeaturesTransformer()
            numerical_transformers_list.append(squared_features_transformer)
        else:
            raise NotImplementedError("No such numerical transformer")
    return numerical_transformers_list
