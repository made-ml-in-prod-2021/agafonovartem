from typing import List
from src.enities.train_pipeline_params import TrainingPipelineParams
from src.enities.pipeline_params import read_pipeline_params, read_pipeline_params_to_dict
from src.enities.validation_params import TrainTestSplitParams, KFoldParams, read_val_params
from src.enities.feature_params import FeatureParams, StandardScalerTransformer, PCATransformer, read_feature_params
from src.enities.model_params import read_model_params
from src.enities._test_pipeline_params import _TestingPipelineParams


def test_read_config(config_val_train_test_split_params_path: str):
    yaml_dict = read_pipeline_params_to_dict(config_val_train_test_split_params_path)
    expected_yaml_dict_length = 3
    assert len(yaml_dict.get("validation_params")) == expected_yaml_dict_length, (
        f".yaml file was loaded incorrectly."
    )


def test_read_train_test_split_params_from_dict(config_val_train_test_split_params_path: str):
    yaml_dict = read_pipeline_params_to_dict(config_val_train_test_split_params_path)
    train_test_split_params = read_val_params(yaml_dict)
    assert isinstance(train_test_split_params, TrainTestSplitParams), (
        ".yaml config was parsed incorrectly. Wrong type."
    )
    expected_train_test_split_params_test_size = 0.5
    assert train_test_split_params.test_size == expected_train_test_split_params_test_size, (
        f".yaml config was parsed incorrectly. Test size is {train_test_split_params.test_size}, "
        f"while expected test size is {expected_train_test_split_params_test_size}"
    )
    expected_train_test_split_params_random_state = 10
    assert train_test_split_params.random_state == expected_train_test_split_params_random_state, (
        f".yaml config was parsed incorrectly. Number of splits is {train_test_split_params.random_state}, "
        f"while expected number of splits is {expected_train_test_split_params_random_state}"
    )


def test_read_k_fold_cross_validation_shuffle_params_from_dict(config_k_fold_cross_validation_shuffle_params_path: str):
    yaml_dict = read_pipeline_params_to_dict(config_k_fold_cross_validation_shuffle_params_path)
    train_test_split_params = read_val_params(yaml_dict)
    assert isinstance(train_test_split_params, KFoldParams), (
        ".yaml config was parsed incorrectly. Wrong type."
    )
    expected_train_test_split_params_n_splits = 7
    assert train_test_split_params.n_splits == expected_train_test_split_params_n_splits, (
        f".yaml config was parsed incorrectly. Number of splits is {train_test_split_params.n_splits}, "
        f"while expected number of splits is {expected_train_test_split_params_n_splits}"
    )
    expected_train_test_split_params_random_state = 10
    assert train_test_split_params.random_state == expected_train_test_split_params_random_state, (
        f".yaml config was parsed incorrectly. Random state is {train_test_split_params.random_state}, "
        f"while expected random state is {expected_train_test_split_params_random_state}"
    )
    expected_train_test_split_params_shuffle = True
    assert train_test_split_params.shuffle == expected_train_test_split_params_shuffle, (
        f".yaml config was parsed incorrectly. Shuffle is {train_test_split_params.shuffle}, "
        f"while expected shuffle is {expected_train_test_split_params_shuffle}"
    )


def test_read_k_fold_cross_validation_no_shuffle_params_from_dict(config_k_fold_cross_validation_no_shuffle_params_path: str):
    yaml_dict = read_pipeline_params_to_dict(config_k_fold_cross_validation_no_shuffle_params_path)
    train_test_split_params = read_val_params(yaml_dict)
    assert isinstance(train_test_split_params, KFoldParams), (
        ".yaml config was parsed incorrectly. Wrong type."
    )
    expected_train_test_split_params_n_splits = 3
    assert train_test_split_params.n_splits == expected_train_test_split_params_n_splits, (
        f".yaml config was parsed incorrectly. Test size is {train_test_split_params.n_splits}, "
        f"while expected type test size is {expected_train_test_split_params_n_splits}"
    )
    expected_train_test_split_params_random_state = None
    assert train_test_split_params.random_state == expected_train_test_split_params_random_state, (
        f".yaml config was parsed incorrectly. Random state is {train_test_split_params.random_state}, "
        f"while expected random state is {expected_train_test_split_params_random_state}"
    )
    expected_train_test_split_params_shuffle = False
    assert train_test_split_params.shuffle == expected_train_test_split_params_shuffle, (
        f".yaml config was parsed incorrectly. Shuffle is {train_test_split_params.shuffle}, "
        f"while expected shuffle is {expected_train_test_split_params_shuffle}"
    )


def test_read_feature_params_from_dict(config_feature_params_path: str):
    yaml_dict = read_pipeline_params_to_dict(config_feature_params_path)
    feature_params = read_feature_params(yaml_dict)
    assert isinstance(feature_params, FeatureParams), (
        f".yaml config was parsed incorrectly. Wrong type."
    )
    expected_numerical_features_length = 11
    assert len(feature_params.numerical_features) == expected_numerical_features_length, (
        f".yaml config was parsed incorrectly. Number of numerical features is "
        f"{len(feature_params.numerical_features)}, while expected number is "
        f"{expected_numerical_features_length}"
    )
    expected_target_column_name = "SalePrice"
    assert feature_params.target_col == expected_target_column_name, (
        f".yaml config was parsed incorrectly. Target col name is "
        f"{len(feature_params.target_col)}, while expected target col name is "
        f"{expected_target_column_name}"
    )
    assert isinstance(feature_params.numerical_transformers, List), (
        f".yaml config was parsed incorrectly. Wrong type of numerical_transformers."
    )
    assert isinstance(feature_params.numerical_transformers[0], StandardScalerTransformer), (
        f".yaml config was parsed incorrectly. Wrong type of standard scaler transformer."
    )
    assert isinstance(feature_params.numerical_transformers[1], PCATransformer), (
        f".yaml config was parsed incorrectly. Wrong type of pca transformer."
    )
    assert feature_params.numerical_transformers[1].n_components == 6, (
        f".yaml config was parsed incorrectly. Wrong number of n_components in PCATransformer."
    )
    assert feature_params.numerical_transformers[2].n_components is None, (
        f".yaml config was parsed incorrectly. Wrong number of n_components in PCATransformer."
    )


def test_read_model_params_from_dict(config_model_params_path: str):
    yaml_dict = read_pipeline_params_to_dict(config_model_params_path)
    feature_params = read_model_params(yaml_dict)
    assert feature_params.name == "LogisticRegression", (
        ".yaml config was parsed incorrectly. Wrong name."
    )
    assert isinstance(feature_params.params, dict), (
        ".yaml config was parsed incorrectly. Wrong type of feature_params.params."
    )
    assert feature_params.params.get("C") == 3.5, (
        ".yaml config was parsed incorrectly. Wrong value of feature_params.params C."
    )


def test_read_all_params(config_all_path):
    training_pipeline_params = read_pipeline_params(config_all_path)
    assert isinstance(training_pipeline_params, TrainingPipelineParams), (
        ".yaml config was parsed incorrectly. Wrong type of training_pipeline_params."
    )


def test_read_test_params(config_test_path):
    testing_pipeline_params = read_pipeline_params(config_test_path)
    assert isinstance(testing_pipeline_params, _TestingPipelineParams), (
        ".yaml config was parsed incorrectly. Wrong type of testing_pipeline_params."
    )
    expected_output_predictions_path = "output_predictions_path.txt"
    assert testing_pipeline_params.output_predictions_path == expected_output_predictions_path
    expected_transformer_path = "input_transformer.pth"
    assert testing_pipeline_params.input_transformer_path == expected_transformer_path
