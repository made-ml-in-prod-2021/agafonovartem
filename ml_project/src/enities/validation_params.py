from typing import Optional
from dataclasses import dataclass, field


@dataclass()
class ValParams:
    val_name: str
    random_state: Optional[int] = field(default=None)


@dataclass()
class TrainTestSplitParams(ValParams):
    val_name: str = field(default="TrainTestSplit")
    random_state: int = field(default=52)
    test_size: float = field(default=0.2)


@dataclass()
class KFoldParams(ValParams):
    val_name: str = field(default="KFoldCrossValidation")
    n_splits: int = field(default=3)
    shuffle: bool = field(default=False)


def read_train_test_split_params(validation_params_dict: dict) -> TrainTestSplitParams:
    train_test_split_params = TrainTestSplitParams()
    random_state = validation_params_dict.get("random_state")
    test_size = validation_params_dict.get("test_size")
    if random_state:
        train_test_split_params.random_state = random_state
    if test_size:
        train_test_split_params.test_size = test_size
    return train_test_split_params


def read_k_fold_cross_validation_params(validation_params_dict: dict) -> KFoldParams:
    shuffle = validation_params_dict.get("shuffle")
    n_splits = validation_params_dict.get("n_splits")
    random_state = validation_params_dict.get("random_state")
    if shuffle is True:
        k_fold_params = KFoldParams(random_state=random_state, shuffle=True)
    else:
        k_fold_params = KFoldParams(random_state=None)
    if n_splits:
        k_fold_params.n_splits = n_splits
    return k_fold_params


def read_val_params(yaml_dict: dict) -> ValParams:
    validation_params_dict = yaml_dict.get("validation_params")
    if validation_params_dict.get("val_name") == "TrainTestSplit":
        return read_train_test_split_params(validation_params_dict)
    elif yaml_dict.get("validation_params").get("val_name") == "KFoldCrossValidation":
        return read_k_fold_cross_validation_params(validation_params_dict)
    else:
        raise NotImplementedError("No such validation type")

