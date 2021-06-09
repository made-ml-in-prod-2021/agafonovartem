import pickle
import os

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def dump(object_, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(object_, f)


def load(filepath):
    with open(filepath, 'rb') as fin:
        return pickle.load(fin)


def calculate_metrics(target, preds):
    return {"rmse": np.sqrt(mean_squared_error(target, preds)),
            "mae": mean_absolute_error(target, preds),
            "r2": r2_score(target, preds)}


def _wait_for_file(path: str):
    return os.path.exists(path)
