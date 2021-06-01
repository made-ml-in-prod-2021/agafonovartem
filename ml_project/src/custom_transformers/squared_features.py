import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SquaredFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, data: np.ndarray, y=None):
        return self

    def transform(self, data: np.ndarray, y=None):
        data_ = data.T.copy()
        already_in = []
        for i, column1 in enumerate(data.T):
            for j, column2 in enumerate(data.T):
                if (i, j) not in already_in:
                    data_ = np.vstack([data_, column1 * column2])
                    already_in.append((j, i))
        return data_.T
