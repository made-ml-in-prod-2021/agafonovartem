from typing import Optional, List

import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA


from src.enities import (FeatureParams, PCATransformer, StandardScalerTransformer,
                         NumericalTransformer, SquaredFeaturesTransformer)

from src.custom_transformers.squared_features import SquaredFeatures


def build_numerical_pipeline(numerical_transformers: Optional[List[NumericalTransformer]]
                             ) -> Pipeline:
    transformers_list = [("impute", SimpleImputer(missing_values=np.nan, strategy="mean"))]
    if numerical_transformers:
        for transformer in numerical_transformers:
            if isinstance(transformer, StandardScalerTransformer):
                standard_scaler = ("standard_scaler", StandardScaler())
                transformers_list.append(standard_scaler)
            elif isinstance(transformer, PCATransformer):
                pca = ("pca", PCA(n_components=transformer.n_components))
                transformers_list.append(pca)
            elif isinstance(transformer, SquaredFeaturesTransformer):
                squared_features = ("squared_features", SquaredFeatures())
                transformers_list.append(squared_features)
            else:
                raise NotImplementedError(transformer)

    num_pipeline = Pipeline(
        transformers_list
    )
    return num_pipeline


def process_numerical_features(numerical_df: pd.DataFrame,
                               numerical_transformers: Optional[List[NumericalTransformer]],
                               ) -> pd.DataFrame:
    num_pipeline = build_numerical_pipeline(numerical_transformers)
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:

    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(params.numerical_transformers),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def dump_transformer(transformer: ColumnTransformer, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(transformer, f)
    return output


def load_transformer(input_: str) -> ColumnTransformer:
    with open(input_, "rb") as f:
        transformer = pickle.load(f)
    return transformer


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target
