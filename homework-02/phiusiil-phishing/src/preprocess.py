"""Leak-safe preprocessing pipelines built with scikit-learn transformers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from . import config as C


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Map each category to its training-fold frequency.

    Unseen categories map to 0.  Fitted statistics come only from the data
    passed to ``fit``, so the encoder is safe inside cross-validation.
    """

    def __init__(self, column: str | None = None):
        self.column = column

    def fit(self, X, y=None):
        s = self._to_series(X)
        self.counts_ = s.value_counts()
        self.n_ = len(s)
        return self

    def transform(self, X):
        s = self._to_series(X)
        vals = s.map(self.counts_).fillna(0.0).to_numpy(dtype=float)
        return vals.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        name = self.column or (input_features[0] if input_features is not None else "category")
        return np.asarray([f"{name}_freq"])

    @staticmethod
    def _to_series(X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].astype(str)
        arr = np.asarray(X)
        if arr.ndim == 2:
            arr = arr[:, 0]
        return pd.Series(arr).astype(str)


def make_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale: bool,
    scaler: str = "standard",
) -> ColumnTransformer:
    Scaler = RobustScaler if scaler == "robust" else StandardScaler
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", Scaler()))
    transformers = [("num", Pipeline(num_steps), numeric_cols)]

    if categorical_cols:
        cat_steps = [("freq", FrequencyEncoder())]
        if scale:
            cat_steps.append(("scaler", Scaler()))
        transformers.append(("cat", Pipeline(cat_steps), categorical_cols))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_pipeline(model, scheme: str, scale: bool, scaler: str = "standard") -> Pipeline:
    numeric, categorical = C.scheme_columns(scheme)
    return Pipeline(
        [
            ("pre", make_preprocessor(numeric, categorical, scale=scale, scaler=scaler)),
            ("model", model),
        ]
    )


def feature_names_out(pipeline: Pipeline, scheme: str) -> list[str]:
    numeric, categorical = C.scheme_columns(scheme)
    names = list(numeric)
    if categorical:
        names += [f"{c}_freq" for c in categorical]
    return names
