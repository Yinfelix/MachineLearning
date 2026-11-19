"""按字段类型组装预处理步骤，供训练、验证和预测共用。"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config
from .dataset import normalize_missing_values


def promotion_type(data):
    """促销状态决定空白类型的含义。"""
    values = normalize_missing_values(data)["Promotion_Type"]
    fill = pd.Series(config.MISSING_VALUES["unspecified"], index=data.index)
    fill.loc[data["Promotion_Applied"].eq("No")] = config.MISSING_VALUES["no_promotion"]
    return values.fillna(fill)


class PromotionTypeFiller(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return promotion_type(X).to_frame()

    def get_feature_names_out(self, input_features=None):
        return np.array(["Promotion_Type"])


def build_preprocessor(scale_numeric=True, extra_numeric=(), extra_categories=(), drop=()):
    """加入派生列或移除指定字段。各步骤的参数由 fit 时传入的数据决定。"""
    normalize = FunctionTransformer(normalize_missing_values, feature_names_out="one-to-one")
    numeric = Pipeline([
        ("normalize", normalize), ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler() if scale_numeric else "passthrough"),
    ])
    categorical = Pipeline([
        ("normalize", normalize), ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    binary_cols = [c for c in config.BINARY_FEATURES if c not in drop]
    binary = Pipeline([
        ("normalize", normalize), ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[config.BINARY_VALUES] * len(binary_cols))),
    ])
    festival = Pipeline([
        ("normalize", normalize),
        ("imputer", SimpleImputer(strategy="constant", fill_value=config.MISSING_VALUES["unspecified"])),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    promo = Pipeline([
        ("imputer", PromotionTypeFiller()), ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    numbers = [c for c in [*config.NUMERIC_FEATURES, *extra_numeric] if c not in drop]
    categories = [c for c in [*config.CATEGORICAL_FEATURES, *extra_categories] if c not in drop]
    steps = [("numeric", numeric, numbers), ("categorical", categorical, categories),
             ("binary", binary, binary_cols)]
    if "Festival" not in drop:
        steps.append(("festival", festival, ["Festival"]))
    if "Promotion_Type" not in drop:
        steps.append(("promotion_type", promo, ["Promotion_Type", "Promotion_Applied"]))
    return ColumnTransformer(steps)
