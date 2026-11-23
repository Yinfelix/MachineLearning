"""构造回归模型，并统一验证件数与金额预测的误差。"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeRegressor

from .features import CALENDAR, CROSSES, make_features, quantity_mi
from .preprocessing import build_preprocessor


SEED = 42
MODELS = {
    "median": DummyRegressor(strategy="median"),
    "ridge": Ridge(solver="lsqr", alpha=1),
    "tree": DecisionTreeRegressor(random_state=SEED),
    "forest": RandomForestRegressor(n_estimators=64, random_state=SEED, n_jobs=4),
}
GRIDS = {
    "ridge": {"model__alpha": [0.1, 1, 10, 100]},
    "tree": {"model__max_depth": [4, 8, 12], "model__min_samples_leaf": [5, 30]},
    "forest": {"model__max_depth": [5, 12], "model__min_samples_leaf": [2, 20]},
}


class ProductMedian(RegressorMixin, BaseEstimator):
    """按商品估计件数中位数，新商品使用训练集的总体中位数。"""

    def fit(self, X, y):
        self.medians_ = y.groupby(X["Product"]).median()
        self.median_ = y.median()
        return self

    def predict(self, X):
        return X["Product"].map(self.medians_).fillna(self.median_).to_numpy()


def pipeline(name, level="base", k=None, drop=()):
    features = FunctionTransformer(make_features, kw_args={"level": level})
    prep = build_preprocessor(
        scale_numeric=name == "ridge", extra_numeric=CALENDAR if level != "base" else (),
        extra_categories=CROSSES if level == "cross" else (), drop=drop,
    )
    steps = [("features", features), ("prep", prep)]
    if k is not None:
        steps.append(("select", SelectKBest(quantity_mi, k=k)))
    steps.append(("model", clone(MODELS[name])))
    return Pipeline(steps)


def metrics(y, pred):
    return {"MAE": mean_absolute_error(y, pred), "RMSE": np.sqrt(mean_squared_error(y, pred)),
            "R2": r2_score(y, pred)}
