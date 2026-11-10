"""构造缺失填补、类别编码和数值标准化的统一预处理器。"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from . import config
from .dataset import normalize_missing_values


class PromotionTypeFiller(TransformerMixin, BaseEstimator):
    """根据促销状态填补缺失的促销类型。"""

    def fit(self, X: pd.DataFrame, y: object = None) -> "PromotionTypeFiller":
        """
        记录输入字段，满足 sklearn 拟合接口。

        参数：
            self
            X: 包含 Promotion_Type 和 Promotion_Applied 的数据表。
            y: 供 sklearn 接口兼容的目标值。

        返回：
            当前填补器实例。
        """
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        根据促销状态填补空类型。

        参数：
            self
            X: 包含促销类型和促销状态的数据表。

        返回：
            按配置标签填补缺失值后的 Promotion_Type 单列表。
        """
        values = normalize_missing_values(X)["Promotion_Type"].astype(object)
        fill_values = pd.Series(config.MISSING_VALUES["unspecified"], index=X.index)
        fill_values.loc[X["Promotion_Applied"].eq(config.BINARY_VALUES[0])] = (
            config.MISSING_VALUES["no_promotion"]
        )
        return values.fillna(fill_values).to_frame()

    def get_feature_names_out(
        self, input_features: list[str] | np.ndarray | None = None
    ) -> np.ndarray:
        """
        提供转换后的字段名。

        参数：
            self
            input_features: 供 sklearn 接口兼容的输入字段名。

        返回：
            包含 Promotion_Type 的字符串数组。
        """
        return np.asarray(["Promotion_Type"], dtype=object)


def _normalizer() -> FunctionTransformer:
    """
    构造保持字段名的缺失值规范化步骤。

    参数：
        无

    返回：
        将空白统一为 NaN 的 sklearn FunctionTransformer。
    """
    return FunctionTransformer(normalize_missing_values, feature_names_out="one-to-one")


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    按字段白名单构建尚未拟合的预处理器。

    参数：
        scale_numeric: 是否在中位数填补后标准化数值列。

    返回：
        包含缺失填补、编码和可选标准化的 ColumnTransformer。
    """
    numeric = Pipeline(
        [
            ("normalize", _normalizer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler() if scale_numeric else "passthrough"),
        ]
    )
    categorical = Pipeline(
        [
            ("normalize", _normalizer()),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    binary = Pipeline(
        [
            ("normalize", _normalizer()),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[config.BINARY_VALUES for _ in config.BINARY_FEATURES],
                    dtype=np.float64,
                ),
            ),
        ]
    )
    festival = Pipeline(
        [
            ("normalize", _normalizer()),
            (
                "imputer",
                SimpleImputer(
                    strategy="constant", fill_value=config.MISSING_VALUES["unspecified"]
                ),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    promotion_type = Pipeline(
        [
            ("imputer", PromotionTypeFiller()),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, config.NUMERIC_FEATURES),
            ("categorical", categorical, config.CATEGORICAL_FEATURES),
            ("binary", binary, config.BINARY_FEATURES),
            ("festival", festival, ["Festival"]),
            (
                "promotion_type",
                promotion_type,
                ["Promotion_Type", "Promotion_Applied"],
            ),
        ],
        remainder="drop",
    )
