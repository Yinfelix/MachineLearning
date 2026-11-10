"""读取、清理和划分数据，并返回可写入审计文件的跳过记录统计。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class DatasetSplits:
    """分别保存训练集、验证集和测试集，保留清理后记录的索引。"""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def normalize_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    将文本列的空白及缺失值统一为 NaN。

    参数：
        data: 待规范化的数据表。

    返回：
        规范化后的数据表副本。
    """
    normalized = data.copy()
    for column in normalized.select_dtypes(include=["object", "string", "category"]):
        values = normalized[column].astype(object)
        blank = values.astype("string").str.fullmatch(r"\s*", na=False)
        values.loc[values.isna() | blank] = np.nan
        normalized[column] = values
    return normalized


def _partition_labels(dates: pd.Series) -> pd.Series:
    """
    按配置为日期分配数据集名称。

    参数：
        dates: datetime 类型的日期列。

    返回：
        与输入索引对应的数据集标签列，范围外或缺失日期对应空值。
    """
    labels = pd.Series(index=dates.index, dtype=object)
    for name, (start, end) in config.SPLIT_PERIODS.items():
        labels.loc[dates.between(start, end)] = name
    return labels


def clean_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    清理数据，按日期异常、目标异常、范围外和跨集合订单的顺序统计跳过原因。

    参数：
        data: 原始订单明细，至少包含 Date 和配置指定的目标列。

    返回：
        清理后的数据表与输入、保留和跳过记录的统计字典。
    """
    cleaned = normalize_missing_values(data)
    cleaned["Date"] = pd.to_datetime(
        cleaned["Date"], format=config.DATE_FORMAT, errors="coerce"
    )
    quantity = pd.to_numeric(cleaned[config.TARGET_COLUMN], errors="coerce")
    cleaned[config.TARGET_COLUMN] = quantity
    labels = _partition_labels(cleaned["Date"])
    reasons = pd.Series("", index=cleaned.index, dtype=object)
    checks = {
        "invalid_date": cleaned["Date"].isna(),
        "invalid_quantity": (
            quantity.isna()
            | ~np.isfinite(quantity)
            | quantity.le(0)
            | quantity.mod(1).ne(0)
        ),
        "outside_periods": labels.isna(),
    }
    for reason, mask in checks.items():
        reasons.loc[reasons.eq("") & mask] = reason

    if "Order_ID" in cleaned.columns:
        eligible = reasons.eq("")
        order_periods = labels.loc[eligible].groupby(
            cleaned.loc[eligible, "Order_ID"]
        ).nunique()
        cross_period_orders = order_periods.index[order_periods.gt(1)]
        reasons.loc[
            eligible & cleaned["Order_ID"].isin(cross_period_orders)
        ] = "cross_split_order"

    kept = reasons.eq("")
    report = {
        "input_rows": len(data),
        "kept_rows": int(kept.sum()),
        "skipped_rows": int((~kept).sum()),
        "skipped_by_reason": {
            reason: int(reasons.eq(reason).sum())
            for reason in (*checks, "cross_split_order")
        },
    }
    return cleaned.loc[kept].copy(), report


def load_dataset(path: str | Path = config.DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """
    读取 CSV 并清理异常记录。

    参数：
        path: CSV 文件路径，默认使用项目配置中的数据路径。

    返回：
        清理后的数据表与包含跳过原因及数量的统计字典。
    """
    return clean_dataset(pd.read_csv(path, encoding="utf-8-sig"))


def split_by_date(data: pd.DataFrame) -> DatasetSplits:
    """
    按配置的日期区间划分并排序数据。

    参数：
        data: clean_dataset 或 load_dataset 返回的数据表。

    返回：
        包含训练集、验证集和测试集的 DatasetSplits，无记录的区间对应空表。
    """
    labels = _partition_labels(data["Date"])
    parts = {
        name: data.loc[labels.eq(name)].sort_values("Date", kind="stable").copy()
        for name in config.SPLIT_PERIODS
    }
    return DatasetSplits(
        train=parts["train"], validation=parts["validation"], test=parts["test"]
    )
