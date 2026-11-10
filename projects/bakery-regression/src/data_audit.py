"""生成数据质量、清理统计与训练集分析报告，命令行运行时写入 JSON。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .dataset import DatasetSplits, load_dataset, split_by_date


def _date_text(value: pd.Timestamp) -> str | None:
    """
    将日期统计量转换为可保存的文本。

    参数：
        value: 日期最小值或最大值，可以为 NaT。

    返回：
        配置格式的日期字符串，NaT 对应 None。
    """
    return value.strftime(config.DATE_FORMAT) if pd.notna(value) else None


def _split_summary(splits: DatasetSplits) -> dict:
    """
    汇总三个数据集的记录数和时间范围。

    参数：
        splits: 清理后按时间划分的数据集容器。

    返回：
        以数据集名称为键的摘要，空集合的日期范围为 None。
    """
    result = {}
    for name in config.SPLIT_PERIODS:
        part = getattr(splits, name)
        result[name] = {
            "rows": len(part),
            "start": _date_text(part["Date"].min()),
            "end": _date_text(part["Date"].max()),
        }
    return result


def _quantity_summary(quantity: pd.Series) -> dict:
    """
    计算训练集购买数量的描述性统计。

    参数：
        quantity: 清理后的训练集购买数量列。

    返回：
        包含均值、中位数、标准差、极值、上四分位数和偏度的字典，未定义的值为 None。
    """
    values = {
        "mean": quantity.mean(),
        "median": quantity.median(),
        "std": quantity.std(),
        "min": quantity.min(),
        "q75": quantity.quantile(0.75),
        "max": quantity.max(),
        "skew": quantity.skew(),
    }
    return {
        key: round(float(value), 4) if pd.notna(value) and np.isfinite(value) else None
        for key, value in values.items()
    }


def audit_dataset(data: pd.DataFrame, cleaning_report: dict) -> dict:
    """
    汇总清理结果、全量数据质量及训练集目标关系。

    参数：
        data: load_dataset 或 clean_dataset 返回的清理后订单明细。
        cleaning_report: 同次清理返回的输入、保留、跳过数量及原因统计。

    返回：
        JSON 可序列化的审计字典。
    """
    splits = split_by_date(data)
    train = splits.train
    quantity = train[config.TARGET_COLUMN]
    sold = pd.to_numeric(train["Units_Sold"], errors="coerce")
    price = pd.to_numeric(train["Selling_Price"], errors="coerce")
    cost = pd.to_numeric(train["Cost_Price"], errors="coerce")
    bill = pd.to_numeric(train["Total_Bill"], errors="coerce")
    profit = pd.to_numeric(train["Profit"], errors="coerce")
    missing_counts = data.isna().sum()
    missing_values = {
        column: {
            "count": int(count),
            "percentage": round(100 * int(count) / len(data), 2),
        }
        for column, count in missing_counts.items()
        if count
    }
    dates = data["Date"]
    times = pd.to_datetime(data["Time"], format=config.TIME_FORMAT, errors="coerce")
    weekend = dates.dt.dayofweek.ge(5).map(
        {False: config.BINARY_VALUES[0], True: config.BINARY_VALUES[1]}
    )
    return {
        "cleaning": cleaning_report,
        "rows": len(data),
        "columns": len(data.columns),
        "date_range": {
            "start": _date_text(dates.min()),
            "end": _date_text(dates.max()),
            "days_with_orders": int(dates.nunique()),
        },
        "unique_counts": {
            "orders": int(data["Order_ID"].nunique()),
            "customers": int(data["Customer_ID"].nunique()),
            "products": int(data["Product"].nunique()),
            "stores": int(data["Store_ID"].nunique()),
        },
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_values": missing_values,
        "calendar_feature_mismatches": {
            "Year": int(data["Year"].ne(dates.dt.year).sum()),
            "Month": int(data["Month"].ne(dates.dt.month).sum()),
            "Day": int(data["Day"].ne(dates.dt.day).sum()),
            "Day_of_Week": int(data["Day_of_Week"].ne(dates.dt.day_name()).sum()),
            "Weekend": int(data["Weekend"].ne(weekend).sum()),
            "Hour": int(data["Hour"].ne(times.dt.hour).sum()),
        },
        "splits": _split_summary(splits),
        "training_quantity": _quantity_summary(quantity),
        "training_quantity_vs_units_sold": {
            "sold_greater": int((sold > quantity).sum()),
            "equal": int((sold == quantity).sum()),
            "sold_less": int((sold < quantity).sum()),
        },
        "training_leakage_relationship_mismatches": {
            "total_bill": int(
                (~np.isclose(
                    bill, price * quantity, rtol=0, atol=config.EQUALITY_TOLERANCE
                )).sum()
            ),
            "profit": int(
                (~np.isclose(
                    profit, (price - cost) * quantity,
                    rtol=0, atol=config.EQUALITY_TOLERANCE,
                )).sum()
            ),
        },
        "promotion_type_missing": {
            "without_promotion": int(
                (data["Promotion_Type"].isna()
                 & data["Promotion_Applied"].eq(config.BINARY_VALUES[0])).sum()
            ),
            "with_promotion": int(
                (data["Promotion_Type"].isna()
                 & data["Promotion_Applied"].eq(config.BINARY_VALUES[1])).sum()
            ),
        },
    }


def write_audit_report(
    report: dict, output: str | Path = config.AUDIT_OUTPUT_PATH
) -> Path:
    """
    创建父目录并将审计结果写入 UTF-8 JSON 文件。

    参数：
        report: audit_dataset 返回的审计字典。
        output: 输出文件路径，默认使用配置中的审计路径。

    返回：
        已写入的文件路径。
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    """
    解析 --data 和 --output 参数并写入审计报告。

    参数：
        无

    返回：
        None
    """
    parser = argparse.ArgumentParser(description="审计烘焙订单数据并生成 JSON 报告")
    parser.add_argument("--data", type=Path, default=config.DEFAULT_DATA_PATH, help="原始 CSV 路径")
    parser.add_argument("--output", type=Path, default=config.AUDIT_OUTPUT_PATH, help="审计 JSON 路径")
    args = parser.parse_args()
    data, cleaning_report = load_dataset(args.data)
    write_audit_report(audit_dataset(data, cleaning_report), args.output)


if __name__ == "__main__":
    main()
