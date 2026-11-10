from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import config
from src.dataset import clean_dataset, load_dataset, split_by_date


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        读取真实数据供时间划分测试使用。

        参数：
            cls

        返回：
            None
        """
        cls.data, cls.cleaning = load_dataset()

    def test_real_split_counts_bounds_and_order(self):
        """
        验证真实数据的记录数、时间边界及订单隔离。

        参数：
            self

        返回：
            None
        """
        splits = split_by_date(self.data)
        expected = {
            "train": (14340, "2023-01-01", "2024-06-30"),
            "validation": (2533, "2024-07-01", "2024-09-30"),
            "test": (2448, "2024-10-01", "2024-12-31"),
        }
        all_indices = set()
        all_orders = set()
        for name, (count, start, end) in expected.items():
            part = getattr(splits, name)
            self.assertEqual(len(part), count)
            self.assertEqual(part["Date"].min(), pd.Timestamp(start))
            self.assertEqual(part["Date"].max(), pd.Timestamp(end))
            self.assertTrue(part["Date"].is_monotonic_increasing)
            self.assertFalse(all_indices.intersection(part.index))
            self.assertFalse(all_orders.intersection(part["Order_ID"]))
            all_indices.update(part.index)
            all_orders.update(part["Order_ID"])
        self.assertEqual(len(all_indices), len(self.data))
        self.assertEqual(self.cleaning["skipped_rows"], 0)

    def test_invalid_dates_and_targets_are_counted_once(self):
        """
        验证日期、目标异常的静默跳过与原因计数互斥。

        参数：
            self

        返回：
            None
        """
        invalid_targets = [None, "invalid", 0, -1, 1.5, np.inf]
        data = pd.DataFrame({
            "Date": ["2023-01-01"] * 7 + [None, "invalid", "2022-12-31", "2025-01-01"],
            "Quantity": [2] + invalid_targets + [None, 2, 2, 2],
        })
        before = data.copy(deep=True)
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            cleaned, report = clean_dataset(data)
        self.assertEqual(cleaned.index.tolist(), [0])
        self.assertEqual(report["input_rows"], 11)
        self.assertEqual(report["kept_rows"], 1)
        self.assertEqual(report["skipped_rows"], 10)
        self.assertEqual(report["skipped_by_reason"], {
            "invalid_date": 2, "invalid_quantity": 6,
            "outside_periods": 2, "cross_split_order": 0,
        })
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        pd.testing.assert_frame_equal(data, before)

    def test_cross_split_orders_are_recorded(self):
        """
        验证跨集合订单整体跳过并计入审计原因。

        参数：
            self

        返回：
            None
        """
        data = pd.DataFrame({
            "Date": ["2023-01-01", "2024-07-01", "2024-10-01"],
            "Quantity": [1, 2, 3],
            "Order_ID": ["same", "same", "other"],
        })
        cleaned, report = clean_dataset(data)
        self.assertEqual(cleaned["Order_ID"].tolist(), ["other"])
        self.assertEqual(report["skipped_by_reason"]["cross_split_order"], 2)

    def test_empty_periods_are_preserved(self):
        """
        验证无记录的日期区间以空数据集返回。

        参数：
            self

        返回：
            None
        """
        cleaned, _ = clean_dataset(pd.DataFrame({"Date": ["2023-01-01"], "Quantity": [1]}))
        splits = split_by_date(cleaned)
        self.assertEqual(len(splits.train), 1)
        self.assertTrue(splits.validation.empty)
        self.assertTrue(splits.test.empty)

    def test_period_configuration_drives_cleaning_and_split(self):
        """
        验证清理与划分共用配置中的时间区间。

        参数：
            self

        返回：
            None
        """
        periods = {
            "train": ["2023-01-01", "2024-03-31"],
            "validation": ["2024-04-01", "2024-09-30"],
            "test": ["2024-10-01", "2024-12-31"],
        }
        with patch.dict(config.SPLIT_PERIODS, periods, clear=True):
            cleaned, _ = clean_dataset(pd.DataFrame({
                "Date": ["2024-03-31", "2024-04-01"], "Quantity": [1, 2],
            }))
            splits = split_by_date(cleaned)
        self.assertEqual(splits.train["Quantity"].tolist(), [1])
        self.assertEqual(splits.validation["Quantity"].tolist(), [2])


if __name__ == "__main__":
    unittest.main()
