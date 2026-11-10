import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from src.data_audit import audit_dataset, write_audit_report
from src.dataset import clean_dataset, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        准备真实数据及对应的审计结果。

        参数：
            cls

        返回：
            None
        """
        cls.data, cls.cleaning = load_dataset()
        cls.report = audit_dataset(cls.data, cls.cleaning)

    def test_real_dataset_overview_and_missing_values(self):
        """
        验证真实数据的概览、缺失数和时间字段一致性。

        参数：
            self

        返回：
            None
        """
        self.assertEqual((self.report["rows"], self.report["columns"]), (19321, 40))
        self.assertEqual(self.report["date_range"]["days_with_orders"], 731)
        self.assertEqual(
            self.report["unique_counts"],
            {"orders": 6576, "customers": 500, "products": 35, "stores": 3},
        )
        self.assertEqual(self.report["duplicate_rows"], 0)
        self.assertEqual(
            self.report["missing_values"],
            {
                "Festival": {"count": 16006, "percentage": 82.84},
                "Promotion_Type": {"count": 13826, "percentage": 71.56},
            },
        )
        self.assertEqual(
            self.report["promotion_type_missing"],
            {"without_promotion": 13826, "with_promotion": 0},
        )
        self.assertEqual(
            self.report["calendar_feature_mismatches"],
            {
                "Year": 0,
                "Month": 0,
                "Day": 0,
                "Day_of_Week": 0,
                "Weekend": 0,
                "Hour": 0,
            },
        )

    def test_calendar_inconsistencies_are_reported(self):
        """
        验证时间字段不一致会写入审计结果。

        参数：
            self

        返回：
            None
        """
        changed = self.data.copy()
        first = changed.index[0]
        changed.loc[first, "Year"] = 2025
        changed.loc[first, "Time"] = "00:00"
        mismatches = audit_dataset(changed, self.cleaning)["calendar_feature_mismatches"]
        self.assertEqual(mismatches["Year"], 1)
        self.assertEqual(mismatches["Hour"], 1)
        self.assertEqual(mismatches["Month"], 0)

    def test_training_quantity_relationships(self):
        """
        验证数量、账单与利润关系只基于训练集统计。

        参数：
            self

        返回：
            None
        """
        self.assertEqual(
            self.report["training_quantity_vs_units_sold"],
            {"sold_greater": 3779, "equal": 10561, "sold_less": 0},
        )
        self.assertEqual(
            self.report["training_leakage_relationship_mismatches"],
            {"total_bill": 0, "profit": 0},
        )

    def test_training_statistics_do_not_use_test_targets(self):
        """
        验证测试目标值变化不会影响训练集分析。

        参数：
            self

        返回：
            None
        """
        changed = self.data.copy()
        test = changed["Date"] >= pd.Timestamp("2024-10-01")
        changed.loc[test, ["Quantity", "Units_Sold", "Total_Bill", "Profit"]] = 999
        changed_report = audit_dataset(changed, self.cleaning)
        for key in (
            "training_quantity",
            "training_quantity_vs_units_sold",
            "training_leakage_relationship_mismatches",
        ):
            self.assertEqual(changed_report[key], self.report[key])

    def test_missing_audit_column_uses_native_error(self):
        """
        验证缺少必需审计列时保留 Pandas 原生错误。

        参数：
            self

        返回：
            None
        """
        with self.assertRaisesRegex(KeyError, "Order_ID"):
            audit_dataset(self.data.drop(columns=["Order_ID"]), self.cleaning)

    def test_all_rows_skipped_produces_valid_json(self):
        """
        验证全部记录异常时仍可保存清理结果。

        参数：
            self

        返回：
            None
        """
        invalid = self.data.iloc[:3].copy()
        invalid["Date"] = None
        data, cleaning = clean_dataset(invalid)
        report = audit_dataset(data, cleaning)
        self.assertEqual(report["cleaning"]["skipped_rows"], 3)
        self.assertEqual(report["rows"], 0)
        self.assertIsNone(report["date_range"]["start"])
        self.assertTrue(all(value is None for value in report["training_quantity"].values()))
        json.dumps(report, allow_nan=False)

    def test_json_report_and_command_line(self):
        """
        验证报告保存和命令行静默执行。

        参数：
            self

        返回：
            None
        """
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "reports" / "audit.json"
            self.assertEqual(write_audit_report(self.report, destination), destination)
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8"))["rows"], 19321
            )
            cli_destination = Path(directory) / "cli.json"
            environment = os.environ.copy()
            environment["PYTHONUTF8"] = "1"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.data_audit",
                    "--output",
                    str(cli_destination),
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            cli_report = json.loads(cli_destination.read_text(encoding="utf-8"))
            self.assertEqual(cli_report["splits"], self.report["splits"])
            self.assertEqual(cli_report["cleaning"]["skipped_rows"], 0)
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, "")

    def test_skipped_rows_are_persisted_by_command_line(self):
        """
        验证命令行将异常记录原因保存到审计文件。

        参数：
            self

        返回：
            None
        """
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dirty.csv"
            destination = Path(directory) / "audit.json"
            invalid = self.data.iloc[:5].copy()
            invalid.loc[invalid.index[0], "Date"] = pd.NaT
            invalid.loc[invalid.index[1], "Quantity"] = 0
            invalid.to_csv(source, index=False)
            result = subprocess.run(
                [sys.executable, "-m", "src.data_audit",
                 "--data", str(source), "--output", str(destination)],
                cwd=PROJECT_ROOT, capture_output=True, check=True,
            )
            report = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(report["cleaning"]["kept_rows"], 3)
            self.assertEqual(report["cleaning"]["skipped_rows"], 2)
            self.assertEqual(report["cleaning"]["skipped_by_reason"]["invalid_date"], 1)
            self.assertEqual(report["cleaning"]["skipped_by_reason"]["invalid_quantity"], 1)
            self.assertEqual(result.stdout, b"")
            self.assertEqual(result.stderr, b"")


if __name__ == "__main__":
    unittest.main()
