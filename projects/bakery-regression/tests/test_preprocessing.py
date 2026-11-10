from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.config import (
    BINARY_FEATURES,
    EXCLUDED_FIELDS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.dataset import load_dataset, split_by_date
from src.preprocessing import PromotionTypeFiller, build_preprocessor


def dense(matrix):
    """
    将稀疏或稠密矩阵转换为 NumPy 数组。

    参数：
        matrix: 预处理结果矩阵。

    返回：
        NumPy 数组。
    """
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        准备共享的真实数据和预处理器。

        参数：
            cls

        返回：
            None
        """
        cls.data, _ = load_dataset()
        cls.splits = split_by_date(cls.data)
        cls.preprocessor = build_preprocessor()
        cls.preprocessor.fit(cls.splits.train)

    def test_feature_roles_cover_original_schema_without_overlap(self):
        """
        验证配置覆盖原始字段且输入与目标、排除字段隔离。

        参数：
            self

        返回：
            None
        """
        self.assertEqual(len(FEATURE_COLUMNS), len(set(FEATURE_COLUMNS)))
        self.assertFalse(set(FEATURE_COLUMNS) & set(EXCLUDED_FIELDS))
        self.assertNotIn(TARGET_COLUMN, FEATURE_COLUMNS)
        self.assertEqual(
            set(self.data.columns),
            set(FEATURE_COLUMNS) | set(EXCLUDED_FIELDS) | {TARGET_COLUMN},
        )

    def test_target_and_excluded_fields_cannot_affect_transformation(self):
        """
        验证目标及排除字段的变化不会影响输入矩阵。

        参数：
            self

        返回：
            None
        """
        sample = self.splits.validation.iloc[:5].copy()
        changed = sample.copy()
        for column in EXCLUDED_FIELDS:
            changed[column] = "excluded"
        changed[TARGET_COLUMN] = 99999
        expected = dense(self.preprocessor.transform(sample))
        actual = dense(self.preprocessor.transform(changed))
        np.testing.assert_allclose(actual, expected)

    def test_scaler_is_fitted_only_on_training_data(self):
        """
        验证标准化统计量只来自训练集。

        参数：
            self

        返回：
            None
        """
        scaler = self.preprocessor.named_transformers_["numeric"].named_steps["scaler"]
        expected = self.splits.train[NUMERIC_FEATURES].mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected)
        validation = self.splits.validation.iloc[:3].copy()
        validation["Temperature"] = 10000
        self.preprocessor.transform(validation)
        np.testing.assert_allclose(scaler.mean_, expected)

    def test_numeric_imputation_uses_training_median_without_scaling(self):
        """
        验证缺失数值使用训练集中位数填补。

        参数：
            self

        返回：
            None
        """
        train = self.splits.train.iloc[:20].copy()
        train.loc[train.index[0], "Temperature"] = np.nan
        median = train["Temperature"].median()
        preprocessor = build_preprocessor(scale_numeric=False).fit(train)
        self.assertEqual(
            preprocessor.named_transformers_["numeric"].named_steps["imputer"].statistics_[
                NUMERIC_FEATURES.index("Temperature")
            ],
            median,
        )
        validation = self.splits.validation.iloc[:1].copy()
        validation["Temperature"] = np.nan
        transformed = dense(preprocessor.transform(validation))
        index = list(preprocessor.get_feature_names_out()).index("numeric__Temperature")
        self.assertEqual(transformed[0, index], median)

    def test_unknown_category_is_ignored_without_refitting(self):
        """
        验证新商品类别可以转换且不会更新训练类别表。

        参数：
            self

        返回：
            None
        """
        validation = self.splits.validation.iloc[:1].copy()
        validation["Product"] = "New Product"
        transformed = dense(self.preprocessor.transform(validation))
        names = self.preprocessor.get_feature_names_out()
        product_columns = [
            index for index, name in enumerate(names) if name.startswith("categorical__Product_")
        ]
        self.assertGreater(len(product_columns), 0)
        self.assertEqual(transformed[0, product_columns].sum(), 0)
        self.assertNotIn("categorical__Product_New Product", names)

    def test_binary_encoding_and_missing_category_labels(self):
        """
        验证二元编码及节日、促销缺失标签。

        参数：
            self

        返回：
            None
        """
        sample = pd.concat([self.splits.train.iloc[:1]] * 2, ignore_index=True)
        for column in BINARY_FEATURES:
            sample[column] = ["No", "Yes"]
        sample["Promotion_Type"] = [np.nan, "Festival"]
        sample["Festival"] = [np.nan, "Diwali"]
        preprocessor = build_preprocessor().fit(sample)
        transformed = dense(preprocessor.transform(sample))
        names = list(preprocessor.get_feature_names_out())
        for column in BINARY_FEATURES:
            index = names.index(f"binary__{column}")
            np.testing.assert_array_equal(transformed[:, index], [0, 1])
        self.assertIn("festival__Festival_Not Specified", names)
        self.assertIn("promotion_type__Promotion_Type_No Promotion", names)

    def test_invalid_binary_value_is_rejected(self):
        """
        验证非法二元类别由编码器原生错误提示。

        参数：
            self

        返回：
            None
        """
        sample = self.splits.validation.iloc[:1].copy()
        sample["Promotion_Applied"] = "Maybe"
        sample["Promotion_Type"] = "Festival"
        with self.assertRaisesRegex(ValueError, "unknown categories"):
            self.preprocessor.transform(sample)

    def test_promotion_type_is_filled_for_each_status(self):
        """
        验证所有促销状态的空类型都按语义填补。

        参数：
            self

        返回：
            None
        """
        filler = PromotionTypeFiller()
        valid = pd.DataFrame(
            {
                "Promotion_Type": [None, "", "Festival", "  ", None, pd.NA],
                "Promotion_Applied": ["No", "No", "Yes", "Yes", None, "No"],
            }
        )
        result = filler.fit_transform(valid)
        self.assertEqual(
            result["Promotion_Type"].tolist(),
            ["No Promotion", "No Promotion", "Festival", "Not Specified",
             "Not Specified", "No Promotion"],
        )

    def test_blank_features_are_imputed(self):
        """
        验证空白类别、二元和数值字段均可完成填补。

        参数：
            self

        返回：
            None
        """
        sample = self.splits.validation.iloc[:1].copy()
        sample["Product"] = " "
        sample["Temperature"] = ""
        sample["Loyalty_Member"] = None
        sample["Festival"] = "  "
        sample["Promotion_Type"] = None
        sample["Promotion_Applied"] = "Yes"
        transformed = dense(self.preprocessor.transform(sample))
        names = list(self.preprocessor.get_feature_names_out())
        mode = self.splits.train["Product"].mode().iloc[0]
        self.assertEqual(transformed[0, names.index(f"categorical__Product_{mode}")], 1)
        self.assertEqual(transformed[0, names.index("festival__Festival_Not Specified")], 1)
        self.assertTrue(np.isfinite(transformed).all())

    def test_model_pipeline_can_be_cloned_and_serialized(self):
        """
        验证预处理器能与回归模型组合、克隆并序列化。

        参数：
            self

        返回：
            None
        """
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", Ridge())])
        train = self.splits.train.iloc[:300]
        validation = self.splits.validation.iloc[:5]
        fitted = clone(pipeline).fit(train, train[TARGET_COLUMN])
        predictions = fitted.predict(validation)
        self.assertEqual(predictions.shape, (5,))
        self.assertTrue(np.isfinite(predictions).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.joblib"
            joblib.dump(fitted, path)
            restored = joblib.load(path)
            np.testing.assert_allclose(restored.predict(validation), predictions)


if __name__ == "__main__":
    unittest.main()
