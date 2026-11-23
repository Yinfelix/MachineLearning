import unittest

import numpy as np
import pandas as pd

from src.dataset import load_dataset, split_by_date
from src.features import make_features, sale_price
from src.models import pipeline
from src.train import INPUTS, date_folds


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = split_by_date(load_dataset()[0]).train

    def test_folds_keep_dates_and_orders_together(self):
        for past, future in date_folds(self.train.Date):
            a, b = self.train.iloc[past], self.train.iloc[future]
            self.assertLess(a.Date.max(), b.Date.min())
            self.assertTrue(set(a.Order_ID).isdisjoint(b.Order_ID))

    def test_cross_features_and_sale_price_use_only_order_inputs(self):
        sample = self.train.iloc[:100]
        pd.testing.assert_frame_equal(make_features(sample, "cross"), make_features(sample[INPUTS], "cross"))
        altered = sample.assign(Quantity=100, Total_Bill=9999, Selling_Price=9999, Profit=9999)
        pd.testing.assert_series_equal(sale_price(sample), sale_price(altered))
        pd.testing.assert_frame_equal(make_features(sample, "cross"), make_features(altered, "cross"))

    def test_top_k_features_are_fixed_after_training(self):
        a, b = self.train.iloc[:400], self.train.iloc[400:410].copy()
        model = pipeline("ridge", "cross", 10).fit(a[INPUTS], a.Quantity)
        mask = model["select"].get_support().copy()
        b["Product"] = "New bakery item"
        self.assertTrue(np.isfinite(model.predict(b[INPUTS])).all())
        np.testing.assert_array_equal(mask, model["select"].get_support())
        self.assertEqual(mask.sum(), 10)
