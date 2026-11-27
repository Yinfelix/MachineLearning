import json

import pandas as pd

from src import config as C


def test_results_table_is_complete():
    res = pd.read_csv(C.METRIC_DIR / "results_all.csv")
    assert len(res) == len(C.MODEL_ORDER + ["Linear SVM"]) * len(C.SCHEMES)
    assert res["test_f1"].notna().all()
    assert res["test_f1"].between(0, 1).all()
    assert set(res["scheme"]) == set(C.SCHEMES)


def test_shortcut_removal_does_not_improve_f1():
    res = pd.read_csv(C.METRIC_DIR / "results_all.csv")
    a = res[res.scheme == "A"]["test_f1"].mean()
    t = res[res.scheme == "T"]["test_f1"].mean()
    v = res[res.scheme == "V"]["test_f1"].mean()
    assert t <= a and v <= a


def test_confusion_matrices_sum_to_test_size():
    meta = json.loads((C.METRIC_DIR / "test_meta.json").read_text(encoding="utf-8"))
    cms = json.loads((C.METRIC_DIR / "confusion_matrices.json").read_text(encoding="utf-8"))
    for cm in cms.values():
        assert sum(sum(row) for row in cm) == meta["n_test"]
