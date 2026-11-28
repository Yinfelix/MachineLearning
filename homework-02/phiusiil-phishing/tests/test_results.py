import json

import pandas as pd

from src import config as C


def test_results_table_is_complete():
    res = pd.read_csv(C.METRIC_DIR / "results_all.csv")
    assert len(res) == len(C.MODEL_ORDER + ["Linear SVM"]) * len(C.SCHEMES)
    assert res["test_f1"].notna().all()
    assert res["test_f1"].between(0, 1).all()
    assert set(res["scheme"]) == set(C.SCHEMES)


def test_saved_predictions_match_reported_confusion_counts():
    import numpy as np
    from sklearn.metrics import confusion_matrix

    results = pd.read_csv(C.METRIC_DIR / "results_all.csv")
    with np.load(C.METRIC_DIR / "test_scores.npz") as scores:
        for row in results.itertuples():
            pred = scores[f"{row.model}|{row.scheme}|pred"]
            counts = confusion_matrix(scores["y_test"], pred, labels=[0, 1]).ravel()
            assert tuple(counts) == (row.test_tn, row.test_fp, row.test_fn, row.test_tp)


def test_confusion_matrices_sum_to_test_size():
    meta = json.loads((C.METRIC_DIR / "test_meta.json").read_text(encoding="utf-8"))
    cms = json.loads((C.METRIC_DIR / "confusion_matrices.json").read_text(encoding="utf-8"))
    for cm in cms.values():
        assert sum(sum(row) for row in cm) == meta["n_test"]

def test_polynomial_comparisons_use_paired_validation_folds():
    folds = pd.read_csv(C.METRIC_DIR / "polynomial_folds.csv")
    assert len(folds) == 2 * C.N_INNER_FOLDS
    assert set(folds["fold"]) == set(range(C.N_INNER_FOLDS))
    assert not folds.duplicated(["fold", "mode"]).any()
    assert set(folds["mode"]) == {"linear", "quadratic"}
    assert folds.groupby("fold")["top10"].nunique().eq(1).all()
    assert all(len(set(columns.split("|"))) == 10 for columns in folds["top10"])
    summary = pd.read_csv(C.METRIC_DIR / "polynomial_results.csv").set_index("mode")
    for mode, group in folds.groupby("mode"):
        assert abs(summary.loc[mode, "cv_f1_mean"] - group["f1"].mean()) <= 0.0000005
