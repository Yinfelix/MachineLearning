"""Post-hoc diagnostics for the two collapse cases on scheme V.

Both choices (tree depth, NB threshold) are made on training data only; the
test set is used once to report the resulting performance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import data, evaluate
from src.evaluate import save_json
from src.preprocess import make_pipeline


def main():
    df = data.load_raw()
    X, y, _ = data.get_xy(df)
    splits = data.ensure_splits(df)
    train_idx, test_idx = splits["train_idx"], splits["test_idx"]
    fold_id = splits["inner_fold_id"]
    X_tr, y_tr = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
    X_te, y_te = X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

    # ---------------------------------------------- DT depth re-selected on V
    val_idx = train_idx[fold_id[train_idx] == 0]
    pool_idx = train_idx[fold_id[train_idx] != 0]
    X_pool, y_pool = X.iloc[pool_idx].reset_index(drop=True), y.iloc[pool_idx].reset_index(drop=True)
    X_val, y_val = X.iloc[val_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True)
    depth_rows = []
    for depth in [3, 4, 5, 6, 8, 10, 12, 16, 20, None]:
        pipe = make_pipeline(DecisionTreeClassifier(max_depth=depth, random_state=C.RANDOM_STATE), "V", scale=False)
        est = clone(pipe).fit(X_pool, y_pool)
        m = evaluate.metrics_dict(y_val, est.predict(X_val))
        depth_rows.append({"max_depth": 99 if depth is None else depth, "val_f1": m["f1"]})
    best = max(depth_rows, key=lambda r: r["val_f1"])
    best_depth = None if best["max_depth"] == 99 else best["max_depth"]
    pipe = make_pipeline(DecisionTreeClassifier(max_depth=best_depth, random_state=C.RANDOM_STATE), "V", scale=False)
    est = clone(pipe).fit(X_tr, y_tr)
    test_m = evaluate.metrics_dict(y_te, est.predict(X_te))

    # ------------------------------------------------- NB threshold on V
    nb = make_pipeline(GaussianNB(var_smoothing=1e-9), "V", scale=False)
    cv = data.inner_cv_splits(train_idx, fold_id)
    oof = cross_val_predict(clone(nb), X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    thresholds = np.unique(np.concatenate([np.logspace(-6, -1, 60), np.linspace(0.1, 0.99, 90)]))
    rows = []
    for t in thresholds:
        m = evaluate.metrics_dict(y_tr, (oof >= t).astype(int))
        rows.append((t, m["f1"], m["precision"], m["recall"], m["fpr"]))
    best_nb = max(rows, key=lambda r: r[1])
    nb_fit = clone(nb).fit(X_tr, y_tr)
    prob = nb_fit.predict_proba(X_te)[:, 1]
    nb_default = evaluate.metrics_dict(y_te, (prob >= 0.5).astype(int), prob)
    nb_chosen = evaluate.metrics_dict(y_te, (prob >= best_nb[0]).astype(int), prob)

    save_json(
        {
            "dt_V": {
                "depth_selection": depth_rows,
                "best_depth_by_validation": best_depth,
                "test_f1_at_best_depth": test_m["f1"],
                "test_recall_at_best_depth": test_m["recall"],
                "test_fpr_at_best_depth": test_m["fpr"],
                "test_f1_at_depth3": None,
            },
            "nb_V": {
                "best_threshold_oof": float(best_nb[0]),
                "default_0.5": nb_default,
                "chosen_threshold": nb_chosen,
            },
        },
        C.METRIC_DIR / "posthoc_analysis.json",
    )
    print(json.dumps({"dt_best_depth": best_depth, "dt_test_f1": test_m["f1"],
                      "nb_threshold": best_nb[0], "nb_default_f1": nb_default["f1"],
                      "nb_chosen_f1": nb_chosen["f1"]}, indent=2))


if __name__ == "__main__":
    main()
