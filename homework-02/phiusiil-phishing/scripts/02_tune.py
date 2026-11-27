"""Step 2: exhaustive grid search on scheme A with grouped 5-fold CV.

Hyper-parameters are selected once on the full-feature scheme (A) and then held
fixed across feature schemes, so the ablation reflects the feature sets rather
than a second round of tuning.  The RBF SVM is tuned on its fixed 20k subsample;
the linear SVM is the full-data control.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import data, models
from src.evaluate import save_json


def cv_for_subset(splits: dict, subset_idx):
    ids = splits["inner_fold_id"][subset_idx]
    out = []
    for k in sorted(set(ids.tolist())):
        val = (ids == k).nonzero()[0]
        tr = (ids != k).nonzero()[0]
        out.append((tr, val))
    return out


def main() -> None:
    df = data.load_raw()
    X, y, _ = data.get_xy(df)
    splits = data.ensure_splits(df)
    train_idx = splits["train_idx"]
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)

    summary = {}
    for name in C.MODEL_ORDER + ["Linear SVM"]:
        t0 = time.perf_counter()
        spec = models.MODEL_SPECS[name]
        if spec["subsample"] == "svc":
            sub = splits["svc_idx"]
            X_fit = X.iloc[sub].reset_index(drop=True)
            y_fit = y.iloc[sub].reset_index(drop=True)
            cv = cv_for_subset(splits, sub)
            n_samples = len(sub)
        else:
            X_fit, y_fit = X_train, y_train
            cv = data.inner_cv_splits(train_idx, splits["inner_fold_id"])
            n_samples = len(train_idx)

        pipe = models.build_pipeline(name, "A")
        search = GridSearchCV(
            pipe,
            models.build_param_grid(name),
            scoring="f1",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=True,
            verbose=2,
            error_score="raise",
        )
        search.fit(X_fit, y_fit)
        elapsed = time.perf_counter() - t0

        best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        tag = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        pd.DataFrame(search.cv_results_).drop(columns=["params"]).to_csv(
            C.METRIC_DIR / f"tuning_grid_{tag}.csv", index=False, float_format="%.6f"
        )
        joblib.dump(search.best_estimator_, C.MODEL_DIR / f"best_A_{tag}.joblib")
        summary[name] = {
            "best_params": best,
            "best_cv_f1": float(search.best_score_),
            "n_samples": int(n_samples),
            "n_candidates": int(len(search.cv_results_["params"])),
            "n_fits": int(len(search.cv_results_["params"]) * len(cv)),
            "elapsed_seconds": float(elapsed),
            "scale_features": bool(spec["scale"]),
            "subsample": spec["subsample"],
        }
        save_json(summary, C.METRIC_DIR / "tuning_summary.json")
        print(f"[done] {name}: f1={search.best_score_:.5f} params={best} ({elapsed:.1f}s)", flush=True)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
