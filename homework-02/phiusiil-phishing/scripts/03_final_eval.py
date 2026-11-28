"""Step 3: 5-fold CV and one-shot test evaluation for every model × scheme."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import data, evaluate, models
from src.evaluate import save_json


def cv_for_subset(splits: dict, subset_idx):
    ids = splits["inner_fold_id"][subset_idx]
    out = []
    for k in sorted(set(ids.tolist())):
        val = (ids == k).nonzero()[0]
        tr = (ids != k).nonzero()[0]
        out.append((tr, val))
    return out


def tag(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def main() -> None:
    tuning = json.loads((C.METRIC_DIR / "tuning_summary.json").read_text(encoding="utf-8"))
    df = data.load_raw()
    X, y, _ = data.get_xy(df)
    splits = data.ensure_splits(df)
    train_idx, test_idx = splits["train_idx"], splits["test_idx"]
    y_train_all = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)

    all_models = C.MODEL_ORDER + ["Linear SVM"]
    rows, fold_rows, cms = [], [], {}
    score_store = {"y_test": y_test.to_numpy()}

    for name in all_models:
        spec = models.MODEL_SPECS[name]
        best = tuning[name]["best_params"]
        for scheme in C.SCHEMES:
            t0 = time.perf_counter()
            pipe = models.build_pipeline(name, scheme)
            pipe.set_params(**{f"model__{k}": v for k, v in best.items()})

            if spec["subsample"] == "svc":
                sub = splits["svc_idx"]
                X_fit = X.iloc[sub].reset_index(drop=True)
                y_fit = y.iloc[sub].reset_index(drop=True)
                cv = cv_for_subset(splits, sub)
            else:
                X_fit = X.iloc[train_idx].reset_index(drop=True)
                y_fit = y_train_all
                cv = data.inner_cv_splits(train_idx, splits["inner_fold_id"])

            cv_df = evaluate.cross_validate_pipeline(pipe, X_fit, y_fit, cv)
            agg = evaluate.aggregate_cv(cv_df)

            test_metrics, scores, preds = evaluate.evaluate_on_test(
                pipe, X_fit, y_fit, X_test, y_test, return_predictions=True
            )
            row = {
                "model": name,
                "scheme": scheme,
                "scheme_label": C.SCHEMES[scheme]["label"],
                "best_params": json.dumps(best, ensure_ascii=False),
                "n_train": int(len(X_fit)),
                "n_features": len(C.scheme_columns(scheme)[0]) + len(C.scheme_columns(scheme)[1]),
                "cv_f1_mean": agg["f1_mean"],
                "cv_f1_std": agg["f1_std"],
                "cv_accuracy_mean": agg["accuracy_mean"],
                "cv_precision_mean": agg["precision_mean"],
                "cv_recall_mean": agg["recall_mean"],
                "cv_roc_auc_mean": agg["roc_auc_mean"],
                "cv_pr_auc_mean": agg["pr_auc_mean"],
                "cv_fpr_mean": agg["fpr_mean"],
                "cv_balanced_accuracy_mean": agg["balanced_accuracy_mean"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_specificity": test_metrics["specificity"],
                "test_fpr": test_metrics["fpr"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_tn": test_metrics["tn"],
                "test_fp": test_metrics["fp"],
                "test_fn": test_metrics["fn"],
                "test_tp": test_metrics["tp"],
                "fit_seconds": test_metrics["fit_seconds"],
                "predict_seconds": test_metrics["predict_seconds"],
                "wall_seconds": time.perf_counter() - t0,
            }
            rows.append(row)
            cv_df.insert(0, "model", name)
            cv_df.insert(1, "scheme", scheme)
            fold_rows.append(cv_df)
            cms[f"{name}|{scheme}"] = [[test_metrics["tn"], test_metrics["fp"]], [test_metrics["fn"], test_metrics["tp"]]]
            score_store[f"{name}|{scheme}|score"] = np.asarray(scores)
            score_store[f"{name}|{scheme}|pred"] = np.asarray(preds)
            print(
                f"[{name:22s} {scheme}] cvF1={agg['f1_mean']:.5f}±{agg['f1_std']:.5f} "
                f"testF1={test_metrics['f1']:.5f} AUC={test_metrics['roc_auc']:.5f}",
                flush=True,
            )

    res = pd.DataFrame(rows)
    res.to_csv(C.METRIC_DIR / "results_all.csv", index=False, float_format="%.6f")
    res[res["scheme"].isin(C.SCHEME_ORDER)].to_csv(C.METRIC_DIR / "results_main.csv", index=False, float_format="%.6f")
    res[res["scheme"].isin(C.SUPPLEMENTARY_SCHEME_ORDER)].to_csv(
        C.METRIC_DIR / "results_supplementary.csv", index=False, float_format="%.6f"
    )
    pd.concat(fold_rows, ignore_index=True).to_csv(C.METRIC_DIR / "cv_folds_all.csv", index=False, float_format="%.6f")
    save_json(cms, C.METRIC_DIR / "confusion_matrices.json")
    np.savez_compressed(C.METRIC_DIR / "test_scores.npz", **score_store)
    save_json(
        {
            "n_test": int(len(test_idx)),
            "test_phishing": int(y_test.sum()),
            "test_legitimate": int((1 - y_test).sum()),
            "test_phishing_rate": float(y_test.mean()),
        },
        C.METRIC_DIR / "test_meta.json",
    )
    print("final evaluation complete", flush=True)


if __name__ == "__main__":
    main()
