"""Step 4: model optimisation — RF tuning, tree pruning, threshold selection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import data, evaluate, models, plots
from src.evaluate import save_json
from src.preprocess import make_pipeline, make_preprocessor


def cv_setup(splits):
    train_idx = splits["train_idx"]
    return (
        data.inner_cv_splits(train_idx, splits["inner_fold_id"]),
        train_idx,
    )


def eval_pipeline(pipe, X_train, y_train, cv, X_test, y_test) -> dict:
    cv_df = evaluate.cross_validate_pipeline(pipe, X_train, y_train, cv)
    agg = evaluate.aggregate_cv(cv_df)
    test, _ = evaluate.evaluate_on_test(pipe, X_train, y_train, X_test, y_test)
    return {
        "cv_f1_mean": agg["f1_mean"],
        "cv_f1_std": agg["f1_std"],
        "cv_accuracy_mean": agg["accuracy_mean"],
        "cv_recall_mean": agg["recall_mean"],
        "cv_fpr_mean": agg["fpr_mean"],
        "test_f1": test["f1"],
        "test_accuracy": test["accuracy"],
        "test_recall": test["recall"],
        "test_precision": test["precision"],
        "test_fpr": test["fpr"],
        "test_tn": test["tn"],
        "test_fp": test["fp"],
        "test_fn": test["fn"],
        "test_tp": test["tp"],
    }


def build_depth_curve(X_train, y_train, cv) -> pd.DataFrame:
    rows = []
    for depth in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, None]:
        tr_scores, va_scores = [], []
        for tr_pos, va_pos in cv:
            pipe = make_pipeline(DecisionTreeClassifier(max_depth=depth, random_state=C.RANDOM_STATE), "B", scale=False)
            pipe.fit(X_train.iloc[tr_pos], y_train.iloc[tr_pos])
            tr_scores.append(evaluate.metrics_dict(y_train.iloc[tr_pos], pipe.predict(X_train.iloc[tr_pos]))["f1"])
            va_scores.append(evaluate.metrics_dict(y_train.iloc[va_pos], pipe.predict(X_train.iloc[va_pos]))["f1"])
        rows.append(
            {
                "max_depth": 99 if depth is None else depth,
                "train_f1": float(np.mean(tr_scores)),
                "val_f1": float(np.mean(va_scores)),
                "val_f1_std": float(np.std(va_scores)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = data.load_raw()
    X, y, _ = data.get_xy(df)
    splits = data.ensure_splits(df)
    train_idx, test_idx = splits["train_idx"], splits["test_idx"]
    cv, _ = cv_setup(splits)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    tuning = json.loads((C.METRIC_DIR / "tuning_summary.json").read_text(encoding="utf-8"))
    rf_a_params = tuning["Random Forest"]["best_params"]

    # ------------------------------------------------------- RF on scheme B
    if (C.METRIC_DIR / "optimization_rf.csv").exists() and (C.METRIC_DIR / "rf_tuning_B.json").exists():
        rows = pd.read_csv(C.METRIC_DIR / "optimization_rf.csv").to_dict("records")
        best_b = json.loads((C.METRIC_DIR / "rf_tuning_B.json").read_text(encoding="utf-8"))["rf_best_params_B"]
        print("resume: loaded existing RF optimisation results")
    else:
        rows = []
        default_rf = models.default_forest()
        pipe_default = make_pipeline(default_rf, "B", scale=False)
        r = eval_pipeline(pipe_default, X_train, y_train, cv, X_test, y_test)
        r.update({"config": "默认参数（100 棵树）", "scheme": "B"})
        rows.append(r)

        pipe_a = models.build_pipeline("Random Forest", "B")
        pipe_a.set_params(**{f"model__{k}": v for k, v in rf_a_params.items()})
        r = eval_pipeline(pipe_a, X_train, y_train, cv, X_test, y_test)
        r.update({"config": "方案 A 调参结果", "scheme": "B"})
        rows.append(r)

        grid_b = GridSearchCV(
            models.build_pipeline("Random Forest", "B"),
            models.build_param_grid("Random Forest"),
            scoring="f1",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=True,
            verbose=2,
        )
        grid_b.fit(X_train, y_train)
        pd.DataFrame(grid_b.cv_results_).drop(columns=["params"]).to_csv(
            C.METRIC_DIR / "tuning_grid_rf_schemeB.csv", index=False, float_format="%.6f"
        )
        best_b = {k.replace("model__", ""): v for k, v in grid_b.best_params_.items()}
        r = eval_pipeline(grid_b.best_estimator_, X_train, y_train, cv, X_test, y_test)
        r.update({"config": "方案 B 重新调参", "scheme": "B"})
        rows.append(r)
        pd.DataFrame(rows).to_csv(C.METRIC_DIR / "optimization_rf.csv", index=False, float_format="%.6f")
        save_json({"rf_best_params_B": best_b, "rf_best_cv_f1_B": float(grid_b.best_score_)},
                  C.METRIC_DIR / "rf_tuning_B.json")

    # ------------------------------------------------------ DT pruning curves
    depth_path = C.METRIC_DIR / "dt_depth_curve.csv"
    if depth_path.exists():
        depth_df = pd.read_csv(depth_path)
        print("resume: loaded existing DT depth curve")
    else:
        depth_df = build_depth_curve(X_train, y_train, cv)
        depth_df.to_csv(depth_path, index=False, float_format="%.6f")

    # cost-complexity pruning path
    pre_tree = make_preprocessor(*C.scheme_columns("B"), scale=False)
    Xt_train = pre_tree.fit_transform(X_train)
    base_tree = DecisionTreeClassifier(random_state=C.RANDOM_STATE).fit(Xt_train, y_train)
    path = base_tree.cost_complexity_pruning_path(Xt_train, y_train)
    alphas = np.unique(np.quantile(path.ccp_alphas, np.linspace(0, 1, 12)))
    ccp_rows = []
    for alpha in alphas:
        tr_scores, va_scores = [], []
        for tr_pos, va_pos in cv:
            pipe = make_pipeline(DecisionTreeClassifier(ccp_alpha=float(alpha), random_state=C.RANDOM_STATE), "B", scale=False)
            pipe.fit(X_train.iloc[tr_pos], y_train.iloc[tr_pos])
            tr_scores.append(evaluate.metrics_dict(y_train.iloc[tr_pos], pipe.predict(X_train.iloc[tr_pos]))["f1"])
            va_scores.append(evaluate.metrics_dict(y_train.iloc[va_pos], pipe.predict(X_train.iloc[va_pos]))["f1"])
        ccp_rows.append(
            {
                "ccp_alpha": float(alpha),
                "train_f1": float(np.mean(tr_scores)),
                "val_f1": float(np.mean(va_scores)),
                "val_f1_std": float(np.std(va_scores)),
            }
        )
    ccp_df = pd.DataFrame(ccp_rows)
    ccp_df.to_csv(C.METRIC_DIR / "dt_ccp_curve.csv", index=False, float_format="%.6f")

    plots.plot_pruning(depth_df, ccp_df, "fig09_dt_pruning.png")

    # --------------------------------------------- threshold on scheme V
    rf_best_b = models.build_pipeline("Random Forest", "V")
    rf_best_b.set_params(**{f"model__{k}": v for k, v in best_b.items()})
    oof = cross_val_predict(clone(rf_best_b), X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 197)
    rec, fpr, prec, f1s = [], [], [], []
    for t in thresholds:
        m = evaluate.metrics_dict(y_train, (oof >= t).astype(int), oof)
        rec.append(m["recall"]); fpr.append(m["fpr"]); prec.append(m["precision"]); f1s.append(m["f1"])
    thr_df = pd.DataFrame({"threshold": thresholds, "recall": rec, "fpr": fpr, "precision": prec, "f1": f1s})
    thr_df.to_csv(C.METRIC_DIR / "threshold_curve.csv", index=False, float_format="%.6f")
    eligible = thr_df[thr_df["recall"] >= 0.99]
    if len(eligible):
        chosen = float(eligible["threshold"].max())
    else:
        chosen = float(thr_df.loc[thr_df["recall"].idxmax(), "threshold"])

    final = clone(rf_best_b).fit(X_train, y_train)
    test_prob = final.predict_proba(X_test)[:, 1]
    t05 = evaluate.metrics_dict(y_test, (test_prob >= 0.5).astype(int), test_prob)
    tch = evaluate.metrics_dict(y_test, (test_prob >= chosen).astype(int), test_prob)
    save_json(
        {
            "model": "Random Forest on scheme V",
            "params": best_b,
            "chosen_threshold": chosen,
            "rule": "highest threshold with OOF recall >= 0.99",
            "default_0.5": t05,
            "chosen": tch,
            "n_eligible_thresholds": int(len(eligible)),
        },
        C.METRIC_DIR / "threshold_analysis.json",
    )
    plots.plot_threshold_curve(thresholds, rec, fpr, chosen, "阈值对召回率与误报率的影响（方案 V，OOF）",
                               "fig10_threshold.png")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("chosen threshold", chosen, flush=True)
    print("optimization complete", flush=True)


if __name__ == "__main__":
    main()
