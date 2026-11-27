"""Step 5 (extension 1): generalisation and robustness experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import data, evaluate, models, plots
from src.evaluate import save_json
from src.preprocess import make_pipeline

RNG_SEEDS = [42, 52, 62]
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
LABEL_RATES = [0.0, 0.05, 0.10, 0.20]
FEATURE_LEVELS = [0.0, 0.1, 0.5, 1.0]


def tuned_pipeline(name: str, scheme="V", tuning=None):
    pipe = models.build_pipeline(name, scheme)
    best = tuning[name]["best_params"]
    pipe.set_params(**{f"model__{k}": v for k, v in best.items()})
    # Fairness override: depth=3 was selected on scheme A (a single
    # URLSimilarityIndex split) and severely underfits scheme V.  For the
    # robustness curves we use the depth re-selected on V (unlimited), so the
    # experiment measures robustness rather than a known underfit.
    if name == "Decision Tree" and scheme == "V":
        pipe.set_params(model__max_depth=None)
    return pipe


def evaluate_val(pipe, X_pool, y_pool, X_val, y_val) -> dict:
    est = clone(pipe).fit(X_pool, y_pool)
    pred = est.predict(X_val)
    score = evaluate.score_of(est, X_val)
    return evaluate.metrics_dict(y_val, pred, score)


def main() -> None:
    tuning = json.loads((C.METRIC_DIR / "tuning_summary.json").read_text(encoding="utf-8"))
    df = data.load_raw()
    X, y, _ = data.get_xy(df)
    splits = data.ensure_splits(df)
    train_idx, test_idx = splits["train_idx"], splits["test_idx"]
    fold_id = splits["inner_fold_id"]

    # Fixed internal validation = inner fold 0; the remaining four folds form the
    # sampling pool.  The independent test set is never used here.
    val_idx = train_idx[fold_id[train_idx] == 0]
    pool_idx = train_idx[fold_id[train_idx] != 0]
    X_pool, y_pool = X.iloc[pool_idx].reset_index(drop=True), y.iloc[pool_idx].reset_index(drop=True)
    X_val, y_val = X.iloc[val_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True)
    save_json(
        {"pool": {"n": len(pool_idx), "phishing": int(y_pool.sum())},
         "validation": {"n": len(val_idx), "phishing": int(y_val.sum())}},
        C.METRIC_DIR / "robustness_setup.json",
    )

    # ------------------------------------------------------- 1. learning curve
    rows = []
    for name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        pipe = tuned_pipeline(name, tuning=tuning)
        for frac in FRACTIONS:
            for seed in RNG_SEEDS:
                if frac >= 1.0:
                    sub = pool_idx
                else:
                    n = max(200, int(len(pool_idx) * frac))
                    sub, _ = train_test_split(pool_idx, train_size=n, stratify=y.iloc[pool_idx], random_state=seed)
                m = evaluate_val(pipe, X.iloc[sub], y.iloc[sub], X_val, y_val)
                rows.append({"model": name, "fraction": frac, "seed": seed, "n_train": len(sub),
                             "f1": m["f1"], "roc_auc": m["roc_auc"], "fpr": m["fpr"], "recall": m["recall"]})
                print(f"[LC] {name:20s} {frac:4.0%} seed={seed} n={len(sub):6d} F1={m['f1']:.4f}", flush=True)
    lc = pd.DataFrame(rows)
    lc.to_csv(C.METRIC_DIR / "robustness_learning_curve.csv", index=False, float_format="%.6f")
    plots.plot_learning_curve(lc, "fig11_learning_curve.png")

    # ---------------------------------------------------------- 2. label noise
    rows = []
    rng = np.random.RandomState(C.RANDOM_STATE)
    flip = rng.rand(len(y_pool))
    for name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        pipe = tuned_pipeline(name, tuning=tuning)
        for rate in LABEL_RATES:
            y_noisy = y_pool.copy()
            mask = flip < rate
            y_noisy[mask] = 1 - y_noisy[mask]
            m = evaluate_val(pipe, X_pool, y_noisy, X_val, y_val)
            rows.append({"model": name, "noise_rate": rate, "f1": m["f1"], "roc_auc": m["roc_auc"],
                         "fpr": m["fpr"], "recall": m["recall"]})
            print(f"[LN] {name:20s} {rate:4.0%} F1={m['f1']:.4f}", flush=True)
    ln = pd.DataFrame(rows)
    ln.to_csv(C.METRIC_DIR / "robustness_label_noise.csv", index=False, float_format="%.6f")

    # ------------------------------------------------ 3. test feature noise
    numeric_b, _ = C.scheme_columns("V")
    continuous = [c for c in numeric_b if X_pool[c].nunique() > 10]
    stds = X_pool[continuous].std(ddof=0).replace(0, 1.0)
    rows = []
    noise_rng = np.random.RandomState(C.RANDOM_STATE)
    noise_draws = {lvl: noise_rng.normal(size=(len(X_val), len(continuous))) for lvl in FEATURE_LEVELS}
    for name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        pipe = tuned_pipeline(name, tuning=tuning)
        est = clone(pipe).fit(X_pool, y_pool)
        for lvl in FEATURE_LEVELS:
            Xn = X_val.copy()
            Xn[continuous] = Xn[continuous].to_numpy() + noise_draws[lvl] * stds.to_numpy() * lvl
            pred = est.predict(Xn)
            score = evaluate.score_of(est, Xn)
            m = evaluate.metrics_dict(y_val, pred, score)
            rows.append({"model": name, "noise_sd": lvl, "f1": m["f1"], "roc_auc": m["roc_auc"],
                         "fpr": m["fpr"], "recall": m["recall"]})
            print(f"[FN] {name:20s} {lvl:3.1f}sd F1={m['f1']:.4f}", flush=True)
    fn = pd.DataFrame(rows)
    fn.to_csv(C.METRIC_DIR / "robustness_feature_noise.csv", index=False, float_format="%.6f")
    plots.plot_noise_panels(ln, fn, "fig12_noise.png")

    # ------------------------------------------------------------- 4. outliers
    outlier_feats = [c for c in ["URLLength", "DomainLength", "NoOfOtherSpecialCharsInURL"] if c in numeric_b]
    X_out = X_val.copy()
    out_rng = np.random.RandomState(C.RANDOM_STATE)
    pick = out_rng.choice(len(X_out), size=max(1, int(0.05 * len(X_out))), replace=False)
    for c in outlier_feats:
        mu, sd = X_pool[c].mean(), X_pool[c].std(ddof=0)
        X_out.loc[X_out.index[pick], c] = mu + 15 * sd
    rows = []
    lr_best = tuning["Logistic Regression"]["best_params"]
    for scaler in ["standard", "robust"]:
        pipe = make_pipeline(LogisticRegression(solver="liblinear", max_iter=2000, random_state=C.RANDOM_STATE,
                                                C=lr_best["C"], penalty=lr_best.get("penalty", "l2")),
                             "V", scale=True, scaler=scaler)
        est = clone(pipe).fit(X_pool, y_pool)
        for condition, Xeval in [("干净验证集", X_val), ("5% 极端异常值", X_out)]:
            pred = est.predict(Xeval)
            score = evaluate.score_of(est, Xeval)
            m = evaluate.metrics_dict(y_val, pred, score)
            rows.append({"scaler": scaler, "condition": condition, "f1": m["f1"], "roc_auc": m["roc_auc"],
                         "fpr": m["fpr"], "recall": m["recall"]})
    pd.DataFrame(rows).to_csv(C.METRIC_DIR / "robustness_outliers.csv", index=False, float_format="%.6f")

    # ------------------------------------------------------ 5. TLD transfer
    top_tld = X_pool["TLD"].value_counts().head(20).index
    common_idx = pool_idx[X.iloc[pool_idx]["TLD"].isin(top_tld).to_numpy()]
    rare_test_idx = test_idx[X.iloc[test_idx]["TLD"].notna().to_numpy()]
    rare_test_idx = rare_test_idx[~X.iloc[rare_test_idx]["TLD"].isin(top_tld).to_numpy()]
    common_tr, common_te = train_test_split(
        common_idx, test_size=0.2, stratify=y.iloc[common_idx], random_state=C.RANDOM_STATE
    )
    X_ctr, y_ctr = X.iloc[common_tr], y.iloc[common_tr]
    X_cte, y_cte = X.iloc[common_te], y.iloc[common_te]
    X_rare, y_rare = X.iloc[rare_test_idx], y.iloc[rare_test_idx]
    rows = []
    for name in ["Logistic Regression", "Random Forest"]:
        pipe = tuned_pipeline(name, tuning=tuning)
        est = clone(pipe).fit(X_ctr, y_ctr)
        for split_name, Xeval, yeval in [("随机划分", X_cte, y_cte), ("罕见 TLD", X_rare, y_rare)]:
            pred = est.predict(Xeval)
            score = evaluate.score_of(est, Xeval)
            m = evaluate.metrics_dict(yeval, pred, score)
            rows.append({"model": name, "split": split_name, "n": len(yeval), "phishing_rate": float(yeval.mean()),
                         "f1": m["f1"], "roc_auc": m["roc_auc"], "fpr": m["fpr"], "recall": m["recall"]})
            print(f"[TLD] {name:20s} {split_name:6s} n={len(yeval):5d} F1={m['f1']:.4f}", flush=True)
    tld_df = pd.DataFrame(rows)
    tld_df.to_csv(C.METRIC_DIR / "robustness_tld_transfer.csv", index=False, float_format="%.6f")
    plots.plot_tld_transfer(tld_df, "fig13_tld_transfer.png")

    # ------------------------------------------------------- 6. overfitting gap
    rows = []
    for name in C.MODEL_ORDER:
        spec = models.MODEL_SPECS[name]
        pipe = tuned_pipeline(name, tuning=tuning)
        if spec["subsample"] == "svc":
            sub = splits["svc_idx"]
            ids = fold_id[sub]
            X_fit, y_fit = X.iloc[sub].reset_index(drop=True), y.iloc[sub].reset_index(drop=True)
            cv = []
            for k in sorted(set(ids.tolist())):
                val = (ids == k).nonzero()[0]
                tr = (ids != k).nonzero()[0]
                cv.append((tr, val))
        else:
            X_fit, y_fit = X_pool, y_pool
            cv = []
            ids = fold_id[pool_idx]
            for k in sorted(set(ids.tolist())):
                val = (ids == k).nonzero()[0]
                tr = (ids != k).nonzero()[0]
                cv.append((tr, val))
        tr_s, va_s = [], []
        for tr, va in cv:
            est = clone(pipe).fit(X_fit.iloc[tr], y_fit.iloc[tr])
            tr_s.append(evaluate.metrics_dict(y_fit.iloc[tr], est.predict(X_fit.iloc[tr]))["f1"])
            va_s.append(evaluate.metrics_dict(y_fit.iloc[va], est.predict(X_fit.iloc[va]))["f1"])
        rows.append({"model": name, "train_f1": float(np.mean(tr_s)), "val_f1": float(np.mean(va_s)),
                     "gap": float(np.mean(tr_s) - np.mean(va_s))})
        print(f"[OF] {name:20s} train={rows[-1]['train_f1']:.4f} val={rows[-1]['val_f1']:.4f}", flush=True)
    of = pd.DataFrame(rows)
    of.to_csv(C.METRIC_DIR / "overfitting.csv", index=False, float_format="%.6f")
    plots.plot_overfitting(of, "fig14_overfitting.png")
    print("robustness experiments complete", flush=True)


if __name__ == "__main__":
    main()
