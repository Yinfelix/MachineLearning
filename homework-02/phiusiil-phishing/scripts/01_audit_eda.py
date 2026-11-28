"""Step 1: data audit, split construction, EDA tables and figures."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import data, plots
from src.evaluate import save_json


def feature_group(col: str) -> str:
    if col in C.SHORTCUT_FEATURES:
        return "捷径特征"
    if col in C.URL_NUMERIC:
        return "URL/域名"
    return "页面内容"


def main() -> None:
    df = data.load_raw()
    X, y, groups = data.get_xy(df)
    splits = data.ensure_splits(df, force=True)
    tr, te = splits["train_idx"], splits["test_idx"]
    train = df.iloc[tr].copy()
    train["is_phishing"] = y.iloc[tr].values

    # ------------------------------------------------------------------ audit
    dom_counts = df["Domain"].value_counts()
    dup_dom_rows = int((dom_counts[df["Domain"]].values > 1).sum())
    same_label = (
        train.groupby("Domain")["label"].nunique().value_counts().sort_index().to_dict()
    )
    sim = train.groupby("label")["URLSimilarityIndex"]
    audit = {
        "shape": list(df.shape),
        "class_conditional_scope": "outer training rows only",
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_features_excluding_text": len(C.all_feature_columns()),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_col": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_url": int(df["URL"].nunique()),
        "duplicate_url_rows": int(df["URL"].duplicated().sum()),
        "unique_domains": int(df["Domain"].nunique()),
        "rows_in_repeated_domains": dup_dom_rows,
        "max_domain_group": int(dom_counts.max()),
        "domain_label_nunique_counts": {str(k): int(v) for k, v in same_label.items()},
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "label_meaning": {"1": "legitimate", "0": "phishing"},
        "phishing_rate": float((df["label"] == 0).mean()),
        "url_similarity_by_label": {
            str(k): {
                "mean": float(v.mean()),
                "std": float(v.std()),
                "min": float(v.min()),
                "max": float(v.max()),
                "pct_at_100": float((v == 100).mean()),
            }
            for k, v in sim
        },
        "phishing_with_similarity_100": int(((train["label"] == 0) & (train["URLSimilarityIndex"] == 100)).sum()),
        "unique_tld": int(df["TLD"].nunique()),
        "top_tld": {str(k): int(v) for k, v in df["TLD"].value_counts().head(15).items()},
        "is_https_by_label": {
            str(k): float(v) for k, v in train.groupby("label")["IsHTTPS"].mean().items()
        },
        "dtypes": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
        "train_test": splits["meta"],
    }
    save_json(audit, C.METRIC_DIR / "data_audit.json")

    # --------------------------------------------------------- feature groups
    fg = {
        "url_numeric": C.URL_NUMERIC,
        "page_numeric": C.PAGE_NUMERIC,
        "categorical": C.CATEGORICAL,
        "shortcut_features": C.SHORTCUT_FEATURES,
        "schemes": {k: {"label": v["label"], "numeric": v["numeric"], "categorical": v["categorical"]} for k, v in C.SCHEMES.items()},
        "log_features": C.LOG_FEATURES,
        "n_features": {
            k: len(v["numeric"]) + len(v["categorical"]) for k, v in C.SCHEMES.items()
        },
    }
    save_json(fg, C.CONFIG_DIR / "feature_groups.json")

    # ------------------------------------------------------- univariate stats
    ytr = y.iloc[tr]
    rows = []
    for col in C.all_feature_columns():
        if col == "TLD":
            continue
        x = X.iloc[tr][col].astype(float)
        auc = roc_auc_score(ytr, x)
        rows.append(
            {
                "feature": col,
                "group": feature_group(col),
                "auc": float(auc),
                "auc_oriented": float(max(auc, 1 - auc)),
                "direction": "higher=phishing" if auc >= 0.5 else "lower=phishing",
            }
        )
    auc_df = pd.DataFrame(rows).sort_values("auc_oriented", ascending=False)
    auc_df.to_csv(C.METRIC_DIR / "univariate_auc.csv", index=False, float_format="%.6f")

    # ---------------------------------------------------------- correlations
    num_cols = [c for c in C.all_feature_columns() if c != "TLD"]
    corr = X.iloc[tr][num_cols].corr(method="spearman")
    corr.to_csv(C.METRIC_DIR / "correlation_spearman.csv", float_format="%.5f")
    pairs = []
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            rho = corr.loc[a, b]
            if abs(rho) >= 0.95:
                pairs.append({"feature_a": a, "feature_b": b, "rho": float(rho), "abs_rho": float(abs(rho))})
    pd.DataFrame(pairs).sort_values("abs_rho", ascending=False).to_csv(
        C.METRIC_DIR / "high_correlation_pairs.csv", index=False, float_format="%.5f"
    )
    save_json(
        {
            "n_numeric_features": len(num_cols),
            "n_pairs_abs_rho_ge_095": len(pairs),
            "n_pairs_abs_rho_ge_090": int(sum(abs(corr.loc[a, b]) >= 0.90 for i, a in enumerate(num_cols) for b in num_cols[i + 1:])),
        },
        C.METRIC_DIR / "correlation_summary.json",
    )

    # ---------------------------------------------------------------- figures
    plots.plot_dataset_overview(df, train, "fig01_dataset_overview.png")
    plots.plot_univariate_auc(auc_df, "fig02_univariate_auc.png")
    top_feats = auc_df.head(26)["feature"].tolist()
    plots.plot_correlation(corr.loc[top_feats, top_feats], "fig03_correlation.png",
                           title="单变量判别力最强的 26 个特征：Spearman 相关性")
    print("audit + EDA complete")
    print(auc_df.head(12).to_string(index=False))
    print(audit["train_test"]["train"], audit["train_test"]["test"])


if __name__ == "__main__":
    main()
