"""Continuation for the two figures that failed in 06 (tree fontsize, PCA scatter)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import data, models, plots
from src.evaluate import save_json
from src.preprocess import make_preprocessor


def main():
    tuning = json.loads((C.METRIC_DIR / "tuning_summary.json").read_text(encoding="utf-8"))
    df = data.load_raw()
    X, y, _ = data.get_xy(df)
    splits = data.ensure_splits(df)
    train_idx, test_idx = splits["train_idx"], splits["test_idx"]
    X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)

    pre = make_preprocessor(*C.scheme_columns("B"), scale=False)
    Xt = pre.fit_transform(X_train)
    names = list(pre.get_feature_names_out())

    tree = DecisionTreeClassifier(max_depth=3, random_state=C.RANDOM_STATE).fit(Xt, y_train)
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    plot_tree(tree, feature_names=names, class_names=["合法", "钓鱼"], filled=True, rounded=True,
              fontsize=7, ax=ax, impurity=False, proportion=True)
    ax.set_title("方案 B 上的浅层决策树（max_depth=3）")
    fig.tight_layout()
    plots.savefig(fig, "fig18_tree_rules.png")
    root_feature = names[tree.tree_.feature[0]]

    rf = models.build_pipeline("Random Forest", "B")
    rf.set_params(**{f"model__{k}": v for k, v in tuning["Random Forest"]["best_params"].items()})
    rf.fit(X_train, y_train)
    sample_idx = np.random.RandomState(C.RANDOM_STATE).choice(len(X_test), size=min(12000, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]
    emb = PCA(n_components=2, random_state=C.RANDOM_STATE).fit_transform(
        make_preprocessor(*C.scheme_columns("B"), scale=True).fit(X_train).transform(X_sample)
    )
    plots.plot_pca_scatter(emb, y.iloc[test_idx].iloc[sample_idx].to_numpy(), rf.predict(X_sample), "fig19_pca.png")

    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names)
    numeric_b, _ = C.scheme_columns("B")
    top10 = imp[imp.index.isin(numeric_b)].sort_values(ascending=False).head(10).index.tolist()
    save_json(
        {
            "top10_rf_importance": top10,
            "tree_root_feature": root_feature,
            "shap_sample": 2000,
            "permutation_sample": 20000,
            "permutation_repeats": 5,
        },
        C.METRIC_DIR / "feature_engineering_summary.json",
    )
    print("tail done; root feature:", root_feature)


if __name__ == "__main__":
    main()
