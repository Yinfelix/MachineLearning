"""Step 6 (extension 2): feature engineering, selection, and interpretability."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import data, evaluate, models, plots
from src.evaluate import save_json
from src.preprocess import make_pipeline, make_preprocessor


def tuned_pipeline(name: str, scheme="B", tuning=None):
    pipe = models.build_pipeline(name, scheme)
    best = tuning[name]["best_params"]
    pipe.set_params(**{f"model__{k}": v for k, v in best.items()})
    return pipe


def main() -> None:
    tuning = json.loads((C.METRIC_DIR / "tuning_summary.json").read_text(encoding="utf-8"))
    df = data.load_raw()
    X, y, _ = data.get_xy(df)
    splits = data.ensure_splits(df)
    train_idx, test_idx = splits["train_idx"], splits["test_idx"]
    fold_id = splits["inner_fold_id"]
    cv = data.inner_cv_splits(train_idx, fold_id)
    X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
    X_test, y_test = X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
    val_idx = train_idx[fold_id[train_idx] == 0]
    X_val, y_val = X.iloc[val_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True)

    # ------------------------------------------- 1. univariate / mutual info
    mi_sample = X_train.sample(n=min(50000, len(X_train)), random_state=C.RANDOM_STATE)
    mi_y = y_train.loc[mi_sample.index]
    from sklearn.feature_selection import mutual_info_classif
    numeric_b, _ = C.scheme_columns("B")
    mi = mutual_info_classif(mi_sample[numeric_b], mi_y, random_state=C.RANDOM_STATE, discrete_features="auto")
    mi_df = pd.DataFrame({"feature": numeric_b, "mutual_info": mi}).sort_values("mutual_info", ascending=False)
    mi_df.to_csv(C.METRIC_DIR / "mutual_info.csv", index=False, float_format="%.6f")

    # -------------------------------------------- 2. filter selection (K best)
    rows = []
    for name in ["Logistic Regression", "Random Forest"]:
        for k in [5, 10, 20, "all"]:
            pipe = tuned_pipeline(name, tuning=tuning)
            if k != "all":
                pipe.steps.insert(-1, ("select", SelectKBest(f_classif, k=k)))
            cv_df = evaluate.cross_validate_pipeline(pipe, X_train, y_train, cv)
            rows.append({"model": name, "k": k, "cv_f1_mean": cv_df["f1"].mean(), "cv_f1_std": cv_df["f1"].std(ddof=0),
                         "cv_roc_auc_mean": cv_df["roc_auc"].mean()})
            print(f"[K] {name} k={k} F1={cv_df['f1'].mean():.4f}", flush=True)
    k_df = pd.DataFrame(rows)
    k_df.to_csv(C.METRIC_DIR / "feature_selection_k.csv", index=False, float_format="%.6f")

    # ------------------------------------------------------ 3. L1 LR path
    # A 50k stratified subsample keeps the path experiment fast; the number of
    # non-zero coefficients is stable at this size and is used only qualitatively.
    l1_sample = X_train.sample(n=min(50000, len(X_train)), random_state=C.RANDOM_STATE)
    y_l1 = y_train.loc[l1_sample.index]
    l1_rows = []
    pre = make_preprocessor(*C.scheme_columns("B"), scale=True)
    Xt = pre.fit_transform(l1_sample)
    for Cval in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        lr = LogisticRegression(penalty="l1", solver="liblinear", C=Cval, max_iter=1000, random_state=C.RANDOM_STATE)
        lr.fit(Xt, y_l1)
        l1_rows.append({"C": Cval, "nonzero": int(np.sum(np.abs(lr.coef_) > 1e-8)),
                        "train_f1": float(evaluate.metrics_dict(y_l1, lr.predict(Xt))["f1"])})
    pd.DataFrame(l1_rows).to_csv(C.METRIC_DIR / "l1_path.csv", index=False, float_format="%.6f")

    # ------------------------------------------------------------- 4. PCA
    rows = []
    for name in ["Logistic Regression", "SVM (RBF)"]:
        spec = models.MODEL_SPECS[name]
        if spec["subsample"] == "svc":
            sub = splits["svc_idx"]
            ids = fold_id[sub]
            X_fit, y_fit = X.iloc[sub].reset_index(drop=True), y.iloc[sub].reset_index(drop=True)
            cv_fit = []
            for k in sorted(set(ids.tolist())):
                val = (ids == k).nonzero()[0]
                tr = (ids != k).nonzero()[0]
                cv_fit.append((tr, val))
        else:
            X_fit, y_fit, cv_fit = X_train, y_train, cv
        for var in [0.90, 0.95, 0.99]:
            pipe = tuned_pipeline(name, tuning=tuning)
            pipe.steps.insert(-1, ("pca", PCA(n_components=var, random_state=C.RANDOM_STATE)))
            cv_df = evaluate.cross_validate_pipeline(pipe, X_fit, y_fit, cv_fit)
            full = clone(pipe).fit(X_fit, y_fit)
            ncomp = int(full.named_steps["pca"].n_components_)
            rows.append({"model": name, "variance": var, "n_components": ncomp,
                         "cv_f1_mean": cv_df["f1"].mean(), "cv_f1_std": cv_df["f1"].std(ddof=0),
                         "cv_roc_auc_mean": cv_df["roc_auc"].mean()})
            print(f"[PCA] {name} var={var} ncomp={ncomp} F1={cv_df['f1'].mean():.4f}", flush=True)
    pca_df = pd.DataFrame(rows)
    pca_df.to_csv(C.METRIC_DIR / "pca_results.csv", index=False, float_format="%.6f")

    # ------------------------------------------------- 5. polynomial expansion
    # Fit the RF to rank features, then compare linear vs quadratic LR on top-10.
    rf = tuned_pipeline("Random Forest", tuning=tuning).fit(X_train, y_train)
    names_out = list(rf.named_steps["pre"].get_feature_names_out())
    imp_all = pd.Series(rf.named_steps["model"].feature_importances_, index=names_out).sort_values(ascending=False)
    imp = imp_all[imp_all.index.isin(numeric_b)]
    top10 = imp.head(10).index.tolist()
    poly_rows = []
    for mode in ["linear", "quadratic"]:
        steps = [("num", make_preprocessor(top10, [], scale=True))]
        if mode == "quadratic":
            steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
            steps.append(("scale2", StandardScaler()))
        lr_best = tuning["Logistic Regression"]["best_params"]
        pipe = Pipeline(steps + [("model", LogisticRegression(solver="liblinear", max_iter=3000,
                                                               C=lr_best["C"],
                                                               penalty=lr_best.get("penalty", "l2"),
                                                               random_state=C.RANDOM_STATE))])
        cv_df = evaluate.cross_validate_pipeline(pipe, X_train[top10], y_train, cv)
        poly_rows.append({"mode": mode, "n_features_in": len(top10), "cv_f1_mean": cv_df["f1"].mean(),
                          "cv_f1_std": cv_df["f1"].std(ddof=0), "cv_roc_auc_mean": cv_df["roc_auc"].mean()})
        print(f"[POLY] {mode} F1={cv_df['f1'].mean():.4f}", flush=True)
    pd.DataFrame(poly_rows).to_csv(C.METRIC_DIR / "polynomial_results.csv", index=False, float_format="%.6f")

    # -------------------------------------- 6. impurity vs permutation importance
    tree_imp = pd.DataFrame({"feature": names_out, "importance": rf.named_steps["model"].feature_importances_}).sort_values(
        "importance", ascending=False
    )
    tree_imp.to_csv(C.METRIC_DIR / "rf_impurity_importance.csv", index=False, float_format="%.8f")

    perm_sample = X_val.sample(n=min(20000, len(X_val)), random_state=C.RANDOM_STATE)
    perm_y = y_val.loc[perm_sample.index]
    perm = permutation_importance(rf, perm_sample, perm_y, scoring="f1", n_repeats=5,
                                  random_state=C.RANDOM_STATE, n_jobs=1)
    perm_df = pd.DataFrame({"feature": perm_sample.columns, "importance": perm.importances_mean,
                            "std": perm.importances_std}).sort_values("importance", ascending=False)
    perm_df.to_csv(C.METRIC_DIR / "permutation_importance.csv", index=False, float_format="%.8f")
    plots.plot_horizontal_importance(perm_df, "fig16_permutation_importance.png",
                                     "随机森林的置换重要性（方案 B，验证集，F1 下降）")

    # ------------------------------------------------------------- 7. SHAP
    shap_sample = X_val.sample(n=min(2000, len(X_val)), random_state=C.RANDOM_STATE)
    Xs = rf.named_steps["pre"].transform(shap_sample)
    explainer = shap.TreeExplainer(rf.named_steps["model"])
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):
        sv1 = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv1 = sv[:, :, 1]
    else:
        sv1 = sv
    plt.figure(figsize=(7.4, 6.2))
    shap.summary_plot(sv1, Xs, feature_names=names_out, max_display=20, show=False, plot_size=None)
    ax = plt.gca()
    ax.set_title("随机森林的 SHAP 值（方案 B，验证集 2,000 条）")
    fig = plt.gcf()
    fig.tight_layout()
    plots.savefig(fig, "fig17_shap.png")

    # ------------------------------------------------- 8. shallow tree + PCA 2D
    pre_tree = make_preprocessor(*C.scheme_columns("B"), scale=False)
    Xt_tree = pre_tree.fit_transform(X_train)
    tree = DecisionTreeClassifier(max_depth=3, random_state=C.RANDOM_STATE).fit(Xt_tree, y_train)
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    plot_tree(tree, feature_names=names_out, class_names=["合法", "钓鱼"], filled=True, rounded=True,
              fontsize=7, ax=ax, impurity=False, proportion=True)
    ax.set_title("方案 B 上的浅层决策树（max_depth=3）")
    fig.tight_layout()
    plots.savefig(fig, "fig18_tree_rules.png")

    pre_pca = make_preprocessor(*C.scheme_columns("B"), scale=True)
    Xt_pca = pre_pca.fit_transform(X_train)
    pca2 = PCA(n_components=2, random_state=C.RANDOM_STATE).fit(Xt_pca)
    sample_idx = np.random.RandomState(C.RANDOM_STATE).choice(len(X_test), size=min(12000, len(X_test)), replace=False)
    X_test_sample = X_test.iloc[sample_idx]
    emb = pca2.transform(pre_pca.transform(X_test_sample))
    rf_pred = rf.predict(X_test_sample)
    plots.plot_pca_scatter(emb, y_test.iloc[sample_idx].to_numpy(), rf_pred, "fig19_pca.png")

    save_json(
        {
            "top10_rf_importance": top10,
            "shap_sample": int(len(shap_sample)),
            "permutation_sample": int(len(perm_sample)),
            "permutation_repeats": 5,
        },
        C.METRIC_DIR / "feature_engineering_summary.json",
    )
    print("feature engineering complete", flush=True)


if __name__ == "__main__":
    main()
