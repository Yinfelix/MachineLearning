"""Load every artefact produced by the experiment pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "outputs" / "metrics"
FIG = ROOT / "outputs" / "figures"


def _json(name):
    p = MET / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _csv(name):
    p = MET / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_all() -> dict:
    R = {
        "audit": _json("data_audit.json"),
        "uni": _csv("univariate_auc.csv"),
        "corr_summary": _json("correlation_summary.json"),
        "high_corr": _csv("high_correlation_pairs.csv"),
        "tuning": _json("tuning_summary.json"),
        "res": _csv("results_all.csv"),
        "res_main": _csv("results_main.csv"),
        "res_supp": _csv("results_supplementary.csv"),
        "cv_folds": _csv("cv_folds_all.csv"),
        "cms": _json("confusion_matrices.json"),
        "shortcut": _json("shortcut_rule.json"),
        "rfopt": _csv("optimization_rf.csv"),
        "rf_b": _json("rf_tuning_B.json"),
        "dt_depth": _csv("dt_depth_curve.csv"),
        "dt_ccp": _csv("dt_ccp_curve.csv"),
        "threshold": _json("threshold_analysis.json"),
        "threshold_curve": _csv("threshold_curve.csv"),
        "posthoc": _json("posthoc_analysis.json"),
        "lc": _csv("robustness_learning_curve.csv"),
        "label_noise": _csv("robustness_label_noise.csv"),
        "feat_noise": _csv("robustness_feature_noise.csv"),
        "outliers": _csv("robustness_outliers.csv"),
        "tld": _csv("robustness_tld_transfer.csv"),
        "overfit": _csv("overfitting.csv"),
        "kbest": _csv("feature_selection_k.csv"),
        "pca": _csv("pca_results.csv"),
        "poly": _csv("polynomial_results.csv"),
        "perm": _csv("permutation_importance.csv"),
        "tree_imp": _csv("rf_impurity_importance.csv"),
        "l1": _csv("l1_path.csv"),
        "mi": _csv("mutual_info.csv"),
        "feature_summary": _json("feature_engineering_summary.json"),
        "test_meta": _json("test_meta.json"),
        "key": _json("report_key_numbers.json"),
    }
    if R["res"].empty:
        return R
    R["res"]["model"] = pd.Categorical(R["res"]["model"], categories=[
        "Logistic Regression", "Gaussian NB", "Decision Tree", "Random Forest", "SVM (RBF)", "Linear SVM"
    ], ordered=True)
    R["res_main"] = R["res"][R["res"]["scheme"].isin(["A", "B", "C", "D"])].copy()
    R["scores"] = np.load(MET / "test_scores.npz") if (MET / "test_scores.npz").exists() else {}
    return R
