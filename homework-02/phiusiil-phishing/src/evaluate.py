"""Unified metrics, cross-validation, and result persistence."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config as C

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity", "fpr", "balanced_accuracy"]


def score_of(model, X) -> np.ndarray:
    """Continuous score for ROC/PR: probability when available, else margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel()
    return np.asarray(model.predict(X), dtype=float)


def metrics_dict(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_score is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def cross_validate_pipeline(pipeline, X, y, splits) -> pd.DataFrame:
    rows = []
    for k, (tr, va) in enumerate(splits):
        est = clone(pipeline)
        t0 = time.perf_counter()
        est.fit(X.iloc[tr], y.iloc[tr])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = est.predict(X.iloc[va])
        score = score_of(est, X.iloc[va])
        pred_s = time.perf_counter() - t0
        row = metrics_dict(y.iloc[va], pred, score)
        row.update({"fold": k, "fit_seconds": fit_s, "predict_seconds": pred_s})
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_on_test(pipeline, X_train, y_train, X_test, y_test, *, return_predictions=False):
    est = clone(pipeline)
    t0 = time.perf_counter()
    est.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = est.predict(X_test)
    score = score_of(est, X_test)
    pred_s = time.perf_counter() - t0
    out = metrics_dict(y_test, pred, score)
    out.update({"fit_seconds": fit_s, "predict_seconds": pred_s})
    if return_predictions:
        return out, score, pred
    return out, score


def aggregate_cv(cv_df: pd.DataFrame) -> dict:
    out = {}
    for k in METRIC_KEYS + ["fit_seconds"]:
        if k in cv_df:
            out[f"{k}_mean"] = float(cv_df[k].mean())
            out[f"{k}_std"] = float(cv_df[k].std(ddof=0))
    return out


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
