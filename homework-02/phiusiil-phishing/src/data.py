"""Data loading, target definition, grouped splits, and scheme views."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from . import config as C


def load_raw() -> pd.DataFrame:
    if not C.DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {C.DATA_PATH}. "
            "Extract archive/data/phiusiil+phishing+url+dataset.zip into data/raw/."
        )
    df = pd.read_csv(C.DATA_PATH)
    return df


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return raw feature frame, binary target (1 = phishing), and domain groups."""
    y = (df["label"] == 0).astype(int)
    y.name = "is_phishing"
    X = df[C.all_feature_columns()].copy()
    groups = df["Domain"].astype(str)
    return X, y, groups


def _split_paths() -> dict[str, Path]:
    return {
        "train": C.SPLITS_DIR / "train_idx.npy",
        "test": C.SPLITS_DIR / "test_idx.npy",
        "fold": C.SPLITS_DIR / "inner_fold_id.npy",
        "svc": C.SPLITS_DIR / "svc_subsample_idx.npy",
        "meta": C.SPLITS_DIR / "split_meta.json",
    }


def _describe_split(y: pd.Series, idx: np.ndarray) -> dict:
    yy = y.iloc[idx]
    return {
        "n": int(len(idx)),
        "phishing": int(yy.sum()),
        "legitimate": int((1 - yy).sum()),
        "phishing_rate": float(yy.mean()),
    }


def ensure_splits(df: pd.DataFrame, force: bool = False) -> dict:
    """Create (or load) one grouped, stratified outer test split plus inner folds.

    Outer: StratifiedGroupKFold(5), fold 0 held out as the independent test set
    (20%), the other four folds form the training set.  Groups (Domains) never
    cross the boundary.  Inner: StratifiedGroupKFold(5) on the training set.
    """
    paths = _split_paths()
    if all(paths[k].exists() for k in ("train", "test", "fold", "svc")) and not force:
        return load_splits()

    X, y, groups = get_xy(df)
    n = len(df)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    folds = list(outer.split(X, y, groups=groups))

    # Choose the fold whose test size is closest to 20% as the independent test.
    test_fold = int(np.argmin([abs(len(te) / n - C.TEST_FRACTION) for _, te in folds]))
    train_idx = np.sort(folds[test_fold][0])
    test_idx = np.sort(folds[test_fold][1])

    inner_fold_id = np.full(n, -1, dtype=np.int8)
    inner = StratifiedGroupKFold(n_splits=C.N_INNER_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    for k, (_, val_idx) in enumerate(inner.split(X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx])):
        inner_fold_id[train_idx[val_idx]] = k

    # Fixed stratified 20k subsample of the training set for the RBF SVM.
    svc_idx, _ = train_test_split(
        train_idx,
        train_size=min(C.SVC_SUBSAMPLE, len(train_idx) // 2),
        stratify=y.iloc[train_idx],
        random_state=C.RANDOM_STATE,
    )
    svc_idx = np.sort(svc_idx)

    np.save(paths["train"], train_idx)
    np.save(paths["test"], test_idx)
    np.save(paths["fold"], inner_fold_id)
    np.save(paths["svc"], svc_idx)

    meta = {
        "random_state": C.RANDOM_STATE,
        "outer_test_fold": test_fold,
        "train": _describe_split(y, train_idx),
        "test": _describe_split(y, test_idx),
        "groups_train": int(groups.iloc[train_idx].nunique()),
        "groups_test": int(groups.iloc[test_idx].nunique()),
        "group_overlap": int(len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))),
        "inner_folds": {k: _describe_split(y, train_idx[inner_fold_id[train_idx] == k]) for k in range(C.N_INNER_FOLDS)},
        "svc_subsample": _describe_split(y, svc_idx),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return load_splits()


def load_splits() -> dict:
    p = _split_paths()
    meta = json.loads(p["meta"].read_text(encoding="utf-8")) if p["meta"].exists() else {}
    return {
        "train_idx": np.load(p["train"]),
        "test_idx": np.load(p["test"]),
        "inner_fold_id": np.load(p["fold"]),
        "svc_idx": np.load(p["svc"]),
        "meta": meta,
    }


def inner_cv_splits(train_idx: np.ndarray, fold_id: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_positions, val_positions) relative to train_idx."""
    out = []
    ids = fold_id[train_idx]
    for k in sorted(set(ids.tolist())):
        val = np.flatnonzero(ids == k)
        tr = np.flatnonzero(ids != k)
        out.append((tr, val))
    return out


def scheme_frame(X: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Select the original-scale inputs used by the fitted pipelines."""
    numeric, categorical = C.scheme_columns(scheme)
    return X[numeric + categorical].copy()
