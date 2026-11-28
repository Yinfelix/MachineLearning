import numpy as np
import pandas as pd

from src import config as C
from src import data


def test_saved_splits_are_disjoint_and_complete():
    splits = data.load_splits()
    tr, te, fold = splits["train_idx"], splits["test_idx"], splits["inner_fold_id"]
    assert len(tr) + len(te) == 235795
    assert len(np.intersect1d(tr, te)) == 0
    assert np.all(fold[tr] >= 0)
    assert np.all(fold[te] == -1)
    assert set(np.unique(fold[tr])) == set(range(C.N_INNER_FOLDS))


def test_groups_do_not_cross_train_test():
    df = pd.read_csv(C.DATA_PATH, usecols=["Domain"])
    splits = data.load_splits()
    tr_domains = set(df["Domain"].iloc[splits["train_idx"]])
    te_domains = set(df["Domain"].iloc[splits["test_idx"]])
    assert len(tr_domains & te_domains) == 0


def test_svc_subsample_is_stratified_subset_of_train():
    df = pd.read_csv(C.DATA_PATH, usecols=["label"])
    y = (df["label"] == 0).astype(int).to_numpy()
    splits = data.load_splits()
    sub = splits["svc_idx"]
    assert len(sub) == C.SVC_SUBSAMPLE
    assert set(sub).issubset(set(splits["train_idx"].tolist()))
    assert abs(y[sub].mean() - y[splits["train_idx"]].mean()) < 0.01


def test_inner_validation_domains_are_disjoint():
    domains = pd.read_csv(C.DATA_PATH, usecols=["Domain"])["Domain"].str.lower().str.rstrip(".")
    splits = data.load_splits()
    tr = splits["train_idx"]
    groups = pd.DataFrame({"domain": domains.iloc[tr], "fold": splits["inner_fold_id"][tr]})
    assert groups.groupby("domain")["fold"].nunique().max() == 1
