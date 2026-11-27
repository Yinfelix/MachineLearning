import numpy as np
import pandas as pd

from src import config as C
from src.preprocess import FrequencyEncoder, make_pipeline
from sklearn.linear_model import LogisticRegression


def test_frequency_encoder_unseen_maps_to_zero():
    enc = FrequencyEncoder().fit(pd.DataFrame({"TLD": ["a", "a", "b"]}))
    out = enc.transform(pd.DataFrame({"TLD": ["a", "b", "c"]})).ravel()
    assert out.tolist() == [2.0, 1.0, 0.0]


def _tiny_frame(scheme):
    numeric, categorical = C.scheme_columns(scheme)
    rng = np.random.RandomState(0)
    data = {c: rng.rand(40) for c in numeric}
    for c in categorical:
        data[c] = rng.choice(["com", "org", "net"], size=40)
    return pd.DataFrame(data)


def test_pipeline_output_width_matches_scheme():
    for scheme in C.SCHEMES:
        numeric, categorical = C.scheme_columns(scheme)
        pipe = make_pipeline(LogisticRegression(solver="liblinear"), scheme, scale=True)
        X = _tiny_frame(scheme)
        y = np.array([0, 1] * 20)
        pipe.fit(X, y)
        assert pipe.named_steps["pre"].transform(X).shape[1] == len(numeric) + len(categorical)


def test_robust_scaler_option_runs():
    pipe = make_pipeline(LogisticRegression(solver="liblinear"), "B", scale=True, scaler="robust")
    X = _tiny_frame("B")
    y = np.array([0, 1] * 20)
    pipe.fit(X, y)
    assert hasattr(pipe.named_steps["pre"].named_transformers_["num"].named_steps["scaler"], "center_")
