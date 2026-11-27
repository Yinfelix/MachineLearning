"""Model zoo, search grids, and pipeline builders."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .preprocess import make_pipeline


def _logreg() -> LogisticRegression:
    return LogisticRegression(solver="liblinear", max_iter=2000, random_state=C.RANDOM_STATE)


def _nb() -> GaussianNB:
    return GaussianNB()


def _tree() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=C.RANDOM_STATE)


def _forest(n_estimators: int = 300, n_jobs: int = -1) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=C.RANDOM_STATE)


def _svc() -> SVC:
    return SVC(kernel="rbf", cache_size=1000, random_state=C.RANDOM_STATE)


def _linear_svc() -> LinearSVC:
    return LinearSVC(max_iter=5000, random_state=C.RANDOM_STATE)




MODEL_SPECS = {
    "Logistic Regression": {
        "factory": _logreg,
        # L2 only in the main grid: liblinear+L1 on 150k x 45 can take minutes
        # per fold.  L1 is studied separately as a regularisation path in
        # scripts/06_feature_engineering.py.
        "grid": {"model__C": [0.01, 0.1, 1, 10, 100]},
        "scale": True,
        "subsample": None,
    },
    "Gaussian NB": {
        "factory": _nb,
        "grid": {"model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
        "scale": False,
        "subsample": None,
    },
    "Decision Tree": {
        "factory": _tree,
        "grid": {
            "model__max_depth": [3, 5, 8, 12, None],
            "model__min_samples_leaf": [1, 5, 20, 50],
        },
        "scale": False,
        "subsample": None,
    },
    "Random Forest": {
        "factory": _forest,
        "grid": {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [None, 20],
            "model__max_features": ["sqrt", 0.3, 0.5],
            "model__min_samples_leaf": [1, 5],
        },
        "scale": False,
        "subsample": None,
    },
    "SVM (RBF)": {
        "factory": _svc,
        "grid": {"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01, 0.1]},
        "scale": True,
        "subsample": "svc",
    },
    "Linear SVM": {
        "factory": _linear_svc,
        "grid": {"model__C": [0.01, 0.1, 1, 10]},
        "scale": True,
        "subsample": None,
    },
}

# The five models compared in the main experiment; Linear SVM is a full-data
# control for the RBF SVM and appears in the SVM discussion only.
MAIN_MODELS = C.MODEL_ORDER


def build_pipeline(name: str, scheme: str):
    spec = MODEL_SPECS[name]
    return make_pipeline(spec["factory"](), scheme, scale=spec["scale"])


def build_param_grid(name: str) -> dict:
    return MODEL_SPECS[name]["grid"]


def default_forest() -> RandomForestClassifier:
    """Untuned reference configuration used in the optimisation section."""
    return RandomForestClassifier(n_estimators=100, random_state=C.RANDOM_STATE, n_jobs=-1)


def default_tree() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=C.RANDOM_STATE)
