"""Paired diagnostic controls evaluated exclusively inside the outer training set."""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, data, evaluate, models
from src.preprocess import FrequencyEncoder, make_pipeline, make_preprocessor


def main():
    X, y, _ = data.get_xy(data.load_raw())
    splits = data.load_splits()
    train, folds = splits["train_idx"], splits["inner_fold_id"]
    fit = train[folds[train] != 1]
    val = train[folds[train] == 1]
    tuning = json.loads((C.METRIC_DIR / "tuning_summary.json").read_text(encoding="utf8"))
    rows = []

    def record(name, condition, scheme, idx, validation, estimator):
        score = evaluate.score_of(estimator, X.iloc[validation])
        metrics = evaluate.metrics_dict(y.iloc[validation], estimator.predict(X.iloc[validation]), score)
        if hasattr(estimator, "predict_proba"):
            metrics["brier"] = brier_score_loss(y.iloc[validation], score)
        rows.append(dict(model=name, condition=condition, scheme=scheme, validation_fold=1,
                         n_train=len(idx), n_val=len(validation), **metrics))

    # Hyperparameters fixed across the single-column ablation.
    for drop in [False, True]:
        pipe = models.build_pipeline("Logistic Regression", "A")
        pipe.set_params(**{"model__" + k: v for k, v in tuning["Logistic Regression"]["best_params"].items()})
        if drop:
            numeric, categorical = C.scheme_columns("A")
            numeric.remove("URLSimilarityIndex")
            pipe.set_params(pre=make_preprocessor(numeric, categorical, scale=True))
        pipe.fit(X.iloc[fit], y.iloc[fit])
        record("Logistic Regression", "without similarity" if drop else "all", "A", fit, val, pipe)

    # Compare kernels using precisely the same grouped training AND validation rows.
    sub = splits["svc_idx"]
    for name in ["SVM (RBF)", "Linear SVM"]:
        pipe = models.build_pipeline(name, "V")
        pipe.set_params(**{"model__" + k: v for k, v in tuning[name]["best_params"].items()})
        matched_fit = sub[folds[sub] != 1]
        pipe.fit(X.iloc[matched_fit], y.iloc[matched_fit])
        record(name, "matched 20k subset", "V", matched_fit, sub[folds[sub] == 1], pipe)

    # Select smoothing separately for each representation inside the fit pool.
    for condition in ["counts", "relative frequency", "standardized"]:
        pipe = make_pipeline(models.MODEL_SPECS["Gaussian NB"]["factory"](), "V",
                             scale=condition == "standardized")
        if condition == "relative frequency":
            pipe.named_steps["pre"].transformers[1][1].set_params(freq=FrequencyEncoder(normalize=True))
        search = GridSearchCV(pipe, models.build_param_grid("Gaussian NB"),
                              cv=data.inner_cv_splits(fit, folds), scoring="f1", error_score="raise")
        search.fit(X.iloc[fit], y.iloc[fit])
        record("Gaussian NB", condition, "V", fit, val, search)
        rows[-1]["var_smoothing"] = search.best_params_["model__var_smoothing"]

    # Repeat the common-to-rare comparison with TLD omitted, holding rows fixed.
    common = X.iloc[fit]["TLD"].value_counts().head(20).index
    common_fit = fit[X.iloc[fit]["TLD"].isin(common)]
    is_common = X.iloc[val]["TLD"].isin(common)
    for name in ["Logistic Regression", "Random Forest"]:
        pipe = models.build_pipeline(name, "V")
        numeric, _ = C.scheme_columns("V")
        pipe.set_params(pre=make_preprocessor(numeric, [], scale=models.MODEL_SPECS[name]["scale"]))
        pipe.set_params(**{"model__" + k: v for k, v in tuning[name]["best_params"].items()})
        pipe.fit(X.iloc[common_fit], y.iloc[common_fit])
        for group, subset in [("common", val[is_common]), ("rare", val[~is_common])]:
            record(name, "without TLD: " + group, "V", common_fit, subset, pipe)
    pd.DataFrame(rows).to_csv(C.METRIC_DIR / "training_controls.csv", index=False, float_format="%.12g")


if __name__ == "__main__":
    main()
