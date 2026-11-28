import numpy as np

from src.evaluate import metrics_dict


def test_confusion_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 0])
    m = metrics_dict(y, pred, np.array([0.1, 0.9, 0.8, 0.2]))
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5
    assert m["fpr"] == 0.5
    assert m["specificity"] == 0.5
    assert m["roc_auc"] == 0.5


def test_perfect_separation_auc_is_one():
    y = np.array([0, 0, 1, 1])
    m = metrics_dict(y, y.copy(), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1.0
    assert m["f1"] == 1.0
    assert m["fpr"] == 0.0


def test_test_scores_and_labels_come_from_one_fit():
    from sklearn.base import BaseEstimator
    from src.evaluate import evaluate_on_test

    class CountingClassifier(BaseEstimator):
        fits = 0

        def fit(self, X, y):
            type(self).fits += 1
            return self

        def predict(self, X):
            return X[:, 0]

        def predict_proba(self, X):
            return np.column_stack((1 - X[:, 0], X[:, 0]))

    X = np.array([[0], [1]])
    y = X[:, 0]
    metrics, scores, pred = evaluate_on_test(CountingClassifier(), X, y, X, y, return_predictions=True)
    assert CountingClassifier.fits == 1
    assert np.array_equal(scores, pred)
    assert metrics["f1"] == 1


def test_unavailable_auc_is_json_null():
    import json

    metrics = metrics_dict([0, 1], [0, 1])
    assert json.loads(json.dumps(metrics, allow_nan=False))["roc_auc"] is None
