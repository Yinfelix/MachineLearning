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
