import numpy as np
from sklearn.datasets import make_classification

from driftmamba.evaluation.metrics import evaluate_predictions
from driftmamba.models.baselines import train_baseline


def test_baselines_train_and_reject_low_confidence_samples():
    features, labels = make_classification(
        n_samples=180, n_features=12, n_informative=8, n_redundant=2,
        n_classes=3, class_sep=2.0, random_state=42,
    )
    text_labels = np.asarray([f"app_{label}" for label in labels])
    for name in ["random_forest", "hist_gradient_boosting"]:
        bundle = train_baseline(
            name, features[:100], text_labels[:100], features[100:140], seed=42
        )
        predictions, confidence = bundle.predict(features[140:])
        assert len(predictions) == 40
        assert np.all((confidence >= 0) & (confidence <= 1))
        assert 0 <= bundle.rejection_threshold <= 1


def test_open_set_metrics_have_expected_ranges():
    report = evaluate_predictions(
        known_true=np.array(["a", "a", "b", "b"]),
        known_pred=np.array(["a", "a", "b", "a"]),
        known_confidence=np.array([0.95, 0.9, 0.8, 0.6]),
        unknown_pred=np.array(["a", "b", "b"]),
        unknown_confidence=np.array([0.1, 0.2, 0.3]),
        rejection_threshold=0.5,
        class_names=["a", "b"],
    )
    assert report["open_set"]["unknown_auroc"] == 1.0
    assert report["open_set"]["unknown_recall"] == 1.0
    assert 0 <= report["closed_set"]["macro_f1"] <= 1
