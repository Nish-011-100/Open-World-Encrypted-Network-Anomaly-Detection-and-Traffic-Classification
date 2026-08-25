"""Closed-set, open-set, and calibration metrics."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                               confidence: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in pairwise(edges):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            accuracy = np.mean(y_true[selected] == y_pred[selected])
            result += selected.mean() * abs(accuracy - confidence[selected].mean())
    return float(result)


def false_positive_rate_at_recall(y_unknown: np.ndarray, unknown_score: np.ndarray,
                                  recall: float = 0.95) -> float:
    positive_scores = unknown_score[y_unknown]
    negative_scores = unknown_score[~y_unknown]
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return float("nan")
    threshold = np.quantile(positive_scores, 1.0 - recall)
    return float(np.mean(negative_scores >= threshold))


def evaluate_predictions(
    known_true: np.ndarray,
    known_pred: np.ndarray,
    known_confidence: np.ndarray,
    unknown_pred: np.ndarray,
    unknown_confidence: np.ndarray,
    rejection_threshold: float,
    class_names: list[str],
) -> dict[str, object]:
    known_rejected = known_confidence < rejection_threshold
    unknown_rejected = unknown_confidence < rejection_threshold
    open_true = np.concatenate([known_true.astype(str), np.full(len(unknown_confidence), "UNKNOWN")])
    open_pred = np.concatenate([
        np.where(known_rejected, "UNKNOWN", known_pred.astype(str)),
        np.where(unknown_rejected, "UNKNOWN", unknown_pred.astype(str)),
    ])
    y_unknown = np.concatenate([
        np.zeros(len(known_confidence), dtype=bool),
        np.ones(len(unknown_confidence), dtype=bool),
    ])
    unknown_score = 1.0 - np.concatenate([known_confidence, unknown_confidence])
    return {
        "closed_set": {
            "accuracy": float(accuracy_score(known_true, known_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(known_true, known_pred)),
            "macro_f1": float(f1_score(known_true, known_pred, average="macro")),
            "weighted_f1": float(f1_score(known_true, known_pred, average="weighted")),
            "expected_calibration_error": expected_calibration_error(
                known_true, known_pred, known_confidence
            ),
            "confusion_matrix": confusion_matrix(
                known_true, known_pred, labels=class_names
            ).tolist(),
            "class_names": class_names,
        },
        "open_set": {
            "unknown_auroc": float(roc_auc_score(y_unknown, unknown_score)),
            "unknown_average_precision": float(average_precision_score(y_unknown, unknown_score)),
            "fpr_at_95_unknown_recall": false_positive_rate_at_recall(y_unknown, unknown_score),
            "known_acceptance_rate": float((~known_rejected).mean()),
            "unknown_recall": float(unknown_rejected.mean()),
            "open_macro_f1": float(f1_score(open_true, open_pred, average="macro")),
            "rejection_threshold": float(rejection_threshold),
        },
    }
