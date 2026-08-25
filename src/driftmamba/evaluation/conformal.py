"""Split-conformal prediction sets and known/unknown p-values."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


class ConformalCalibrator:
    def __init__(self, alpha: float = 0.10):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between zero and one")
        self.alpha = alpha
        self.class_quantile: float | None = None
        self.known_nonconformity: np.ndarray | None = None

    def fit(self, logits: np.ndarray, similarities: np.ndarray,
            true_indices: np.ndarray) -> ConformalCalibrator:
        probabilities = softmax(logits)
        class_scores = 1.0 - probabilities[np.arange(len(probabilities)), true_indices]
        quantile_level = min(np.ceil((len(class_scores) + 1) * (1 - self.alpha)) / len(class_scores), 1)
        self.class_quantile = float(np.quantile(class_scores, quantile_level, method="higher"))
        self.known_nonconformity = 1.0 - similarities.max(axis=1)
        return self

    def predict(self, logits: np.ndarray, similarities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.class_quantile is None or self.known_nonconformity is None:
            raise RuntimeError("Conformal calibrator has not been fitted")
        probabilities = softmax(logits)
        prediction_sets = (1.0 - probabilities) <= self.class_quantile
        candidate_scores = 1.0 - similarities.max(axis=1)
        p_values = np.asarray([
            (1 + np.sum(self.known_nonconformity >= score)) / (len(self.known_nonconformity) + 1)
            for score in candidate_scores
        ])
        return prediction_sets, p_values


def adaptive_alpha_trace(covered: np.ndarray, target_alpha: float = 0.10,
                         learning_rate: float = 0.02) -> np.ndarray:
    """Audit how Adaptive Conformal Inference would update alpha with delayed labels."""
    alpha = target_alpha
    trace = []
    for is_covered in covered:
        error = float(not is_covered)
        alpha = float(np.clip(alpha + learning_rate * (target_alpha - error), 0.001, 0.999))
        trace.append(alpha)
    return np.asarray(trace)
