"""Chronological feature and representation drift diagnostics."""

from __future__ import annotations

import numpy as np


def population_stability_index(reference: np.ndarray, current: np.ndarray,
                               bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts = np.histogram(reference, bins=edges)[0] / len(reference)
    current_counts = np.histogram(current, bins=edges)[0] / len(current)
    reference_counts = np.clip(reference_counts, 1e-6, None)
    current_counts = np.clip(current_counts, 1e-6, None)
    return float(np.sum((current_counts - reference_counts) * np.log(current_counts / reference_counts)))


def drift_summary(reference_features: np.ndarray, current_features: np.ndarray,
                  reference_embeddings: np.ndarray, current_embeddings: np.ndarray) -> dict[str, object]:
    feature_psi = [
        population_stability_index(reference_features[:, column], current_features[:, column])
        for column in range(reference_features.shape[1])
    ]
    centroid_shift = float(np.linalg.norm(
        reference_embeddings.mean(axis=0) - current_embeddings.mean(axis=0)
    ))
    return {
        "mean_feature_psi": float(np.mean(feature_psi)),
        "maximum_feature_psi": float(np.max(feature_psi)),
        "features_above_0_25": int(np.sum(np.asarray(feature_psi) > 0.25)),
        "embedding_centroid_shift": centroid_shift,
        "severity": "high" if max(feature_psi, default=0) > 0.25 else (
            "moderate" if max(feature_psi, default=0) > 0.10 else "low"
        ),
    }


def chronological_drift_report(reference_features: np.ndarray, current_features: np.ndarray,
                               reference_embeddings: np.ndarray, current_embeddings: np.ndarray,
                               timestamps: np.ndarray, windows: int = 5) -> list[dict[str, object]]:
    order = np.argsort(timestamps)
    groups = np.array_split(order, min(windows, len(order)))
    report = []
    for index, group in enumerate(groups, start=1):
        if len(group) == 0:
            continue
        summary = drift_summary(
            reference_features, current_features[group],
            reference_embeddings, current_embeddings[group],
        )
        summary.update({
            "window": index, "rows": len(group),
            "start_time": float(timestamps[group].min()),
            "end_time": float(timestamps[group].max()),
        })
        report.append(summary)
    return report
