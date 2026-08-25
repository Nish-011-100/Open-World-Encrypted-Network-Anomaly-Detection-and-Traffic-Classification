"""Strong classical baselines with calibration-only unknown rejection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, OneClassSVM


@dataclass
class BaselineBundle:
    name: str
    model: object
    label_encoder: LabelEncoder
    rejection_threshold: float
    anomaly_model: object | None = None
    anomaly_center: float = 0.0
    anomaly_scale: float = 1.0

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.model.predict_proba(features)
        indices = probabilities.argmax(axis=1)
        labels = self.label_encoder.inverse_transform(indices)
        confidence = probabilities.max(axis=1)
        if self.anomaly_model is not None:
            raw = self.anomaly_model.decision_function(features)
            z = np.clip((raw - self.anomaly_center) / self.anomaly_scale, -60.0, 60.0)
            knownness = 1.0 / (1.0 + np.exp(-z))
            confidence = np.sqrt(confidence * knownness)
        return labels, confidence


def build_model(name: str, seed: int = 42) -> object:
    if name in {"random_forest", "lof_random_forest", "ocsvm_random_forest"}:
        return RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            class_weight="balanced",
            random_state=seed,
        )
    if name == "rbf_svm":
        return CalibratedClassifierCV(
            SVC(C=4.0, kernel="rbf", gamma="scale", class_weight="balanced",
                cache_size=1024, random_state=seed),
            method="sigmoid", cv=3, ensemble=False,
        )
    if name == "isolation_extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=seed,
        )
    raise ValueError(f"Unknown baseline: {name}")


def train_baseline(name: str, train_features: np.ndarray, train_labels: np.ndarray,
                   calibration_features: np.ndarray, target_known_acceptance: float = 0.90,
                   seed: int = 42) -> BaselineBundle:
    if not 0 < target_known_acceptance < 1:
        raise ValueError("target_known_acceptance must be between zero and one")
    encoder = LabelEncoder().fit(train_labels.astype(str))
    encoded = encoder.transform(train_labels.astype(str))
    model = build_model(name, seed=seed)
    model.fit(train_features, encoded)
    anomaly_model = None
    anomaly_center, anomaly_scale = 0.0, 1.0
    if name == "isolation_extra_trees":
        anomaly_model = IsolationForest(
            n_estimators=300, max_samples="auto", contamination="auto",
            n_jobs=-1, random_state=seed,
        ).fit(train_features)
    elif name == "lof_random_forest":
        anomaly_model = LocalOutlierFactor(
            n_neighbors=50, novelty=True, contamination="auto", n_jobs=-1
        ).fit(train_features)
    elif name == "ocsvm_random_forest":
        anomaly_model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05, cache_size=1024).fit(
            train_features
        )
    if anomaly_model is not None:
        anomaly_scores = anomaly_model.decision_function(calibration_features)
        anomaly_center = float(np.median(anomaly_scores))
        anomaly_scale = float(max(np.median(np.abs(anomaly_scores - anomaly_center)) * 1.4826, 1e-6))
    provisional = BaselineBundle(
        name, model, encoder, 0.0, anomaly_model, anomaly_center, anomaly_scale
    )
    _, calibration_confidence = provisional.predict(calibration_features)
    threshold = float(np.quantile(calibration_confidence, 1.0 - target_known_acceptance))
    provisional.rejection_threshold = threshold
    return provisional
