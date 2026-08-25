"""Reusable loading and inference for trained DriftMamba artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from driftmamba.evaluation.conformal import ConformalCalibrator
from driftmamba.features import AggregatePreprocessor, path_signature_features, sequence_tensor
from driftmamba.models.driftmamba import (
    DriftMambaClassifier,
    FlowTransformerClassifier,
    HyenaFlowClassifier,
    XLSTMFlowClassifier,
)
from driftmamba.models.training import predict_deep


@dataclass
class DeepInferenceBundle:
    model: DriftMambaClassifier
    calibrator: ConformalCalibrator
    preprocessor: AggregatePreprocessor
    classes: np.ndarray
    maximum_packets: int = 64

    def predict(self, flows: pd.DataFrame, alpha: float | None = None) -> pd.DataFrame:
        sequence, mask = sequence_tensor(flows, maximum_packets=self.maximum_packets)
        data = {
            "sequence": sequence,
            "mask": mask,
            "aggregate": self.preprocessor.transform(flows),
            "signatures": path_signature_features(sequence, mask),
            "labels": np.full(len(flows), "UNLABELLED"),
        }
        logits, similarities, embeddings = predict_deep(self.model, data)
        prediction_sets, known_p_values = self.calibrator.predict(logits, similarities)
        predicted_indices = logits.argmax(axis=1)
        predicted_labels = self.classes[predicted_indices]
        threshold = self.calibrator.alpha if alpha is None else alpha
        if not 0 < threshold < 1:
            raise ValueError("alpha must be between zero and one")
        output = flows.copy()
        output["PredictedApplication"] = predicted_labels
        output["KnownTrafficPValue"] = known_p_values
        output["NearestPrototypeSimilarity"] = similarities.max(axis=1)
        output["PredictionSet"] = [
            "|".join(self.classes[selected].tolist()) if selected.any() else "EMPTY"
            for selected in prediction_sets
        ]
        output["PredictionSetSize"] = prediction_sets.sum(axis=1)
        output["Decision"] = np.where(known_p_values < threshold, "UNKNOWN", predicted_labels)
        output["EmbeddingNorm"] = np.linalg.norm(embeddings, axis=1)
        return output


def load_deep_bundle(models_directory: Path, preprocessor_path: Path,
                     maximum_packets: int = 64) -> DeepInferenceBundle:
    drift_path = models_directory / "driftmamba_prototype.pt"
    candidates = [
        drift_path, models_directory / "transformer_prototype.pt",
        models_directory / "hyena_prototype.pt",
        models_directory / "xlstm_prototype.pt",
    ]
    checkpoint_path = next(path for path in candidates if path.exists())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_class = {
        "driftmamba": DriftMambaClassifier,
        "transformer": FlowTransformerClassifier,
        "hyena": HyenaFlowClassifier,
        "xlstm": XLSTMFlowClassifier,
    }.get(checkpoint.get("encoder_type", "driftmamba"), DriftMambaClassifier)
    model = model_class(**checkpoint["configuration"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    calibrator = joblib.load(models_directory / "conformal_calibrator.joblib")
    preprocessor = joblib.load(preprocessor_path)
    return DeepInferenceBundle(
        model=model,
        calibrator=calibrator,
        preprocessor=preprocessor,
        classes=np.asarray(checkpoint["classes"]),
        maximum_packets=maximum_packets,
    )
