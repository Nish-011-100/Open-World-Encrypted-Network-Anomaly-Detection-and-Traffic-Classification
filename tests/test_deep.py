import numpy as np
import pandas as pd
import torch

from driftmamba.evaluation.conformal import ConformalCalibrator, adaptive_alpha_trace
from driftmamba.evaluation.drift import drift_summary
from driftmamba.features import path_signature_features
from driftmamba.models.driftmamba import DriftMambaClassifier, prototype_loss
from driftmamba.models.inference import DeepInferenceBundle
from driftmamba.models.training import predict_deep, train_deep_model


def tensor_data(rows: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    sequence = rng.normal(size=(rows, 12, 3)).astype("float32")
    mask = np.ones((rows, 12), dtype=bool)
    mask[:, -2:] = False
    return {
        "sequence": sequence,
        "mask": mask,
        "aggregate": rng.normal(size=(rows, 7)).astype("float32"),
        "signatures": path_signature_features(sequence, mask),
        "labels": np.asarray(["video" if index % 2 else "cloud" for index in range(rows)]),
    }


def test_selective_model_forward_and_loss():
    model = DriftMambaClassifier(aggregate_dimension=7, number_classes=2,
                                 model_dimension=16, embedding_dimension=8, blocks=1)
    data = tensor_data(6)
    logits, embeddings, similarities = model(
        torch.from_numpy(data["sequence"]), torch.from_numpy(data["mask"]),
        torch.from_numpy(data["aggregate"]), torch.from_numpy(data["signatures"]),
    )
    assert logits.shape == (6, 2)
    assert embeddings.shape == (6, 8)
    assert torch.isfinite(prototype_loss(logits, similarities, torch.tensor([0, 1, 0, 1, 0, 1])))


def test_training_conformal_and_drift_pipeline():
    train, calibration = tensor_data(32), tensor_data(16, seed=7)
    result = train_deep_model(
        train, calibration, epochs=1, batch_size=8, model_dimension=16,
        embedding_dimension=8, blocks=1, seed=5,
    )
    logits, similarities, embeddings = predict_deep(result.model, calibration, batch_size=8)
    targets = result.label_encoder.transform(calibration["labels"])
    calibrator = ConformalCalibrator(alpha=0.1).fit(logits, similarities, targets)
    prediction_sets, p_values = calibrator.predict(logits, similarities)
    assert prediction_sets.shape == logits.shape
    assert np.all((p_values > 0) & (p_values <= 1))
    coverage = prediction_sets[np.arange(len(targets)), targets]
    assert len(adaptive_alpha_trace(coverage)) == len(coverage)
    drift = drift_summary(train["aggregate"], calibration["aggregate"],
                          predict_deep(result.model, train)[2], embeddings)
    assert drift["severity"] in {"low", "moderate", "high"}


def test_inference_bundle_returns_open_world_fields():
    train, calibration = tensor_data(24), tensor_data(12, seed=11)
    result = train_deep_model(
        train, calibration, epochs=1, batch_size=8, model_dimension=16,
        embedding_dimension=8, blocks=1, seed=9,
    )
    logits, similarities, _ = predict_deep(result.model, calibration)
    targets = result.label_encoder.transform(calibration["labels"])
    calibrator = ConformalCalibrator().fit(logits, similarities, targets)

    class FixedPreprocessor:
        def transform(self, frame):
            return calibration["aggregate"][: len(frame)]

    flows = pd.DataFrame({
        "packet_sizes": ["[100, 200, 150]"] * 3,
        "directions": ["[1, -1, 1]"] * 3,
        "inter_arrival_times": ["[0.0, 0.1, 0.2]"] * 3,
    })
    bundle = DeepInferenceBundle(
        result.model, calibrator, FixedPreprocessor(), result.label_encoder.classes_, 12
    )
    predictions = bundle.predict(flows)
    assert {"PredictedApplication", "KnownTrafficPValue", "PredictionSet", "Decision"}.issubset(
        predictions.columns
    )
    assert len(predictions) == 3
