"""Train DriftMamba, calibrate open-world decisions, and audit temporal drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

from driftmamba.evaluation.conformal import ConformalCalibrator, adaptive_alpha_trace
from driftmamba.evaluation.drift import chronological_drift_report, drift_summary
from driftmamba.evaluation.metrics import evaluate_predictions
from driftmamba.models.training import load_tensors, predict_deep, train_deep_model


def prefix_copy(data: dict[str, np.ndarray], packets: int) -> dict[str, np.ndarray]:
    copied = {name: value.copy() for name, value in data.items()}
    copied["sequence"][:, packets:] = 0
    copied["mask"][:, packets:] = False
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--models-directory", type=Path, default=Path("models/deep"))
    parser.add_argument("--reports-directory", type=Path, default=Path("reports/deep"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument(
        "--encoder", choices=["driftmamba", "transformer", "hyena", "xlstm"],
        default="driftmamba"
    )
    args = parser.parse_args()
    args.models_directory.mkdir(parents=True, exist_ok=True)
    args.reports_directory.mkdir(parents=True, exist_ok=True)
    train = load_tensors(args.data_directory / "train.npz")
    calibration = load_tensors(args.data_directory / "calibration.npz")
    known = load_tensors(args.data_directory / "test_known.npz")
    unknown = load_tensors(args.data_directory / "test_unknown.npz")
    result = train_deep_model(
        train, calibration, epochs=args.epochs, batch_size=args.batch_size,
        balanced_sampling=args.balanced_sampling, encoder_type=args.encoder, seed=args.seed,
    )
    calibration_logits, calibration_similarities, _ = predict_deep(
        result.model, calibration
    )
    calibration_targets = result.label_encoder.transform(calibration["labels"].astype(str))
    calibrator = ConformalCalibrator(alpha=args.alpha).fit(
        calibration_logits, calibration_similarities, calibration_targets
    )
    known_logits, known_similarities, known_embeddings = predict_deep(result.model, known)
    unknown_logits, unknown_similarities, _ = predict_deep(result.model, unknown)
    known_sets, known_p_values = calibrator.predict(known_logits, known_similarities)
    unknown_sets, unknown_p_values = calibrator.predict(unknown_logits, unknown_similarities)
    known_pred = result.label_encoder.inverse_transform(known_logits.argmax(axis=1))
    unknown_pred = result.label_encoder.inverse_transform(unknown_logits.argmax(axis=1))
    report = evaluate_predictions(
        known["labels"].astype(str), known_pred, known_p_values,
        unknown_pred, unknown_p_values, args.alpha, result.label_encoder.classes_.tolist(),
    )
    known_targets = result.label_encoder.transform(known["labels"].astype(str))
    coverage = known_sets[np.arange(len(known_sets)), known_targets]
    report["conformal"] = {
        "target_coverage": 1.0 - args.alpha,
        "empirical_known_coverage": float(coverage.mean()),
        "mean_prediction_set_size": float(known_sets.sum(axis=1).mean()),
        "empty_unknown_set_rate": float((unknown_sets.sum(axis=1) == 0).mean()),
        "final_adaptive_alpha": float(adaptive_alpha_trace(coverage, args.alpha)[-1]),
    }
    train_sample = {name: value[: min(len(value), 20_000)] for name, value in train.items()}
    _, _, train_embeddings = predict_deep(result.model, train_sample)
    report["drift"] = drift_summary(
        train_sample["aggregate"], known["aggregate"], train_embeddings, known_embeddings
    )
    known_times = pd.read_csv(
        args.data_directory / "test_known.csv", usecols=["start_time"]
    )["start_time"].to_numpy()
    report["drift"]["chronological_windows"] = chronological_drift_report(
        train_sample["aggregate"], known["aggregate"], train_embeddings,
        known_embeddings, known_times,
    )
    early_results = {}
    combined_unknown = np.ones(len(unknown["labels"]), dtype=int)
    for packets in [8, 16, 32, 64]:
        if packets > known["sequence"].shape[1]:
            continue
        prefix_known = prefix_copy(known, packets)
        prefix_unknown = prefix_copy(unknown, packets)
        prefix_known_logits, prefix_known_similarity, _ = predict_deep(result.model, prefix_known)
        prefix_unknown_logits, prefix_unknown_similarity, _ = predict_deep(result.model, prefix_unknown)
        _, prefix_known_p = calibrator.predict(prefix_known_logits, prefix_known_similarity)
        _, prefix_unknown_p = calibrator.predict(prefix_unknown_logits, prefix_unknown_similarity)
        prefix_prediction = result.label_encoder.inverse_transform(prefix_known_logits.argmax(axis=1))
        y_unknown = np.concatenate([np.zeros(len(prefix_known_p)), combined_unknown])
        unknown_score = 1.0 - np.concatenate([prefix_known_p, prefix_unknown_p])
        early_results[str(packets)] = {
            "known_macro_f1": float(f1_score(known["labels"], prefix_prediction, average="macro")),
            "unknown_auroc": float(roc_auc_score(y_unknown, unknown_score)),
        }
    report["early_classification"] = early_results
    report["training_history"] = result.history
    report["encoder_type"] = args.encoder
    artifact_name = {
        "driftmamba": "driftmamba_prototype.pt",
        "transformer": "transformer_prototype.pt",
        "hyena": "hyena_prototype.pt",
        "xlstm": "xlstm_prototype.pt",
    }[args.encoder]
    torch.save({
        "state_dict": result.model.state_dict(), "configuration": result.configuration,
        "classes": result.label_encoder.classes_.tolist(), "encoder_type": args.encoder,
    }, args.models_directory / artifact_name)
    joblib.dump(calibrator, args.models_directory / "conformal_calibrator.joblib")
    (args.reports_directory / "deep_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    predictions = pd.DataFrame({
        "true_label": np.concatenate([known["labels"], unknown["labels"]]),
        "predicted_label": np.concatenate([known_pred, unknown_pred]),
        "known_p_value": np.concatenate([known_p_values, unknown_p_values]),
        "prediction_set_size": np.concatenate([known_sets.sum(1), unknown_sets.sum(1)]),
        "is_unknown_test": np.concatenate([
            np.zeros(len(known_pred), dtype=bool), np.ones(len(unknown_pred), dtype=bool)
        ]),
    })
    predictions["decision"] = np.where(
        predictions["known_p_value"] < args.alpha, "UNKNOWN", predictions["predicted_label"]
    )
    predictions.to_csv(args.reports_directory / "deep_predictions.csv", index=False)
    print(json.dumps({
        "macro_f1": report["closed_set"]["macro_f1"],
        "unknown_auroc": report["open_set"]["unknown_auroc"],
        "conformal_coverage": report["conformal"]["empirical_known_coverage"],
        "drift_severity": report["drift"]["severity"],
    }, indent=2))


if __name__ == "__main__":
    main()
