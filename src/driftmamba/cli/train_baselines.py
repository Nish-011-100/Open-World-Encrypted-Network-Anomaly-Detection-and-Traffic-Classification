"""Train and evaluate classical closed/open-world traffic baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from driftmamba.evaluation.metrics import evaluate_predictions
from driftmamba.models.baselines import train_baseline


def load_partition(directory: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(directory / f"{name}.npz", allow_pickle=False)
    return data["aggregate"], data["labels"].astype(str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("reports/baselines"))
    parser.add_argument("--models-directory", type=Path, default=Path("models/baselines"))
    parser.add_argument("--target-known-acceptance", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.models_directory.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_partition(args.data_directory, "train")
    x_calibration, _ = load_partition(args.data_directory, "calibration")
    x_known, y_known = load_partition(args.data_directory, "test_known")
    x_unknown, y_unknown = load_partition(args.data_directory, "test_unknown")
    comparison: dict[str, object] = {}
    for name in [
        "random_forest", "hist_gradient_boosting", "lof_random_forest",
        "ocsvm_random_forest", "rbf_svm",
    ]:
        bundle = train_baseline(
            name, x_train, y_train, x_calibration,
            target_known_acceptance=args.target_known_acceptance, seed=args.seed,
        )
        known_pred, known_confidence = bundle.predict(x_known)
        unknown_pred, unknown_confidence = bundle.predict(x_unknown)
        report = evaluate_predictions(
            y_known, known_pred, known_confidence, unknown_pred, unknown_confidence,
            bundle.rejection_threshold, bundle.label_encoder.classes_.tolist(),
        )
        report["model"] = name
        report["training_rows"] = len(y_train)
        report["calibration_rows"] = len(x_calibration)
        report["known_test_rows"] = len(y_known)
        report["unknown_test_rows"] = len(y_unknown)
        comparison[name] = report
        joblib.dump(bundle, args.models_directory / f"{name}.joblib")
        (args.output_directory / f"{name}_metrics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        predictions = pd.DataFrame({
            "true_label": np.concatenate([y_known, y_unknown]),
            "predicted_label": np.concatenate([known_pred, unknown_pred]),
            "confidence": np.concatenate([known_confidence, unknown_confidence]),
            "is_unknown_test": np.concatenate([
                np.zeros(len(y_known), dtype=bool), np.ones(len(y_unknown), dtype=bool)
            ]),
        })
        predictions["decision"] = np.where(
            predictions["confidence"] < bundle.rejection_threshold,
            "UNKNOWN", predictions["predicted_label"],
        )
        predictions.to_csv(args.output_directory / f"{name}_predictions.csv", index=False)
    (args.output_directory / "baseline_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    summary = {
        name: {
            "macro_f1": values["closed_set"]["macro_f1"],
            "unknown_auroc": values["open_set"]["unknown_auroc"],
            "unknown_recall": values["open_set"]["unknown_recall"],
        }
        for name, values in comparison.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
