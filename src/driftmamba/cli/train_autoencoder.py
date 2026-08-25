"""Train and evaluate an autoencoder-gated open-world classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from driftmamba.evaluation.metrics import evaluate_predictions
from driftmamba.models.autoencoder import train_autoencoder
from driftmamba.models.baselines import train_baseline


def load(directory: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(directory / f"{name}.npz", allow_pickle=False)
    return data["aggregate"], data["labels"].astype(str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("reports/autoencoder"))
    parser.add_argument("--models-directory", type=Path, default=Path("models/autoencoder"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.models_directory.mkdir(parents=True, exist_ok=True)
    train_x, train_y = load(args.data_directory, "train")
    cal_x, _ = load(args.data_directory, "calibration")
    known_x, known_y = load(args.data_directory, "test_known")
    unknown_x, _ = load(args.data_directory, "test_unknown")
    classifier = train_baseline("random_forest", train_x, train_y, cal_x, seed=args.seed)
    detector = train_autoencoder(train_x, cal_x, epochs=args.epochs, seed=args.seed)
    cal_pred, cal_class = classifier.predict(cal_x)
    del cal_pred
    threshold = float(np.quantile(np.sqrt(cal_class * detector.knownness(cal_x)), 0.10))
    known_pred, known_class = classifier.predict(known_x)
    unknown_pred, unknown_class = classifier.predict(unknown_x)
    known_score = np.sqrt(known_class * detector.knownness(known_x))
    unknown_score = np.sqrt(unknown_class * detector.knownness(unknown_x))
    report = evaluate_predictions(
        known_y, known_pred, known_score, unknown_pred, unknown_score, threshold,
        classifier.label_encoder.classes_.tolist(),
    )
    report["model"] = "autoencoder_gated_random_forest"
    report["training_history"] = detector.history
    torch_path = args.models_directory / "flow_autoencoder.pt"
    import torch
    torch.save({"state_dict": detector.model.state_dict(), "input_dimension": train_x.shape[1],
                "error_center": detector.error_center, "error_scale": detector.error_scale}, torch_path)
    joblib.dump(classifier, args.models_directory / "random_forest.joblib")
    (args.output_directory / "autoencoder_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"accuracy": report["closed_set"]["accuracy"],
                      "macro_f1": report["closed_set"]["macro_f1"],
                      "unknown_auroc": report["open_set"]["unknown_auroc"],
                      "unknown_recall": report["open_set"]["unknown_recall"]}, indent=2))


if __name__ == "__main__":
    main()
