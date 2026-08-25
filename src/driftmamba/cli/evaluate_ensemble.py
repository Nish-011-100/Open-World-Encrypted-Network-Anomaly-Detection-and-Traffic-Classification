"""Evaluate a heterogeneous neural ensemble from aligned saved prediction reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    frames = [pd.read_csv(path) for path in args.predictions]
    if any(len(frame) != len(frames[0]) for frame in frames[1:]):
        raise ValueError("All ensemble prediction reports must contain aligned rows")
    labels = np.stack([frame["predicted_label"].astype(str).to_numpy() for frame in frames])
    votes = []
    for column in labels.T:
        values, counts = np.unique(column, return_counts=True)
        votes.append(values[np.argmax(counts)])
    votes = np.asarray(votes)
    known_p = np.mean(np.stack([frame["known_p_value"].to_numpy() for frame in frames]), axis=0)
    unknown = frames[0]["is_unknown_test"].astype(bool).to_numpy()
    truth = frames[0]["true_label"].astype(str).to_numpy()
    known = ~unknown
    report = {
        "model": "heterogeneous_deep_ensemble",
        "members": [str(path) for path in args.predictions],
        "accuracy": float(accuracy_score(truth[known], votes[known])),
        "balanced_accuracy": float(balanced_accuracy_score(truth[known], votes[known])),
        "macro_f1": float(f1_score(truth[known], votes[known], average="macro")),
        "unknown_auroc": float(roc_auc_score(unknown.astype(int), 1.0 - known_p)),
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"true_label": truth, "predicted_label": votes,
                  "known_p_value": known_p, "is_unknown_test": unknown}).to_csv(
        args.output_directory / "ensemble_predictions.csv", index=False
    )
    (args.output_directory / "ensemble_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
