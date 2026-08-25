"""Classify new canonical encrypted flows and reject unfamiliar applications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from driftmamba.models.inference import load_deep_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Canonical flow CSV")
    parser.add_argument("--models-directory", type=Path, default=Path("models/deep"))
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/inference/predictions.csv"))
    parser.add_argument("--maximum-packets", type=int, default=64)
    parser.add_argument("--alpha", type=float)
    args = parser.parse_args()
    flows = pd.read_csv(args.input)
    bundle = load_deep_bundle(
        args.models_directory, args.preprocessor, maximum_packets=args.maximum_packets
    )
    results = bundle.predict(flows, alpha=args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    summary = {
        "flows": len(results),
        "known": int((results["Decision"] != "UNKNOWN").sum()),
        "unknown": int((results["Decision"] == "UNKNOWN").sum()),
        "mean_prediction_set_size": float(results["PredictionSetSize"].mean()),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
