"""Validate canonical flows and create leakage-safe model-ready partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from driftmamba.data.cesnet import convert_cesnet_csv
from driftmamba.data.materialize import materialize_partitions
from driftmamba.data.splits import chronological_open_world_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("data/processed/experiment"))
    parser.add_argument("--source-format", choices=["canonical", "cesnet"], default="cesnet")
    parser.add_argument("--maximum-rows", type=int)
    parser.add_argument("--maximum-packets", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    canonical_path = output / "canonical_flows.csv"
    if args.source_format == "cesnet":
        convert_cesnet_csv(args.input, canonical_path, maximum_packets=args.maximum_packets,
                           maximum_rows=args.maximum_rows)
    else:
        canonical_path = args.input
    flows = pd.read_csv(canonical_path)
    splits = chronological_open_world_split(flows, seed=args.seed)
    partitions = {
        "train": splits.train, "calibration": splits.calibration,
        "test_known": splits.test_known, "test_unknown": splits.test_unknown,
    }
    manifest = materialize_partitions(partitions, output, maximum_packets=args.maximum_packets,
                                      metadata={
        "seed": args.seed, "maximum_packets": args.maximum_packets,
        "known_classes": list(splits.known_classes),
        "unknown_classes": list(splits.unknown_classes),
    })
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
