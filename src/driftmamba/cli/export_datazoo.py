"""Download an official CESNET DataZoo edition and export a bounded QUIC22 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driftmamba.data.cesnet import convert_chunk
from driftmamba.data.materialize import materialize_partitions


def main() -> None:
    try:
        from cesnet_datazoo.config import AppSelection, DatasetConfig
        from cesnet_datazoo.datasets import CESNET_QUIC22
    except ImportError as error:
        raise RuntimeError("Install the official loader with `pip install cesnet-datazoo`") from error

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/cesnet-quic22"))
    parser.add_argument("--output-directory", type=Path, default=Path("data/processed/datazoo_xs"))
    parser.add_argument("--edition", choices=["XS", "S", "M", "L"], default="XS")
    parser.add_argument("--known-applications", type=int, default=15)
    parser.add_argument("--train-rows", type=int, default=100_000)
    parser.add_argument("--calibration-rows", type=int, default=20_000)
    parser.add_argument("--known-test-rows", type=int, default=20_000)
    parser.add_argument("--unknown-test-rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = CESNET_QUIC22(str(args.data_root), size=args.edition)
    config = DatasetConfig(
        dataset=dataset,
        train_period_name="W-2022-44",
        test_period_name="W-2022-45",
        apps_selection=AppSelection.TOPX_KNOWN,
        apps_selection_topx=args.known_applications,
        train_size=args.train_rows,
        val_known_size=args.calibration_rows,
        test_known_size=args.known_test_rows,
        test_unknown_size=args.unknown_test_rows,
        return_other_fields=True,
        disable_label_encoding=True,
        random_state=args.seed,
    )
    # A cancelled DataZoo initialization can leave a partial ``test_indices.npz``.
    # Regenerating bounded experiment indices is cheap and avoids trusting stale cache state.
    dataset.set_dataset_config_and_initialize(config, disable_indices_cache=True)
    train = convert_chunk(dataset.get_train_df(), maximum_packets=64)
    calibration = convert_chunk(dataset.get_val_df(), maximum_packets=64)
    raw_test = dataset.get_test_df()
    known_apps = set(dataset.get_known_apps())
    test_known_raw = raw_test[raw_test["APP"].astype(str).isin(known_apps)]
    test_unknown_raw = raw_test[~raw_test["APP"].astype(str).isin(known_apps)]
    partitions = {
        "train": train,
        "calibration": calibration,
        "test_known": convert_chunk(test_known_raw, maximum_packets=64),
        "test_unknown": convert_chunk(test_unknown_raw, maximum_packets=64),
    }
    manifest = materialize_partitions(
        partitions, args.output_directory, maximum_packets=64,
        metadata={
            "source": "CESNET DataZoo CESNET-QUIC22",
            "edition": args.edition,
            "train_period": "W-2022-44", "test_period": "W-2022-45",
            "known_classes": sorted(known_apps),
            "unknown_classes": sorted(partitions["test_unknown"]["label"].unique().tolist()),
            "seed": args.seed,
        },
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
