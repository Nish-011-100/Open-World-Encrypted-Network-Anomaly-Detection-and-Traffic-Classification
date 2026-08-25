"""Persist canonical partitions as model-ready CSV, arrays, and audit metadata."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from driftmamba.features import AggregatePreprocessor, path_signature_features, sequence_tensor


def materialize_partitions(partitions: dict[str, pd.DataFrame], output: Path, *,
                           maximum_packets: int = 64, metadata: dict | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    preprocessor = AggregatePreprocessor().fit(partitions["train"])
    for name, partition in partitions.items():
        partition.to_csv(output / f"{name}.csv", index=False)
        aggregate = preprocessor.transform(partition)
        sequence, mask = sequence_tensor(partition, maximum_packets=maximum_packets)
        signatures = path_signature_features(sequence, mask)
        np.savez_compressed(
            output / f"{name}.npz", aggregate=aggregate, sequence=sequence, mask=mask,
            signatures=signatures,
            labels=np.asarray(partition["label"].astype(str).tolist(), dtype=str),
        )
    joblib.dump(preprocessor, output / "aggregate_preprocessor.joblib")
    manifest = {
        "maximum_packets": maximum_packets,
        "rows": {name: len(value) for name, value in partitions.items()},
        "classes": {
            name: sorted(value["label"].astype(str).unique().tolist())
            for name, value in partitions.items()
        },
        "time_ranges": {
            name: [float(value["start_time"].min()), float(value["start_time"].max())]
            for name, value in partitions.items()
        },
        **(metadata or {}),
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
