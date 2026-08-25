"""Leakage-safe chronological and open-world dataset partitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test_known: pd.DataFrame
    test_unknown: pd.DataFrame
    known_classes: tuple[str, ...]
    unknown_classes: tuple[str, ...]


def chronological_open_world_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
    unknown_class_fraction: float = 0.20,
    seed: int = 42,
) -> DatasetSplits:
    """Hold out complete labels as unknown, then split known flows by time.

    Unknown labels are selected before row splitting and never occur in training or calibration.
    The final known test partition is later in time than both fitted partitions.
    """
    if frame.empty:
        raise ValueError("Cannot split an empty dataset")
    if "label" not in frame or frame["label"].isna().any() or (frame["label"] == "").any():
        raise ValueError("Open-world classification requires a non-empty label for every flow")
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("Train and calibration fractions must leave a test partition")

    ordered = frame.sort_values("start_time", kind="stable").reset_index(drop=True)
    counts = ordered["label"].value_counts()
    eligible = counts[counts >= 3].index.to_numpy()
    if len(eligible) < 2:
        raise ValueError("At least two labels with three flows each are required")
    rng = np.random.default_rng(seed)
    unknown_count = min(max(1, round(len(eligible) * unknown_class_fraction)), len(eligible) - 1)
    unknown_classes = tuple(sorted(rng.choice(eligible, size=unknown_count, replace=False).tolist()))
    train_boundary = ordered["start_time"].quantile(train_fraction)
    calibration_boundary = ordered["start_time"].quantile(train_fraction + calibration_fraction)
    unknown_mask = ordered["label"].isin(unknown_classes)
    known = ordered[~unknown_mask].copy()
    train = known[known["start_time"] < train_boundary].copy()
    calibration = known[
        (known["start_time"] >= train_boundary)
        & (known["start_time"] < calibration_boundary)
    ].copy()
    test_known = known[known["start_time"] >= calibration_boundary].copy()
    unknown = ordered[
        unknown_mask & (ordered["start_time"] >= calibration_boundary)
    ].copy()
    trained_classes = set(train["label"].unique())
    if len(trained_classes) < 2:
        raise ValueError("The chronological training period must contain at least two classes")
    calibration = calibration[calibration["label"].isin(trained_classes)].copy()
    late_test = test_known[~test_known["label"].isin(trained_classes)]
    test_known = test_known[test_known["label"].isin(trained_classes)].copy()
    unknown = pd.concat([unknown, late_test], ignore_index=True).sort_values(
        "start_time", kind="stable"
    )
    if min(len(train), len(calibration), len(test_known), len(unknown)) == 0:
        raise ValueError("One or more partitions are empty; use more data or different fractions")
    return DatasetSplits(
        train=train,
        calibration=calibration,
        test_known=test_known,
        test_unknown=unknown,
        known_classes=tuple(sorted(trained_classes)),
        unknown_classes=tuple(sorted(unknown["label"].unique().tolist())),
    )
