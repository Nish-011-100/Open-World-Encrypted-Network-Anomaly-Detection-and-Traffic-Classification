"""Privacy-preserving aggregate and packet-sequence feature construction."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

BASE_NUMERIC = [
    "duration", "packet_count", "total_bytes", "forward_packets", "reverse_packets",
    "forward_bytes", "reverse_bytes", "mean_packet_size", "std_packet_size",
    "mean_inter_arrival", "std_inter_arrival",
]
DERIVED_NUMERIC = [
    "log_duration", "log_packet_count", "log_total_bytes", "byte_direction_ratio",
    "packet_direction_ratio", "bytes_per_packet", "packets_per_second",
]
CATEGORICAL = ["protocol", "destination_port_group"]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float) / denominator.replace(0, np.nan).astype(float)


def _port_group(port: object) -> str:
    value = int(port)
    if value == 443:
        return "HTTPS_QUIC"
    if value == 53:
        return "DNS"
    if 0 <= value <= 1023:
        return "WELL_KNOWN"
    if value <= 49151:
        return "REGISTERED"
    return "EPHEMERAL"


def aggregate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # CSV readers infer numeric-looking protocol values (for example QUIC/UDP as 17) as integers,
    # while dataset adapters deliberately fit categorical values as strings. Normalize both paths
    # so saved preprocessors remain usable for CLI inference after a CSV round trip.
    result["protocol"] = result["protocol"].astype(str)
    result["log_duration"] = np.log1p(result["duration"].clip(lower=0))
    result["log_packet_count"] = np.log1p(result["packet_count"].clip(lower=0))
    result["log_total_bytes"] = np.log1p(result["total_bytes"].clip(lower=0))
    result["byte_direction_ratio"] = _safe_ratio(
        result["forward_bytes"], result["total_bytes"]
    ).fillna(0)
    result["packet_direction_ratio"] = _safe_ratio(
        result["forward_packets"], result["packet_count"]
    ).fillna(0)
    result["bytes_per_packet"] = _safe_ratio(
        result["total_bytes"], result["packet_count"]
    ).fillna(0)
    result["packets_per_second"] = _safe_ratio(
        result["packet_count"], result["duration"].clip(lower=1e-6)
    ).fillna(0).clip(upper=1e7)
    result["destination_port_group"] = result["port_b"].map(_port_group)
    return result


@dataclass
class AggregatePreprocessor:
    transformer: ColumnTransformer | None = None

    def fit(self, training: pd.DataFrame) -> AggregatePreprocessor:
        self.transformer = ColumnTransformer([
            ("numeric", RobustScaler(), BASE_NUMERIC + DERIVED_NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])
        self.transformer.fit(aggregate_frame(training)[BASE_NUMERIC + DERIVED_NUMERIC + CATEGORICAL])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("Preprocessor must be fitted on training data first")
        features = aggregate_frame(frame)[BASE_NUMERIC + DERIVED_NUMERIC + CATEGORICAL]
        return self.transformer.transform(features).astype("float32")


def sequence_tensor(frame: pd.DataFrame, maximum_packets: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return [batch, packets, size/direction/time] features and a validity mask."""
    values = np.zeros((len(frame), maximum_packets, 3), dtype="float32")
    mask = np.zeros((len(frame), maximum_packets), dtype=bool)
    for index, row in enumerate(frame.itertuples(index=False)):
        sizes = np.asarray(json.loads(row.packet_sizes), dtype=float)[:maximum_packets]
        directions = np.asarray(json.loads(row.directions), dtype=float)[:maximum_packets]
        times = np.asarray(json.loads(row.inter_arrival_times), dtype=float)[:maximum_packets]
        length = min(len(sizes), len(directions), len(times), maximum_packets)
        values[index, :length, 0] = np.log1p(np.abs(sizes[:length]))
        values[index, :length, 1] = np.sign(directions[:length])
        values[index, :length, 2] = np.log1p(np.maximum(times[:length], 0))
        mask[index, :length] = True
    return values, mask


def path_signature_features(sequence: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute a normalized second-order signature of signed-size/time paths.

    Level one captures net displacement; level two captures ordered interactions between traffic
    direction/volume and elapsed time. This is dependency-free and differentiable inputs are not
    required because signatures form the proposal's explicit second view.
    """
    output = np.zeros((len(sequence), 6), dtype="float32")
    for sample in range(len(sequence)):
        valid = sequence[sample, mask[sample]]
        if len(valid) == 0:
            continue
        increments = np.column_stack((valid[:, 1] * valid[:, 0], valid[:, 2]))
        scale = np.maximum(np.abs(increments).sum(axis=0), 1e-6)
        increments = increments / scale
        level_one = increments.sum(axis=0)
        prefix = np.zeros(2, dtype=float)
        level_two = np.zeros((2, 2), dtype=float)
        for delta in increments:
            level_two += np.outer(prefix + 0.5 * delta, delta)
            prefix += delta
        output[sample] = np.concatenate([level_one, level_two.ravel()])
    return output
