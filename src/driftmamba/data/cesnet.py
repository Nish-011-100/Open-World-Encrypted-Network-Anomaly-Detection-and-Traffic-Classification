"""Streaming adapter for the published CESNET-QUIC22 CSV schema."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "ID", "SRC_IP", "DST_IP", "SRC_PORT", "DST_PORT", "PROTOCOL",
    "TIME_FIRST", "TIME_LAST", "DURATION", "BYTES", "BYTES_REV",
    "PACKETS", "PACKETS_REV", "PPI", "APP",
}


def _sequence(value: object) -> tuple[list[float], list[int], list[int]]:
    """Parse CESNET PPI: [[inter-packet times], [directions], [packet sizes]]."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
    else:
        parsed = value
    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
        raise ValueError("PPI must contain [times, directions, sizes]")
    times = [float(item) for item in parsed[0]]
    directions = [int(item) for item in parsed[1]]
    sizes = [int(item) for item in parsed[2]]
    length = min(len(times), len(directions), len(sizes))
    if length == 0:
        raise ValueError("PPI sequence is empty")
    return times[:length], directions[:length], sizes[:length]


def _timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(f"Could not parse {int(parsed.isna().sum())} flow timestamps")
    return parsed.astype("int64") / 1_000_000_000


def convert_chunk(frame: pd.DataFrame, maximum_packets: int = 64) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing CESNET-QUIC22 columns: {sorted(missing)}")
    result: list[dict[str, object]] = []
    start_times = _timestamp(frame["TIME_FIRST"])
    end_times = _timestamp(frame["TIME_LAST"])
    for position, (_, row) in enumerate(frame.iterrows()):
        inter_arrivals, directions, sizes = _sequence(row["PPI"])
        sequence_length = min(len(sizes), maximum_packets)
        sizes_array = np.asarray(sizes, dtype=float)
        times_array = np.asarray(inter_arrivals, dtype=float)
        flow_id = hashlib.sha256(str(row["ID"]).encode()).hexdigest()[:20]
        result.append({
            "flow_id": flow_id,
            "start_time": float(start_times.iloc[position]),
            "end_time": float(end_times.iloc[position]),
            "endpoint_a": "REDACTED", "endpoint_b": "REDACTED",
            "port_a": int(row["SRC_PORT"]), "port_b": int(row["DST_PORT"]),
            "protocol": str(row["PROTOCOL"]),
            "packet_sizes": json.dumps(sizes[:sequence_length]),
            "directions": json.dumps(directions[:sequence_length]),
            "inter_arrival_times": json.dumps(inter_arrivals[:sequence_length]),
            "duration": float(row["DURATION"]),
            "packet_count": int(row["PACKETS"]) + int(row["PACKETS_REV"]),
            "total_bytes": int(row["BYTES"]) + int(row["BYTES_REV"]),
            "forward_packets": int(row["PACKETS"]),
            "reverse_packets": int(row["PACKETS_REV"]),
            "forward_bytes": int(row["BYTES"]), "reverse_bytes": int(row["BYTES_REV"]),
            "mean_packet_size": float(sizes_array.mean()),
            "std_packet_size": float(sizes_array.std()),
            "mean_inter_arrival": float(times_array.mean()),
            "std_inter_arrival": float(times_array.std()),
            "label": str(row["APP"]).strip(),
        })
    return pd.DataFrame(result)


def convert_cesnet_csv(input_path: Path, output_path: Path, *, maximum_packets: int = 64,
                       chunksize: int = 100_000, maximum_rows: int | None = None) -> int:
    """Convert large CSVs incrementally without loading the 89 GB dataset into memory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    first = True
    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        if maximum_rows is not None:
            chunk = chunk.iloc[: max(maximum_rows - written, 0)]
        if chunk.empty:
            break
        canonical = convert_chunk(chunk, maximum_packets=maximum_packets)
        canonical.to_csv(output_path, mode="w" if first else "a", header=first, index=False)
        first = False
        written += len(canonical)
        if maximum_rows is not None and written >= maximum_rows:
            break
    return written
