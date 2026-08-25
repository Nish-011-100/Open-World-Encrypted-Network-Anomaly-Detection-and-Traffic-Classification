"""Convert packet-level Wireshark CSV exports into bidirectional flows."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"Time", "Source", "Destination", "Protocol", "Length", "Info"}
PORT_PATTERN = re.compile(r"(?P<src>\d+)\s*>\s*(?P<dst>\d+)")


@dataclass
class FlowState:
    endpoint_a: str
    port_a: int
    endpoint_b: str
    port_b: int
    protocol: str
    session_index: int
    times: list[float] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    directions: list[int] = field(default_factory=list)


def _ports(info: str) -> tuple[int, int]:
    match = PORT_PATTERN.search(info)
    return (int(match.group("src")), int(match.group("dst"))) if match else (0, 0)


def _canonical_key(source: str, destination: str, source_port: int,
                   destination_port: int, protocol: str) -> tuple[tuple, int]:
    source_endpoint = (source, source_port)
    destination_endpoint = (destination, destination_port)
    if source_endpoint <= destination_endpoint:
        return (source, source_port, destination, destination_port, protocol), 1
    return (destination, destination_port, source, source_port, protocol), -1


def load_packets(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="latin-1")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing Wireshark columns: {sorted(missing)}")
    frame = frame.copy()
    frame["Time"] = pd.to_numeric(frame["Time"], errors="coerce")
    frame["Length"] = pd.to_numeric(frame["Length"], errors="coerce")
    frame = frame.dropna(subset=["Time", "Length", "Source", "Destination"])
    frame = frame[(frame["Time"] >= 0) & (frame["Length"] > 0)]
    frame["Protocol"] = frame["Protocol"].fillna("UNKNOWN").astype(str).str.upper()
    frame["Info"] = frame["Info"].fillna("").astype(str)
    return frame.sort_values("Time", kind="stable").reset_index(drop=True)


def _finish(state: FlowState, maximum_packets: int) -> dict[str, object]:
    times = np.asarray(state.times, dtype=float)
    sizes = np.asarray(state.sizes, dtype=int)
    directions = np.asarray(state.directions, dtype=int)
    inter_arrivals = np.diff(times, prepend=times[0])
    forward = directions == 1
    identity = (f"{state.endpoint_a}:{state.port_a}-{state.endpoint_b}:{state.port_b}-"
                f"{state.protocol}-{state.session_index}-{times[0]:.9f}")
    return {
        "flow_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
        "start_time": float(times[0]), "end_time": float(times[-1]),
        "endpoint_a": state.endpoint_a, "endpoint_b": state.endpoint_b,
        "port_a": state.port_a, "port_b": state.port_b, "protocol": state.protocol,
        "packet_sizes": json.dumps(sizes[:maximum_packets].tolist()),
        "directions": json.dumps(directions[:maximum_packets].tolist()),
        "inter_arrival_times": json.dumps(inter_arrivals[:maximum_packets].tolist()),
        "duration": float(max(times[-1] - times[0], 0.0)),
        "packet_count": len(sizes), "total_bytes": int(sizes.sum()),
        "forward_packets": int(forward.sum()), "reverse_packets": int((~forward).sum()),
        "forward_bytes": int(sizes[forward].sum()), "reverse_bytes": int(sizes[~forward].sum()),
        "mean_packet_size": float(sizes.mean()), "std_packet_size": float(sizes.std()),
        "mean_inter_arrival": float(inter_arrivals.mean()),
        "std_inter_arrival": float(inter_arrivals.std()), "label": "",
    }


def packets_to_flows(packets: pd.DataFrame, *, timeout_seconds: float = 60.0,
                     maximum_packets: int = 64) -> pd.DataFrame:
    active: dict[tuple, FlowState] = {}
    session_counts: dict[tuple, int] = {}
    completed: list[dict[str, object]] = []
    for row in packets.itertuples(index=False):
        source_port, destination_port = _ports(str(row.Info))
        key, direction = _canonical_key(str(row.Source), str(row.Destination), source_port,
                                        destination_port, str(row.Protocol))
        state = active.get(key)
        if state is not None and float(row.Time) - state.times[-1] > timeout_seconds:
            completed.append(_finish(state, maximum_packets))
            state = None
        if state is None:
            session_index = session_counts.get(key, 0)
            session_counts[key] = session_index + 1
            state = FlowState(*key, session_index=session_index)
            active[key] = state
        state.times.append(float(row.Time))
        state.sizes.append(int(row.Length))
        state.directions.append(direction)
    completed.extend(_finish(state, maximum_packets) for state in active.values())
    return pd.DataFrame(completed).sort_values("start_time", kind="stable").reset_index(drop=True)


def convert_wireshark_csv(input_path: Path, output_path: Path, *,
                          timeout_seconds: float = 60.0, maximum_packets: int = 64) -> pd.DataFrame:
    flows = packets_to_flows(load_packets(input_path), timeout_seconds=timeout_seconds,
                             maximum_packets=maximum_packets)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flows.to_csv(output_path, index=False)
    return flows
