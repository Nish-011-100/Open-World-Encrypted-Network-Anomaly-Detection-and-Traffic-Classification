import json

import numpy as np
import pandas as pd

from driftmamba.data.cesnet import convert_chunk
from driftmamba.data.splits import chronological_open_world_split
from driftmamba.features import AggregatePreprocessor, sequence_tensor


def cesnet_row(index: int, label: str) -> dict:
    timestamp = pd.Timestamp("2022-11-01", tz="UTC") + pd.Timedelta(index, unit="min")
    return {
        "ID": index, "SRC_IP": "private", "DST_IP": "service", "SRC_PORT": 50000,
        "DST_PORT": 443, "PROTOCOL": 17,
        "TIME_FIRST": timestamp.isoformat(),
        "TIME_LAST": (timestamp + pd.Timedelta(1, unit="s")).isoformat(),
        "DURATION": 1.0, "BYTES": 300, "BYTES_REV": 600,
        "PACKETS": 2, "PACKETS_REV": 2,
        "PPI": json.dumps([[0.0, 0.1, 0.2, 0.3], [1, 1, -1, -1], [100, 200, 300, 300]]),
        "APP": label,
    }


def test_cesnet_adapter_and_features():
    flows = convert_chunk(pd.DataFrame([cesnet_row(0, "video")]))
    assert flows.loc[0, "endpoint_a"] == "REDACTED"
    assert flows.loc[0, "label"] == "video"
    sequence, mask = sequence_tensor(flows, maximum_packets=8)
    assert sequence.shape == (1, 8, 3)
    assert mask.sum() == 4
    assert np.all(sequence[0, :4, 1] == [1, 1, -1, -1])


def test_preprocessor_accepts_protocol_inferred_as_integer_after_csv_round_trip():
    flows = convert_chunk(pd.DataFrame([cesnet_row(index, "video") for index in range(3)]))
    transformer = AggregatePreprocessor().fit(flows)
    csv_like = flows.copy()
    csv_like["protocol"] = csv_like["protocol"].astype(int)
    transformed = transformer.transform(csv_like)
    assert transformed.shape[0] == len(csv_like)
    assert transformed.dtype == np.float32


def test_open_world_split_is_class_disjoint_and_chronological():
    raw = pd.DataFrame([cesnet_row(i, ["video", "cloud", "social"][i % 3]) for i in range(90)])
    flows = convert_chunk(raw)
    splits = chronological_open_world_split(flows, seed=7)
    assert not set(splits.train.label).intersection(splits.unknown_classes)
    assert not set(splits.calibration.label).intersection(splits.unknown_classes)
    assert splits.train.start_time.max() < splits.calibration.start_time.min()
    assert splits.calibration.start_time.max() < splits.test_known.start_time.min()
    assert splits.test_unknown.start_time.min() >= splits.test_known.start_time.min()
    transformer = AggregatePreprocessor().fit(splits.train)
    assert transformer.transform(splits.test_known).dtype == np.float32
