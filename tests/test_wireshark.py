import json

import pandas as pd

from driftmamba.data.wireshark import packets_to_flows


def packet(time, source, destination, length, info, protocol="QUIC"):
    return {"Time": time, "Source": source, "Destination": destination,
            "Protocol": protocol, "Length": length, "Info": info}


def test_bidirectional_packets_form_one_flow():
    packets = pd.DataFrame([
        packet(0.0, "client", "server", 100, "50000 > 443 Len=58"),
        packet(0.2, "server", "client", 200, "443 > 50000 Len=158"),
    ])
    flows = packets_to_flows(packets)
    assert len(flows) == 1
    assert flows.loc[0, "packet_count"] == 2
    assert {flows.loc[0, "forward_packets"], flows.loc[0, "reverse_packets"]} == {1}
    assert json.loads(flows.loc[0, "directions"]) == [1, -1]


def test_timeout_starts_new_session():
    packets = pd.DataFrame([
        packet(0.0, "a", "b", 80, "1 > 2", "TCP"),
        packet(61.0, "a", "b", 90, "1 > 2", "TCP"),
    ])
    assert len(packets_to_flows(packets, timeout_seconds=60.0)) == 2


def test_truncation_does_not_change_aggregates():
    packets = pd.DataFrame([packet(float(i), "a", "b", 10 + i, "3 > 4", "UDP")
                            for i in range(5)])
    flow = packets_to_flows(packets, maximum_packets=2).iloc[0]
    assert len(json.loads(flow["packet_sizes"])) == 2
    assert flow["packet_count"] == 5
    assert flow["total_bytes"] == 60
