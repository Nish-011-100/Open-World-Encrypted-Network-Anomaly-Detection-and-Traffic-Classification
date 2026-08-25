"""Build canonical bidirectional flows from a Wireshark CSV."""

import argparse
import json
from pathlib import Path

from driftmamba.data.wireshark import convert_wireshark_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--maximum-packets", type=int, default=64)
    args = parser.parse_args()
    flows = convert_wireshark_csv(args.input, args.output,
                                  timeout_seconds=args.timeout_seconds,
                                  maximum_packets=args.maximum_packets)
    print(json.dumps({"flows": len(flows), "output": str(args.output.resolve()),
                      "labelled": bool(flows["label"].astype(bool).any())}, indent=2))


if __name__ == "__main__":
    main()
