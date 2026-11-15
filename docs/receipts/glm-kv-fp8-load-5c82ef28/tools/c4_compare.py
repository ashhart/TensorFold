#!/usr/bin/env python3
"""Compare suite-c4.sh launches: 4-stream aggregate and steady decode a prompt and temperature.

usage: c4_compare.py OUT_DIR LABEL... (labels whose name holds "-b" are the base, the others the head)
"""
import json
import statistics
import sys
from pathlib import Path


def load(d):
    cells = json.loads((d / "c4.json").read_text())["cells"]
    assert all(c["failed"] == 0 and c["timing_valid"] for c in cells), d
    return {(c["prompt"], c["temperature"]): c for c in cells}


def main():
    root, labels = Path(sys.argv[1]), sys.argv[2:]
    runs = {lb: load(root / lb) for lb in labels}
    base = [lb for lb in labels if "-b" in lb]
    head = [lb for lb in labels if lb not in base]
    print("| 4 streams, tok/s | " + " | ".join(labels) + " | base median | head median | change |")
    print("| --- |" + " ---: |" * (len(labels) + 3))
    for key in runs[labels[0]]:
        for field, name in (("aggregate_tps", "aggregate"), ("steady_tps", "steady")):
            vals = {lb: runs[lb][key][field] for lb in labels}
            b = statistics.median(vals[lb] for lb in base)
            h = statistics.median(vals[lb] for lb in head)
            print(f"| {key[0]}, temperature {key[1]:g}, {name} | " + " | ".join(f"{vals[lb]:.1f}" for lb in labels)
                  + f" | {b:.1f} | {h:.1f} | {(h / b - 1) * 100:+.1f}% |")
    for key in runs[labels[0]]:
        shas = {lb: sorted(e["token_sha256"] for r in runs[lb][key]["reps"] for e in r["evidence"]) for lb in labels}
        same = len({tuple(v) for v in shas.values()}) == 1
        print(f"replies {key[0]}, temperature {key[1]:g}: {len(shas[labels[0]])} a launch, "
              f"{len(set().union(*map(set, shas.values())))} distinct token sha256, "
              + ("the same multiset in every launch" if same else "launches DIFFER"))


if __name__ == "__main__":
    main()
