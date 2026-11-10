#!/usr/bin/env python3
"""Summarize paired cache-on/off decode observations without accepting a regression."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def compare(path):
    groups = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            prefix, effort, kind, fixture = row["key"].split("/")
            if prefix != "speed" or not fixture.isdigit():
                raise ValueError("invalid synthetic workload identifier")
            on, off, ratio = (float(row[k]) for k in ("on_tps", "off_tps", "ratio"))
            if not all(math.isfinite(v) and v > 0 for v in (on, off, ratio)):
                raise ValueError("invalid throughput")
            if not math.isclose(on / off, ratio, rel_tol=1e-12):
                raise ValueError("ratio does not match the observations")
            groups.setdefault((effort, kind, row["label"]), []).append(ratio)
    return [{"effort": key[0], "kind": key[1], "request": key[2],
             "pairs": len(values), "geomean_on_over_off": statistics.geometric_mean(values),
             "median_on_over_off": statistics.median(values),
             "pairs_slower_with_cache": sum(value < 1 for value in values)}
            for key, values in sorted(groups.items())]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.pairs), indent=2))
