"""Summarize retained local served receipts without contacting the API."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent.parent / "results"


def read(name):
    return [json.loads(line) for line in (root / name).read_text().splitlines()]


baseline = {x["cell"]: x for x in read("perf-v5-before-v6.jsonl")}
for name in ("perf-v5-before-v6.jsonl", "perf-v6-off.jsonl", "perf-v6-all.jsonl",
             "perf-v6-off-repeat.jsonl", "perf-v6-all-repeat.jsonl"):
    if not (root / name).exists():
        continue
    groups = defaultdict(list)
    for row in read(name):
        old = baseline[row["cell"]]
        assert old["request"] == row["request"], (name, row["cell"], "request")
        assert (old["result"]["tensorfold"]["token_sha256"] ==
                row["result"]["tensorfold"]["token_sha256"]), (name, row["cell"], "tokens")
        groups[row["cell"]].append(row["result"]["tensorfold"]["tokens_per_second"])
    print(json.dumps({"file": name, "token_equal": True, "cells": {
        k: {"median": statistics.median(v), "runs": v} for k, v in groups.items()}}))

for path in sorted(root.glob("served-v6-all-*.jsonl")):
    rows = read(path.name)
    exact = [x for x in rows if "equivalence" in x["test"]]
    for row in exact:
        assert all(v for k, v in row["result"].items() if k.endswith("equal")), row
    print(json.dumps({"file": path.name, "receipts": len(rows),
                      "equivalence_checks": len(exact)}))
