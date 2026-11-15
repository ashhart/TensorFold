#!/usr/bin/env python3
"""Compare suite-kv.sh launches: plain decode, cold prompt time and drafted decode after it, and reply equality.

usage: kvload_compare.py OUT_DIR LABEL... (labels whose name holds "-b" are the base, the others the head)
"""
import json
import statistics
import sys
from pathlib import Path


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def load(d):
    out = {"nodraft": {}, "prompt": {}}
    for r in json.loads((d / "nodraft.json").read_text()):
        out["nodraft"][r["prompt"]] = (r["decode_tps_median"], r["decode_tps_all"])
    for f in sorted(d.glob("prefill*.json")):
        n = int(f.stem[len("prefill"):])
        rows = [r for r in json.loads(f.read_text())["rows"] if r["phase"] in ("first", "repeated")]
        assert all(r["cache_state"] == "cold" and r["complete"] for r in rows), f
        out["prompt"][n] = {
            "ttft": [r["ttft_s"] for r in rows],
            "tps": [r["decode_tps"] for r in rows],
            "sha": {r["token_sha256"] for r in rows},
        }
    return out


def main():
    root, labels = Path(sys.argv[1]), sys.argv[2:]
    runs = {lb: load(root / lb) for lb in labels}
    base = [lb for lb in labels if "-b" in lb]
    head = [lb for lb in labels if lb not in base]
    print("| Row | " + " | ".join(labels) + " | base median | head median | change |")
    print("| --- |" + " ---: |" * (len(labels) + 3))

    def row(name, vals, fmt, lower_better):
        b = med([v for lb, v in vals.items() if lb in base])
        h = med([v for lb, v in vals.items() if lb in head])
        ch = (h / b - 1) * 100
        print(f"| {name} | " + " | ".join(fmt.format(vals[lb]) for lb in labels)
              + f" | {fmt.format(b)} | {fmt.format(h)} | {ch:+.1f}% |")

    for p in runs[labels[0]]["nodraft"]:
        row(f'`"draft": false` decode, {p} (tok/s)', {lb: runs[lb]["nodraft"][p][0] for lb in labels}, "{:.2f}", False)
    for n in sorted(runs[labels[0]]["prompt"]):
        row(f"Cold prompt of {n:,} tokens (s)", {lb: med(runs[lb]["prompt"][n]["ttft"]) for lb in labels}, "{:.2f}", True)
        row(f"Drafted decode after {n:,} (tok/s)", {lb: med(runs[lb]["prompt"][n]["tps"]) for lb in labels}, "{:.1f}", False)
    same = True
    for n in sorted(runs[labels[0]]["prompt"]):
        shas = set().union(*(runs[lb]["prompt"][n]["sha"] for lb in labels))
        same &= len(shas) == 1
        print(f"replies after {n}: {len(shas)} distinct token sha256 over {len(labels)} launches")
    print("RESULT replies " + ("equal" if same else "DIFFER"))


if __name__ == "__main__":
    main()
