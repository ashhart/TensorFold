#!/usr/bin/env python3
"""ab_compare.py OUT_DIR LABEL...: suite-ab launches; labels holding "-b" are the base. Decode tok/s, cold prompt s, token sha256."""
import json, statistics, sys
from pathlib import Path
root, labels = Path(sys.argv[1]), sys.argv[2:]
base = [l for l in labels if "-b" in l]; head = [l for l in labels if l not in base]
def load(l):
    d = root / l; out = {}
    for r in json.loads((d / "nodraft.json").read_text()):
        out[f"decode {r['prompt']} tok/s"] = (r["decode_tps_median"], None)
    for n in (8192, 32768):
        rows = [r for r in json.loads((d / f"prefill{n}.json").read_text())["rows"] if r["phase"] in ("first", "repeated")]
        assert all(r["cache_state"] == "cold" for r in rows)
        out[f"cold {n} s"] = (statistics.median(r["ttft_s"] for r in rows), {r["token_sha256"] for r in rows})
    return out
runs = {l: load(l) for l in labels}
print("| row | " + " | ".join(labels) + " | base med | head med | change |"); print("|---|" + "---:|" * (len(labels) + 3))
for k in runs[labels[0]]:
    v = {l: runs[l][k][0] for l in labels}
    b = statistics.median(v[l] for l in base); h = statistics.median(v[l] for l in head)
    print(f"| {k} | " + " | ".join(f"{v[l]:.3f}" for l in labels) + f" | {b:.3f} | {h:.3f} | {(h / b - 1) * 100:+.2f}% |")
for k in runs[labels[0]]:
    if runs[labels[0]][k][1] is not None:
        s = set().union(*(runs[l][k][1] for l in labels)); print(f"{k}: {len(s)} distinct token sha256 over {len(labels)} launches")
