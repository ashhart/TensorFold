#!/usr/bin/env python3
"""pdl_ab.py RUNS LABEL...: one row a serve from its suite files; labels holding "b" before a digit are main's."""
import json
import statistics
import sys
from pathlib import Path


def med(xs):
    return statistics.median(xs) if xs else None


def row(d):
    r = {}
    if (d / "nodraft.json").exists():
        nd = json.loads((d / "nodraft.json").read_text())[0]
        r["plain tok/s"] = nd["decode_tps_median"]
        r["ITL p50 ms"] = nd["itl_p50_ms_median"]
        r["plain sha"] = ",".join(sorted({e["token_sha"] for e in nd["evidence"]}))
    for n in (8192, 32768):
        p = d / f"prefill{n}.json"
        if p.exists():
            rows = json.loads(p.read_text())["rows"]
            r[f"cold {n // 1024}k s"] = med([x["ttft_s"] for x in rows if x["phase"] in ("first", "repeated")])
            r[f"{n // 1024}k sha"] = ",".join(sorted({x["token_sha"] for x in rows}))
    if (d / "drafted.json").exists():
        dr = json.loads((d / "drafted.json").read_text())
        dr = dr[0] if isinstance(dr, list) else dr
        r["drafted tok/s"] = dr["decode_tps_median"]
    if (d / "c4.json").exists():
        c4 = json.loads((d / "c4.json").read_text())
        r["4 streams tok/s"] = c4_rate(c4)
    if (d / "form.txt").exists():
        r["experts"] = (d / "form.txt").read_text().split()[-1]
    return r


def c4_rate(c4):
    """The 4-stream cells' summed reply rate (`aggregate_tps`), median over the prompts."""
    return med([c["aggregate_tps"] for c in c4["cells"] if c["streams"] == 4 and not c["failed"]])


def main(argv):
    root, labels = Path(argv[1]), argv[2:]
    rows = {lab: row(root / lab) for lab in labels}
    cols = [k for k in dict.fromkeys(k for r in rows.values() for k in r)]
    print("| serve | " + " | ".join(cols) + " |")
    print("| --- |" + " --- |" * len(cols))
    for lab, r in rows.items():
        cells = [f"{r[c]:.3f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols]
        print(f"| {lab} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
