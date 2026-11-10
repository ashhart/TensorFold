#!/usr/bin/env python3
"""Fingerprints and speed from the per-request lines. usage: compare.py RUNS_DIR OUT_MD

RUNS_DIR holds <kv>-<order>-<side>.jsonl (kv: bf16|int8, order: 1..4, side: main|branch).
Equal means every request of a cell, drafted and plain, on both sides, has the same tensorfold.token_sha256.
Speed: median of the measured requests (warm-up excluded) of both runs of a side, drafted and plain.
"""
import collections, glob, json, os, statistics as st, sys

D, OUTMD = sys.argv[1], sys.argv[2]
rows = collections.defaultdict(list)
for f in sorted(glob.glob(f"{D}/*.jsonl")):
    kv, order, side = os.path.basename(f)[:-6].split("-")
    for line in open(f):
        r = json.loads(line)
        rows[(kv, r["depth"], r["kind"])].append(dict(r, side=side, order=order))
lines, bad = [], 0
for kv in ("bf16", "int8"):
    lines += [f"### {kv} KV", "", "| prompt tokens | kind | requests | distinct token_sha256 | main = branch = plain |",
              "|---|---|---|---|---|"]
    for (k, depth, kind), rs in sorted(rows.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        if k != kv:
            continue
        shas = {r["token_sha256"] for r in rs}
        ok = len(shas) == 1 and None not in shas and all(r["ok"] for r in rs)
        bad += not ok
        pt = sorted({r["prompt_tokens"] for r in rs})
        lines.append(f"| {pt[0]} | {kind} | {len(rs)} | {len(shas)} | {'yes' if ok else 'NO'} |")
    lines += ["", "| prompt tokens | kind | mode | main tok/s | branch tok/s | branch / main | n main | n branch |",
              "|---|---|---|---|---|---|---|---|"]
    for (k, depth, kind), rs in sorted(rows.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        if k != kv:
            continue
        for mode in ("draft", "plain"):
            v = {s: [r["decode_tok_s"] for r in rs if r["side"] == s and r["mode"] == mode and not r["warmup"]
                     and r["decode_tok_s"]] for s in ("main", "branch")}
            if not v["main"] or not v["branch"]:
                continue
            a, b = st.median(v["main"]), st.median(v["branch"])
            pt = sorted({r["prompt_tokens"] for r in rs})[0]
            lines.append(f"| {pt} | {kind} | {mode} | {a:.2f} | {b:.2f} | {b / a:.3f} | {len(v['main'])} | {len(v['branch'])} |")
    lines.append("")
open(OUTMD, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("MISMATCH CELLS:", bad)
sys.exit(1 if bad else 0)
