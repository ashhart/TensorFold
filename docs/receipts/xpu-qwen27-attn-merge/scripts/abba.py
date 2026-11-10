#!/usr/bin/env python3
"""Per-pass medians and dispersion for the plain (non-drafted) and drafted decode. usage: abba.py RUNS_DIR"""
import collections, glob, json, os, statistics as st, sys
D = sys.argv[1]
cell = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{D}/*.jsonl")):
    kv, order, side = os.path.basename(f)[:-6].split("-")
    for line in open(f):
        r = json.loads(line)
        if r["warmup"] or not r["decode_tok_s"]:
            continue
        cell[(kv, r["depth"], r["kind"], r["mode"])].setdefault((int(order), side), []).append(r["decode_tok_s"])
print("| kv | prompt depth | kind | mode | main pass 1 | branch pass 2 | branch pass 3 | main pass 4 | ratio (b2+b3)/(m1+m4) | ratio pass2/pass1 | ratio pass3/pass4 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(cell):
    c = cell[k]
    m = {o: st.median(v) for o, v in ((o, c[(o, s)]) for o, s in ((1, "main"), (2, "branch"), (3, "branch"), (4, "main")))}
    r = (m[2] + m[3]) / (m[1] + m[4])
    print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {m[1]:.2f} | {m[2]:.2f} | {m[3]:.2f} | {m[4]:.2f} | {r:.3f} | {m[2]/m[1]:.3f} | {m[3]/m[4]:.3f} |")
