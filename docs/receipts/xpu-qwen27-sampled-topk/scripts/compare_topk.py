#!/usr/bin/env python3
"""usage: compare_topk.py RUNS_DIR   exactness (token_sha256 per cell and stream) and speed (A-B-B-A pairs) of the sampled top-k."""
import glob, json, os, re, statistics as st, sys
from collections import defaultdict

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "p*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        m = re.match(r"p(\d)-(main|branch)-par(\d+)", r["label"])
        r["pass"], r["side"] = int(m.group(1)), m.group(2)
        rows.append(r)

# (a) exactness: one token_sha256 for a (mode, parallel, depth, kind, stream) over every pass, side and wave
key = lambda r: (("greedy" if r["greedy"] else "sampled"), r["parallel"], r["depth"], r["kind"], r["stream"])
sha = defaultdict(lambda: defaultdict(set))
for r in rows:
    if not r["ok"]:
        print("NOT OK:", r["label"], r["depth"], r["kind"], r["stream"], r.get("error"))
        continue
    sha[key(r)]["all"].add(r["token_sha256"])
    sha[key(r)][r["side"]].add(r["token_sha256"])
    sha[key(r)]["n"].add(r["completion_tokens"])
bad_cells = [k for k, v in sha.items() if len(v["all"]) != 1]
print(f"streams checked: {len(sha)} (mode, streams, depth, kind, stream); requests: {len(rows)}")
print("MISMATCH main/branch/passes/waves (distinct token_sha256 != 1):", len(bad_cells))
for k in bad_cells:
    print("  ", k, {s: len(v) for s, v in sha[k].items()})
short = [k for k, v in sha.items() if v["n"] != {256}]
print("streams not at 256 tokens:", len(short), [(k, sorted(sha[k]["n"])) for k in short][:5])
for mode in ("sampled", "greedy"):
    ks = [k for k in sha if k[0] == mode]
    print(f"{mode}: {len(ks)} streams, {len({next(iter(sha[k]['all'])) for k in ks if len(sha[k]['all']) == 1})} distinct hashes")

# (c) a stream drawn in a shared round equals the same request (seed, prompt) drawn alone or in a round of another width
alone = defaultdict(set)
for r in rows:
    if r["ok"]:
        alone[(("greedy" if r["greedy"] else "sampled"), r["depth"], r["kind"], r["stream"])].add(r["token_sha256"])
diff = [k for k, v in alone.items() if len(v) != 1]
print("MISMATCH across 1, 4 and 8 streams (same prompt and seed, distinct token_sha256 != 1):", len(diff), "of", len(alone), "(prompt, seed) pairs")

# (b) speed: per pass, cell (parallel, depth, kind): per-stream decode tok/s (mean of the wave's streams, median of the 3 measured
# waves) and the wave's aggregate (completion tokens over the wave's wall time, prefill included)
cell = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["greedy"] or r["warmup"] or not r["ok"]:
        continue
    cell[(r["parallel"], r["depth"], r["kind"])][(r["pass"], r["side"], r["rep"])].append(r)
speed = {}
for c, waves in cell.items():
    per = defaultdict(lambda: ([], [], []))
    for (p, side, rep), rs in waves.items():
        mean = sum(r["decode_tok_s"] for r in rs) / len(rs)
        agg = sum(r["completion_tokens"] for r in rs) / rs[0]["wave_wall_s"]
        per[(p, side)][0].append(mean)
        per[(p, side)][1].append(agg)
        per[(p, side)][2].append(rs[0]["wave_wall_s"])
    for (p, side), (m, a, w) in per.items():
        speed[(c, p, side)] = (st.median(m), st.median(a), st.median(w))
print()
print("cell (streams, prompt tokens, kind) | per-stream tok/s main / branch = ratio (pairs) | aggregate tok/s main / branch = ratio (pairs)")
for c in sorted({k[0] for k in speed}):
    def pair(idx):
        # passes 1 (main) 2 (branch) 3 (branch) 4 (main): pairs (1,2) and (4,3)
        m = [speed[(c, 1, "main")][idx], speed[(c, 4, "main")][idx]]
        b = [speed[(c, 2, "branch")][idx], speed[(c, 3, "branch")][idx]]
        r1, r2 = b[0] / m[0], b[1] / m[1]
        mm, bm = sum(m) / 2, sum(b) / 2
        return f"{mm:.2f} / {bm:.2f} = {bm / mm:.3f} ({r1:.3f}, {r2:.3f}; gap {abs(r1 - r2):.3f})"
    print(f"{c[0]} streams, {c[1] or 'short'} {c[2]}: {pair(0)} | {pair(1)}")
