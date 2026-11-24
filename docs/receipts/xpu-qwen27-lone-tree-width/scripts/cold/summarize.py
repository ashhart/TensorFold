#!/usr/bin/env python3
"""Summary of a cold-start series. usage: summarize.py OUTDIR  (reads OUTDIR/raw/*.jsonl, one line per cell)

Per prompt and arm: first-request decode tok/s of each fresh server in run order, median and mean, nodes a round
(drafted / rounds), TTFT, distinct token hashes. Verdict per prompt: all hashes equal, and the adaptive arm's median
against the best fixed tree's (the acceptance criteria), plus against fixed 15 alone (the main-against-head comparison).
"""
import glob, json, statistics as st, sys

OUT = sys.argv[1]
rows = [json.loads(l) for f in sorted(glob.glob(f"{OUT}/raw/*.jsonl")) for l in open(f) if l.strip()]
ARMS = ["A", "F15", "F7", "F3"]
NAMES = {"A": "adaptive (head)", "F15": "fixed 15", "F7": "fixed 7", "F3": "fixed 3"}
print(f"# Cold-start series: {len(rows)} cells, the first request of each fresh server\n")
bad = [r for r in rows if not r.get("ok") or not r.get("decode_tok_s")]
if bad:
    print("FAILED CELLS:", ", ".join(f"{r['prompt']}-{r['arm']}-s{r['server']}" for r in bad), "\n")
for prompt in ("short", "long", "p28946", "p121022"):
    cell = {a: sorted((r for r in rows if r["prompt"] == prompt and r["arm"] == a and r.get("decode_tok_s")),
                      key=lambda r: int(r["server"])) for a in ARMS}
    if not any(cell.values()):
        continue
    pt = {r["prompt_tokens"] for r in rows if r["prompt"] == prompt}
    hashes = {r["token_sha256"] for r in rows if r["prompt"] == prompt and r.get("token_sha256")}
    print(f"## {prompt} prompt ({', '.join(str(x) for x in sorted(pt))} tokens)\n")
    print("| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    med = {}
    for a in ARMS:
        c = cell[a]
        if not c:
            continue
        v = [r["decode_tok_s"] for r in c]
        med[a] = st.median(v)
        nodes = [r["nodes_per_round"] for r in c if r.get("nodes_per_round")]
        print(f"| {NAMES[a]} | {len(c)} | {', '.join(f'{x:.2f}' for x in v)} | {med[a]:.2f} | {st.mean(v):.2f} | "
              f"{(st.mean(nodes) if nodes else 0):.2f} | {st.median(r['ttft_s'] for r in c):.2f} | "
              f"{len({r['token_sha256'] for r in c})} |")
    fixed = {a: m for a, m in med.items() if a != "A"}
    print(f"\nToken hashes across every cell of this prompt: {len(hashes)} distinct "
          f"({'EQUAL' if len(hashes) == 1 else 'NOT EQUAL'}).")
    if "A" in med and fixed:
        best = max(fixed, key=fixed.get)
        worst_a = min(r["decode_tok_s"] for r in cell["A"])
        print(f"Adaptive / best fixed tree ({NAMES[best]}, {fixed[best]:.2f}): {med['A'] / fixed[best]:.3f} on the medians "
              f"({'at or above' if med['A'] >= fixed[best] else 'BELOW'}); the slowest adaptive server is "
              f"{worst_a / fixed[best]:.3f} of it.")
        if "F15" in fixed:
            print(f"Adaptive / fixed 15 (head against main's default): {med['A'] / fixed['F15']:.3f}.")
    print()
