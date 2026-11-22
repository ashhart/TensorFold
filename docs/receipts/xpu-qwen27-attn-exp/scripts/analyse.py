#!/usr/bin/env python3
"""Served table from the pass files. usage: analyse.py RUNS_DIR   (files p<N>-main.jsonl and p<N>-branch.jsonl)"""
import collections, glob, json, os, statistics as st, sys

R = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "p*.jsonl"))):
    p, side = os.path.basename(f)[:-6].split("-", 1)
    for l in open(f):
        r = json.loads(l)
        r["pass"], r["side"] = int(p[1:]), side
        R.append(r)
DEPTHS = sorted({r["depth"] for r in R})
ptok = {d: sorted({r["prompt_tokens"] for r in R if r["depth"] == d}) for d in DEPTHS}
failed = sum(not r["ok"] for r in R)
print(f"requests {len(R)}, not ok {failed}, with no token_sha256 {sum(not r['token_sha256'] for r in R)}\n")

print("## Tokens: distinct token_sha256 per cell (all four passes, drafted and plain, main and branch)\n")
print("| prompt tokens | mode | requests | distinct hashes | token_sha256 (first 16 hex) | prompt_sha256 (first 16) |\n|---|---|---|---|---|---|")
bad = 0
for d in DEPTHS:
    for mode in ("draft", "plain"):
        rs = [r for r in R if r["depth"] == d and r["mode"] == mode]
        sh = {r["token_sha256"] for r in rs}
        ps = {r["prompt_sha256"] for r in rs}
        bad += len(sh) != 1
        print(f"| {ptok[d][0]} | {mode} | {len(rs)} | {len(sh)} | {', '.join(sorted(x[:16] for x in sh))} | {', '.join(sorted(x[:16] for x in ps))} |")
print(f"\nCELLS WITH MORE THAN ONE HASH (main against branch, per mode): {bad}")
for d in DEPTHS:
    sh = {r["token_sha256"] for r in R if r["depth"] == d}
    print(f"prompt tokens {ptok[d][0]}: drafted and plain together, main and branch: {len(sh)} hash(es)")

print("\n## Decode tok/s: the median of a pass's three measured requests (warm-up request excluded)\n")
per = collections.defaultdict(list)
for r in R:
    if not r["warmup"]:
        per[(r["depth"], r["mode"], r["pass"], r["side"])].append(r["decode_tok_s"])
print("| prompt tokens | mode | main p1 | branch p2 | main p3 | branch p4 | main mean | branch mean | branch / main | p2/p1 | p4/p3 |\n|---|---|---|---|---|---|---|---|---|---|---|")
for d in DEPTHS:
    for mode in ("draft", "plain"):
        m = {p: st.median(per[(d, mode, p, "main" if p % 2 else "branch")]) for p in (1, 2, 3, 4)}
        mm, bm = (m[1] + m[3]) / 2, (m[2] + m[4]) / 2
        print(f"| {ptok[d][0]} | {'drafted' if mode == 'draft' else 'plain'} | {m[1]:.2f} | {m[2]:.2f} | {m[3]:.2f} | {m[4]:.2f} | "
              f"{mm:.2f} | {bm:.2f} | {bm / mm:.3f} | {m[2] / m[1]:.3f} | {m[4] / m[3]:.3f} |")

print("\n## Every measured request (tok/s)\n")
print("| prompt tokens | mode | pass | side | reps 1-3 tok/s |\n|---|---|---|---|---|")
for d in DEPTHS:
    for mode in ("draft", "plain"):
        for p in (1, 2, 3, 4):
            s = "main" if p % 2 else "branch"
            print(f"| {ptok[d][0]} | {mode} | {p} | {s} | {', '.join('%.2f' % x for x in per[(d, mode, p, s)])} |")
