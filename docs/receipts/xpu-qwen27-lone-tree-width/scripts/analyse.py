#!/usr/bin/env python3
"""Tables from the pass files. usage: analyse.py RUNS_DIR   (files <pass>-<variant>.jsonl, and -sampled.jsonl)"""
import collections, glob, json, os, statistics as st, sys

R = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "p*.jsonl"))):
    name = os.path.basename(f)[:-6]
    sampled = name.endswith("-sampled")
    name = name.replace("-sampled", "")
    p, variant = name.split("-", 1)
    for l in open(f):
        r = json.loads(l)
        r["pass"], r["variant"] = int(p[1:]), variant
        R.append(r)
DEPTHS = sorted({r["depth"] for r in R})
names = {0: "34", 32768: "28,946", 65536: "58,543", 131072: "121,022"}

print("## Tokens: distinct token_sha256 per prompt and mode\n")
print("| prompt tokens | mode | requests | distinct hashes | variants |\n|---|---|---|---|---|")
bad = 0
for d in DEPTHS:
    for mode in ("draft", "plain", "sampled"):
        rs = [r for r in R if r["depth"] == d and r["mode"] == mode]
        if not rs:
            continue
        sh = {r["token_sha256"] for r in rs}
        bad += len(sh) != 1
        print(f"| {names.get(d, d)} | {mode} | {len(rs)} | {len(sh)} | {len({r['variant'] for r in rs})} |")
print(f"\nCELLS WITH MORE THAN ONE HASH: {bad}")
# greedy: drafted and plain are one hash per prompt together
for d in DEPTHS:
    sh = {r["token_sha256"] for r in R if r["depth"] == d and r["mode"] in ("draft", "plain")}
    print(f"greedy {names.get(d, d)} (draft + plain together): {len(sh)} hash(es)")

print("\n## Decode tok/s, median of a pass's three measured requests, mean of the variant's passes (passes in brackets)\n")
cols = ["main-t15", "main-t7", "main-t3", "pr", "plain"]
print("| prompt tokens | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
per = collections.defaultdict(list)
for r in R:
    if r["warmup"] or not r["decode_tok_s"] or r["mode"] == "sampled":
        continue
    v = "plain" if r["mode"] == "plain" else r["variant"]
    per[(r["depth"], v, r["pass"])].append(r["decode_tok_s"])
best = {}
for d in DEPTHS:
    cells = []
    for c in cols:
        ps = sorted(k[2] for k in per if k[0] == d and k[1] == c)
        meds = [st.median(per[(d, c, p)]) for p in ps]
        if meds:
            best[(d, c)] = st.mean(meds)
        cells.append("%.2f (%s)" % (st.mean(meds), ", ".join("%.2f" % m for m in meds)) if meds else "-")
    print(f"| {names.get(d, d)} | " + " | ".join(cells) + " |")

print("\n## PR default against the best fixed tree (unrounded means)\n")
print("| prompt tokens | best fixed | its tok/s | PR default tok/s | PR / best fixed | PR / plain |\n|---|---|---|---|---|---|")
for d in DEPTHS:
    fixed = {c: best[(d, c)] for c in cols[:3] if (d, c) in best}
    if (d, "pr") not in best or not fixed:
        continue
    b = max(fixed, key=fixed.get)
    pl = best.get((d, "plain"))
    print(f"| {names.get(d, d)} | {b} | {fixed[b]:.2f} | {best[(d, 'pr')]:.2f} | {best[(d, 'pr')] / fixed[b]:.3f} | {best[(d, 'pr')] / pl:.3f} |" if pl else "")

print("\n## Drafts verified a round (drafted / rounds) and accepted a round, mean of the measured requests\n")
print("| prompt tokens | variant | nodes a round | accepted a round |\n|---|---|---|---|")
for d in DEPTHS:
    for c in cols[:4]:
        xs = [r["speculative"] for r in R if r["depth"] == d and not r["warmup"] and r["mode"] == "draft" and r["variant"] == c and r.get("speculative")]
        if xs:
            print(f"| {names.get(d, d)} | {c} | {st.mean(x['drafted'] / x['rounds'] for x in xs):.2f} | {st.mean(x['accepted'] / x['rounds'] for x in xs):.2f} |")

print("\n## PR default, every request: nodes a round and tok/s (the warm-up request is rep 0)\n")
print("| prompt tokens | pass | rep | nodes a round | tok/s |\n|---|---|---|---|---|")
for r in R:
    if r["variant"] == "pr" and r["mode"] == "draft" and r.get("speculative"):
        s = r["speculative"]
        print(f"| {names.get(r['depth'], r['depth'])} | {r['pass']} | {r['rep']} | {s['drafted'] / s['rounds']:.2f} | {r['decode_tok_s']:.2f} |")
