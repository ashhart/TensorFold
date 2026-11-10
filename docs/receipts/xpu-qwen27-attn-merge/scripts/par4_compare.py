#!/usr/bin/env python3
"""Controls and speed of the four-stream passes. usage: par4_compare.py PAR4_DIR SOLO_RUNS_DIR

PAR4_DIR holds par4-<kv>-<side>.jsonl (par4.py); SOLO_RUNS_DIR holds <kv>-<order>-<side>.jsonl (bench_merge.py).
(a) main = branch: every request of a (kv, prompt, mode) cell, both sides and all waves, has one token_sha256.
(b) repeats: on each side, the four waves of a cell (warm-up included) have one token_sha256.
(c) four streams = solo: that token_sha256 is the one of the solo passes (same prompt_sha256, kv and mode).
Speed: per wave, the four streams' tokens over the wave's wall time (first request sent to last token, prefill
included); median of the three measured waves. A stream's own tok/s is not reported: the prefills take turns, so it
depends on which stream the server serves first.
"""
import collections, glob, json, os, statistics as st, sys

P, S = sys.argv[1], sys.argv[2]
par = collections.defaultdict(list)
for f in sorted(glob.glob(f"{P}/par4-*.jsonl")):
    _, kv, side = os.path.basename(f)[:-6].split("-")
    for line in open(f):
        r = json.loads(line)
        par[(kv, r["depth"], r["kind"], r["mode"])].append(dict(r, side=side))
solo = collections.defaultdict(set)
for f in sorted(glob.glob(f"{S}/*.jsonl")):
    kv = os.path.basename(f).split("-")[0]
    for line in open(f):
        r = json.loads(line)
        solo[(kv, r["prompt_sha256"], r["mode"])].add(r["token_sha256"])
lines, bad_a, bad_b, bad_c = [], 0, 0, 0
lines += ["| KV | prompt tokens | kind | mode | requests | (a) main = branch | (b) repeats equal | (c) = solo (main / branch) |",
          "|---|---|---|---|---|---|---|---|"]
for key in sorted(par, key=lambda k: (k[0], k[1], k[2], k[3] != "draft")):
    kv, depth, kind, mode = key
    rs = par[key]
    allsha = {r["token_sha256"] for r in rs}
    a = len(allsha) == 1 and None not in allsha and all(r["ok"] for r in rs)
    b = all(len({r["token_sha256"] for r in rs if r["side"] == s}) == 1 for s in ("main", "branch"))
    c = {}
    for s in ("main", "branch"):
        ps = {r["prompt_sha256"] for r in rs}
        ref = solo.get((kv, next(iter(ps)), mode), set())
        mine = {r["token_sha256"] for r in rs if r["side"] == s}
        c[s] = len(ref) == 1 and mine == ref
    bad_a += not a
    bad_b += not b
    bad_c += not (c["main"] and c["branch"])
    pt = sorted({r["prompt_tokens"] for r in rs})[0]
    yn = lambda x: "yes" if x else "NO"
    lines.append(f"| {kv} | {pt} | {kind} | {mode} | {len(rs)} | {yn(a)} | {yn(b)} | {yn(c['main'])} / {yn(c['branch'])} |")
lines += ["", "| KV | mode | main wave s | branch wave s | main tok/s | branch tok/s | branch / main | waves per side |",
          "|---|---|---|---|---|---|---|---|"]
for kv in ("bf16", "int8"):
    for mode in ("draft", "plain"):
        w, t = {}, {}
        for s in ("main", "branch"):
            waves = collections.defaultdict(list)
            for k, rs in par.items():
                if k[0] == kv and k[3] == mode:
                    for r in rs:
                        if r["side"] == s and not r["warmup"]:
                            waves[r["rep"]].append(r)
            w[s] = st.median(x[0]["wave_wall_s"] for x in waves.values())
            t[s] = st.median(sum(r["completion_tokens"] for r in x) / x[0]["wave_wall_s"] for x in waves.values())
        lines.append(f"| {kv} | {mode} | {w['main']:.1f} | {w['branch']:.1f} | {t['main']:.2f} | {t['branch']:.2f} | "
                     f"{t['branch'] / t['main']:.3f} | {len(waves)} |")
lines.append("")
lines.append(f"MISMATCH CELLS: (a) {bad_a}, (b) {bad_b}, (c) {bad_c}")
print("\n".join(lines))
sys.exit(1 if bad_a else 0)
