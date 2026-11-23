#!/usr/bin/env python3
"""Compare native and MLX. python fid_compare.py mlx.json native.json"""
import json, sys, math, statistics as st
M, N = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
print("Teacher-forced: MLX's greedy ids forced position by position; top-1 agreement, forced-token logprob difference")
print("| prompt | prompt tokens | top-1 agree | mean abs dlogprob (agreeing) | max abs dlogprob | mismatches: j, MLX top1-top2 gap, native-vs-MLX lp of MLX token |")
print("|---|---|---|---|---|---|")
tot = agree = 0; gaps = []; posbins = [0, 0, 0, 0]; posmis = [0, 0, 0, 0]
for name, m in M.items():
    if name not in N: continue
    s = N[name]["steps"]; assert [x["j"] for x in s] == list(range(256)), name
    diffs = [abs(x["lp"] - m["logprob"][x["j"]]) for x in s if x["agree"]]
    mis = [x for x in s if not x["agree"]]
    desc = []
    for x in mis:
        j = x["j"]; g = m["top5"][j][0][1] - m["top5"][j][1][1]
        gaps.append(g)
        desc.append(f"{j}: {g:.3f}, " + ("n/a" if x["lp"] is None else f"{x['lp']-m['logprob'][j]:+.3f}"))
    for x in s:
        b = min(3, x["j"] // 64); posbins[b] += 1; posmis[b] += (not x["agree"])
    a = 256 - len(mis); tot += 256; agree += a
    print(f"| {name} | {len(m['prompt_ids'])} | {a}/256 ({100*a/256:.1f} %) | {st.mean(diffs):.4f} | {max(diffs):.4f} | {'; '.join(desc) or '-'} |")
print(f"\nall: {agree}/{tot} = {100*agree/tot:.2f} %; mismatches by 64-token block of the continuation: {posmis} of {posbins}")
if gaps: print(f"MLX top1-top2 gap at mismatches (nats): median {st.median(gaps):.3f}, max {max(gaps):.3f}, <0.1: {sum(g<0.1 for g in gaps)} of {len(gaps)}")
print("\nFree greedy (first request per prompt, no forcing), native against MLX")
print("| prompt | first divergence (j) | MLX top1-top2 gap | MLX top1 / native token |")
print("|---|---|---|---|")
for name, m in M.items():
    if name not in N: continue
    s = N[name]["steps"]; f = next((x for x in s if not x["agree"]), None)
    if f is None: print(f"| {name} | none in 256 | - | - |"); continue
    j = f["j"]; print(f"| {name} | {j} | {m['top5'][j][0][1]-m['top5'][j][1][1]:.3f} | {N[name]['strs'][j]!r} / {f['native']!r} |")
