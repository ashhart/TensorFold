#!/usr/bin/env python3
"""MLX against itself (prefill_step_size 2048 vs 256): first divergence and logprob difference on the common prefix.
python fid_selfnoise.py mlx.json mlx.step256.json"""
import json, sys, statistics as st
A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
print("| prompt | common greedy prefix (of 256) | MLX gap at divergence | mean abs dlogprob on prefix | max |")
print("|---|---|---|---|---|")
for k in B:
    a, b = A[k], B[k]
    n = next((i for i in range(256) if a["ids"][i] != b["ids"][i]), 256)
    d = [abs(a["logprob"][i] - b["logprob"][i]) for i in range(n)]
    gap = "-" if n == 256 else f"{a['top5'][n][0][1]-a['top5'][n][1][1]:.3f}"
    print(f"| {k} | {n} | {gap} | {st.mean(d):.4f} | {max(d):.4f} |")
