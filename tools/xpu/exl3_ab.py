# SPDX-License-Identifier: Apache-2.0
"""Interleaved A/B of exl3.cl variants: each round runs every variant on every case in turn; median over rounds.
Usage: exl3_ab.py <sage dir> <rounds> name=repo:NS[:hs][:ksN][:opts] ...   (one GPU job at a time)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import exl3_ref as ref, exl3_check as ck

sage, rounds = sys.argv[1], int(sys.argv[2])
ROWS = tuple(int(r) for r in os.environ.get("AB_ROWS", "1,8").split(","))
variants = []
for spec in sys.argv[3:]:
    name, rest = spec.split("=", 1)
    repo, ns, *opt = rest.split(":")
    e = ck.Exl3(repo); e.ns_fixed = int(ns); e.hs_fixed = int("hs" in opt)
    e.ks_fixed = next((int(o[2:]) for o in opt if o.startswith("ks")), None)
    e.extra_opts = " ".join(o for o in opt if o != "hs" and not o.startswith("ks"))
    variants.append((name, e))
st = ref.SafeTensors(sage)
rng = np.random.default_rng(6)
cases = []
LAYERS = int(os.environ.get("AB_LAYERS", "40"))  # 1: one layer's top-8, the size of a real decode launch
for proj in ("gate_proj", "down_proj"):
    cases.append((f"experts {proj} 8x{LAYERS}", [st.group(f"model.language_model.layers.{l}.mlp.experts.{x}.{proj}") for l in range(LAYERS) for x in range(8)]))
cases.append(("lm_head", [st.group("lm_head")]))
res = {}
for rnd in range(rounds):
    for label, groups in cases:
        for rows in ROWS:
            xs = [ck.to_bf16(rng.standard_normal((rows, g[0].shape[0] * 16)).astype(np.float32)) for g in groups]
            y0 = None
            for name, e in variants:
                y, t = ck.project(e, groups, xs, rows, reps=20)
                if rnd == 0:  # every variant must give the same bits as the first
                    yb = np.concatenate([np.asarray(a).ravel() for a in y]).view(np.uint32)
                    if y0 is None: y0 = yb
                    else: print(f"{label} rows={rows} {name}: bit-exact vs first: {(yb == y0).all()}", flush=True)
                res.setdefault((label, rows, name), []).append(t[1])
for label, groups in cases:
    tb = sum(g[0].nbytes for g in groups)
    for rows in ROWS:
        line = f"{label:24s} rows={rows:2d}"
        for name, _ in variants:
            ts = res[(label, rows, name)]
            line += f" | {name} {tb/np.median(ts)/1e9:6.1f} GB/s (range {tb/max(ts)/1e9:.0f}-{tb/min(ts)/1e9:.0f})"
        print(line)
