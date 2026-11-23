#!/usr/bin/env python3
"""MLX side: greedy 256 tokens per prompt (ids, logprob of each chosen token, top-5), no EOS stop.
python fid_mlx.py CKPT prompts.json out.json [prefill_step_size] [prompt names...]"""
import json, sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
ck, pf, out = sys.argv[1:4]
step = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
only = sys.argv[5:]
model, tok = load(ck)
P = json.load(open(pf))
R = {}
for name, text in P.items():
    if only and name not in only: continue
    ids = tok._tokenizer(text)["input_ids"]
    t0 = time.perf_counter()
    toks, lps, tops = [], [], []
    for tk, lp in generate_step(mx.array(ids), model, max_tokens=256, prefill_step_size=step):
        toks.append(int(tk))
        top = mx.argsort(-lp)[:5].tolist()
        lps.append(float(lp[int(tk)]))
        tops.append([(i, float(lp[i])) for i in top])
    R[name] = {"prompt_ids": ids, "ids": toks, "logprob": lps, "top5": tops, "seconds": time.perf_counter() - t0}
    print(name, len(ids), round(R[name]["seconds"], 1), flush=True)
    json.dump(R, open(out, "w"))
