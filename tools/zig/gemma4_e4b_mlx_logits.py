"""Oracle C for Gemma 4 E4B: mlx-lm on a checkpoint, the reference's tokens forced, logits in the reference's format."""
# Writes OUT/<name>.logits.npy [steps, 262144] fp32 as gemma4_e4b_reference.py did, for gemma4_e4b_compare.py.

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("ref", type=Path, help="the reference directory (prompt and generated ids)")
    p.add_argument("out", type=Path)
    p.add_argument("--only")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model, _ = load(str(args.model))
    only = set(args.only.split(",")) if args.only else None
    for f in sorted(args.ref.glob("*.json")):
        meta = json.loads(f.read_text())
        if not isinstance(meta, dict) or "prompt_ids" not in meta or (only and meta["name"] not in only):
            continue
        ids, gen = meta["prompt_ids"], meta["generated_ids"]
        t0 = time.time()
        cache = model.make_cache()
        x = mx.array([ids + gen[:-1]])
        rows = []
        chunk = 1024
        for at in range(0, x.shape[1], chunk):
            out = model(x[:, at : at + chunk], cache=cache)
            mx.eval(out)
            lo = max(len(ids) - 1 - at, 0)
            if at + out.shape[1] > len(ids) - 1:
                rows.append(np.array(out[0, lo:].astype(mx.float32)))
        logits = np.concatenate(rows)[: len(gen)]
        np.save(args.out / f"{meta['name']}.logits.npy", logits.astype(np.float32))
        print(f"{meta['name']:16s} {len(ids)} + {len(gen)} tokens {time.time() - t0:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
