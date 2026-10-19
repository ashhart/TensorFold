#!/usr/bin/env python3
"""The aot-fallback check's fixture: one tiny kernel, a folded and a runtime specialization, as an aot set."""
# Launches tf_aot_fallback_k with R=1 (Triton folds the 1 into a constexpr) and R=2 (a runtime parameter),

# then packs both cached cubins into the aot.json + cubins/ layout `tf-cuda-test aot-fallback` loads,

# compiled for the running device. Run inside the GPU image.

# Ours (Gustav / GustavBlack), 2026-10-18, for #1003.
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def tf_aot_fallback_k(X, OUT, R, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(OUT + i, v * R + R)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    (out / "cubins").mkdir(parents=True)

    if not torch.cuda.is_available():
        print("no CUDA device: nothing to compile", flush=True)
        return 1

    dev = "cuda"
    x = torch.arange(128, device=dev, dtype=torch.int32)
    o = torch.empty(128, device=dev, dtype=torch.int32)
    for r in (1, 2):
        tf_aot_fallback_k[(1,)](x, o, r, BLOCK=128)
        torch.cuda.synchronize()

    # classify the cache's entries by the cubin's own ABI: the folded one has no u32 parameter for R

    # (this Triton's cache json carries no signature, the PTX does)
    root = cache_root()
    entries = [p for p in root.iterdir() if (p / "tf_aot_fallback_k.cubin").is_file()] if root.is_dir() else []
    folded = [p for p in entries if ".param .u32" not in (p / "tf_aot_fallback_k.ptx").read_text()]
    runtime = [p for p in entries if p not in folded]
    assert len(folded) == 1 and len(runtime) == 1, f"expected one folded and one runtime entry, got {len(folded)}/{len(runtime)}"
    seen = {1: folded[0], 2: runtime[0]}

    kernels = []
    for r in (1, 2):
        d = seen[r]
        meta = json.loads((d / "tf_aot_fallback_k.json").read_text())
        cubin = d / "tf_aot_fallback_k.cubin"
        shutil.copy(cubin, out / "cubins" / f"{d.name}.cubin")
        consts = {"BLOCK": {"int": 128}}
        params = [
            {"name": "X", "type": "*i32", "div16": True, "nospec": False},
            {"name": "OUT", "type": "*i32", "div16": True, "nospec": False},
        ]
        if r == 1:
            consts["R"] = {"int": 1}
        else:
            params.append({"name": "R", "type": "i32", "div16": False, "nospec": False})
        kernels.append({
            "fn": "tf_aot_fallback_k",
            "hash": d.name,
            "name": meta["name"],
            "num_warps": meta["num_warps"],
            "num_ctas": meta.get("num_ctas", 1),
            "shared": meta.get("shared", 0),
            "global_scratch": 0,
            "global_align": 1,
            "profile_scratch": 0,
            "pdl": bool(meta.get("launch_pdl", False)),
            "params": params,
            "consts": consts,
        })
    (out / "aot.json").write_text(json.dumps({"kernels": kernels}) + "\n")
    print(f"aot-fallback fixture: {len(kernels)} variants in {out}", flush=True)
    return 0


def cache_root() -> Path:
    env = os.environ.get("TRITON_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".triton" / "cache"


if __name__ == "__main__":
    raise SystemExit(main())
