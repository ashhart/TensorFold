#!/usr/bin/env python3
"""The Triton kernels Flash Next's MTP head adds to the main model's (its RMSNorms and _add_streams), no checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", required=True, help="folder of triton_aot_manifest.py")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.tools)
    import triton_aot_manifest as aot

    rec = aot.Recorder().install()
    import torch
    from tensorfold.families.qwen4_exp.cuda import glue

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    streams, dims = 4, 2560
    wide = streams * dims
    with rec.scope("mtp"):
        for m in (1, 7):
            emb4 = torch.randn((m, wide), device=dev, generator=g).to(torch.bfloat16)
            norm_e = torch.rand((dims,), device=dev, generator=g)
            mixed = torch.empty((m, dims), dtype=torch.bfloat16, device=dev)
            xs = torch.empty((m, dims // 32), dtype=torch.float32, device=dev)
            glue.rmsnorm(emb4[:, :dims], norm_e, 1e-6, out=mixed, xs=xs)
            h = torch.randn((m, wide), device=dev, generator=g).to(torch.bfloat16)
            norm_h = torch.rand((wide,), device=dev, generator=g)
            hn = torch.empty((m, wide), dtype=torch.bfloat16, device=dev)
            xh = torch.empty((m, wide // 32), dtype=torch.float32, device=dev)
            glue.rmsnorm(h, norm_h, 1e-6, out=hn, xs=xh)
            eo = torch.randn((m, dims), device=dev, generator=g).to(torch.bfloat16)
            hs = torch.randn((m * streams, dims), device=dev, generator=g).to(torch.bfloat16)
            hout = torch.empty((m, wide), dtype=torch.bfloat16, device=dev)
            glue.add_streams(eo, hs, hout, streams)
        torch.cuda.synchronize()
    rec.dump(out / "launches.json")
    from triton.runtime.jit import JITFunction
    import gc

    spec = {}
    for fn in gc.get_objects():
        if isinstance(fn, JITFunction):
            name = f"{fn.fn.__module__}.{fn.fn.__qualname__}"
            spec[name] = {"params": [p.name for p in fn.params],
                          "do_not_specialize": [p.name for p in fn.params if p.do_not_specialize],
                          "no_align": [p.name for p in fn.params if p.do_not_specialize_on_alignment]}
    (out / "jit.json").write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
