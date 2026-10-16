#!/usr/bin/env python3
"""The decode kernels Flash Next's batched prompt pass runs at M > 1 (M 17 and 32), no checkpoint."""

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
    from tensorfold.cuda import moe
    from tensorfold.families.qwen4_exp.cuda import glue, qmm

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    streams, dims, low = 4, 2560, 320
    wide = streams * dims

    def q4(n: int, k: int):
        w = torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int32, device=dev, generator=g)
        s = (torch.rand((n, k // 32), device=dev, generator=g) * 0.01).to(torch.bfloat16)
        b = (torch.rand((n, k // 32), device=dev, generator=g) * 0.01 - 0.005).to(torch.bfloat16)
        return qmm.make_q4(w, s, b, layout="tiled")

    downs = {n: q4(n, wide) for n in (low + streams, low)}
    up = q4(wide, low)
    router_rows = (torch.randn((513, dims), device=dev, generator=g) * 0.02).to(torch.bfloat16)
    with rec.scope("rows"):
        for m in (17, 32):
            h = torch.randn((m, wide), device=dev, generator=g).to(torch.bfloat16)
            pss = torch.rand((m, dims // 256, streams), device=dev, generator=g) * 100
            scale = torch.rand((wide,), device=dev, generator=g)
            normed = torch.empty((m, wide), dtype=torch.bfloat16, device=dev)
            act = torch.empty((m, low), dtype=torch.bfloat16, device=dev)
            xs_act = torch.empty((m, low // 32), dtype=torch.float32, device=dev)
            inj = torch.empty((m, streams), dtype=torch.bfloat16, device=dev)
            for n, q in downs.items():
                dn = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                part = torch.empty((32 * m * n,), dtype=torch.float32, device=dev)
                got = qmm.hc_down(h, pss, scale, normed, q, 1e-6, streams, out=dn, part=part)
                glue.hc_reduce_act(got, act, xs_act, inj if n == low + streams else None, streams, low)
            mixed = torch.empty((m, dims), dtype=torch.bfloat16, device=dev)
            xs_mixed = torch.empty((m, dims // 32), dtype=torch.float32, device=dev)
            qmm.hc_upmix(act, xs_act, up, normed, mixed, xs_mixed, streams)
            # the up projection alone (a training pass keeps it): qmm.matmul's tiled kernel at SHAPES16
            up_rows = torch.empty((m, wide), dtype=torch.bfloat16, device=dev)
            qmm._qmm[(-(-m // 16), wide // 64, 1)](act, xs_act, up.weight, up.scales, up.biases, up_rows, up_rows, m,
                                                    act.stride(0), N=wide, K=low, SK=1, BM=16, BLOCK_N=64, GPI=2,
                                                    F32=False, SBN=64, num_warps=4, num_stages=3)
            x = torch.randn((m, dims), device=dev, generator=g).to(torch.bfloat16)
            logits = torch.empty((m, 513), dtype=torch.float32, device=dev)
            grid = (-(-m // 16), -(-513 // 32))
            moe._router[grid](x, router_rows, logits, m, x.stride(0), D=dims, NE=513, BM=16, BLOCK_E=32, BK=256,
                              num_warps=4, num_stages=4)
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
