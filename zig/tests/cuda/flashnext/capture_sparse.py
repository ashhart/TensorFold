#!/usr/bin/env python3
"""Flash Next's sparse attention kernels and _select fixtures on synthetic tensors, no checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", required=True, help="folder of triton_aot_manifest.py")
    ap.add_argument("--fixtures", default="", help="folder for the selection fixtures (empty: none)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.tools)
    import triton_aot_manifest as aot

    rec = aot.Recorder().install()
    import torch
    import triton
    from tensorfold.families.qwen4_exp.cuda import attention as A
    from tensorfold.families.qwen4_exp.cuda import glue, kvcache

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    heads, kv, hd, ih, idim, ratio = 24, 2, 256, 4, 128, 4
    cap = 8192                                   # nb = 2048: NB % 16 == 0, as the Zig scratch keeps it
    inv = (10_000_000.0 ** (-torch.arange(32, dtype=torch.float64) / 32)).to(torch.float32).to(dev)
    with rec.scope("sparse"):
        for rows in (1, 17, 32):
            sc = A.AttnScratch(rows, heads, hd, cap, dev)
            assert sc.qsa and sc.nch == 5 and sc.nb % 16 == 0
            cache = kvcache.KVCache(cap, kv, hd, dev, "bf16")
            cache.k.copy_(torch.randn(cache.k.shape, device=dev, generator=g).to(torch.bfloat16))
            cache.v.copy_(torch.randn(cache.v.shape, device=dev, generator=g).to(torch.bfloat16))
            ikc = torch.randn((cap, idim), device=dev, generator=g).to(torch.bfloat16)
            pooled = torch.zeros((sc.nb, idim), dtype=torch.bfloat16, device=dev)
            w = torch.rand((idim,), device=dev, generator=g) + 0.5
            # every block before the window pooled first (one row at a time would do it the same way)
            pos0 = torch.zeros((1,), dtype=torch.int32, device=dev)
            p0 = 3001
            A.qsa_pool(ikc, pooled, pos0, w, inv, 1e-6, sc, p0 - p0 % 32) if rows != 1 else None
            pos0.fill_(p0)
            A.qsa_pool(ikc, pooled, pos0, w, inv, 1e-6, sc, rows)
            iq = torch.randn((rows, ih, idim), device=dev, generator=g).to(torch.bfloat16)
            q = torch.randn((rows, heads, hd), device=dev, generator=g).to(torch.bfloat16)
            ends = p0 + rows
            blocks = min(sc.nb, max(1, triton.cdiv(ends, ratio)))
            A._scores[(rows, triton.cdiv(blocks, 64))](iq, pooled, pos0, sc.scores, sc.nb, HI=ih, DI=idim,
                                                      RATIO=ratio, TOP=sc.budget // ratio, BB=64, num_warps=4)
            A._launch_select(sc, pos0, rows, blocks)
            A.attention(q, cache.k, cache.v, pos0, sc, rows, hd ** -0.5, context=ends, ks=cache.ks, vs=cache.vs)
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
    if a.fixtures:
        fixtures(Path(a.fixtures), A, torch, triton, g)
    print("done", flush=True)
    return 0


def fixtures(fx: Path, A, torch, triton, g) -> None:
    """Score rows (few distinct values: ties everywhere) and the lists _select / _select_tiles make of them."""

    fx.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    cases = []
    # (capacity, pos0, rows, distinct score values): register select, then rows past SELECT_REGS blocks (tiles)
    for i, (cap, p0, rows, levels) in enumerate([(8192, 2040, 24, 7), (8192, 5000, 9, 0), (65536, 40000, 5, 3),
                                                 (262144, 150000, 3, 11), (262144, 262000, 2, 0)]):
        sc = A.AttnScratch(rows, 24, 256, cap, dev)
        if levels:
            s = torch.randint(0, levels, (rows, sc.nb), device=dev, generator=g).float() / 7.0
        else:
            s = torch.rand((rows, sc.nb), device=dev, generator=g) * 3.0
        s[:, ::97] = 0.0                       # zeros among them, as relu sums give
        sc.scores.copy_(s)
        sc.ids.fill_(-1)
        pos0 = torch.full((1,), p0, dtype=torch.int32, device=dev)
        blocks = min(sc.nb, max(1, triton.cdiv(p0 + rows, 4)))
        A._launch_select(sc, pos0, rows, blocks)
        torch.cuda.synchronize()
        name = f"case{i}"
        sc.scores.cpu().numpy().tofile(fx / f"{name}.scores")
        sc.ids.cpu().numpy().tofile(fx / f"{name}.ids")
        sc.nk.cpu().numpy().tofile(fx / f"{name}.nk")
        sc.sparse.cpu().numpy().tofile(fx / f"{name}.sparse")
        cases.append({"name": name, "rows": rows, "pos0": p0, "nb": sc.nb, "idw": sc.idw})
    (fx / "cases.json").write_text(json.dumps(cases, indent=1) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
