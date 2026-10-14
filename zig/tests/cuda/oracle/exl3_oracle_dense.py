"""The dense prompt fixture: one real projection's 300-row prompt chunk through forward.py's exl3_mm path."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

ROWS = 300
SEED = 1235
SIZE_LIMIT = 8192  # prefer layers with N, K at or under this
PREFER = ("self_attn.qkv_proj", "shared_experts.gate_proj", "self_attn.q_b_proj")


def save(out: Path, name: str, t: torch.Tensor) -> None:
    (out / f"{name}.bin").write_bytes(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def write(out: Path, model_dir: Path) -> dict:
    from tensorfold.cuda.exl3 import format as fmt
    from tensorfold.cuda.exl3.linear import Exl3Linear

    cp = fmt.scan(model_dir)
    cands = [t for t in cp.groups.values()
             if t.bits in (4.0, 5.0, 6.0, 7.0, 8.0) and t.in_scales == "suh" and t.out_scales == "svh"]

    def pref(t):
        rank = next((i for i, pat in enumerate(PREFER) if pat in t.prefix), len(PREFER))
        return (rank, t.k * t.n)

    modest = [t for t in cands if t.k <= SIZE_LIMIT and t.n <= SIZE_LIMIT]
    t = min(modest or cands, key=pref)
    marker = cp.markers.get(f"{t.prefix}.{t.codebook}")
    if t.codebook in fmt.MARKERS and marker is not None and marker != fmt.MARKERS[t.codebook]:
        raise ValueError(f"{t.prefix}: {t.codebook} marker is 0x{marker:08X}, want 0x{fmt.MARKERS[t.codebook]:08X}")
    lin = Exl3Linear.load(model_dir, t.prefix, layout="stored")
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn((ROWS, lin.k), generator=g).to(torch.bfloat16)

    # forward.py's exl3_mm: the decode-row linear over 128-row chunks, a fresh output tile each chunk
    y = torch.empty((ROWS, lin.n), dtype=torch.bfloat16)
    o = torch.empty((min(ROWS, 128), lin.n), dtype=torch.bfloat16, device="cuda")
    for r in range(0, ROWS, 128):
        e = min(ROWS - r, 128)
        lin(x[r:r + e].cuda(), out=o[:e])
        torch.cuda.synchronize()
        y[r:r + e] = o[:e].cpu()

    lid = "dense"
    save(out, f"{lid}.trellis", lin.words)
    save(out, f"{lid}.suh", lin.suh)
    save(out, f"{lid}.svh", lin.svh)
    if lin.bias is not None:
        save(out, f"{lid}.bias", lin.bias)
    save(out, f"{lid}.x", x)
    save(out, f"{lid}.y", y)
    sha = hashlib.sha256(y.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
    entry = {"id": lid, "name": t.prefix, "bits": int(t.bits), "codebook": t.codebook,
             "k": lin.k, "n": lin.n, "rows": ROWS, "trellis": f"{lid}.trellis.bin",
             "suh": f"{lid}.suh.bin", "svh": f"{lid}.svh.bin",
             "bias": None if lin.bias is None else f"{lid}.bias.bin",
             "x": f"{lid}.x.bin", "y": f"{lid}.y.bin", "x_dtype": "bf16", "y_dtype": "bf16", "sha256": sha}
    if marker is not None:
        entry["marker"] = f"0x{marker:08X}"
    print(f"{lid}: {t.prefix} EXL3 {t.bits:g}-bit {t.codebook} K={lin.k} N={lin.n} sha256={sha[:16]}…")
    return entry
