#!/usr/bin/env python3
"""Prompt-row MLA attention (fused._attn_chunks + latent merge) at GLM-5.3's rank shapes: 16 heads, 512 latent + 64
rope, per-row key lists of 2048. Times tile settings for dense rows (p < 2048) and sparse rows (p >= 2048)."""
import itertools
import sys

import torch
import triton

sys.path.insert(0, "/tf/src")
from tensorfold.families.glm_moe_dsa.cuda import fused  # noqa: E402
from tensorfold.families.glm5_next.cuda import latent  # noqa: E402

H, LW, RD, K = 16, 512, 64, 2048


def run(R, P, T, CHK, KT, warps, stages, iters=5):
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(0)
    qa = (torch.randn(R, H, LW, generator=g) * 0.1).to(dev, torch.bfloat16)
    qr = (torch.randn(R, H, RD, generator=g) * 0.1).to(dev, torch.bfloat16)
    lc = (torch.randn(T, LW + RD, generator=g) * 0.1).to(dev, torch.bfloat16)
    if P >= K:
        tok = torch.sort(torch.stack([torch.randperm(P, generator=g)[:K] for _ in range(R)]), dim=-1).values
    else:
        tok = torch.arange(K).repeat(R, 1)
    tok = tok.to(dev, torch.int32)
    pos = torch.tensor([P], dtype=torch.int32, device=dev)
    nch = K // CHK
    po = torch.empty((nch * R * H * LW,), device=dev)
    pm = torch.empty((nch * R * H,), device=dev)
    pl = torch.empty((nch * R * H,), device=dev)
    out = torch.empty((R, H, LW), dtype=torch.bfloat16, device=dev)
    dummy = torch.zeros((1,), dtype=torch.int32, device=dev)

    def step():
        fused._attn_chunks[(R, nch)](qa, qr, lc, tok, pos, po, pm, pl, R, H=H, LW=LW, RD=RD, K=K, CHK=CHK, KTT=KT,
                                     SCALE=0.0625, num_warps=warps, num_stages=stages)
        latent._merge[(R, H)](po, pm, pl, out, dummy, R, H=H, LW=LW, NCH=nch, SPARSE=False, num_warps=4)
    step()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        step()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters, out


def main():
    for label, P, R in (("dense rows (p<2048)", 0, 2048), ("decode window sparse", 8000, 3),
                        ("decode window dense", 1500, 3)):
        ref = None
        res = []
        for CHK, KT, warps, stages in itertools.product((256, 512, 1024), (32, 64), (4, 8), (1, 2, 3)):
            if KT > CHK or (R < 16 and KT == 64 and CHK == 1024):
                continue
            try:
                ms, out = run(R, P, 10240, CHK, KT, warps, stages, iters=5 if R > 16 else 200)
            except Exception as exc:                     # noqa: BLE001  resources a tiling cannot fit
                res.append((1e9, CHK, KT, warps, stages, type(exc).__name__))
                continue
            if ref is None:
                ref = out.float()
            err = float((out.float() - ref).abs().max())
            res.append((ms, CHK, KT, warps, stages, f"maxdiff {err:.2e}"))
        res.sort()
        print(f"== {label}, R={R}: best first (current: CHK 512, KT 32, warps 8, stages 1)")
        for ms, CHK, KT, warps, stages, note in res[:8]:
            print(f"  {ms:8.3f} ms/layer  CHK {CHK:4d} KT {KT:2d} warps {warps} stages {stages}  {note}")
        cur = [r for r in res if r[1:5] == (512, 32, 8, 1)]
        if cur:
            print(f"  current: {cur[0][0]:.3f} ms/layer")


if __name__ == "__main__":
    main()
