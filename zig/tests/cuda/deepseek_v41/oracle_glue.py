"""oracle.py's glue fixtures (plan, top-k, pointwise, prefill glue, window glue, mHC decode, KV store) and save."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tensorfold.families.deepseek_v41.cuda import x3gm as gm

DEV = torch.device("cuda:0")


def save(out: Path, params: dict, arrays: dict[str, torch.Tensor]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"params": params, "arrays": {}}
    for name, t in arrays.items():
        t = t.detach().contiguous().cpu()
        (out / f"{name}.bin").write_bytes(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        manifest["arrays"][name] = {"file": f"{name}.bin", "dtype": str(t.dtype).replace("torch.", ""),
                                    "shape": list(t.shape)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def plan_case(out: Path, g) -> dict:
    """x3gm.plan at full sizes (2,048 / 4,096-row blocks, 384 experts) with pairs past the table, for x3gm_plan.cu."""
    cases = [(2048, 6, 384, 64, 0.0), (4096, 6, 384, 128, 0.3), (1, 6, 384, 64, 0.0), (2048, 6, 128, 64, 0.5),
             (3, 7, 13, 64, 0.0), (700, 6, 384, 128, 0.9)]
    arrays, params = {}, {"kind": "plan", "cases": len(cases)}
    for i, (R, S, E, bm, past) in enumerate(cases):
        pk = torch.randint(0, E, (R, S), generator=g, dtype=torch.int32)
        pk[torch.rand(R, S, generator=g) < past] = E
        pk = pk.to(DEV)
        order, pe, poff, pcnt, npass = gm.plan(pk, E, bm)
        params.update({f"c{i}_E": E, f"c{i}_bm": bm, f"c{i}_P": R * S})
        arrays.update({f"c{i}_pk": pk, f"c{i}_order": order, f"c{i}_pe": pe, f"c{i}_poff": poff, f"c{i}_pcnt": pcnt,
                       f"c{i}_npass": npass})
    torch.cuda.synchronize()
    save(out, params, arrays)
    return {"cases": len(cases)}


def topk_case(out: Path, g) -> dict:
    """pick.top on a rank's vocabulary half (64,640 columns) of bf16-valued logits with many exact ties."""
    from tensorfold.families.deepseek_v41.cuda import pick
    R, C = 6, 64640
    lg = (torch.randint(-96, 97, (R, C), generator=g).float() / 16.0)          # bf16-exact, heavy ties
    lg[1] = torch.randn(C, generator=g).to(torch.bfloat16).float()               # a realistic row
    lg[2, ::7] = 6.0                                                              # one value at many columns
    lg[3] = -0.0
    lg = lg.to(DEV)
    arrays, params = {"lg": lg}, {"kind": "topk", "R": R, "C": C, "ks": 4}
    for i, k in enumerate((1, 5, 64, 1024)):
        vals, cols = pick.top(lg, k)
        arrays.update({f"vals{i}": vals, f"cols{i}": cols})
        params[f"k{i}"] = k
    torch.cuda.synchronize()
    save(out, params, arrays)
    return {"rows": R}


def pointwise_case(out: Path, g) -> dict:
    """prefill_moe's torch pointwise ops (clamps / silu / mul, adds) on edge values: past the limit, NaN, inf, -0.0."""
    R, n, limit = 7, 2304, 10.0
    gg = torch.randn(R, n, generator=g) * 8
    uu = torch.randn(R, n, generator=g) * 8
    gg[0, :6] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 100.0, -100.0])
    uu[0, :6] = torch.tensor([1.0, float("nan"), 2.0, float("inf"), -float("inf"), -0.0])
    gg, uu = gg.to(DEV), uu.to(DEV)
    act = torch.nn.functional.silu(gg.clone().clamp_(max=limit)).mul_(uu.clone().clamp_(min=-limit, max=limit))
    a = (torch.randn(R, 5120, generator=g) * 3).to(DEV)
    b = (torch.randn(R, 5120, generator=g) * 3).to(DEV)
    s32 = a.clone()
    s32 += b
    y16 = torch.empty((R, 5120), dtype=torch.bfloat16, device=DEV)
    torch.add(a, b, out=y16)
    torch.cuda.synchronize()
    save(out, {"kind": "pointwise", "R": R, "n": n, "D": 5120, "limit": limit},
         {"g": gg, "u": uu, "act": act, "a": a, "b": b, "sum32": s32, "sum16": y16})
    return {"rows": R}

def glue_case(out: Path, g) -> dict:
    """The window's torch glue on edge values: out-of-range ids, -0 / +0, NaN, inf, halfway and denormal values."""
    V, D, lo, W, n = 200, 512, 1000, 3, 7
    embed = torch.randn(V, D, generator=g).to(torch.bfloat16)
    embed[3, :4] = torch.tensor([-0.0, float("nan"), float("inf"), 1e-40]).to(torch.bfloat16)
    embed = embed.to(DEV)
    ids = torch.tensor([1003, 999, 1199, 1200, 1000, 5, 1003], dtype=torch.long, device=DEV)
    local = ids - lo                                                       # forward.embed_rows, then the bf16 send
    ok = (local >= 0) & (local < V)
    rows = embed.index_select(0, local.clamp(0, V - 1)).to(torch.float32)
    rows = torch.where(ok[:, None], rows, torch.zeros_like(rows))
    send = rows.to(torch.bfloat16)
    recv = torch.randn(W, n, D, generator=g).to(torch.bfloat16)
    recv[0, :, :8] = -0.0
    recv[1, :, :4] = 0.0
    recv[2, 0, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")]).to(torch.bfloat16)
    recv = recv.to(DEV)
    parts = recv.float()                                                   # forward.allsum (parts fp32, rank order)
    acc = parts[0].clone()
    for r in range(1, W):
        acc += parts[r]
    summed = acc.to(torch.bfloat16).repeat(1, 4).contiguous()
    x = torch.randn(n, D, generator=g) * 3
    x[0, :8] = torch.tensor([float("nan"), float("inf"), -0.0, 1e-41, 70000.0, -70000.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -9])
    x[1, :4] = torch.tensor([2 ** -20, 6.1e-5, 5.9e-8, 1.0 + 2 ** -11])
    x = x.to(DEV)
    cast = x.to(torch.bfloat16)
    kw = x.to(torch.float16).to(torch.float32)                             # pick.kit_weights
    kl = x.to(torch.bfloat16).to(x.dtype)                                  # pick.kit_logits
    gc = torch.randn(W, n, 24, generator=g).to(torch.bfloat16).to(DEV)
    cols = gc.view(W, n, 24).permute(1, 0, 2).reshape(n, -1).contiguous()  # forward.gather_cols
    src = torch.randn(5, 1100, generator=g).to(torch.bfloat16).to(DEV)[:, :1024]
    carry = src[2].to(torch.float32).contiguous()
    torch.cuda.synchronize()
    save(out, {"kind": "glue", "V": V, "D": D, "lo": lo, "W": W, "n": n, "c": 24, "ld": 1100, "row": 2, "start": 3000},
         {"embed": embed, "ids": ids, "send": send, "recv": recv, "summed": summed, "x": x, "cast": cast, "kw": kw,
          "kl": kl, "gc": gc, "cols": cols, "src": src.as_strided((5, 1100), (1100, 1)),   # the whole [5, 1100] rows (row stride 1,100)
          "carry": carry})
    return {"rows": n}

