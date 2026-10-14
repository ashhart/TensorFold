"""GLM-5.3-Flash gap fixtures on real weights ("gaps"): 1536/512 rmsnorms, fp32-y combine, MoE per-slot y."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def save(out: Path, name: str, t: torch.Tensor) -> None:
    (out / f"{name}.bin").write_bytes(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def write(out: Path, model_dir: Path) -> dict:
    from tensorfold.cuda.exl3 import experts as generic
    from tensorfold.cuda.exl3.linear import Exl3Linear
    from tensorfold.families.glm5_next.cuda import exl3_generic, glue
    from tensorfold.families.glm5_next.cuda.split import RankReader

    dev = "cuda"
    L, EPS, LIM, SCALE, TOPK, NE, D, I = 3, 1e-5, 10.0, 2.5, 8, 288, 4096, 1024
    P = f"model.language_model.layers.{L}.mlp."
    rd = RankReader(model_dir, 0)
    g = torch.Generator(device="cpu").manual_seed(70707)
    man: dict = {"layer": L, "eps": EPS}

    def as_int16(t: torch.Tensor) -> torch.Tensor:
        return t.view(torch.int16) if t.element_size() == 4 else t

    def lin_mm(lin: Exl3Linear, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """forward.py exl3_mm: the dense kernel takes 1..128 rows and is row-invariant."""
        m = x.shape[0]
        if m <= 128:
            return lin(x, out=out)
        o = torch.empty((min(m, 128), lin.n), dtype=out.dtype, device=out.device)
        for r in range(0, m, 128):
            n = min(m - r, 128)
            lin(x[r:r + n], out=o[:n])
            out[r:r + n].copy_(o[:n])
        return out
    MP = "model.language_model."
    nw1 = rd.get(MP + f"layers.{L}.self_attn.q_a_layernorm.weight").to(torch.bfloat16).to(dev)
    nw2 = rd.get(MP + f"layers.{L}.self_attn.kv_a_layernorm.weight").to(torch.bfloat16).to(dev)
    rows = 4
    if tuple(nw1.shape) != (1536,) or tuple(nw2.shape) != (512,):
        raise ValueError(f"layernorm widths {tuple(nw1.shape)} {tuple(nw2.shape)}")
    dp = torch.randn((rows, 2048), generator=g).to(torch.bfloat16).to(dev)
    out1 = torch.empty((rows, 1536), dtype=torch.bfloat16, device=dev)
    sums1 = torch.empty((rows, 24), dtype=torch.float32, device=dev)
    glue.rmsnorm(dp[:, :1536], nw1, EPS, out1, sums1)
    out2 = torch.empty((rows, 512), dtype=torch.bfloat16, device=dev)
    sums2 = torch.empty((rows, 8), dtype=torch.float32, device=dev)
    glue.rmsnorm(dp[:, 1536:], nw2, EPS, out2, sums2)
    for name, t in (("rms_dp", dp), ("rms_nw1", nw1), ("rms_nw2", nw2), ("rms_out1", out1),
                    ("rms_sums1", sums1), ("rms_out2", out2), ("rms_sums2", sums2)):
        save(out, name, t)
    man["rms"] = {"rows": rows, "d1": 1536, "d2": 512, "eps": EPS, "dp": "rms_dp.bin", "nw1": "rms_nw1.bin",
                  "nw2": "rms_nw2.bin", "out1": "rms_out1.bin", "sums1": "rms_sums1.bin",
                  "out2": "rms_out2.bin", "sums2": "rms_sums2.bin"}

    # -- glm.glue.combine: the fp32 per-slot outputs (the EXL3 path's ey is scratch.y, fp32) -------------
    yy = torch.randn((rows * 9, D), generator=g).to(torch.float32).to(dev)
    wts = torch.rand((rows, 9), generator=g).to(torch.float32).to(dev)
    cout = torch.empty((rows, D), dtype=torch.float32, device=dev)
    glue.combine(yy, wts, cout)
    for name, t in (("c_y32", yy), ("c_wts32", wts), ("c_out32", cout)):
        save(out, name, t)
    man["combine"] = {"rows": rows, "d": D, "y": "c_y32.bin", "wts": "c_wts32.bin", "out": "c_out32.bin"}

    # -- the real MoE layer: 288 routed experts, the router, the shared expert, per-slot y and combine ---
    moe: dict = {"layer": L, "experts": NE, "dims": D, "width": I, "codebook": "mul1", "limit": LIM,
                 "act_mode": 0, "slots": TOPK + 1}
    gate, up, down = [], [], []
    trellis: dict[str, list[str]] = {"gate": [], "up": [], "down": []}
    for e in range(NE):
        for part, short, mats in (("gate_proj", "gate", gate), ("up_proj", "up", up), ("down_proj", "down", down)):
            mats.append((as_int16(rd.get(P + f"experts.{e}.{part}.trellis")).to(dev),
                         rd.get(P + f"experts.{e}.{part}.suh").to(dev),
                         rd.get(P + f"experts.{e}.{part}.svh").to(dev)))
            trellis[short].append(f"moe_{short}{e:03d}.bin")
            save(out, f"moe_{short}{e:03d}", mats[-1][0])
    ex = generic.prepare(gate, up, down, "mul1", device=dev)
    moe["gate_k2"] = [int(v) for v in ex.gate_k2.cpu()]
    moe["up_k2"] = [int(v) for v in ex.up_k2.cpu()]
    moe["down_k2"] = [int(v) for v in ex.down_k2.cpu()]
    moe["gate"] = trellis["gate"]
    moe["up"] = trellis["up"]
    moe["down"] = trellis["down"]
    for name, t in (("suh_g", ex.suh_g), ("suh_u", ex.suh_u), ("svh_g", ex.svh_g), ("svh_u", ex.svh_u),
                    ("suh_d", ex.suh_d), ("svh_d", ex.svh_d)):
        save(out, f"moe_{name}", t)
        moe[name] = f"moe_{name}.bin"
    gate_w = rd.get(P + "gate.weight").to(torch.bfloat16).to(dev)
    bias = rd.get(P + "gate.e_score_correction_bias").to(torch.float32).to(dev)
    if tuple(gate_w.shape) != (NE, D) or tuple(bias.shape) != (NE,):
        raise ValueError(f"router shapes {tuple(gate_w.shape)} {tuple(bias.shape)}")
    save(out, "moe_gate_w", gate_w)
    save(out, "moe_bias", bias)
    moe["gate_w"] = "moe_gate_w.bin"
    moe["bias"] = "moe_bias.bin"
    s_lin = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        s_lin[proj] = Exl3Linear.from_tensors(
            as_int16(rd.get(P + f"shared_experts.{proj}.trellis")),
            rd.get(P + f"shared_experts.{proj}.suh").to(torch.float16).to(dev),
            rd.get(P + f"shared_experts.{proj}.svh").to(torch.float16).to(dev), "mul1", device=dev)
    if s_lin["gate_proj"].n != I or s_lin["up_proj"].n != I or s_lin["down_proj"].k != I:
        raise ValueError(f"shared expert widths {s_lin['gate_proj'].n} {s_lin['up_proj'].n} {s_lin['down_proj'].k}")
    xm, pm, wm, ym, sym, partm, hy, hp = {}, {}, {}, {}, {}, {}, {}, {}
    for m in (1, 3, 16, 300):
        x = torch.randn((m, D), generator=g).to(torch.bfloat16).to(dev)
        logits = torch.empty((m, NE), dtype=torch.float32, device=dev)
        glue.router(x, gate_w, logits)
        pick = torch.empty((m, TOPK + 1), dtype=torch.int32, device=dev)
        wts = torch.empty((m, TOPK + 1), dtype=torch.float32, device=dev)
        glue.select(logits, bias, pick, wts, TOPK, NE, SCALE, True)
        scratch = generic.Scratch(ex, m, TOPK + 1)
        y = exl3_generic.routed(x, pick, ex, scratch, m, LIM).clone()          # [m * 9, D] fp32
        smg = torch.empty((m, I), dtype=torch.bfloat16, device=dev)
        smu = torch.empty((m, I), dtype=torch.bfloat16, device=dev)
        lin_mm(s_lin["gate_proj"], x, smg)
        lin_mm(s_lin["up_proj"], x, smu)
        sgu = torch.cat((smg, smu), dim=1)
        sact = torch.empty((m, I), dtype=torch.bfloat16, device=dev)
        glue.swiglu(sgu, sact, torch.empty((m,), dtype=torch.float32, device=dev), LIM)
        sy = lin_mm(s_lin["down_proj"], sact, torch.empty((m, D), dtype=torch.float32, device=dev))
        y.view(m, TOPK + 1, D)[:, TOPK] = sy
        part = torch.empty((m, D), dtype=torch.float32, device=dev)
        glue.combine(y, wts, part)
        k = str(m)
        for name, t, bag in (("x", x, xm), ("pick", pick, pm), ("wts", wts, wm), ("y", y, ym),
                             ("sy", sy, sym), ("part", part, partm)):
            save(out, f"moe_{name}{m}", t)
            bag[k] = f"moe_{name}{m}.bin"
        hy[k], hp[k] = sha(y), sha(part)
        del scratch
    moe.update({"rows": [1, 3, 16, 300], "x": xm, "pick": pm, "wts": wm, "y": ym, "sy": sym, "part": partm,
                "hash": {"y": hy, "part": hp}})
    man["moe"] = moe
    return man
