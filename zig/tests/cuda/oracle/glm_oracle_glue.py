"""GLM-5.3-Flash glue fixtures from the engine's Triton kernels on real weights, for glm_glue_tests.zig ("glue")."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _load(model_dir: Path, want: dict[str, tuple]) -> dict[str, torch.Tensor]:
    """The tensors named in `want` (name -> (dtype, shape)) gathered from the checkpoint's shards."""
    from safetensors import safe_open

    found: dict[str, torch.Tensor] = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(f, framework="pt") as s:
            for k in s.keys():
                short = k.removeprefix("model.language_model.")
                if short in want and short not in found:
                    found[short] = s.get_tensor(k)
        if len(found) == len(want):
            break
    missing = set(want) - set(found)
    if missing:
        raise KeyError(f"missing checkpoint tensors: {sorted(missing)}")
    out = {}
    for name, (dt, shape) in want.items():
        t = found[name]
        assert tuple(t.shape) == shape, f"{name}: {tuple(t.shape)} != {shape}"
        out[name] = t.to(dt).contiguous().to("cuda")
    return out


def save(out: Path, name: str, t: torch.Tensor) -> None:
    (out / f"{name}.bin").write_bytes(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def write(out: Path, model_dir: Path) -> dict:
    dev = "cuda"
    L = 5  # the layer whose hyper-connection and norm weights the fixtures carry
    w = _load(model_dir, {
        f"layers.{L}.hc_attn_fn": (torch.float32, (24, 16384)),
        f"layers.{L}.hc_attn_base": (torch.float32, (24,)),
        f"layers.{L}.hc_attn_scale": (torch.float32, (3,)),
        f"layers.{L}.input_layernorm.weight": (torch.bfloat16, (4096,)),
        f"layers.{L}.mlp.gate.weight": (torch.bfloat16, (288, 4096)),
        f"layers.{L}.mlp.gate.e_score_correction_bias": (torch.float32, (288,)),
        "lm_head.weight": (torch.bfloat16, (154880, 4096)),
        "embed_tokens.weight": (torch.bfloat16, (154880, 4096)),
    })
    from tensorfold.families.glm5_next.cuda import glue, qmm

    EPS, HC_EPS, ITERS = 1e-6, 1e-6, 20
    LIM, SCALE, TOPK, NE, D = 6144.0, 2.5, 8, 144, 4096
    g = torch.Generator(device="cpu").manual_seed(4242)
    rows = 4

    def bf16(*shape) -> torch.Tensor:
        return torch.randn(shape, generator=g).to(torch.bfloat16).to(dev)

    nw = w[f"layers.{L}.input_layernorm.weight"]
    fn = w[f"layers.{L}.hc_attn_fn"]
    base, scale = w[f"layers.{L}.hc_attn_base"], w[f"layers.{L}.hc_attn_scale"]
    gate = w[f"layers.{L}.mlp.gate.weight"][:NE].contiguous()
    bias = w[f"layers.{L}.mlp.gate.e_score_correction_bias"][:NE].contiguous()
    head = w["lm_head.weight"][:4096].contiguous()
    table = w["embed_tokens.weight"][:8192].contiguous()
    # glm.glue.embed
    ids = torch.randint(0, 8192, (rows,), generator=g, dtype=torch.int32).to(dev)
    eout = torch.empty((rows, 4, D), dtype=torch.bfloat16, device=dev)
    glue.embed(ids, table, D, 4, eout)
    save(out, "glue_e_ids", ids)
    save(out, "glue_e_tab", table)
    save(out, "glue_e_out", eout)

    # glm.glue.rmsnorm
    rx = bf16(rows, D)
    rout = torch.empty_like(rx)
    rsums = torch.empty((rows, D // 64), dtype=torch.float32, device=dev)
    glue.rmsnorm(rx, nw, EPS, rout, rsums)
    save(out, "glue_r_x", rx)
    save(out, "glue_r_out", rout)
    save(out, "glue_r_sums", rsums)

    # glm.glue.hc_pre (partial dots + finish)
    hx = bf16(rows, 4 * D)
    hout = torch.empty((rows, D), dtype=torch.bfloat16, device=dev)
    xs = torch.empty((rows, D // 64), dtype=torch.float32, device=dev)
    post = torch.empty((rows, 4), dtype=torch.float32, device=dev)
    comb = torch.empty((rows, 16), dtype=torch.float32, device=dev)
    part = torch.empty((rows, 16, 32), dtype=torch.float32, device=dev)
    glue.hc_pre(hx, fn, base, scale, nw, hout, xs, post, comb, part, EPS, HC_EPS, ITERS)
    save(out, "glue_h_x", hx)
    save(out, "glue_h_out", hout)
    save(out, "glue_h_xs", xs)
    save(out, "glue_h_post", post)
    save(out, "glue_h_comb", comb)
    save(out, "glue_h_part", part)

    # glm.glue.hc_post (world partials, contiguous and held apart)
    gx = bf16(rows, 4 * D)
    gg = torch.randn((2, rows, D), generator=g).to(dev)
    gout = torch.empty_like(gx)
    glue.hc_post(gx, gout, gg, post, comb)
    save(out, "glue_p_x", gx)
    save(out, "glue_p_out", gout)
    save(out, "glue_p_g", gg)
    g0, g1 = gg[0].contiguous(), gg[1].contiguous()
    pout = torch.empty_like(gx)
    glue.hc_post_pair(gx, pout, g0, g1, post, comb)
    save(out, "glue_p_pair_out", pout)

    # glm.glue.residual (residual_add and stream_mean)
    resx = bf16(rows, D)
    resout = torch.empty_like(resx)
    glue.residual_add(resx, resout, gg)
    save(out, "glue_res_x", resx)
    save(out, "glue_res_out", resout)
    smx = bf16(rows, 4 * D)
    smout = torch.empty((rows, D), dtype=torch.bfloat16, device=dev)
    glue.stream_mean(smx, smout)
    save(out, "glue_sm_x", smx)
    save(out, "glue_sm_out", smout)

    # glm.glue.swiglu
    gu = bf16(rows, 2 * 6144)
    sout = torch.empty((rows, 6144), dtype=torch.bfloat16, device=dev)
    sxs = torch.empty((rows, 96), dtype=torch.float32, device=dev)
    glue.swiglu(gu, sout, sxs, LIM)
    save(out, "glue_s_gu", gu)
    save(out, "glue_s_out", sout)
    save(out, "glue_s_xs", sxs)

    # glm.glue.router (the sliced path: d % (8*64) == 0)
    mm_x = bf16(rows, D)
    lrout = torch.empty((rows, NE), dtype=torch.float32, device=dev)
    glue.router(mm_x, gate, lrout)
    save(out, "glue_router_x", mm_x)
    save(out, "glue_router_w", gate)
    save(out, "glue_router_out", lrout)

    # glm.glue.select (the router's logits, the real bias)
    pick = torch.empty((rows, TOPK + 1), dtype=torch.int32, device=dev)
    wts = torch.empty((rows, TOPK + 1), dtype=torch.float32, device=dev)
    glue.select(lrout, bias, pick, wts, TOPK, NE, SCALE, True)
    save(out, "glue_sel_logits", lrout)
    save(out, "glue_sel_bias", bias)
    save(out, "glue_sel_pick", pick)
    save(out, "glue_sel_wts", wts)

    # glm.glue.combine (9 slots: the picked experts' outputs + the shared one)
    yy = bf16(rows * (TOPK + 1), D)
    cout = torch.empty((rows, D), dtype=torch.float32, device=dev)
    glue.combine(yy, wts, cout)
    save(out, "glue_c_y", yy)
    save(out, "glue_c_wts", wts)
    save(out, "glue_c_out", cout)

    # glm.bmm (the lm_head slice through qmm.matmul: the split-K shape)
    bx = bf16(rows, D)
    bm = qmm.B16(head, 4096, D)
    by = qmm.matmul(bx, bm)
    save(out, "glue_bmm_x", bx)
    save(out, "glue_bmm_w", head)
    save(out, "glue_bmm_out", by)

    # glm.sample.greedy (bf16 logits from the head slice, argmax with the tie rule)
    lg = qmm.matmul(bx, bm).to(torch.bfloat16).contiguous()
    tok = torch.argmax(lg.float(), dim=-1).to(torch.int32)
    save(out, "glue_g_logits", lg)
    save(out, "glue_g_tok", tok)

    entry = {
        "rows": rows, "d": D, "wide": 4 * D, "copies": 4, "ne": NE, "topk": TOPK,
        "eps": EPS, "hc_eps": HC_EPS, "iters": ITERS, "lim": LIM, "scale": SCALE,
        "ids": "glue_e_ids.bin", "e_tab": "glue_e_tab.bin", "e_out": "glue_e_out.bin",
        "r_x": "glue_r_x.bin", "r_w": "glue_nw.bin", "r_out": "glue_r_out.bin", "r_sums": "glue_r_sums.bin",
        "h_x": "glue_h_x.bin", "h_out": "glue_h_out.bin", "h_xs": "glue_h_xs.bin",
        "h_post": "glue_h_post.bin", "h_comb": "glue_h_comb.bin",
        "p_x": "glue_p_x.bin", "p_g": "glue_p_g.bin", "p_out": "glue_p_out.bin", "p_pair_out": "glue_p_pair_out.bin",
        "h_part": "glue_h_part.bin",
        "res_x": "glue_res_x.bin", "res_out": "glue_res_out.bin", "sm_x": "glue_sm_x.bin", "sm_out": "glue_sm_out.bin",
        "s_gu": "glue_s_gu.bin", "s_out": "glue_s_out.bin", "s_xs": "glue_s_xs.bin",
        "router_x": "glue_router_x.bin", "router_w": "glue_router_w.bin", "router_out": "glue_router_out.bin",
        "sel_logits": "glue_sel_logits.bin", "sel_bias": "glue_sel_bias.bin",
        "sel_pick": "glue_sel_pick.bin", "sel_wts": "glue_sel_wts.bin",
        "c_y": "glue_c_y.bin", "c_wts": "glue_c_wts.bin", "c_out": "glue_c_out.bin",
        "bmm_x": "glue_bmm_x.bin", "bmm_w": "glue_bmm_w.bin", "bmm_out": "glue_bmm_out.bin",
        "g_logits": "glue_g_logits.bin", "g_tok": "glue_g_tok.bin",
        "fn": "glue_fn.bin", "base": "glue_base.bin", "hc_scale": "glue_scale.bin", "nw": "glue_nw.bin",
    }
    save(out, "glue_fn", fn)
    save(out, "glue_base", base)
    save(out, "glue_scale", scale)
    save(out, "glue_nw", nw)
    print(f"glue: {rows} rows, d={D}, ne={NE} -> {len(entry) - 12} tensors")
    return entry
