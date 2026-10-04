"""Cases for check_mlx_ops.py: the MTP head's MLX kernels at Nemotron's shapes, and its captured draft level."""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from check_mlx_ops import Buf, Case, Launch
from mlx_op_cases import (D, HEAD_DIM, HEADS, KV_HEADS, Checkpoint, argmax_split, chain_for, embed_launch, fixture,
                          normal, qmv_launch, rms_launch, u32)

def gather_qmv_cases(ck: Checkpoint, quick: bool) -> list[list[Case]]:
    cases = []
    tables = {"L1": "backbone.layers.1.mixer.switch_mlp", "mtp": "mtp.layers.1.mixer.switch_mlp"}
    for tname, prefix in tables.items():
        for fc in ("fc1", "fc2"):
            w_all, s_all, b_all = ck.linear(f"{prefix}.{fc}")
            n, k = int(w_all.shape[1]), int(w_all.shape[2]) * 8
            picks = [[17, 3, 101, 64, 5, 127], [0, 1, 2, 3, 4, 5], [90, 12, 77, 31, 44, 120]]
            if tname == "mtp":
                picks.insert(0, [int(i) for i in fixture("mtp/008_moe_experts/in_inds.npy").reshape(-1).tolist()])
            for p, picked in enumerate(picks[:2] if quick else picks):
                x = (fixture("mtp/008_moe_experts/in_x.npy").reshape(1, k) if (fc == "fc1" and p == 0)
                     else normal((1, k), 300 + p + k, (1.0, 0.3, 3.0)[p % 3]))
                sel = sorted(set(picked))                  # a sub-table of the picked experts (same per-expert bits)
                sub = mx.array(sel, dtype=mx.int32)
                w, s, b = w_all[sub], s_all[sub], b_all[sub]
                mx.eval(w, s, b)
                e = len(sel)
                ids = [sel.index(i) for i in picked]
                ref = mx.gather_qmm(x.reshape(1, 1, 1, 1, k), w, s, b, rhs_indices=mx.array(ids, dtype=mx.uint32)
                                    .reshape(1, 1, 6), transpose=True, group_size=64, bits=4).reshape(6, n)
                assert k % 512, "no Nemotron expert shape takes MLX's gather_qmv_fast"
                ours = Launch("tf_gather_qmv_b4_g64",
                              [Buf(0, w), Buf(1, s), Buf(2, b), Buf(3, x), Buf(4, u32(ids)), Buf(5, u32([0] * 6)),
                               Buf(6, i32=[k, n]), Buf(7, out=((6, n), mx.bfloat16))], (1, n // 8, 6), (32, 2, 1),
                              groups=True)
                theirs = Launch("affine_gather_qmv_bfloat16_t_gs_64_b_4",
                                [Buf(0, w), Buf(1, s), Buf(2, b), Buf(3, x), Buf(4, u32([0] * 6)), Buf(5, u32(ids)),
                                 Buf(6, out=((6, n), mx.bfloat16)), Buf(7, i32=[k]), Buf(8, i32=[n]), Buf(9, i32=[1]),
                                 Buf(10, i32=[1, 1, k]), Buf(11, u64=[k]), Buf(12, i32=[1]), Buf(13, i32=[e]),
                                 Buf(14, u64=[n * k // 8]), Buf(15, u64=[n * k // 64]), Buf(16, u64=[n * k // 64]),
                                 Buf(17, i32=[1]), Buf(18, i32=[6]), Buf(19, u64=[1]), Buf(20, u64=[1])],
                                (1, n // 8, 6), (32, 2, 1), groups=True)
                cases.append(Case("gather_qmv", f"{tname}.{fc} K={k} N={n} pick{p}",
                                  "gather_qmv", "qmv", [ours], [theirs], [ref], time=p == 0,
                                  chain=chain_for(w.nbytes // e * 6)))
                del w, s, b
    return [cases]


def gemv_cases(ck: Checkpoint, quick: bool) -> list[list[Case]]:
    cases = []
    gates = {"mtp.gate": fixture("mtp/007_moe_gate/in_gate_w.npy"),
             "L1.gate": ck.get("backbone.layers.1.mixer.gate.weight")}
    for gname, w in gates.items():
        rows, k = int(w.shape[0]), int(w.shape[1])
        for p in range(2 if quick else 5):
            x = fixture("mtp/007_moe_gate/out_normed.npy").reshape(1, k) if p == 0 else normal((1, k), 700 + p,
                                                                                               (1.0, 0.2, 5.0)[p % 3])
            ref = (x.reshape(1, 1, k) @ w.T).reshape(1, rows)
            ours = Launch("tf_gemv_bf16_tm4", [Buf(0, w), Buf(1, x), Buf(2, i32=[k, rows, k]),
                                               Buf(3, out=((1, rows), mx.bfloat16))],
                          (-(-rows // 4), 1, 1), (32, 8, 1), groups=True)
            theirs = Launch("gemv_bfloat16_bm1_bn8_sm1_sn32_tm4_tn4_nc0_axpby0",
                            [Buf(0, w), Buf(1, x), Buf(3, out=((1, rows), mx.bfloat16)), Buf(4, i32=[k]),
                             Buf(5, i32=[rows]), Buf(6, i32=[k]), Buf(9, i32=[1]), Buf(10, i32=[1]), Buf(11, u64=[k]),
                             Buf(12, u64=[0])], (-(-rows // 4), 1, 1), (32, 8, 1), groups=True)
            cases.append(Case("gemv", f"{gname} K={k} N={rows} x{p}", "gemv bm1 bn8 sm1 sn32 tm4 tn4", "gemv",
                              [ours], [theirs], [ref], time=p == 0, chain=chain_for(w.nbytes)))
    # the multi-row MTP attention's scores: (bf16(scale) * q) @ k^T over 2 kv heads x 16 query heads
    for rows in (3, 4, 5):
        for n in ((60, 600) if quick else (7, 60, 300, 1000, 2500)):
            cap = -(-n // 256) * 256 + 256
            kc = normal((1, KV_HEADS, cap, HEAD_DIM), n + rows, 1.5)
            qraw = normal((1, rows, HEADS, HEAD_DIM), 2 * n + rows, 2.0)
            qs = mx.multiply(mx.array(HEAD_DIM ** -0.5, dtype=mx.bfloat16), qraw.transpose(0, 2, 1, 3))
            # MLX's unfused attention (fast.cpp fallback): unflatten the heads, expand the keys, matmul
            ref = mx.matmul(mx.unflatten(qs, 1, (KV_HEADS, HEADS // KV_HEADS)),
                            mx.swapaxes(mx.expand_dims(kc[..., :n, :], 2), -1, -2))
            qs_buf = mx.contiguous(qs.transpose(0, 2, 1, 3))          # its memory: [1, R, 32, 128]
            nv = rows
            ours = Launch(f"tf_gemv_wide_bf16_v{nv}_kl32",
                          [Buf(0, kc), Buf(1, qs_buf), Buf(2, i32=[HEAD_DIM, n, rows, HEAD_DIM, HEADS * HEAD_DIM]),
                           Buf(3, u64=[HEADS // KV_HEADS, (HEADS // KV_HEADS) * HEAD_DIM, HEAD_DIM, cap * HEAD_DIM,
                                       0]),
                           Buf(4, out=((1, KV_HEADS, HEADS // KV_HEADS, rows, n), mx.bfloat16))],
                          (1, -(-n // 4), HEADS), (32, 4, 1), groups=True)
            cases.append(Case("gemv_wide", f"scores R={rows} keys={n}", f"gemv_wide nv{nv} kl32 nc1", "gemv",
                              [ours], None, [ref], time=False))
    return [cases]


def softmax_cases(ck: Checkpoint, quick: bool) -> list[list[Case]]:
    cases = []
    for n in ((5, 60, 300, 5000) if quick else (1, 5, 59, 60, 300, 1023, 4096, 4097, 9000, 20000)):
        for rows in (8, 64):
            x = normal((rows, n), n + rows, 4.0)
            x = mx.where(mx.arange(n)[None] > (n - 4 + mx.arange(rows)[:, None] % 4),
                         mx.array(-3.3895313892515355e38, dtype=mx.bfloat16), x)        # a causal mask's fill
            ref = mx.softmax(x, axis=-1, precise=True)
            looped = n > 4096
            tg = 1024 if looped else 32 * -(-(-(-n // 4)) // 32)
            name = "looped" if looped else "block"
            ours = Launch(f"tf_softmax{'_looped' if looped else ''}_bf16",
                          [Buf(0, x), Buf(1, i32=[n]), Buf(2, out=((rows, n), mx.bfloat16))], (tg * rows, 1, 1),
                          (tg, 1, 1))
            theirs = Launch(f"{name}_softmax_precise_bfloat16",
                            [Buf(0, x), Buf(1, out=((rows, n), mx.bfloat16)), Buf(2, i32=[n])], (tg * rows, 1, 1),
                            (tg, 1, 1))
            cases.append(Case("softmax", f"rows={rows} n={n}", f"{name}_softmax_precise", "softmax", [ours],
                              [theirs], [ref], time=rows == 64 and n in (60, 1023, 9000),
                              chain=chain_for(4 * rows * n)))
    return [cases]


def route_cases(ck: Checkpoint, quick: bool) -> list[list[Case]]:
    from mlx_lm.models.nemotron_h import group_expert_select  # noqa: PLC0415 - mlx_lm only for the reference

    bias = fixture("mtp/007_moe_gate/in_bias.npy")
    eps_scale = [1e-20, 2.5]
    cases = []
    gate_sets = [("fixture", fixture("mtp/007_moe_gate/out_gates.npy").reshape(1, 128))]
    for p in range(3 if quick else 24):
        g = normal((1, 128), 900 + p, (1.0, 4.0, 0.25)[p % 3])
        if p % 4 == 3:
            g = mx.round(g * 2) / 2                     # ties in the scores
            g = g.astype(mx.bfloat16)
        gate_sets.append((f"random{p}", g))
    for gname, g in gate_sets:
        inds, scores = group_expert_select(g.reshape(1, 1, 128), bias, 6, 1, 1, 2.5, True)
        inds, scores = inds.reshape(1, 6).astype(mx.uint32), scores.reshape(1, 6)
        fused = Launch("tf_route_topk", [Buf(0, g), Buf(1, bias), Buf(2, f32=eps_scale), Buf(3, i32=[128, 6]),
                                         Buf(4, out=((1, 6), mx.uint32)), Buf(5, out=((1, 6), mx.float32))],
                       (128, 1, 1), (128, 1, 1))
        cases.append(Case("route", f"{gname} ids+weights", "group_expert_select (fusions, sort, take, sum)", "route",
                          [fused], None, [inds, scores], time=gname == "fixture", chain=64))
    return [cases]


def elementwise_cases(ck: Checkpoint, quick: bool) -> list[list[Case]]:
    cases = []
    n = 6 * 1856
    for p in range(2 if quick else 4):
        a, b = normal((n,), 40 + p, 3.0), normal((n,), 50 + p, 0.5)

        def one(label: str, launch: Launch, ref: mx.array) -> None:
            cases.append(Case("elementwise", f"{label} p{p}", label.split()[0], "elementwise", [launch], None, [ref],
                              time=False))

        one("vv_Add bf16", Launch("tf_add_bf16", [Buf(0, a), Buf(1, b), Buf(2, i32=[n]),
                                                  Buf(3, out=((n,), mx.bfloat16))], (n, 1, 1), (256, 1, 1)), a + b)
        one("relu2 fusion", Launch("tf_relu2_bf16", [Buf(0, a), Buf(1, i32=[n]), Buf(2, out=((n,), mx.bfloat16))],
                                   (n, 1, 1), (256, 1, 1)), nn.relu2(a))
        scale = mx.array([HEAD_DIM ** -0.5], dtype=mx.bfloat16)
        one("sv_Multiply bf16", Launch("tf_scale_bf16", [Buf(0, a), Buf(1, scale), Buf(2, i32=[n]),
                                                         Buf(3, out=((n,), mx.bfloat16))], (n, 1, 1), (256, 1, 1)),
            mx.multiply(mx.array(HEAD_DIM ** -0.5, dtype=mx.bfloat16), a))
        y = fixture("mtp/009_moe_combine/in_fc2.npy").reshape(6, D) if p == 0 else normal((6, D), 60 + p, 2.0)
        w = fixture("mtp/009_moe_combine/in_scores.npy").reshape(6) if p == 0 else \
            mx.random.uniform(shape=(6,), key=mx.random.key(70 + p))
        ref = (y.reshape(1, 1, 6, D) * w.reshape(1, 1, 6)[..., None]).sum(axis=-2).astype(mx.bfloat16).reshape(1, D)
        one("combine (v_copy, g2_Multiply, col_reduce_small, v_copy)",
            Launch("tf_moe_combine", [Buf(0, y), Buf(1, w), Buf(2, i32=[6, D]), Buf(3, out=((1, D), mx.bfloat16))],
                   (D, 1, 1), (256, 1, 1)), ref)
        for rows, keys in ((3, 60), (4, 59), (5, 300)):
            sc4 = normal((KV_HEADS * 16, rows, keys), 80 + p + rows, 3.0)
            mask = mx.arange(keys - rows, keys)[:, None] >= mx.arange(keys)[None]
            refm = mx.where(mask, sc4, mx.array(float(mx.finfo(mx.bfloat16).min), dtype=mx.bfloat16))
            one(f"causal where R={rows}", Launch("tf_causal_mask_bf16",
                                                 [Buf(0, sc4), Buf(1, i32=[keys, rows]),
                                                  Buf(2, out=((KV_HEADS * 16, rows, keys), mx.bfloat16))],
                                                 (keys, rows, KV_HEADS * 16), (64, 1, 1)), refm)
    ids = fixture("mtp/010_draft_head/in_draft_ids.npy")
    pick = mx.array([int(fixture("mtp/010_draft_head/out_argmax.npy")[0].item()), 5, 32767], dtype=mx.uint32)
    cases.append(Case("elementwise", "gather draft_ids[argmax]", "gather_front", "elementwise",
                      [Launch("tf_gather_u32", [Buf(0, ids), Buf(1, pick), Buf(2, i32=[3]),
                                                Buf(3, out=((3,), mx.uint32))], (3, 1, 1), (32, 1, 1))], None,
                      [ids[pick[:3]]], time=False))
    return [cases]


def gemv_wide_plan(m: int) -> tuple[int, int]:
    """(vectors a threadgroup, grid x) of MLX's gemv_wide for M vectors (N under 65536)."""

    passes = (m + 4) // 5
    return -(-m // passes), passes


def attention_fallback(rows: int, n: int, seed: int) -> Case:
    """MLX's unfused attention for 3-8 query rows with GQA 16 (rows x 16 > 32), our five kernels end to end."""

    cap = -(-n // 256) * 256 + 256
    kc, vc = normal((1, KV_HEADS, cap, HEAD_DIM), seed, 1.5), normal((1, KV_HEADS, cap, HEAD_DIM), seed + 1)
    qraw = normal((1, rows, HEADS, HEAD_DIM), seed + 2, 2.0)
    ref = mx.fast.scaled_dot_product_attention(qraw.transpose(0, 2, 1, 3), kc[..., :n, :], vc[..., :n, :],
                                               scale=HEAD_DIM ** -0.5, mask="causal")
    g = HEADS // KV_HEADS
    nv, gx = gemv_wide_plan(rows)
    looped = n > 4096
    tg = 1024 if looped else 32 * -(-(-(-n // 4)) // 32)
    scores = (1, KV_HEADS, g, rows, n)
    launches = [
        Launch("tf_scale_bf16", [Buf(0, qraw), Buf(1, mx.array([HEAD_DIM ** -0.5], dtype=mx.bfloat16)),
                                 Buf(2, i32=[rows * HEADS * HEAD_DIM]),
                                 Buf(3, out=((1, rows, HEADS, HEAD_DIM), mx.bfloat16))],
               (rows * HEADS * HEAD_DIM, 1, 1), (256, 1, 1), stem="elementwise"),
        Launch(f"tf_gemv_wide_bf16_v{nv}_kl32",
               [Buf(0, kc), Buf(1, src=(0, 0)), Buf(2, i32=[HEAD_DIM, n, rows, HEAD_DIM, HEADS * HEAD_DIM]),
                Buf(3, u64=[g, g * HEAD_DIM, HEAD_DIM, cap * HEAD_DIM, 0]), Buf(4, out=(scores, mx.bfloat16))],
               (gx, -(-n // 4), HEADS), (32, 4, 1), groups=True, stem="gemv"),
        Launch("tf_causal_mask_bf16", [Buf(0, src=(1, 0)), Buf(1, i32=[n, rows]), Buf(2, out=(scores, mx.bfloat16))],
               (n, rows, HEADS), (64, 1, 1), stem="elementwise"),
        Launch(f"tf_softmax{'_looped' if looped else ''}_bf16", [Buf(0, src=(2, 0)), Buf(1, i32=[n]),
                                                                 Buf(2, out=(scores, mx.bfloat16))],
               (tg * HEADS * rows, 1, 1), (tg, 1, 1), stem="softmax"),
        Launch("tf_nax_gemm_nn_bf16", [Buf(0, src=(3, 0)), Buf(1, vc), Buf(2, i32=[rows, HEAD_DIM, n, n, HEAD_DIM,
                                                                                   HEAD_DIM]),
                                       Buf(3, u64=[g, g * rows * n, rows * n, cap * HEAD_DIM, 0, rows * HEAD_DIM]),
                                       Buf(4, out=((1, HEADS, rows, HEAD_DIM), mx.bfloat16))],
               (1, 1, HEADS), (32, 4, 1), groups=True, stem="nax_gemm")]
    return Case("attn4", f"rows={rows} keys={n}", "unfused fallback (sv_Multiply, gemv_wide, Select, "
                "softmax, steel_gemm_fused_nax_nn)", "attention", launches, None, [ref], time=False)


def nax_cases(ck: Checkpoint, quick: bool) -> list[list[Case]]:
    cases = []
    for rows in (3, 4, 5, 8):
        for n in ((7, 60, 300) if quick else (1, 7, 31, 32, 60, 255, 256, 300, 1000, 4096, 5000, 8200)):
            cap = -(-n // 256) * 256 + 256
            vc = normal((1, KV_HEADS, cap, HEAD_DIM), 3 * n + rows)
            probs = mx.softmax(normal((1, KV_HEADS, 16, rows, n), 5 * n + rows, 3.0), axis=-1, precise=True)
            ref = mx.matmul(probs, mx.expand_dims(vc[..., :n, :], 2))
            ours = Launch("tf_nax_gemm_nn_bf16",
                          [Buf(0, probs), Buf(1, vc), Buf(2, i32=[rows, HEAD_DIM, n, n, HEAD_DIM, HEAD_DIM]),
                           Buf(3, u64=[16, 16 * rows * n, rows * n, cap * HEAD_DIM, 0, rows * HEAD_DIM]),
                           Buf(4, out=((1, KV_HEADS, 16, rows, HEAD_DIM), mx.bfloat16))],
                          (1, 1, HEADS), (32, 4, 1), groups=True)
            cases.append(Case("nax_gemm", f"P@V rows={rows} keys={n}", "steel_gemm_fused_nax_nn bm64 bn128",
                              "nax_gemm", [ours], None, [ref], time=False))
        for n in ((60, 600) if quick else (3, 60, 300, 1023, 2000, 5000)):
            if rows <= 8 and n >= rows:
                cases.append(attention_fallback(rows, n, 17 * n + rows))
    return [cases]




def mtp_level_cases(ck: Checkpoint, quick: bool) -> list[list[Case]]:
    """The captured MTP draft level stage by stage from its inputs; the stages chain, so all passing is the level."""

    f = lambda name: fixture(f"mtp/{name}")                                        # noqa: E731
    cases: list[Case] = []

    def proj(x: Buf, prefix: str) -> Launch:
        return qmv_launch(*ck.linear(prefix), x, 1, 4)

    def add(label: str, launches: list[Launch], refs: list[mx.array]) -> None:
        cases.append(Case("mtp_level", label, "fixture", "embed_norm", launches, None, refs, time=False))

    w4, s4, b4 = ck.linear("backbone.embeddings")
    token = f("000_embed/in_token.npy").astype(mx.uint32)
    add("000 embed", [embed_launch([int(token[0].item())], w4, s4, b4)],
        [f("000_embed/out_embedding.npy").reshape(1, D)])
    for name, x in (("enorm", "embedding"), ("hnorm", "hidden")):
        add(f"001 {name}", [rms_launch(Buf(0, f(f"001_enorm_hnorm/in_{x}.npy").reshape(1, D)),
                                       f(f"001_enorm_hnorm/in_{name}_w.npy"), 1)],
            [f(f"001_enorm_hnorm/out_{name}.npy").reshape(1, D)])
    add("002 eh_proj", [proj(Buf(3, f("002_eh_proj/in_concat.npy").reshape(1, 2 * D)), "mtp.layers.0.eh_proj")],
        [f("002_eh_proj/out_x.npy").reshape(1, D)])
    nrm = rms_launch(Buf(0, f("003_attn_norm_qkv/in_x.npy").reshape(1, D)), f("003_attn_norm_qkv/in_norm_w.npy"), 1)
    add("003 attn norm", [nrm], [f("003_attn_norm_qkv/out_normed.npy").reshape(1, D)])
    for name, n in (("q", 4096), ("k", 256), ("v", 256)):
        add(f"003 {name}_proj", [nrm, proj(Buf(3, src=(0, 0)), f"mtp.layers.0.mixer.{name}_proj")],
            [f(f"003_attn_norm_qkv/out_{name}.npy").reshape(1, n)])
    keys, values = f("005_sdpa/in_keys.npy"), f("005_sdpa/in_values.npy")
    n_keys = int(keys.shape[2])
    add("005 sdpa", [Launch("tf_sdpa_vec_d128", [Buf(0, f("005_sdpa/in_q.npy")), Buf(1, keys), Buf(2, values),
                                                  Buf(3, i32=[16, n_keys]),
                                                  Buf(4, u64=[n_keys * 128, 128, n_keys * 128, 128]),
                                                  Buf(5, f32=[HEAD_DIM ** -0.5]),
                                                  Buf(6, out=((1, HEADS, 1, HEAD_DIM), mx.bfloat16))],
                            (HEADS, 1, 1), (1024, 1, 1), groups=True, stem="sdpa")], [f("005_sdpa/out_out.npy")])
    oproj = proj(Buf(3, f("006_o_proj_residual/in_attn.npy").reshape(1, 4096)), "mtp.layers.0.mixer.o_proj")
    add("006 o_proj", [oproj], [f("006_o_proj_residual/out_o_proj.npy").reshape(1, D)])
    add("006 residual", [oproj, Launch("tf_add_bf16", [Buf(0, f("006_o_proj_residual/in_x.npy").reshape(1, D)),
                                                        Buf(1, src=(0, 0)), Buf(2, i32=[D]),
                                                        Buf(3, out=((1, D), mx.bfloat16))], (D, 1, 1), (256, 1, 1),
                                       stem="elementwise")], [f("006_o_proj_residual/out_x.npy").reshape(1, D)])
    gnorm = rms_launch(Buf(0, f("007_moe_gate/in_x.npy").reshape(1, D)), f("007_moe_gate/in_norm_w.npy"), 1)
    gate = Launch("tf_gemv_bf16_tm4", [Buf(0, f("007_moe_gate/in_gate_w.npy")), Buf(1, src=(0, 0)),
                                       Buf(2, i32=[D, 128, D]), Buf(3, out=((1, 128), mx.bfloat16))],
                  (32, 1, 1), (32, 8, 1), groups=True, stem="gemv")
    route = Launch("tf_route_topk", [Buf(0, src=(1, 0)), Buf(1, f("007_moe_gate/in_bias.npy")),
                                     Buf(2, f32=[1e-20, 2.5]),
                                     Buf(3, i32=[128, 6]), Buf(4, out=((1, 6), mx.uint32)),
                                     Buf(5, out=((1, 6), mx.float32))], (128, 1, 1), (128, 1, 1), stem="route")
    add("007 gate norm", [gnorm], [f("007_moe_gate/out_normed.npy").reshape(1, D)])
    add("007 gate gemv", [gnorm, gate], [f("007_moe_gate/out_gates.npy").reshape(1, 128)])
    add("007 route", [gnorm, gate, route], [f("007_moe_gate/out_inds.npy").reshape(1, 6),
                                             f("007_moe_gate/out_scores.npy").reshape(1, 6)])
    ids = [int(i) for i in f("008_moe_experts/in_inds.npy").reshape(-1).tolist()]
    fc1, fc2 = ck.linear("mtp.layers.1.mixer.switch_mlp.fc1"), ck.linear("mtp.layers.1.mixer.switch_mlp.fc2")
    sel = sorted(set(ids))
    pick = mx.array(sel, dtype=mx.int32)
    t1, t2 = [t[pick] for t in fc1], [t[pick] for t in fc2]
    sub = u32([sel.index(i) for i in ids])
    up = Launch("tf_gather_qmv_b4_g64", [Buf(0, t1[0]), Buf(1, t1[1]), Buf(2, t1[2]),
                                          Buf(3, f("008_moe_experts/in_x.npy").reshape(1, D)), Buf(4, sub),
                                          Buf(5, u32([0] * 6)), Buf(6, i32=[D, 1856]),
                                          Buf(7, out=((6, 1856), mx.bfloat16))], (1, 1856 // 8, 6), (32, 2, 1),
                groups=True, stem="qmv")
    act = Launch("tf_relu2_bf16", [Buf(0, src=(0, 0)), Buf(1, i32=[6 * 1856]), Buf(2, out=((6, 1856), mx.bfloat16))],
                 (6 * 1856, 1, 1), (256, 1, 1), stem="elementwise")
    down = Launch("tf_gather_qmv_b4_g64", [Buf(0, t2[0]), Buf(1, t2[1]), Buf(2, t2[2]), Buf(3, src=(1, 0)),
                                            Buf(4, sub), Buf(5, u32([0, 1, 2, 3, 4, 5])), Buf(6, i32=[1856, D]),
                                            Buf(7, out=((6, D), mx.bfloat16))], (1, D // 8, 6), (32, 2, 1),
                  groups=True, stem="qmv")
    add("008 fc1", [up], [f("008_moe_experts/out_fc1.npy").reshape(6, 1856)])
    add("008 relu2", [up, act], [f("008_moe_experts/out_relu2.npy").reshape(6, 1856)])
    add("008 fc2", [up, act, down], [f("008_moe_experts/out_fc2.npy").reshape(6, D)])
    combine = Launch("tf_moe_combine", [Buf(0, f("009_moe_combine/in_fc2.npy").reshape(6, D)),
                                        Buf(1, f("009_moe_combine/in_scores.npy").reshape(6)), Buf(2, i32=[6, D]),
                                        Buf(3, out=((1, D), mx.bfloat16))], (D, 1, 1), (256, 1, 1), stem="elementwise")
    sup = proj(Buf(3, f("009_moe_combine/in_normed.npy").reshape(1, D)), "mtp.layers.1.mixer.shared_experts.up_proj")
    sact = Launch("tf_relu2_bf16", [Buf(0, src=(1, 0)), Buf(1, i32=[3712]), Buf(2, out=((1, 3712), mx.bfloat16))],
                  (3712, 1, 1), (256, 1, 1), stem="elementwise")
    sdown = proj(Buf(3, src=(2, 0)), "mtp.layers.1.mixer.shared_experts.down_proj")
    moe = Launch("tf_add_bf16", [Buf(0, src=(0, 0)), Buf(1, src=(3, 0)), Buf(2, i32=[D]),
                                 Buf(3, out=((1, D), mx.bfloat16))], (D, 1, 1), (256, 1, 1), stem="elementwise")
    resid = Launch("tf_add_bf16", [Buf(0, f("009_moe_combine/in_x.npy").reshape(1, D)), Buf(1, src=(4, 0)),
                                   Buf(2, i32=[D]), Buf(3, out=((1, D), mx.bfloat16))], (D, 1, 1), (256, 1, 1),
                   stem="elementwise")
    final = rms_launch(Buf(0, src=(5, 0)), f("009_moe_combine/in_final_w.npy"), 1)
    chain = [combine, sup, sact, sdown, moe, resid, final]
    for upto, name in ((1, "routed"), (2, "shared_up"), (4, "shared"), (5, "moe"), (6, "x"), (7, "out")):
        add(f"009 {name}", chain[:upto], [f(f"009_moe_combine/out_{name}.npy").reshape(1, -1)])
    draft_ids = f("010_draft_head/in_draft_ids.npy")
    w, s, b = ck.linear("lm_head")
    rows = mx.array(draft_ids, dtype=mx.int32)
    hw, hs, hb = w[rows], s[rows], b[rows]
    head = qmv_launch(hw, hs, hb, Buf(3, f("010_draft_head/in_out.npy").reshape(1, D)), 1, 4)
    amax = argmax_split((-1, 0), 1, 32768)
    tok = Launch("tf_gather_u32", [Buf(0, draft_ids), Buf(1, src=(-1, 0)), Buf(2, i32=[1]),
                                   Buf(3, out=((1,), mx.uint32))], (1, 1, 1), (32, 1, 1), stem="elementwise")
    add("010 draft logits", [head], [f("010_draft_head/out_logits.npy").reshape(1, 32768)])
    add("010 argmax", [head, *amax], [f("010_draft_head/out_argmax.npy")])
    add("010 draft token", [head, *amax, tok], [f("010_draft_head/out_token.npy")])
    return [cases]


BUILDERS: dict[str, Callable[[Checkpoint, bool], list[list[Case]]]] = {
    "gather_qmv": gather_qmv_cases, "gemv": gemv_cases, "softmax": softmax_cases, "route": route_cases,
    "elementwise": elementwise_cases, "nax": nax_cases, "mtp_level": mtp_level_cases,
}
