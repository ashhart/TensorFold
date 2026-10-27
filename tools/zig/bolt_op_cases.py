"""Cases for check_mlx_ops.py: Nemotron 3.5 Bolt's ops at its own shapes, each row bit for bit as MLX runs it alone."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx

from check_mlx_ops import Buf, Case, Launch
from mlx_op_cases import Checkpoint, chain_for, normal

MODEL = Path(os.environ.get("TF_ZIG_BOLT", "models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit"))
ROWS = (1, 2, 9, 16)      # one decode row, and a pass's rows: every row must equal its own one-row op
TOP_K = 9


class Bolt:
    """The Bolt checkpoint's tensors and the first MoE layer of each expert count."""

    def __init__(self, folder: Path) -> None:
        self.ck = Checkpoint(folder)
        config = json.loads((folder / "config.json").read_text())
        config = config.get("llm_config", config)
        self.scale = float(config["routed_scaling_factor"])
        self.layers: dict[int, int] = {}
        for i, block in enumerate(config["block_configs"]):
            if block["block_type"] == "moe":
                self.layers.setdefault(int(block["n_routed_experts"]), i)

    def get(self, name: str) -> mx.array:
        return self.ck.get(name)


_bolt: Bolt | None = None


def bolt() -> Bolt:
    global _bolt
    if _bolt is None:
        _bolt = Bolt(MODEL)
    return _bolt


def mlx_gemv(k: int, n: int) -> tuple[int, int]:
    """(bm, bn) of MLX 0.32.3's gemv for x [1, k] @ W.T, W [n, k] (matmul.cpp: bn 8 when K >= 16 N, else bm by N)."""

    if k >= 16 * n:
        return 1, 8
    return (8 if n >= 4096 else 4), 1


def gemv_launch(w: mx.array, x: mx.array, rows: int) -> tuple[Launch, Launch]:
    """Our launch over `rows` vectors (one a threadgroup z) and MLX's own kernel for the first, at its tiles."""

    n, k = int(w.shape[0]), int(w.shape[1])
    bm, bn = mlx_gemv(k, n)
    name = "tf_gemv_bf16_tm4" if bn == 8 else f"tf_gemv_bf16_bm{bm}_tm4"
    groups = (-(-n // (4 * bm)), 1, rows)
    ours = Launch(name, [Buf(0, w), Buf(1, x), Buf(2, i32=[k, n, k]), Buf(3, out=((rows, n), mx.bfloat16))],
                  groups, (32, 8, 1) if bn == 8 else (32, 1, bm), groups=True, stem="gemv")
    theirs = Launch(f"gemv_bfloat16_bm{bm}_bn{bn}_sm1_sn32_tm4_tn4_nc0_axpby0",
                    [Buf(0, w), Buf(1, x[:1]), Buf(3, out=((1, n), mx.bfloat16)), Buf(4, i32=[k]), Buf(5, i32=[n]),
                     Buf(6, i32=[k]), Buf(9, i32=[1]), Buf(10, i32=[1]), Buf(11, u64=[k]), Buf(12, u64=[0])],
                    (groups[0], 1, 1), (32, bn, bm), groups=True)
    return ours, theirs


def per_row(x: mx.array, op) -> mx.array:
    """op applied to each row of x alone (as one decode step runs it), stacked."""

    return mx.concatenate([op(x[r:r + 1]) for r in range(int(x.shape[0]))], axis=0)


def gemv_cases(ck: Checkpoint | None, quick: bool) -> list[list[Case]]:
    b = bolt()
    mats = {f"L{i}.gate E={e}": f"backbone.layers.{i}.mixer.gate.weight" for e, i in sorted(b.layers.items())}
    first = min(b.layers.values())
    mats["fc1_latent_proj"] = f"backbone.layers.{first}.mixer.fc1_latent_proj.weight"
    mats["fc2_latent_proj"] = f"backbone.layers.{first}.mixer.fc2_latent_proj.weight"
    mats["lm_head"] = "lm_head.weight"
    groups = []
    for label, name in mats.items():
        w = b.get(name)
        mx.eval(w)
        n, k = int(w.shape[0]), int(w.shape[1])
        bm, bn = mlx_gemv(k, n)
        cases = []
        for rows in ((1, 16) if quick else ROWS):
            for seed in range(1 if quick else 3):
                x = normal((rows, k), 7000 + 31 * rows + seed + k, (1.0, 0.1, 6.0)[seed % 3])
                ours, theirs = gemv_launch(w, x, rows)
                ref = per_row(x, lambda v: v @ w.T)
                cases.append(Case("gemv", f"{label} K={k} N={n} rows={rows} s{seed}",
                                  f"gemv bm{bm} bn{bn} sn32 tm4 tn4", "gemv", [ours], [theirs] if rows == 1 else None,
                                  [ref], time=seed == 0 and rows in (1, 16), chain=chain_for(w.nbytes)))
        groups.append(cases)
        del w
    return groups


def route_cases(ck: Checkpoint | None, quick: bool) -> list[list[Case]]:
    from mlx_lm.models.nemotron_h import group_expert_select  # noqa: PLC0415 - mlx_lm only for the reference

    b = bolt()
    groups = []
    for e, layer in sorted(b.layers.items()):
        w = b.get(f"backbone.layers.{layer}.mixer.gate.weight")
        bias = b.get(f"backbone.layers.{layer}.mixer.gate.e_score_correction_bias")
        cases = []
        for rows in ((1, 16) if quick else ROWS):
            for p in range(2 if quick else 6):
                x = normal((rows, int(w.shape[1])), 9100 + 17 * rows + p + e, (1.0, 0.3, 3.0)[p % 3])
                g = (x @ w.T).astype(mx.bfloat16) if p % 3 != 2 else normal((rows, e), 9300 + p + e, 4.0)
                if p == 5:
                    g = (mx.round(g.astype(mx.float32) * 2) / 2).astype(mx.bfloat16)      # ties in the scores

                def select(v: mx.array) -> tuple[mx.array, mx.array]:
                    i, s = group_expert_select(v.reshape(1, 1, e), bias, TOP_K, 1, 1, b.scale, True)
                    return i.reshape(1, TOP_K).astype(mx.uint32), s.reshape(1, TOP_K)

                picks = [select(g[r:r + 1]) for r in range(rows)]
                ref = [mx.concatenate([q[0] for q in picks]), mx.concatenate([q[1] for q in picks])]
                fused = Launch("tf_route_topk", [Buf(0, g), Buf(1, bias), Buf(2, f32=[1e-20, b.scale]),
                                                 Buf(3, i32=[e, TOP_K]), Buf(4, out=((rows, TOP_K), mx.uint32)),
                                                 Buf(5, out=((rows, TOP_K), mx.float32))], (e, rows, 1), (e, 1, 1))
                cases.append(Case("route", f"E={e} k={TOP_K} rows={rows} p{p}",
                                  "group_expert_select (fusions, sort, take, sum)", "route", [fused], None, ref,
                                  time=p == 0 and rows == 1, chain=64))
        groups.append(cases)
    return groups


def combine(y: mx.array, w: mx.array, latent: int) -> mx.array:
    """mlx_lm's (y * scores[..., None]).sum(axis=-2).astype(bf16) for one row's picks."""

    return (y.reshape(1, 1, TOP_K, latent) * w.reshape(1, 1, TOP_K)[..., None]).sum(axis=-2).astype(mx.bfloat16) \
        .reshape(1, latent)


def combine_cases(ck: Checkpoint | None, quick: bool) -> list[list[Case]]:
    latent = int(bolt().get(f"backbone.layers.{min(bolt().layers.values())}.mixer.fc1_latent_proj.weight").shape[0])
    cases = []
    for rows in ((1, 16) if quick else ROWS + (64,)):      # 64 rows at scale 8 meet ties only the lane order breaks
        for p in range(2 if quick else 4):
            y = normal((rows, TOP_K, latent), 9500 + rows + p, (2.0, 0.2, 8.0, 1.0)[p])
            w = mx.random.uniform(shape=(rows, TOP_K), key=mx.random.key(9600 + rows + p)) * bolt().scale / TOP_K
            ref = mx.concatenate([combine(y[r:r + 1], w[r:r + 1], latent) for r in range(rows)])
            ours = Launch("tf_moe_combine_rows_lanes8", [Buf(0, y), Buf(1, w), Buf(2, i32=[TOP_K, latent]),
                                                  Buf(3, out=((rows, latent), mx.bfloat16))], (latent, rows, 1),
                          (256, 1, 1))
            cases.append(Case("combine", f"k={TOP_K} latent={latent} rows={rows} p{p}",
                              "v_copy, g2_Multiply, col_reduce_small, v_copy", "elementwise", [ours], None, [ref],
                              time=False))
    return [cases]


BUILDERS = {"bolt_gemv": gemv_cases, "bolt_route": route_cases, "bolt_combine": combine_cases}
