"""GLM-5.3-Flash (model_type ``glm5_next``): TensorFold's own forward pass, text only.

Written from the checkpoint layout and checked against the GLM-5.3 implementation merged into mlx-vlm
(Blaizzy/mlx-vlm#2030, MIT) as oMLX vendors it:

- four residual streams joined by hyper-connections (mHC: a sinkhorn-normalised 4x4 mix, a pre-weighted
  collapse into the block and a gated write-back into every stream);
- 34 Kimi Delta Attention layers (64 heads of 128, a width-4 causal conv over q/k/v, per-channel decays with a
  lower bound, a gated RMSNorm on the output) and 11 DeepSeek sparse-attention layers (NoPE MLA: a 512-wide
  latent key per token, q through a 1536 low rank; an indexer of 32 heads scores pooled blocks of 4 keys and,
  past 2,048 keys, each query reads its best 512 blocks plus its unfinished tail);
- MoE in layers 3-44: 288 routed experts (sigmoid scores with a correction bias, top 8, normalised, x2.5) and a
  shared expert, SwiGLU clamped at 10; layers 0-2 are dense;
- one MTP layer (DeepSeek-V3 style, ``layers.45``) that ``mtp.py`` serves as the draft head.

Weights are the MLX 4-bit conversion as shipped (groups of 64). The MLA ``kv_b_proj`` stays quantized and is
applied per head in its stored layout (keys absorbed into the query, values after attention), so no weight
is quantized a second time.

Two paths:

- ``decode`` (up to ``DECODE_ROWS`` rows): every row gets the bits a one-row step gives it. Projections with
  4-bit, group-64 weights read their weights once for all rows (``kernels.qmv_rows``: MLX's one-row bits); the
  router, the routed experts, the hyper-connection mix and the indexer gate take the window in one kernel that
  gives each row its one-row call's bits (``ROW_KERNELS``); the rest whose kernel could depend on the row count
  (attention, the indexer's choice, the small KDA projections) runs row by row with one-row shapes; the rest is
  elementwise, a per-row norm, or an MLX batch dimension (where each row keeps its bits, checked).
- ``prefill`` (longer inputs): MLX's batched kernels. Its last bits can differ from the decode path's (the
  same known limit as Flash Next and Nemotron: a reply can depend on which prefix was cached).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.kernels.glm.flash.v1 import fused as F
from tensorfold.kernels.glm.flash.v1 import kda as KDA_K
from tensorfold.kernels.glm.flash.v1 import kernels as K
from tensorfold.kernels.glm.flash.v1 import sparse_attention as SA

MODEL_TYPE = "glm5_next"
DECODE_ROWS = 16

# Decode-row paths that take a window's rows in one kernel instead of one MLX call a row (``kernels``: each row
# keeps the bits its one-row call gives it). TF_GLM5_ROW_KERNELS: "all" (default), "none" (row by row), or a
# comma-separated list. The load-time row check (runtime.rows_match_serial) guards them like everything else.
ROW_KERNELS = ("experts", "router", "hc", "igate", "kda_proj", "mla_proj", "indexer")


def _row_kernels() -> frozenset[str]:
    import os

    value = os.environ.get("TF_GLM5_ROW_KERNELS", "all").strip()
    if value == "all":
        return frozenset(ROW_KERNELS)
    if value in ("", "none"):
        return frozenset()
    names = frozenset(v.strip() for v in value.split(","))
    unknown = names - set(ROW_KERNELS)
    if unknown:
        raise ValueError(f"TF_GLM5_ROW_KERNELS: unknown {sorted(unknown)} (known: {', '.join(ROW_KERNELS)})")
    return names


ENABLED = _row_kernels()


def _flag(name: str, default: str = "1") -> bool:
    import os

    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off", "")


# The decode path's fused kernels (they set the serial arithmetic, so they are not row kernels): a KDA layer's whole
# post-projection step in one launch (mlx-vlm #2105, ported: kernels/glm/flash/v1/kda.py) and sparse MLA reading its
# chosen keys straight from the cache (mlx-vlm #2245, ported: kernels/glm/flash/v1/sparse_attention.py).
FUSED_KDA = _flag("TF_GLM5_FUSED_KDA")
SPARSE_KERNEL = _flag("TF_GLM5_SPARSE_KERNEL")

# Fused decode blocks (``kernels/glm/flash/v1/fused.py``): the MoE block in five kernels and each
# hyper-connection boundary in three, each reproducing the row-by-row path's bits. TF_GLM5_FUSED: "all" (default),
# "none", or a comma-separated list.
FUSED_KERNELS = ("moe", "hc")


def _fused() -> frozenset[str]:
    import os

    value = os.environ.get("TF_GLM5_FUSED", "all").strip()
    if value in ("", "none"):
        return frozenset()
    names: set[str] = set()
    for v in value.split(","):
        v = v.strip()
        names |= set(FUSED_KERNELS) if v == "all" else {v}
    unknown = names - set(FUSED_KERNELS)
    if unknown:
        raise ValueError(f"TF_GLM5_FUSED: unknown {sorted(unknown)} (known: {', '.join(FUSED_KERNELS)})")
    return frozenset(names)


FUSED = _fused()


def _eval_every() -> int:
    import os

    return int(os.environ.get("TF_GLM5_EVAL_EVERY", "2"))


# The decode step hands its graph to the GPU every EVAL_EVERY layers (mx.async_eval), so the GPU runs the first
# layers while Python builds the rest (recipe book, step 5). Scheduling only: no arithmetic changes. 0: one eval.
EVAL_EVERY = _eval_every()


def row_kernel(name: str, rows: int, rows_exact: bool) -> bool:
    return rows_exact and rows > 1 and name in ENABLED


@dataclass
class Config:
    hidden_size: int
    num_hidden_layers: int
    layer_types: list[str]
    mlp_layer_types: list[str]
    vocab_size: int
    rms_norm_eps: float
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_tail: bool
    linear_num_heads: int
    linear_head_dim: int
    linear_conv: int
    linear_lower_bound: float
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    intermediate_size: int
    n_shared_experts: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    first_k_dense_replace: int
    swiglu_limit: float
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int
    num_nextn_predict_layers: int
    eos_token_id: list[int]

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Config":
        t = dict(config.get("text_config") or config)
        lin = t.get("linear_attn_config") or {}
        if int(t.get("n_group", 1)) != 1 or int(t.get("topk_group", 1)) != 1:
            raise ValueError("glm5_next: grouped expert selection (n_group > 1) is not implemented")
        if int(t.get("qk_rope_head_dim", 0)) or not t.get("mla_use_nope", True):
            raise ValueError("glm5_next: only NoPE MLA (qk_rope_head_dim 0) is implemented")
        eos = t.get("eos_token_id", config.get("eos_token_id"))
        return cls(
            hidden_size=int(t["hidden_size"]), num_hidden_layers=int(t["num_hidden_layers"]),
            layer_types=list(t["layer_types"]), mlp_layer_types=list(t["mlp_layer_types"]),
            vocab_size=int(t["vocab_size"]), rms_norm_eps=float(t["rms_norm_eps"]),
            num_attention_heads=int(t["num_attention_heads"]), q_lora_rank=int(t["q_lora_rank"]),
            kv_lora_rank=int(t["kv_lora_rank"]), qk_nope_head_dim=int(t["qk_nope_head_dim"]),
            v_head_dim=int(t["v_head_dim"]), index_n_heads=int(t["index_n_heads"]),
            index_head_dim=int(t["index_head_dim"]), index_topk=int(t["index_topk"]),
            index_kpool=int(t.get("index_kpool", 4)), index_tail=bool(t.get("index_kpool_always_select_tail", True)),
            linear_num_heads=int(lin.get("num_heads", 64)), linear_head_dim=int(lin.get("head_dim", 128)),
            linear_conv=int(lin.get("short_conv_kernel_size", 4)),
            linear_lower_bound=float(lin.get("gate_lower_bound", -5.0)),
            n_routed_experts=int(t["n_routed_experts"]), num_experts_per_tok=int(t["num_experts_per_tok"]),
            moe_intermediate_size=int(t["moe_intermediate_size"]), intermediate_size=int(t["intermediate_size"]),
            n_shared_experts=int(t.get("n_shared_experts") or 0),
            routed_scaling_factor=float(t["routed_scaling_factor"]), norm_topk_prob=bool(t.get("norm_topk_prob", True)),
            first_k_dense_replace=int(t.get("first_k_dense_replace", 0)),
            swiglu_limit=float(t.get("swiglu_limit") or 0.0), hc_mult=int(t.get("hc_mult", 4)),
            hc_eps=float(t.get("hc_eps", 1e-6)), hc_sinkhorn_iters=int(t.get("hc_sinkhorn_iters", 20)),
            num_nextn_predict_layers=int(t.get("num_nextn_predict_layers", 0)),
            eos_token_id=list(eos) if isinstance(eos, list) else ([int(eos)] if eos is not None else []))


# -- weights -----------------------------------------------------------------------------------------------------
class Q:
    """A quantized linear's weights (MLX affine layout: [out, in * bits / 32] uint32, scales/biases [out, groups])."""

    def __init__(self, weight: mx.array, scales: mx.array, biases: mx.array) -> None:
        self.weight, self.scales, self.biases = weight, scales, biases
        groups = int(scales.shape[-1])
        packed = int(weight.shape[-1])
        # in_dims = groups * group; bits = packed * 32 / in_dims, with group in {32, 64, 128}
        for group in (64, 32, 128):
            ins = groups * group
            if (packed * 32) % ins == 0 and (packed * 32) // ins in (2, 3, 4, 5, 6, 8):
                self.group, self.bits, self.ins = group, (packed * 32) // ins, ins
                break
        else:
            raise ValueError(f"cannot infer the quantization of a {weight.shape} weight with {scales.shape} scales")

    @property
    def outs(self) -> int:
        return int(self.weight.shape[-2])

    def arrays(self) -> list[mx.array]:
        return [self.weight, self.scales, self.biases]

    def __call__(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases, transpose=True, group_size=self.group,
                                   bits=self.bits)

    @classmethod
    def stack(cls, parts: list["Q"]) -> "Q":
        """Projections that read the same input as one matrix (rows concatenated)."""

        return cls(mx.concatenate([p.weight for p in parts]), mx.concatenate([p.scales for p in parts]),
                   mx.concatenate([p.biases for p in parts]))


def _rows(q: Q, lo: int, hi: int) -> Q:
    """Output rows lo .. hi - 1 of a quantized linear (views of its arrays)."""

    return Q(q.weight[lo:hi], q.scales[lo:hi], q.biases[lo:hi])


def project(x: mx.array, q: Q, *, rows_exact: bool) -> mx.array:
    """x [R, K] through a quantized linear. One row: MLX's quantized matmul. Several rows on the decode path: the
    rows share the weight reads through ``kernels.qmv_rows`` (MLX's one-row bits) where the weights fit it, else
    one MLX call per row."""

    rows = int(x.shape[0])
    if rows == 1 or not rows_exact:
        return q(x)
    if K.metal() and K.qmv_rows_fits(q, rows):
        return K.qmv_rows(x, q)
    return mx.concatenate([q(x[r:r + 1]) for r in range(rows)])


def per_row(fn: Any, x: mx.array, rows_exact: bool) -> mx.array:
    rows = int(x.shape[0])
    if rows == 1 or not rows_exact:
        return fn(x)
    return mx.concatenate([fn(x[r:r + 1]) for r in range(rows)])


def silu(x: mx.array) -> mx.array:
    return nn.silu(x)


# -- caches ---------------------------------------------------------------------------------------------------------
class KDACache:
    """A KDA layer: the conv window (last ``taps - 1`` q/k/v rows) and the recurrent state [1, H, Dv, Dk] fp32.
    ``_replay`` holds the last decode call's entry state and inputs, so ``keep`` can rebuild the state after any
    prefix of it with the same kernel."""

    transient = ("_replay",)

    def __init__(self) -> None:
        self.conv: mx.array | None = None
        self.ssm: mx.array | None = None
        self.offset = 0
        self._replay: list[Any] | None = None

    @property
    def state(self) -> list[mx.array]:
        return [a for a in (self.conv, self.ssm) if a is not None]

    def keep(self, rows: int, keep: int) -> None:
        replay = self.__dict__.get("_replay")
        if replay is None or replay[0] != rows:
            raise RuntimeError("KDACache.keep: no record of the last decode call")
        if replay[1] == "fused":
            # the kept rows again, from the window's entry state and window, through the same kernel
            _, _, kda, proj, conv, entry = replay
            if keep == 0:
                self.ssm, self.conv = entry, conv
            else:
                _, self.ssm, self.conv = KDA_K.kda_rows(kda, mx.contiguous(proj[:keep]), conv, entry)
            self.offset -= rows - keep
            self._replay = None
            return
        _, ci, entry, q, k, v, g, beta = replay
        taps = int(ci.shape[0]) - rows + 1
        _, self.ssm = K.gated_delta(q[:, :keep], k[:, :keep], v[:, :keep], g[:, :keep], beta[:, :keep], entry)
        self.conv = mx.contiguous(ci[keep:keep + taps - 1])
        self.offset -= rows - keep
        self._replay = None


class MLACache:
    """A sparse-attention layer: latent keys [cap, 512], the indexer's raw keys and gate scores [cap, 128], and the
    pooled block keys [cap / 4, 128] (block b is valid once position 4 b + 3 is written). Positions past
    ``offset`` are stale; trimming only moves ``offset``."""

    step = 256

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.ik: mx.array | None = None
        self.ig: mx.array | None = None
        self.pool: mx.array | None = None
        self.offset = 0

    @property
    def state(self) -> list[mx.array]:
        return [a for a in (self.keys, self.ik, self.ig, self.pool) if a is not None]

    def _grow(self, end: int) -> None:
        cap = 0 if self.keys is None else int(self.keys.shape[0])
        if end <= cap:
            return
        new = -(-end // self.step) * self.step

        def grown(a: mx.array | None, width: int, rows: int, dtype: Any) -> mx.array:
            pad = mx.zeros((rows, width), dtype=dtype)
            return pad if a is None else mx.concatenate([a, pad[: rows - int(a.shape[0])]])

        self.keys = grown(self.keys, 512 if self.keys is None else int(self.keys.shape[1]), new, mx.bfloat16)
        self.ik = grown(self.ik, 128 if self.ik is None else int(self.ik.shape[1]), new, mx.bfloat16)
        self.ig = grown(self.ig, 128 if self.ig is None else int(self.ig.shape[1]), new, mx.bfloat16)
        self.pool = grown(self.pool, 128 if self.pool is None else int(self.pool.shape[1]), new // 4, mx.bfloat16)

    def append(self, lat: mx.array, ik: mx.array, ig: mx.array, ape: mx.array, kpool: int) -> None:
        """Write rows [offset, offset + R) and pool every block they complete."""

        rows = int(lat.shape[0])
        start, end = self.offset, self.offset + rows
        if self.keys is None:
            self.keys = mx.zeros((0, int(lat.shape[1])), dtype=mx.bfloat16)
            self.ik = mx.zeros((0, int(ik.shape[1])), dtype=mx.bfloat16)
            self.ig = mx.zeros((0, int(ig.shape[1])), dtype=mx.bfloat16)
            self.pool = mx.zeros((0, int(ik.shape[1])), dtype=mx.bfloat16)
        self._grow(end)
        self.keys[start:end] = lat.astype(mx.bfloat16)
        self.ik[start:end] = ik.astype(mx.bfloat16)
        self.ig[start:end] = ig.astype(mx.bfloat16)
        first, last = start // kpool, end // kpool          # blocks [first, last) complete now
        if last > first:
            self.pool[first:last] = pool_blocks(self.ik[first * kpool:last * kpool],
                                                self.ig[first * kpool:last * kpool],
                                                ape, kpool)
        self.offset = end

    def trim(self, count: int) -> None:
        self.offset -= int(count)


def pool_blocks(keys: mx.array, gates: mx.array, ape: mx.array, kpool: int) -> mx.array:
    """Pooled keys of whole blocks: a softmax over each block's positions of gate + ape weights its raw keys.
    Written elementwise (fp32, positions in order) so a block's bits do not depend on how many blocks are
    pooled together."""

    blocks = int(keys.shape[0]) // kpool
    k = keys.reshape(blocks, kpool, -1).astype(mx.float32)
    logit = gates.reshape(blocks, kpool, -1).astype(mx.float32) + ape.astype(mx.float32)[None]
    top = logit[:, 0]
    for j in range(1, kpool):
        top = mx.maximum(top, logit[:, j])
    e = [mx.exp(logit[:, j] - top) for j in range(kpool)]
    total = e[0]
    for j in range(1, kpool):
        total = total + e[j]
    out = (e[0] / total) * k[:, 0]
    for j in range(1, kpool):
        out = out + (e[j] / total) * k[:, j]
    return out.astype(mx.bfloat16)


# -- blocks ---------------------------------------------------------------------------------------------------------
class HC:
    """One hyper-connection: RMS over the flattened streams, a fp32 projection to (2 + S) S mixes, sinkhorn."""

    def __init__(self, fn: mx.array, base: mx.array, scale: mx.array, cfg: Config) -> None:
        self.fn = fn.astype(mx.float32)
        # the stored bf16 mix matrix repacked for the fused mix kernel (exact: bf16 -> fp32 loses nothing)
        self.fn_packed = F.pack_hc_fn(fn) if fn.dtype == mx.bfloat16 and tuple(fn.shape) == (24, 16384) else None
        self.base = base.astype(mx.float32)
        self.scale = scale.astype(mx.float32)
        self.cfg = cfg

    def split(self, x: mx.array, rows_exact: bool) -> tuple[mx.array, mx.array, mx.array]:
        rows = int(x.shape[0])
        z = mx.fast.rms_norm(x.astype(mx.float32).reshape(rows, -1), None, self.cfg.rms_norm_eps)
        if row_kernel("hc", rows, rows_exact):
            mixes = K.matmul_rows(z, self.fn, transposed=False)
        else:
            mixes = per_row(lambda r: r @ self.fn.T, z, rows_exact)
        return K.hc_split(x, mixes, self.scale, self.base, hc=self.cfg.hc_mult, iters=self.cfg.hc_sinkhorn_iters,
                          eps=self.cfg.hc_eps)


def hc_expand(branch: mx.array, x: mx.array, post: mx.array, comb: mx.array, rows_exact: bool = False) -> mx.array:
    """New streams [R, S, D]: post_i * branch + sum_j comb[j, i] * x_j in fp32 (the reference's arithmetic: a
    batched [S, S] x [S, D] matmul). On the decode path the rows are that matmul's batch, where each keeps its
    one-row bits (checked on the M3 Ultra, MLX 0.32.0; "hc" off: one row at a time)."""

    def one(b: mx.array, xs: mx.array, p: mx.array, c: mx.array) -> mx.array:
        y = p[..., None] * b.astype(mx.float32)[:, None, :]
        return (y + mx.matmul(c.swapaxes(-1, -2), xs.astype(mx.float32))).astype(x.dtype)

    rows = int(x.shape[0])
    if rows == 1 or not rows_exact or row_kernel("hc", rows, rows_exact):
        return one(branch, x, post, comb)
    return mx.concatenate([one(branch[r:r + 1], x[r:r + 1], post[r:r + 1], comb[r:r + 1]) for r in range(rows)])


class DenseMLP:
    def __init__(self, gate: Q, up: Q, down: Q, limit: float) -> None:
        self.gate_up = Q.stack([gate, up])
        self.width = gate.outs
        self.down = down
        self.limit = limit

    def __call__(self, x: mx.array, rows_exact: bool) -> mx.array:
        gu = project(x, self.gate_up, rows_exact=rows_exact)
        return project(swiglu(gu[:, :self.width], gu[:, self.width:], self.limit), self.down, rows_exact=rows_exact)


def swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return silu(gate) * up


class MoE:
    def __init__(self, gate_w: mx.array, bias: mx.array, gate: Q, up: Q, down: Q, shared: DenseMLP | None,
                 cfg: Config) -> None:
        self.router = mx.contiguous(gate_w.astype(mx.float32).T)       # [D, E]
        # the stored bf16 weights repacked for the fused router (exact: bf16 -> fp32 loses nothing)
        self.router_packed = (F.pack_router(gate_w) if gate_w.dtype == mx.bfloat16 and gate_w.shape[0] % 16 == 0
                              and gate_w.shape[1] % 32 == 0 else None)
        self.bias = bias.astype(mx.float32)
        self.gate, self.up, self.down = gate, up, down
        self.shared = shared
        self.cfg = cfg
        self.scale_arr = mx.array([cfg.routed_scaling_factor], dtype=mx.float32)
        self.limit_arr = mx.array([cfg.swiglu_limit or 3.0e38], dtype=mx.float32)
        self.fused_ok = F.moe_fits(self)

    def logits(self, x: mx.array, rows_exact: bool) -> mx.array:
        """Router logits [R, E] in fp32, every row with its one-row matmul's bits on the decode path."""

        xf = x.astype(mx.float32)
        if row_kernel("router", int(x.shape[0]), rows_exact):
            return K.matmul_rows(xf, self.router, transposed=True)
        return per_row(lambda r: r @ self.router, xf, rows_exact)

    def route(self, logits: mx.array) -> tuple[mx.array, mx.array]:
        """Top-k experts [R, k] and their weights from the logits (sigmoid, bias-corrected choice, normalised)."""

        cfg = self.cfg
        scores = mx.sigmoid(logits)
        top = cfg.num_experts_per_tok
        idx = mx.argpartition(-(scores + self.bias), kth=top - 1, axis=-1)[..., :top]
        w = mx.take_along_axis(scores, idx, axis=-1)
        if top > 1 and cfg.norm_topk_prob:
            w = w / w.sum(axis=-1, keepdims=True)
        return idx, w * cfg.routed_scaling_factor

    def select(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return self.route(x.astype(mx.float32) @ self.router)

    def experts(self, x: mx.array, idx: mx.array) -> mx.array:
        """Rows x [R, D] through their experts idx [R, k]: [R, k, D]."""

        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        h = mx.expand_dims(x, (-2, -3))
        do_sort = idx.size >= 64
        order = None
        ids = idx
        if do_sort:
            h, ids, order = _gather_sort(h, idx)

        def run(q: Q, inp: mx.array) -> mx.array:
            return mx.gather_qmm(inp, q.weight, q.scales, q.biases, rhs_indices=ids, transpose=True,
                                 group_size=q.group, bits=q.bits, sorted_indices=do_sort)

        act = swiglu(run(self.gate, h), run(self.up, h), self.cfg.swiglu_limit)
        y = run(self.down, act)
        if do_sort:
            y = _scatter_unsort(y, order, idx.shape)
        return y.squeeze(-2)

    def expert_rows(self, x: mx.array, idx: mx.array) -> mx.array:
        """A window's rows x [R, D] through their experts idx [R, k] -> [R, k, D], every pick with the bits its
        one-row call (``experts`` on one row) gives it; each distinct expert's weights read once for the window."""

        group = K.expert_group(idx, int(self.gate.weight.shape[0]))
        g = K.expert_qmv(x, idx, group, self.gate, per_pick=False)
        u = K.expert_qmv(x, idx, group, self.up, per_pick=False)
        act = swiglu(g, u, self.cfg.swiglu_limit)
        return K.expert_qmv(act, idx, group, self.down, per_pick=True)

    @staticmethod
    def combine(w: mx.array, y: mx.array, dtype: Any) -> mx.array:
        y = y.astype(mx.float32)                                        # [R, k, D]
        acc = w[:, 0:1] * y[:, 0]
        for j in range(1, int(y.shape[1])):
            acc = acc + w[:, j:j + 1] * y[:, j]
        return acc.astype(dtype)

    def __call__(self, x: mx.array, rows_exact: bool) -> mx.array:
        rows = int(x.shape[0])
        if rows_exact and "moe" in FUSED and self.fused_ok and F.metal():
            return F.moe_rows(self, x)
        if row_kernel("experts", rows, rows_exact):
            idx, w = self.route(self.logits(x, True))
            out = self.combine(w, self.expert_rows(x, idx), x.dtype)
        elif row_kernel("router", rows, rows_exact):
            idx, w = self.route(self.logits(x, True))
            out = mx.concatenate([self.combine(w[r:r + 1], self.experts(x[r:r + 1], idx[r:r + 1]), x.dtype)
                                  for r in range(rows)])
        else:
            def routed(one: mx.array) -> mx.array:
                idx, w = self.select(one)
                return self.combine(w, self.experts(one, idx), x.dtype)

            out = per_row(routed, x, rows_exact)
        if self.shared is not None:
            out = out + self.shared(x, rows_exact)
        return out


class KDA:
    """Kimi Delta Attention."""

    def __init__(self, w: dict[str, Any], cfg: Config) -> None:
        self.cfg = cfg
        self.heads, self.dim = cfg.linear_num_heads, cfg.linear_head_dim
        self.width = self.heads * self.dim
        # q, k, v, f_a, g_a and b read the same input: one matrix
        parts = [w["q_proj"], w["k_proj"], w["v_proj"], w["f_a_proj"], w["g_a_proj"], w["b_proj"]]
        self.cuts = []
        at = 0
        for p in parts[:-1]:
            at += p.outs
            self.cuts.append(at)
        self.in_proj = Q.stack(parts)
        self.f_b, self.g_b, self.o_proj = w["f_b_proj"], w["g_b_proj"], w["o_proj"]
        taps = [w[f"{c}_conv1d"] for c in "qkv"]                       # [C, 1, T] (torch) or [C, T, 1]
        conv = mx.concatenate([t.reshape(t.shape[0], -1) for t in taps])  # [3 width, T]
        self.taps = int(conv.shape[1])
        self.conv_w = mx.contiguous(conv.T.astype(mx.float32))          # [T, 3 width]
        self.A = mx.exp(w["A_log"].astype(mx.float32)).reshape(self.heads, 1)
        self.dt_bias = w["dt_bias"].astype(mx.float32).reshape(self.heads, self.dim)
        self.o_norm = w["o_norm"].astype(mx.float32)
        # the fused decode kernel's inputs
        self.A_flat = mx.contiguous(self.A.reshape(-1))
        self.dt_bias_flat = mx.contiguous(self.dt_bias.reshape(-1))
        self.lb_array = mx.array([cfg.linear_lower_bound], dtype=mx.float32)
        self.eps_array = mx.array([cfg.rms_norm_eps], dtype=mx.float32)
        self.fused = None

    @staticmethod
    def _small(q: Q, x: mx.array, decode: bool) -> mx.array:
        """f_b / g_b (128 inputs: MLX's one-row kernel for them is qmv_quad, which qmv_rows does not cover)."""

        rows = int(x.shape[0])
        if row_kernel("kda_proj", rows, decode) and K.qmv_quad_rows_fits(q, rows):
            return K.qmv_quad_rows(x, q)
        return per_row(lambda r: q(r), x, decode)

    def __call__(self, x: mx.array, cache: KDACache, decode: bool) -> mx.array:
        cfg = self.cfg
        rows = int(x.shape[0])
        h, d, width = self.heads, self.dim, self.width
        proj = project(x, self.in_proj, rows_exact=decode)
        if decode and FUSED_KDA:
            if self.fused is None:
                self.fused = KDA_K.fits(self)
            if self.fused:
                taps = self.taps
                conv = cache.conv if cache.conv is not None else mx.zeros((taps - 1, 3 * width), dtype=mx.bfloat16)
                entry = cache.ssm if cache.ssm is not None else mx.zeros((1, h, d, d), dtype=mx.float32)
                y, cache.ssm, cache.conv = KDA_K.kda_rows(self, proj, conv, entry)
                cache.offset += rows
                cache._replay = [rows, "fused", self, proj, conv, entry]
                return project(y, self.o_proj, rows_exact=True)
        mixed = proj[:, :self.cuts[2]]
        fa = proj[:, self.cuts[2]:self.cuts[3]]
        ga = proj[:, self.cuts[3]:self.cuts[4]]
        b = proj[:, self.cuts[4]:]
        taps = self.taps
        conv = cache.conv if cache.conv is not None else mx.zeros((taps - 1, 3 * width), dtype=mixed.dtype)
        ci = mx.concatenate([conv, mixed])                             # [taps - 1 + R, 3 width]
        acc = ci[0:rows].astype(mx.float32) * self.conv_w[0]
        for t in range(1, taps):
            acc = acc + ci[t:t + rows].astype(mx.float32) * self.conv_w[t]
        co = silu(acc.astype(mixed.dtype))
        q = co[:, :width].reshape(1, rows, h, d)
        k = co[:, width:2 * width].reshape(1, rows, h, d)
        v = co[:, 2 * width:].reshape(1, rows, h, d)
        # l2 norms as RMS norms: x / |x| = rms_norm(x, eps / d) / sqrt(d); q also carries d^-1/2
        eps = 1e-6 / d
        q = (mx.fast.rms_norm(q.astype(mx.float32), None, eps) * (1.0 / d)).astype(mx.bfloat16)
        k = (mx.fast.rms_norm(k.astype(mx.float32), None, eps) * (d ** -0.5)).astype(mx.bfloat16)
        a = self._small(self.f_b, fa, decode).reshape(1, rows, h, d)
        g = mx.exp(cfg.linear_lower_bound * mx.sigmoid(self.A * (a.astype(mx.float32) + self.dt_bias)))
        beta = mx.sigmoid(b).reshape(1, rows, h)
        entry = cache.ssm if cache.ssm is not None else mx.zeros((1, h, d, d), dtype=mx.float32)
        y, state = K.gated_delta(q, k, v, g, beta, entry)
        cache.conv = mx.contiguous(ci[rows:])
        cache.ssm = state
        cache.offset += rows
        cache._replay = [rows, ci, entry, q, k, v, g, beta] if decode else None
        gate = self._small(self.g_b, ga, decode).reshape(rows, h, d)
        o = mx.fast.rms_norm(y.reshape(rows, h, d).astype(mx.float32), self.o_norm, cfg.rms_norm_eps)
        o = (o * mx.sigmoid(gate.astype(mx.float32))).astype(mx.bfloat16).reshape(rows, width)
        return project(o, self.o_proj, rows_exact=decode)


class MLA:
    """DeepSeek sparse attention, NoPE MLA over a 512-wide latent, with the pooled-block indexer."""

    def __init__(self, w: dict[str, Any], cfg: Config) -> None:
        self.cfg = cfg
        self.heads = cfg.num_attention_heads
        self.nope, self.vdim, self.rank = cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        self.scale = self.nope ** -0.5
        self.q_a, self.q_b, self.kv_a, self.o_proj = w["q_a_proj"], w["q_b_proj"], w["kv_a_proj_with_mqa"], w["o_proj"]
        self.q_norm, self.kv_norm = w["q_a_layernorm"], w["kv_a_layernorm"]
        kvb: Q = w["kv_b_proj"]                                          # [H (nope + v), rank]
        per = self.nope + self.vdim

        def heads(a: mx.array, lo: int, hi: int) -> mx.array:
            return mx.contiguous(a.reshape(self.heads, per, -1)[:, lo:hi])

        self.wk = Q(heads(kvb.weight, 0, self.nope), heads(kvb.scales, 0, self.nope), heads(kvb.biases, 0, self.nope))
        self.wv = Q(heads(kvb.weight, self.nope, per), heads(kvb.scales, self.nope, per),
                    heads(kvb.biases, self.nope, per))
        # indexer
        self.iq, self.ik_proj, self.iw = w["indexer.wq_b"], w["indexer.wk"], w["indexer.weights_proj"]
        self.ik_norm_w, self.ik_norm_b = w["indexer.k_norm.weight"], w["indexer.k_norm.bias"]
        self.ape = w["indexer.index_kpool_compress_ape"]
        self.igate = mx.contiguous(w["indexer.index_kpool_compress_gate"].T)  # [D, 128]
        self.i_heads, self.i_dim = cfg.index_n_heads, cfg.index_head_dim
        self.i_scale = (self.i_heads ** -0.5) * (self.i_dim ** -0.5)
        # projections that read the same input as one matrix each (a one-row matmul gives every output row the
        # same bits stacked or not, checked); the per-projection names stay as row ranges of the stacked matrices
        self.x_proj = Q.stack([self.q_a, self.kv_a, self.ik_proj, self.iw])
        self.qr_proj = Q.stack([self.q_b, self.iq])
        at = [0]
        for p in (self.q_a, self.kv_a, self.ik_proj, self.iw):
            at.append(at[-1] + p.outs)
        self.x_cuts = at
        self.q_a, self.kv_a, self.ik_proj, self.iw = (_rows(self.x_proj, at[i], at[i + 1]) for i in range(4))
        nq = self.q_b.outs
        self.q_b, self.iq = _rows(self.qr_proj, 0, nq), _rows(self.qr_proj, nq, self.qr_proj.outs)

    # the per-head latent maps (kv_b in its stored layout)
    def absorb(self, q: mx.array) -> mx.array:
        """q_nope [H, n, nope] -> latent queries [H, n, rank]."""

        wk = self.wk
        return mx.quantized_matmul(q, wk.weight, wk.scales, wk.biases, transpose=False, group_size=wk.group,
                                   bits=wk.bits)

    def unabsorb(self, out: mx.array) -> mx.array:
        """latent outputs [H, n, rank] -> values [H, n, v]."""

        wv = self.wv
        return mx.quantized_matmul(out, wv.weight, wv.scales, wv.biases, transpose=True, group_size=wv.group,
                                   bits=wv.bits)

    def keys_values(self, lat: mx.array) -> tuple[mx.array, mx.array]:
        """Per-head keys and values [H, n, 256] of latent keys [n, rank] (prefill)."""

        wk, wv = self.wk, self.wv
        k = mx.quantized_matmul(lat[None], wk.weight, wk.scales, wk.biases, transpose=True, group_size=wk.group,
                                bits=wk.bits)
        v = mx.quantized_matmul(lat[None], wv.weight, wv.scales, wv.biases, transpose=True, group_size=wv.group,
                                bits=wv.bits)
        return k, v

    def index_scores(self, iq: mx.array, iw: mx.array, pool: mx.array) -> mx.array:
        """Block scores [n, P] = sum over indexer heads of w_h relu(q_h . pool) (iq [n, HI, DI], iw [n, HI])."""

        s = iq @ pool.T                                                  # [n, HI, P]
        return mx.sum(iw[..., None] * mx.maximum(s, mx.array(0, s.dtype)), axis=1)

    def selected(self, scores: mx.array, position: int) -> mx.array:
        """Key ids of one query at ``position`` past ``index_topk`` keys: its best blocks' keys, then its tail."""

        cfg = self.cfg
        kp = cfg.index_kpool
        scores = scores.reshape(-1)
        blocks = int(scores.shape[0])
        top = min(cfg.index_topk // kp, blocks)
        pick = mx.argpartition(-scores, kth=top - 1)[:top]
        ids = (pick[:, None] * kp + mx.arange(kp)[None]).reshape(-1)
        tail = (position + 1) % kp
        if cfg.index_tail and tail:
            ids = mx.concatenate([ids, mx.arange(position + 1 - tail, position + 1)])
        return ids

    def __call__(self, x: mx.array, cache: MLACache, decode: bool) -> mx.array:
        cfg = self.cfg
        rows = int(x.shape[0])
        H = self.heads
        c = self.x_cuts
        if decode:
            # the stacked matrices (prefill keeps the separate ones: MLX's batched kernel tiles by width)
            xp = project(x, self.x_proj, rows_exact=True)               # q_a | kv_a | indexer k | indexer weights
            parts = [xp[:, c[i]:c[i + 1]] for i in range(4)]
        else:
            parts = [project(x, p, rows_exact=False) for p in (self.q_a, self.kv_a, self.ik_proj, self.iw)]
        qr = mx.fast.rms_norm(parts[0], self.q_norm, cfg.rms_norm_eps)
        if decode:
            qp = project(qr, self.qr_proj, rows_exact=True)             # q_b | indexer q
            q, iq = qp[:, :self.q_b.outs], qp[:, self.q_b.outs:]
        else:
            q, iq = project(qr, self.q_b, rows_exact=False), project(qr, self.iq, rows_exact=False)
        q = q.reshape(rows, H, self.nope)
        lat = mx.fast.rms_norm(parts[1], self.kv_norm, cfg.rms_norm_eps)
        iq = iq.reshape(rows, self.i_heads, self.i_dim)
        ik = mx.fast.layer_norm(parts[2], self.ik_norm_w, self.ik_norm_b, 1e-6)
        if row_kernel("igate", rows, decode):
            ig = K.matmul_rows(x, self.igate, transposed=True)
        else:
            ig = per_row(lambda r: r @ self.igate, x, decode)
        iw = (parts[3] * self.i_scale).astype(mx.bfloat16)
        start = cache.offset
        cache.append(lat, ik, ig, self.ape, cfg.index_kpool)
        if decode and row_kernel("mla_proj", rows, decode):
            # the latent maps with the rows as a batch (each keeps its one-row bits), attention row by row
            ql = mx.quantized_matmul(q[:, :, None, :], self.wk.weight, self.wk.scales, self.wk.biases,
                                     transpose=False, group_size=self.wk.group, bits=self.wk.bits)   # [R, H, 1, rank]
            sparse = [r for r in range(rows) if SPARSE_KERNEL and start + r + 1 > cfg.index_topk]
            if row_kernel("indexer", rows, decode):
                sels = self._choices(iq, iw, cache, start, skip=sparse)
            else:
                sels = [...] * rows
            parts: list[Any] = [None] * rows
            if sparse:
                at = mx.array(sparse)
                idx = self._sparse_indices(iq[at], iw[at], cache, [start + r for r in sparse])
                got = SA.indexed_attention(ql[at][:, :, 0, :], cache.keys, idx, cache.offset, self.scale)
                for i, r in enumerate(sparse):
                    parts[r] = got[i][None, :, None, :]
            for r in range(rows):
                if parts[r] is None:
                    parts[r] = self._attend(ql[r], iq[r], iw[r], cache, start + r, sels[r])
            att = mx.concatenate(parts) if rows > 1 else parts[0]
            wv = self.wv
            out = mx.quantized_matmul(att, wv.weight, wv.scales, wv.biases, transpose=True, group_size=wv.group,
                                      bits=wv.bits).reshape(rows, -1)
        elif decode:
            outs = [self._decode_row(q[r], iq[r], iw[r], cache, start + r) for r in range(rows)]
            out = mx.concatenate(outs)
        else:
            out = self._prefill(q, iq, iw, cache, start)
        return project(out, self.o_proj, rows_exact=decode)

    def _sparse_indices(self, iq: mx.array, iw: mx.array, cache: MLACache, positions: list[int]) -> mx.array:
        """The key ids rows at ``positions`` (each past ``index_topk`` keys) read, as the sparse attention kernel
        takes them: [m, index_topk + kpool - 1] int32, the chosen blocks' keys then the row's unfinished tail, -1
        past its end. The choice is ``selected``'s, made as ``_choices`` makes it (rows with the same number of
        complete blocks scored and ranked together; each row keeps its one-row bits there)."""

        cfg = self.cfg
        kp = cfg.index_kpool
        width = cfg.index_topk + (kp - 1 if cfg.index_tail else 0)
        out: list[Any] = [None] * len(positions)
        groups: dict[int, list[int]] = {}
        for i, p in enumerate(positions):
            groups.setdefault((p + 1) // kp, []).append(i)
        for blocks, members in groups.items():
            at = mx.array(members) if len(members) < len(positions) else None
            scores = self.index_scores(iq if at is None else iq[at], iw if at is None else iw[at], cache.pool[:blocks])
            top = min(cfg.index_topk // kp, blocks)
            picks = mx.argpartition(-scores, kth=top - 1, axis=-1)[:, :top]
            ids = (picks[:, :, None] * kp + mx.arange(kp)[None, None]).reshape(len(members), -1).astype(mx.int32)
            tails = []
            for i in members:
                p = positions[i]
                t = (p + 1) % kp if cfg.index_tail else 0
                tails.append([p + 1 - t + j for j in range(t)] + [-1] * (width - top * kp - t))
            if width > top * kp:
                ids = mx.concatenate([ids, mx.array(tails, dtype=mx.int32)], axis=1)
            for j, i in enumerate(members):
                out[i] = ids[j:j + 1]
        return mx.concatenate(out) if len(out) > 1 else out[0]

    def _choices(self, iq: mx.array, iw: mx.array, cache: MLACache, start: int, skip: Any = ()) -> list[Any]:
        """The key ids each row of a window reads (None: all its keys), ``selected``'s choice for every row: rows
        with the same number of complete blocks are scored and ranked together, with the rows as the matmul's and
        the partition's batch (each row keeps its bits there; checked on the M3 Ultra, MLX 0.32.0)."""

        cfg = self.cfg
        kp = cfg.index_kpool
        rows = int(iq.shape[0])
        out: list[Any] = [None] * rows
        groups: dict[int, list[int]] = {}
        for r in range(rows):
            if start + r + 1 > cfg.index_topk and r not in skip:
                groups.setdefault((start + r + 1) // kp, []).append(r)
        for blocks, members in groups.items():
            at = mx.array(members)
            scores = self.index_scores(iq[at], iw[at], cache.pool[:blocks])          # [m, P]
            top = min(cfg.index_topk // kp, blocks)
            picks = mx.argpartition(-scores, kth=top - 1, axis=-1)[:, :top]
            for i, r in enumerate(members):
                ids = (picks[i][:, None] * kp + mx.arange(kp)[None]).reshape(-1)
                position = start + r
                tail = (position + 1) % kp
                if cfg.index_tail and tail:
                    ids = mx.concatenate([ids, mx.arange(position + 1 - tail, position + 1)])
                out[r] = ids
        return out

    def _attend(self, ql: mx.array, iq: mx.array, iw: mx.array, cache: MLACache, position: int,
                sel: Any = ...) -> mx.array:
        """One query at ``position`` (latent queries ql [H, 1, rank]) over its own keys: [1, H, 1, rank]. ``sel``:
        its key ids if already chosen (``_choices``; None: all its keys)."""

        cfg = self.cfg
        n = position + 1
        keys = cache.keys[:n]
        if sel is not ...:
            if sel is not None:
                keys = mx.take(keys, sel, axis=0)
        elif n > cfg.index_topk:
            blocks = n // cfg.index_kpool
            scores = self.index_scores(iq[None], iw[None], cache.pool[:blocks])
            keys = mx.take(keys, self.selected(scores, position), axis=0)
        return mx.fast.scaled_dot_product_attention(ql[None], keys[None, None], keys[None, None], scale=self.scale)

    def _decode_row(self, q: mx.array, iq: mx.array, iw: mx.array, cache: MLACache, position: int) -> mx.array:
        """One query at ``position`` (q [H, nope]) over its own keys: absorbed attention on the latents."""

        ql = self.absorb(q[:, None, :])                                  # [H, 1, rank]
        if SPARSE_KERNEL and position + 1 > self.cfg.index_topk:
            idx = self._sparse_indices(iq[None], iw[None], cache, [position])
            out = SA.indexed_attention(ql[:, 0, :][None], cache.keys, idx, cache.offset, self.scale)[:, :, None, :]
        else:
            out = self._attend(ql, iq, iw, cache, position)
        return self.unabsorb(out[0]).reshape(1, -1)                      # [1, H v]

    def _prefill(self, q: mx.array, iq: mx.array, iw: mx.array, cache: MLACache, start: int,
                 chunk: int = 512) -> mx.array:
        cfg = self.cfg
        rows = int(q.shape[0])
        end = start + rows
        k, v = self.keys_values(cache.keys[:end])                       # [H, end, 256]
        qh = q.transpose(1, 0, 2)                                       # [H, rows, nope]
        kp = cfg.index_kpool
        outs = []
        for c0 in range(0, rows, chunk):
            c1 = min(c0 + chunk, rows)
            pos = mx.arange(start + c0, start + c1)                     # query positions
            last = start + c1                                           # keys this chunk reads: [0, last)
            key_pos = mx.arange(last)
            causal = key_pos[None] <= pos[:, None]
            if last > cfg.index_topk:
                blocks = last // kp
                scores = self.index_scores(iq[c0:c1], iw[c0:c1], cache.pool[:blocks])      # [c, P]
                valid = (mx.arange(blocks)[None] * kp + kp - 1) <= pos[:, None]
                scores = mx.where(valid, scores, mx.array(-1e30, scores.dtype))
                top = min(cfg.index_topk // kp, blocks)
                pick = mx.argpartition(-scores, kth=top - 1, axis=-1)[..., :top]
                picked_valid = mx.take_along_axis(valid, pick, axis=-1)
                block_of_key = key_pos // kp
                chosen = mx.zeros((c1 - c0, blocks + 1), dtype=mx.bool_)
                safe = mx.where(picked_valid, pick, blocks)
                chosen = mx.put_along_axis(chosen, safe, mx.array(True), axis=-1)[:, :blocks]
                in_block = mx.take(chosen, mx.minimum(block_of_key, blocks - 1), axis=1) & (block_of_key[None] < blocks)
                tail_start = pos + 1 - (pos + 1) % kp
                in_tail = (key_pos[None] >= tail_start[:, None]) & causal
                if not cfg.index_tail:
                    in_tail = mx.zeros_like(in_tail)
                # queries with at most index_topk keys read them all (every block of theirs is chosen)
                dense = (pos + 1 <= cfg.index_topk)[:, None]
                mask = mx.where(dense, causal, (in_block & causal) | in_tail)
            else:
                mask = causal
            o = mx.fast.scaled_dot_product_attention(qh[None, :, c0:c1], k[None, :, :last], v[None, :, :last],
                                                     scale=self.scale, mask=mask[None, None])
            outs.append(o[0].transpose(1, 0, 2).reshape(c1 - c0, -1))
        return mx.concatenate(outs) if len(outs) > 1 else outs[0]


class Layer:
    def __init__(self, attn: Any, mlp: Any, in_norm: mx.array, post_norm: mx.array, attn_hc: HC | None,
                 ffn_hc: HC | None, cfg: Config) -> None:
        self.attn, self.mlp = attn, mlp
        self.is_linear = isinstance(attn, KDA)
        self.in_norm, self.post_norm = in_norm, post_norm
        self.attn_hc, self.ffn_hc = attn_hc, ffn_hc
        self.eps = cfg.rms_norm_eps

    def __call__(self, x: mx.array, cache: Any, decode: bool) -> mx.array:
        """x [R, S, D] streams (or [R, D] for the plain MTP layer)."""

        if self.attn_hc is None:                                        # plain pre-norm residual block
            x = x + self.attn(mx.fast.rms_norm(x, self.in_norm, self.eps), cache, decode)
            return x + self.mlp(mx.fast.rms_norm(x, self.post_norm, self.eps), decode)
        xc, post, comb = self.attn_hc.split(x, decode)
        x = hc_expand(self.attn(mx.fast.rms_norm(xc, self.in_norm, self.eps), cache, decode), x, post, comb, decode)
        xc, post, comb = self.ffn_hc.split(x, decode)
        return hc_expand(self.mlp(mx.fast.rms_norm(xc, self.post_norm, self.eps), decode), x, post, comb, decode)


class GLM5:
    """The backbone: ``hidden`` (final-normed hidden states) and ``head`` (logits), with the raw (pre-norm, streams
    collapsed) hidden of the last call in ``last_raw`` for the MTP head."""

    def __init__(self, cfg: Config, embed: Q, layers: list[Layer], norm: mx.array, lm_head: Q) -> None:
        self.args = cfg
        self.embed = embed
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.last_raw: mx.array | None = None

    def make_cache(self) -> list[Any]:
        return [KDACache() if layer.is_linear else MLACache() for layer in self.layers]

    def hc_fused_ok(self) -> bool:
        ok = self.__dict__.get("_hc_ok")
        if ok is None:
            dims = int(self.args.hidden_size)
            ok = F.metal() and all(layer.attn_hc is not None and F.hc_fits(layer.attn_hc, dims)
                                   and F.hc_fits(layer.ffn_hc, dims) for layer in self.layers)
            self._hc_ok = ok
        return ok and F.metal()

    def embed_tokens(self, tokens: mx.array) -> mx.array:
        e = self.embed
        ids = tokens.reshape(-1)
        return mx.dequantize(e.weight[ids], e.scales[ids], e.biases[ids], group_size=e.group, bits=e.bits)

    def hidden(self, tokens: Any, cache: list[Any]) -> mx.array:
        ids = mx.array(tokens).reshape(-1).astype(mx.uint32)
        rows = int(ids.shape[0])
        decode = rows <= DECODE_ROWS
        h = self.embed_tokens(ids)                                       # [R, D]
        x = mx.contiguous(mx.broadcast_to(h[:, None, :], (rows, self.args.hc_mult, h.shape[-1])))
        if decode and "hc" in FUSED and self.hc_fused_ok():
            # each block boundary in one fused step: the previous block's write-back, the next block's split + norm
            eps = self.args.rms_norm_eps
            pending = None
            for i, (layer, c) in enumerate(zip(self.layers, cache)):
                x, normed, post, comb = F.hc_step(x, pending, layer.attn_hc, layer.in_norm, eps)
                pending = (layer.attn(normed, c, decode), post, comb)
                x, normed, post, comb = F.hc_step(x, pending, layer.ffn_hc, layer.post_norm, eps)
                pending = (layer.mlp(normed, decode), post, comb)
                if EVAL_EVERY and (i + 1) % EVAL_EVERY == 0 and i + 1 < len(self.layers):
                    mx.async_eval(x, *pending)
            x = F.hc_step(x, pending, None, None, eps)[0]
        else:
            for i, (layer, c) in enumerate(zip(self.layers, cache)):
                x = layer(x, c, decode)
                if decode and EVAL_EVERY and (i + 1) % EVAL_EVERY == 0 and i + 1 < len(self.layers):
                    mx.async_eval(x)
        xs = x.astype(mx.float32)
        raw = xs[:, 0]
        for s in range(1, int(x.shape[1])):
            raw = raw + xs[:, s]
        raw = (raw * (1.0 / int(x.shape[1]))).astype(x.dtype)
        self.last_raw = raw
        return mx.fast.rms_norm(raw, self.norm, self.args.rms_norm_eps)[None]

    def head(self, hidden: mx.array) -> mx.array:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        return project(flat, self.lm_head, rows_exact=int(flat.shape[0]) <= DECODE_ROWS).reshape(*shape[:-1], -1)

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        for c in cache[:len(self.layers)]:
            if isinstance(c, KDACache):
                c.keep(rows, keep)
            else:
                c.trim(rows - keep)


# -- loading ------------------------------------------------------------------------------------------------------
_FP32 = ("A_log", "dt_bias", "mlp.gate.weight", "e_score_correction_bias", "_fn", "_base", "_scale")


class Weights:
    """The checkpoint's language-model tensors by name (``layers.N....``, ``lm_head``, ``embed_tokens``, ``norm``),
    read shard by shard as they are asked for."""

    def __init__(self, model_dir: Path) -> None:
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.dir = model_dir
        self.where: dict[str, str] = {}
        for name, shard in index.items():
            short = _short(name)
            if short is not None:
                self.where[short] = shard
        self._shard: tuple[str, dict[str, mx.array]] | None = None
        self._cache: dict[str, dict[str, mx.array]] = {}

    def has(self, name: str) -> bool:
        return name in self.where

    def get(self, name: str) -> mx.array:
        shard = self.where[name]
        loaded = self._cache.get(shard)
        if loaded is None:
            raw = mx.load(str(self.dir / shard))
            loaded = {}
            for full, value in raw.items():
                short = _short(full)
                if short is not None:
                    loaded[short] = value
            self._cache = {shard: loaded}          # one shard at a time: arrays already taken stay alive
        return loaded[name]

    def q(self, prefix: str) -> Q:
        return Q(self.get(f"{prefix}.weight"), self.get(f"{prefix}.scales"), self.get(f"{prefix}.biases"))


def _short(name: str) -> str | None:
    if name.startswith("model.language_model."):
        return name[len("model.language_model."):]
    if name.startswith("language_model.model."):
        return name[len("language_model.model."):]
    if name.startswith("lm_head.") or name.startswith("language_model.lm_head."):
        return name[name.index("lm_head."):]
    return None


def _materialize(*arrays: Any) -> None:
    flat: list[mx.array] = []
    for a in arrays:
        if isinstance(a, Q):
            flat += a.arrays()
        elif isinstance(a, mx.array):
            flat.append(a)
    mx.eval(*flat)


def load_layer(w: Weights, i: int, cfg: Config, *, plain: bool = False) -> Layer:
    p = f"layers.{i}"
    attn_prefix = f"{p}.self_attn"
    if w.has(f"{attn_prefix}.q_a_proj.weight"):
        names = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "indexer.wq_b", "indexer.wk",
                 "indexer.weights_proj"]
        aw: dict[str, Any] = {n: w.q(f"{attn_prefix}.{n}") for n in names}
        for n in ("q_a_layernorm", "kv_a_layernorm"):
            aw[n] = w.get(f"{attn_prefix}.{n}.weight")
        for n in ("indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.index_kpool_compress_ape",
                  "indexer.index_kpool_compress_gate"):
            aw[n] = w.get(f"{attn_prefix}.{n}")
        attn: Any = MLA(aw, cfg)
        _materialize(attn.x_proj, attn.qr_proj, attn.q_a, attn.q_b, attn.kv_a, attn.o_proj, attn.wk, attn.wv, attn.iq,
                     attn.ik_proj, attn.iw,
                     attn.q_norm, attn.kv_norm, attn.ik_norm_w, attn.ik_norm_b, attn.ape, attn.igate)
    else:
        names = ["q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "b_proj", "o_proj"]
        aw = {n: w.q(f"{attn_prefix}.{n}") for n in names}
        for n in ("q_conv1d", "k_conv1d", "v_conv1d", "o_norm"):
            aw[n] = w.get(f"{attn_prefix}.{n}.weight")
        aw["A_log"] = w.get(f"{attn_prefix}.A_log")
        aw["dt_bias"] = w.get(f"{attn_prefix}.dt_bias")
        attn = KDA(aw, cfg)
        _materialize(attn.in_proj, attn.f_b, attn.g_b, attn.o_proj, attn.conv_w, attn.A, attn.dt_bias, attn.o_norm)
    m = f"{p}.mlp"
    if w.has(f"{m}.gate.weight"):
        shared = None
        if w.has(f"{m}.shared_experts.gate_proj.weight"):
            shared = DenseMLP(w.q(f"{m}.shared_experts.gate_proj"), w.q(f"{m}.shared_experts.up_proj"),
                              w.q(f"{m}.shared_experts.down_proj"), cfg.swiglu_limit)
        stacked = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if w.has(f"{m}.switch_mlp.{proj}.weight"):
                stacked.append(w.q(f"{m}.switch_mlp.{proj}"))
                continue
            parts = [w.q(f"{m}.experts.{e}.{proj}") for e in range(cfg.n_routed_experts)]
            q = Q(mx.stack([x.weight for x in parts]), mx.stack([x.scales for x in parts]),
                  mx.stack([x.biases for x in parts]))
            _materialize(q)
            stacked.append(q)
        mlp: Any = MoE(w.get(f"{m}.gate.weight"), w.get(f"{m}.gate.e_score_correction_bias"), *stacked, shared, cfg)
        _materialize(mlp.router, mlp.bias, mlp.scale_arr, mlp.limit_arr, mlp.router_packed,
                     *(mlp.shared.gate_up, mlp.shared.down) if shared else ())
    else:
        mlp = DenseMLP(w.q(f"{m}.gate_proj"), w.q(f"{m}.up_proj"), w.q(f"{m}.down_proj"), cfg.swiglu_limit)
        _materialize(mlp.gate_up, mlp.down)
    in_norm = w.get(f"{p}.input_layernorm.weight")
    post_norm = w.get(f"{p}.post_attention_layernorm.weight")
    attn_hc = ffn_hc = None
    if not plain:
        attn_hc = HC(w.get(f"{p}.hc_attn_fn"), w.get(f"{p}.hc_attn_base"), w.get(f"{p}.hc_attn_scale"), cfg)
        ffn_hc = HC(w.get(f"{p}.hc_ffn_fn"), w.get(f"{p}.hc_ffn_base"), w.get(f"{p}.hc_ffn_scale"), cfg)
        _materialize(attn_hc.fn, attn_hc.base, attn_hc.scale, ffn_hc.fn, ffn_hc.base, ffn_hc.scale,
                     attn_hc.fn_packed, ffn_hc.fn_packed)
    _materialize(in_norm, post_norm)
    return Layer(attn, mlp, in_norm, post_norm, attn_hc, ffn_hc, cfg)


def load_backbone(model_dir: Path, *, layers: int | None = None) -> GLM5:
    """The backbone from a checkpoint directory, layer by layer (each layer's tensors evaluated as it is built).
    ``layers``: only the first that many (checks on the real weights without the whole model's memory)."""

    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    cfg = Config.from_dict(config)
    w = Weights(model_dir)
    count = cfg.num_hidden_layers if layers is None else min(int(layers), cfg.num_hidden_layers)
    layers = [load_layer(w, i, cfg) for i in range(count)]
    embed = w.q("embed_tokens")
    lm_head = w.q("lm_head")
    norm = w.get("norm.weight")
    _materialize(embed, lm_head, norm)
    model = GLM5(cfg, embed, layers, norm, lm_head)
    model.weights = w                                                    # the MTP head reads its layer from it
    return model


def load(model_dir: Path) -> tuple[GLM5, Any]:
    from mlx_lm.utils import load_tokenizer

    model = load_backbone(Path(model_dir))
    tokenizer = load_tokenizer(Path(model_dir), eos_token_ids=model.args.eos_token_id or None)
    return model, tokenizer


__all__ = ["Config", "GLM5", "KDACache", "MLACache", "Q", "load", "load_backbone", "load_layer", "project"]
