"""The feed-forward blocks: the dense SwiGLU MLP, and the MoE of 288 routed experts and a shared one."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.glm5_next import config as C
from tensorfold.families.glm5_next.config import Config, row_kernel
from tensorfold.families.glm5_next.linear import Q, per_row, project, silu
from tensorfold.kernels.glm.flash.v1 import moe as MK
from tensorfold.kernels.glm.flash.v1 import kernels as K


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
        self.router_packed = (MK.pack_router(gate_w) if gate_w.dtype == mx.bfloat16 and gate_w.shape[0] % 16 == 0
                              and gate_w.shape[1] % 32 == 0 else None)
        self.bias = bias.astype(mx.float32)
        self.gate, self.up, self.down = gate, up, down
        self.shared = shared
        self.cfg = cfg
        self.scale_arr = mx.array([cfg.routed_scaling_factor], dtype=mx.float32)
        self.limit_arr = mx.array([cfg.swiglu_limit or 3.0e38], dtype=mx.float32)
        self.fused_ok = MK.moe_fits(self)

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

    def experts(self, x: mx.array, idx: mx.array, qs: tuple[Q, Q, Q] | None = None) -> mx.array:
        """Rows x [R, D] through their experts idx [R, k] (of ``qs``, else the resident stacks): [R, k, D]."""

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

        gate, up, down = qs or (self.gate, self.up, self.down)
        act = swiglu(run(gate, h), run(up, h), self.cfg.swiglu_limit)
        y = run(down, act)
        if do_sort:
            y = _scatter_unsort(y, order, idx.shape)
        return y.squeeze(-2)

    def expert_rows(self, x: mx.array, idx: mx.array) -> mx.array:
        """A window's rows through their experts, each pick with its one-row call's bits, each expert read once."""

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
        if "streamer" in self.__dict__:                                 # routed experts from the slot pool
            from tensorfold.families.glm5_next import stream

            return stream.moe(self, x, rows_exact)
        rows = int(x.shape[0])
        if rows_exact and "moe" in C.FUSED and self.fused_ok and K.metal():
            return MK.moe_rows(self, x)
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

