"""DeepSeek-V4's MoE: 256 mxfp4 routed experts, 6 a token (by a token-id table in the first layers), one shared."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.families.deepseek_v4 import config as C
from tensorfold.families.deepseek_v4.config import Config, row_kernel
from tensorfold.families.deepseek_v4.dense import dense
from tensorfold.kernels import inputs
from tensorfold.families.glm5_next.linear import Q, per_row
from tensorfold.kernels.deepseek.v4 import moe as V4MK
from tensorfold.kernels.deepseek.v4 import rows as RK
from tensorfold.kernels.glm.flash.v1 import kernels as K
from tensorfold.kernels.glm.flash.v1 import moe as MK


class FP4:
    """A stack of routed experts' matrices in MLX's mxfp4 layout: codes [E, N, K / 8] uint32, scales [E, N, K / 32]."""

    def __init__(self, weight: mx.array, scales: mx.array) -> None:
        if int(weight.shape[-1]) * 8 != int(scales.shape[-1]) * 32 or weight.shape[:-1] != scales.shape[:-1]:
            raise ValueError(f"mxfp4 codes {tuple(weight.shape)} do not fit scales {tuple(scales.shape)}")
        self.weight, self.scales = weight, scales

    def arrays(self) -> list[mx.array]:
        return [self.weight, self.scales]

    def __call__(self, x: mx.array, ids: mx.array, sort: bool) -> mx.array:
        return mx.gather_qmm(x, self.weight, self.scales, None, rhs_indices=ids, transpose=True, group_size=32,
                             bits=4, mode="mxfp4", sorted_indices=sort)


def swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit:
        up = mx.clip(up, -limit, limit)
        gate = mx.minimum(gate, limit)
    return nn.silu(gate) * up


class Shared:
    def __init__(self, gate: Q, up: Q, down: Q, limit: float) -> None:
        self.gate_up = Q.stack([gate, up])
        self.width = gate.outs
        self.down = down
        self.limit = limit

    def __call__(self, x: mx.array, rows_exact: bool) -> mx.array:
        gu = dense(x, self.gate_up, rows_exact)
        act = swiglu(gu[:, :self.width], gu[:, self.width:], self.limit)
        return dense(act, self.down, rows_exact)


class MoE:
    """sqrt(softplus) router scores; the top 6 by score + bias (or the token's table row), renormalised x 1.5."""

    def __init__(self, gate_w: mx.array, bias: mx.array | None, table: mx.array | None, gate: FP4, up: FP4,
                 down: FP4, shared: Shared, cfg: Config) -> None:
        self.router = mx.contiguous(gate_w.astype(mx.float32).T)          # [D, E]
        # the stored bf16 router repacked for GLM's router kernel (exact: bf16 -> fp32 loses nothing)
        self.router_packed = (MK.pack_router(gate_w) if gate_w.dtype == mx.bfloat16 and gate_w.shape[0] % 16 == 0
                              and gate_w.shape[1] % 32 == 0 else None)
        self.bias = None if bias is None else bias.astype(mx.float32)
        self.table = None if table is None else table.astype(mx.int32)
        self.zero_bias = mx.zeros((int(gate_w.shape[0]),), dtype=mx.float32)
        self.no_table = mx.zeros((inputs.MIN_ELEMENTS,), dtype=mx.int32)
        self.gate, self.up, self.down = gate, up, down
        self.shared = shared
        self.top = cfg.num_experts_per_tok
        self.scale = cfg.routed_scaling_factor
        self.limit = cfg.swiglu_limit
        self.scale_arr = mx.array([cfg.routed_scaling_factor], dtype=mx.float32)
        self.limit_arr = mx.array([cfg.swiglu_limit], dtype=mx.float32)
        self._compiled: Any = None

    def scores(self, x: mx.array, rows_exact: bool) -> mx.array:
        """Router scores [R, E] in fp32, each decode row with its one-row matmul's bits."""

        xf = x.astype(mx.float32)
        if row_kernel("router", int(x.shape[0]), rows_exact):
            logits = K.matmul_rows(xf, self.router, transposed=True)
        else:
            logits = per_row(lambda r: r @ self.router, xf, rows_exact)
        return mx.sqrt(mx.logaddexp(logits, mx.array(0.0)))

    def route(self, scores: mx.array, ids: mx.array) -> tuple[mx.array, mx.array]:
        """Experts [R, k] in ascending id (the reference's summation order) and their weights."""

        if self.table is not None:
            idx = self.table[ids]
        else:
            idx = mx.argpartition(-(scores + self.bias), kth=self.top - 1, axis=-1)[..., :self.top]
        idx = mx.sort(idx, axis=-1)
        w = mx.take_along_axis(scores, idx, axis=-1)
        total = w[:, 0:1]
        for j in range(1, self.top):
            total = total + w[:, j:j + 1]
        return idx, w / (total + 1e-20) * self.scale

    def experts(self, x: mx.array, idx: mx.array) -> mx.array:
        """Rows x [R, D] through their experts idx [R, k]: [R, k, D] (MLX's SwitchGLU call)."""

        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        h = mx.expand_dims(x, (-2, -3))
        sort = idx.size >= 64
        order, ids = None, idx
        if sort:
            h, ids, order = _gather_sort(h, idx)
        act = swiglu(self.gate(h, ids, sort), self.up(h, ids, sort), self.limit)
        y = self.down(act, ids, sort)
        if sort:
            y = _scatter_unsort(y, order, idx.shape)
        return y.squeeze(-2)

    @staticmethod
    def combine(w: mx.array, y: mx.array, dtype: Any) -> mx.array:
        """Routed outputs [R, k, D] weighted and summed in fp32 in slot (ascending expert) order, elementwise."""

        y = y.astype(mx.float32)
        acc = w[:, 0:1] * y[:, 0]
        for j in range(1, int(y.shape[1])):
            acc = acc + w[:, j:j + 1] * y[:, j]
        return acc.astype(dtype)

    def expert_rows(self, x: mx.array, idx: mx.array) -> mx.array:
        """A window's rows through their experts, each pick with its one-row call's bits, each expert read once."""

        group = K.expert_group(idx, int(self.gate.weight.shape[0]))
        g = RK.expert_rows_fp4(x, idx, group, self.gate, per_pick=False)
        u = RK.expert_rows_fp4(x, idx, group, self.up, per_pick=False)
        return RK.expert_rows_fp4(swiglu(g, u, self.limit), idx, group, self.down, per_pick=True)

    def __call__(self, x: mx.array, ids: mx.array, rows_exact: bool) -> mx.array:
        """Decode windows through the compiled block (one trace a row count), prompt chunks as they come."""

        if not rows_exact:
            return self.forward(x, ids, False)
        if self._compiled is None:
            self._compiled = mx.compile(lambda xs, ts: self.forward(xs, ts, True))
        return self._compiled(x, ids)

    def forward(self, x: mx.array, ids: mx.array, rows_exact: bool) -> mx.array:
        rows = int(x.shape[0])
        if rows_exact and "moe" in C.ENABLED and V4MK.fits(self, rows) and MK.router_fits(self):
            wts, uids, umem, ucount = V4MK.route(MK.router_rows(x, self), self, ids)   # bf16 read as fp32
            y = V4MK.routed(x, self, wts, uids, umem, ucount)
            return V4MK.combine(y, self.shared(x, rows_exact), self.top)
        idx, w = self.route(self.scores(x, rows_exact), ids)
        if row_kernel("experts", rows, rows_exact) and RK.fp4_rows_fits(self.gate, rows):
            y = self.expert_rows(x, idx)
        elif rows_exact and rows > 1:
            y = mx.concatenate([self.experts(x[r:r + 1], idx[r:r + 1]) for r in range(rows)])
        else:
            y = self.experts(x, idx)
        return self.combine(w, y, x.dtype) + self.shared(x, rows_exact)

    def arrays(self) -> list[mx.array]:
        out = [self.router, *self.gate.arrays(), *self.up.arrays(), *self.down.arrays(), *self.shared.gate_up.arrays(),
               *self.shared.down.arrays()]
        return out + [a for a in (self.bias, self.table) if a is not None]
