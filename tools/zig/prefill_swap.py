"""mlx_lm's NemotronH prefill ops with our kernels swapped in where they exist, everything else exactly as mlx_lm writes it."""

from __future__ import annotations

from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn

import prefill_launch as pl


def qmm_split_k(M: int, N: int, K: int) -> int:
    """MLX's qmm_splitk partition count for a transposed 4-bit projection (1: it runs affine_qmm_t_nax instead)."""

    split = max(1, 512 // (-(-N // 32) * -(-M // 32)))
    split = min(split, K // 64)
    while split > 1 and K % (split * 64) != 0:
        split -= 1
    return split


class Swap:
    """Our kernels behind the call sites mlx_lm's NemotronH blocks reach during a prompt chunk."""

    def __init__(self, runner) -> None:
        self.runner = runner
        self.calls: dict[str, int] = {}             # our launches by the MLX kernel each replaced

    def run(self, launches) -> mx.array:
        name = launches[0].split("_has_batch")[0].split("_align")[0].split(" + ")[0]
        self.calls[name] = self.calls.get(name, 0) + 1
        return pl.run(self.runner, launches[1], verify=False)

    def ssm_attn(self, x, A_log, B, C, D, dt, dt_bias, state=None, time_step_limit=(0.001, 100.0), mask=None,
                 lengths=None, step=256):
        from mlx_lm.models import ssm

        assert mask is None and lengths is None, "the engine's chunks carry no SSM mask or lengths"
        b, l, h, dh = x.shape
        dt = ssm.compute_dt(dt, dt_bias, time_step_limit)
        A = -mx.exp(A_log).astype(dt.dtype)
        dtA = dt * A.reshape(1, 1, -1)
        dtx = dt.reshape(b, l, h, 1) * x
        ys = []
        for i in range(0, l, step):
            y, state = self.ssd_step(dtx[:, i:i + step], dtA[:, i:i + step], B[:, i:i + step], C[:, i:i + step],
                                     state, x.dtype)
            ys.append(y)
        return mx.concatenate(ys, axis=1) + x * D.reshape(1, 1, h, 1), state

    def ssd_step(self, dtx, dtA, B, C, state, out_dtype):
        b, s, h, dh = dtx.shape
        g, d = B.shape[2], B.shape[3]
        rep = h // g
        Bt = mx.transpose(B, (0, 2, 3, 1))
        CB = self.run(pl.gemm(C, B, s, s, d, g * d, g * d, False, True, g, (d, d))).reshape(b, g, s, s)
        CB = mx.repeat(CB, rep, axis=1)
        decay = mx.exp(self.segsum(dtA.swapaxes(1, 2)))
        sur = mx.tril(CB * decay, 0)
        y = self.run(pl.gemm(sur, dtx, s, dh, s, s, h * dh, False, False, h, (s * s, dh))).reshape(b, h, s, dh)
        y = mx.swapaxes(y, 1, 2)
        decay = decay[:, :, -1:, :].transpose(0, 3, 1, 2)
        Bf = mx.repeat(Bt, rep, axis=1).astype(mx.float32)
        dtxdecay = dtx * decay
        nxt = self.run(pl.gemm(dtxdecay, Bf, dh, d, s, h * dh, s, True, True, h, (dh, d * s))).reshape(b, h, dh, d)
        if state is not None:
            e = mx.exp(self.run(pl.scan(dtA, 1, s, h)).reshape(b, s, h))
            nxt += e[:, -1, :, None, None] * state
            y_prev = self.run(pl.gemv(state, C.astype(mx.float32), s, g, h, dh, d)).reshape(b, s, h, dh)
            y += e[..., None] * y_prev
        return y.astype(out_dtype), nxt

    def segsum(self, x):
        l = x.shape[-1]
        x = mx.repeat(x[..., None], l, axis=-1)
        x = mx.tril(x, -1)
        outer = x.size // (l * l)
        return self.run(pl.scan(x, outer, l, l)).reshape(x.shape)

    def conv1d(self, conv, x):
        channels, taps = int(conv.weight.shape[0]), int(conv.weight.shape[1])
        y = self.run(pl.conv(x, conv.weight, x.shape[1] - taps + 1, channels, taps))
        return y + conv.bias if "bias" in conv else y

    def attention(self, queries, keys, values, cache, scale, mask, sinks=None):
        assert mask == "causal" and sinks is None and queries.shape[2] > 8, "chunks over 8 rows, causal, no sinks"
        B, H, L, D = queries.shape
        HK, kL = keys.shape[1], keys.shape[2]
        q, kv = H * L * D, HK * kL * D
        strides = (q, L * D, D, kv, kL * D, D, kv, kL * D, D, L * H * D, D, H * D)
        o = self.run(pl.attention(queries, keys, values, L, kL, strides, scale))
        return o.transpose(0, 2, 1, 3)

    def router(self, gate, x):
        from mlx_lm.models.nemotron_h import group_expert_select

        D = x.shape[-1]
        logits = self.run(pl.matmul_nt(x.reshape(-1, D), gate.weight)).reshape(*x.shape[:-1], -1)
        return group_expert_select(logits, gate.e_score_correction_bias, gate.top_k, gate.n_group, gate.topk_group,
                                   gate.routed_scaling_factor, gate.norm_topk_prob)

    def quantized_matmul(self, x, w, scales, biases=None, transpose=True, group_size=64, bits=4, **kw):
        K, N = x.shape[-1], int(w.shape[0])
        M = x.size // K
        if transpose and group_size == 64 and bits == 4 and biases is not None and not kw and \
                qmm_split_k(M, N, K) == 1 and K % 64 == 0 and N % 64 == 0:
            return self.run(pl.qmm(x, w, scales, biases)).reshape(*x.shape[:-1], N)
        return mx.quantized_matmul(x, w, scales, biases, transpose=transpose, group_size=group_size, bits=bits, **kw)

    def gather_qmm(self, x, w, scales, biases=None, lhs_indices=None, rhs_indices=None, transpose=True, group_size=64,
                   bits=4, mode="affine", sorted_indices=False):
        rows, E = int(rhs_indices.size) if rhs_indices is not None else 0, int(w.shape[0])
        if (lhs_indices is None and sorted_indices and transpose and group_size == 64 and bits == 4 and
                mode == "affine" and biases is not None and x.shape[-2] == 1 and rows >= 16 and rows // E >= 4):
            y = self.run(pl.gather_qmm(x, w, scales, biases, rhs_indices.astype(mx.uint32)))
            return y.reshape(rows, 1, int(w.shape[1]))
        return mx.gather_qmm(x, w, scales, biases, lhs_indices=lhs_indices, rhs_indices=rhs_indices,
                             transpose=transpose, group_size=group_size, bits=bits, mode=mode,
                             sorted_indices=sorted_indices)


class _MX:
    """mlx.core with the swap's quantized matmuls in place of MLX's, for the modules that call them."""

    def __init__(self, swap: Swap) -> None:
        self._swap = swap

    def __getattr__(self, name):
        return getattr(self._swap, name) if name in ("quantized_matmul", "gather_qmm") else getattr(mx, name)


@contextmanager
def swapped(runner):
    """Route mlx_lm's NemotronH prefill through our kernels inside the block."""

    from mlx_lm.models import nemotron_h, ssm, switch_layers

    from tensorfold.kernels.qwen.dense.v1 import lane_qmm

    swap = Swap(runner)
    saved = (ssm.ssm_attn, nemotron_h.scaled_dot_product_attention, nemotron_h.MoEGate.__call__,
             nn.Conv1d.__call__, lane_qmm.mx, switch_layers.mx)
    ssm.ssm_attn = swap.ssm_attn
    nemotron_h.scaled_dot_product_attention = swap.attention
    nemotron_h.MoEGate.__call__ = lambda gate, x: swap.router(gate, x)
    nn.Conv1d.__call__ = lambda conv, x: swap.conv1d(conv, x)
    lane_qmm.mx = switch_layers.mx = _MX(swap)
    try:
        yield swap
    finally:
        (ssm.ssm_attn, nemotron_h.scaled_dot_product_attention, nemotron_h.MoEGate.__call__, nn.Conv1d.__call__,
         lane_qmm.mx, switch_layers.mx) = saved
