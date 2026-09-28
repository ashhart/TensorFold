"""Kimi Delta Attention: the linear-attention layers, a recurrent state each."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.glm5_next import config as C
from tensorfold.families.glm5_next.caches import KDACache
from tensorfold.families.glm5_next.config import Config, row_kernel
from tensorfold.families.glm5_next.linear import Q, per_row, project, silu
from tensorfold.kernels.glm.flash.v1 import kda as KDA_K
from tensorfold.kernels.glm.flash.v1 import kernels as K


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

    def __call__(self, x: mx.array, caches: list[KDACache], lengths: tuple[int, ...], decode: bool) -> mx.array:
        """Consecutive streams' rows: projections on all rows at once, conv window and recurrence stream by stream."""

        proj = project(x, self.in_proj, rows_exact=decode)
        if decode and C.FUSED_KDA and self.fused is None:
            self.fused = KDA_K.fits(self)
        fused = decode and C.FUSED_KDA and self.fused
        outs, at = [], 0
        for cache, rows in zip(caches, lengths):
            part = proj if len(lengths) == 1 else proj[at:at + rows]
            outs.append(self._fused(part, cache) if fused else self._step(part, cache, decode))
            at += rows
        return project(outs[0] if len(outs) == 1 else mx.concatenate(outs), self.o_proj, rows_exact=decode)

    def _fused(self, proj: mx.array, cache: KDACache) -> mx.array:
        """One stream's rows through the fused decode kernel; the entry state is kept so ``keep`` can replay."""

        rows, h, d = int(proj.shape[0]), self.heads, self.dim
        conv = cache.conv if cache.conv is not None else mx.zeros((self.taps - 1, 3 * self.width), dtype=mx.bfloat16)
        entry = cache.ssm if cache.ssm is not None else mx.zeros((1, h, d, d), dtype=mx.float32)
        y, cache.ssm, cache.conv = KDA_K.kda_rows(self, proj, conv, entry)
        cache.offset += rows
        cache._replay = [rows, "fused", self, proj, conv, entry]
        return y

    def _step(self, proj: mx.array, cache: KDACache, decode: bool) -> mx.array:
        """One stream's rows through MLX ops (the prefill path, and decode where the fused kernel does not fit)."""

        cfg = self.cfg
        rows = int(proj.shape[0])
        h, d, width = self.heads, self.dim, self.width
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
        return (o * mx.sigmoid(gate.astype(mx.float32))).astype(mx.bfloat16).reshape(rows, width)
