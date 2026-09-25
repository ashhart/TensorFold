"""Gated-delta layers for many short lanes of one prefix, without copying the state.

Every lane forks the same prefix, so every lane starts from the same
recurrent state S0 in each gated-delta layer. The stock step kernel keeps a
full [Hv, Dv, Dk] state per lane and reads and writes all of it per token:
151 MB per lane per round in bfloat16, 19 GB at 128 lanes. A lane's suffix
is short, so its state is better kept as S0 plus the few rank-one updates it
has made. With G_t the product of the step decays,

    S_t = G_t S0 + sum_{i <= t} (G_t / G_i) d_i k_i^T

and the two things the rule reads from the state become

    S_{t-1} decayed, applied to k_t = G_t S0 k_t + sum_{i < t} (G_t / G_i) (k_i . k_t) d_i
    S_t q_t                         = G_t S0 q_t + sum_{i <= t} (G_t / G_i) (k_i . q_t) d_i

S0 is read once per round for all lanes; each lane reads only its own t
past keys and deltas. The recurrence is the one in
``mlx_lm.models.gated_delta`` (scalar decay per value head, keys shared by
Hv / Hk value heads); only the order of the arithmetic changes, so a lane's
tokens can differ from the state-kernel path at near-ties, like any other
change of kernel shape.

A padded suffix position (ragged absorb) has decay 1 and beta 0: its delta
is zero, so it changes nothing.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

F32 = mx.float32
HIST = mx.bfloat16  # stored keys and deltas; the recurrence itself runs in float32

_STEP_SOURCE = """
    // One threadgroup per (lane, value head); thread dv owns output row dv.
    const uint dv = thread_position_in_threadgroup.x;
    const uint group = threadgroup_position_in_grid.x;
    const uint n = group / HV;
    const uint h = group % HV;
    const uint hk = h / (HV / HK);
    const uint sg = dv / 32;
    const uint sl = dv % 32;
    const int t = tlen[0];

    threadgroup float ks[DK];
    threadgroup float qs[DK];
    threadgroup float aw[CAP];
    threadgroup float cw[CAP];
    threadgroup float red[DV / 32];

    ks[dv] = static_cast<float>(k[(n * HK + hk) * DK + dv]);
    qs[dv] = static_cast<float>(q[(n * HK + hk) * DK + dv]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S0 k and S0 q, from one batched matmul over all lanes outside the kernel.
    const float s0k = s0kq[((n * HV + h) * 2 + 0) * DV + dv];
    const float s0q = s0kq[((n * HV + h) * 2 + 1) * DV + dv];

    const float lg = log_prev[n * HV + h] + log_g[n * HV + h];
    const float decay = metal::exp(lg);

    // Past keys against this key and this query; one simdgroup per stride of i.
    for (int i = int(sg); i < t; i += int(DV / 32)) {
        const device HistT* krow = k_hist + ((n * HK + hk) * CAP + i) * DK;
        float a = 0.0f;
        float c = 0.0f;
        for (int j = int(sl); j < DK; j += 32) {
            float kv = static_cast<float>(krow[j]);
            a += kv * ks[j];
            c += kv * qs[j];
        }
        a = simd_sum(a);
        c = simd_sum(c);
        if (sl == 0) {
            float w = metal::exp(lg - lg_hist[(n * HV + h) * CAP + i]);
            aw[i] = w * a;
            cw[i] = w * c;
        }
    }
    float kq = simd_sum(ks[dv] * qs[dv]);
    if (sl == 0) {
        red[sg] = kq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kq = 0.0f;
    for (int r = 0; r < int(DV / 32); ++r) {
        kq += red[r];
    }

    const device HistT* dcol = d_hist + (n * HV + h) * CAP * DV + dv;
    float memory = decay * s0k;
    float yv = decay * s0q;
    for (int i = 0; i < t; ++i) {
        float dval = static_cast<float>(dcol[i * DV]);
        memory += aw[i] * dval;
        yv += cw[i] * dval;
    }
    const float delta = (static_cast<float>(v[(n * HV + h) * DV + dv]) - memory) * beta[n * HV + h];
    yv += kq * delta;
    y[(n * HV + h) * DV + dv] = static_cast<OutT>(yv);
    delta_out[(n * HV + h) * DV + dv] = static_cast<HistT>(delta);
    if (dv == 0) {
        log_out[n * HV + h] = lg;
    }
"""

# One threadgroup per (lane, key head) instead: the R = Hv / Hk value heads that
# share a key head share its past-key dot products, so the key history is read
# once, not R times (a third less history traffic at R = 3), and S0 k, S0 q
# are read in the [Hv, N, 2, Dv] order the batched matmul writes them.
_STEP_SOURCE_KH = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint r = tid / DV;
    const uint dv = tid % DV;
    const uint group = threadgroup_position_in_grid.x;
    const uint n = group / HK;
    const uint hk = group % HK;
    const uint h = hk * R + r;
    const uint sg = tid / 32;
    const uint sl = tid % 32;
    const int t = tlen[0];
    const uint lanes = uint(tlen[1]);
    constexpr int NSG = (R * DV) / 32;

    threadgroup float ks[DK];
    threadgroup float qs[DK];
    threadgroup float dk[CAP];
    threadgroup float dq[CAP];
    threadgroup float aw[R * CAP];
    threadgroup float cw[R * CAP];
    threadgroup float red[NSG];

    if (tid < DK) {
        ks[tid] = static_cast<float>(k[(n * HK + hk) * DK + tid]);
        qs[tid] = static_cast<float>(q[(n * HK + hk) * DK + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float kqp = (tid < DK) ? ks[tid] * qs[tid] : 0.0f;
    kqp = simd_sum(kqp);
    if (sl == 0) {
        red[sg] = kqp;
    }
    // Past keys against this key and this query, once per key head.
    for (int i = int(sg); i < t; i += NSG) {
        const device HistT* krow = k_hist + ((n * HK + hk) * CAP + i) * DK;
        float a = 0.0f;
        float c = 0.0f;
        for (int j = int(sl); j < DK; j += 32) {
            float kv = static_cast<float>(krow[j]);
            a += kv * ks[j];
            c += kv * qs[j];
        }
        a = simd_sum(a);
        c = simd_sum(c);
        if (sl == 0) {
            dk[i] = a;
            dq[i] = c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Decay weights per value head.
    for (int e = int(tid); e < R * t; e += R * DV) {
        const int rr = e / t;
        const int i = e % t;
        const uint hh = hk * R + uint(rr);
        const float lgr = log_prev[n * HV + hh] + log_g[n * HV + hh];
        const float w = metal::exp(lgr - lg_hist[(n * HV + hh) * CAP + i]);
        aw[rr * CAP + i] = w * dk[i];
        cw[rr * CAP + i] = w * dq[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float kq = 0.0f;
    for (int s2 = 0; s2 < DK / 32; ++s2) {
        kq += red[s2];
    }

    const float lg = log_prev[n * HV + h] + log_g[n * HV + h];
    const float decay = metal::exp(lg);
    const float s0k = s0kq[((h * lanes + n) * 2 + 0) * DV + dv];
    const float s0q = s0kq[((h * lanes + n) * 2 + 1) * DV + dv];
    const device HistT* dcol = d_hist + (n * HV + h) * CAP * DV + dv;
    float memory = decay * s0k;
    float yv = decay * s0q;
    for (int i = 0; i < t; ++i) {
        float dval = static_cast<float>(dcol[i * DV]);
        memory += aw[r * CAP + i] * dval;
        yv += cw[r * CAP + i] * dval;
    }
    const float delta = (static_cast<float>(v[(n * HV + h) * DV + dv]) - memory) * beta[n * HV + h];
    yv += kq * delta;
    y[(n * HV + h) * DV + dv] = static_cast<OutT>(yv);
    delta_out[(n * HV + h) * DV + dv] = static_cast<HistT>(delta);
    if (dv == 0) {
        log_out[n * HV + h] = lg;
    }
"""

_STEP_KERNEL = None
_STEP_KERNEL_KH = None
_T_ARRAYS: dict[int, mx.array] = {}
_TN_ARRAYS: dict[tuple[int, int], mx.array] = {}


def _tn_array(t: int, n: int) -> mx.array:
    """History length and lane count as the kernel's small int input; made once per pair."""

    found = _TN_ARRAYS.get((t, n))
    if found is None:
        found = _TN_ARRAYS[(t, n)] = mx.array([t, n], dtype=mx.int32)
    return found


def _step_kernel_kh():
    global _STEP_KERNEL_KH
    if _STEP_KERNEL_KH is None and mx.metal.is_available():
        _STEP_KERNEL_KH = mx.fast.metal_kernel(
            name="tensorfold_lane_gdn_step_kh",
            input_names=["q", "k", "v", "log_g", "beta", "s0kq", "log_prev", "k_hist", "d_hist",
                         "lg_hist", "tlen"],
            output_names=["y", "delta_out", "log_out"],
            source=_STEP_SOURCE_KH,
        )
    return _STEP_KERNEL_KH


def _t_array(t: int) -> mx.array:
    """The history length as the one-element input the kernel reads; made once per value."""

    found = _T_ARRAYS.get(t)
    if found is None:
        found = _T_ARRAYS[t] = mx.array([t], dtype=mx.int32)
    return found


@mx.compile
def _gates(a_log: mx.array, dt_bias: mx.array, a: mx.array, b: mx.array) -> tuple[mx.array, mx.array]:
    """log of the step decay and the write strength, both float32."""

    log_g = -mx.exp(a_log.astype(F32)) * nn.softplus((a + dt_bias).astype(F32))
    return log_g, mx.sigmoid(b.astype(F32))


def _step_kernel():
    global _STEP_KERNEL
    if _STEP_KERNEL is None and mx.metal.is_available():
        _STEP_KERNEL = mx.fast.metal_kernel(
            name="tensorfold_lane_gdn_step",
            input_names=["q", "k", "v", "log_g", "beta", "s0kq", "log_prev", "k_hist", "d_hist",
                         "lg_hist", "tlen"],
            output_names=["y", "delta_out", "log_out"],
            source=_STEP_SOURCE,
        )
    return _STEP_KERNEL


class LaneGDNCache:
    """One gated-delta layer's cache for N lanes that share a prefix state."""

    growth = 32

    def __init__(self, conv: mx.array, s0: mx.array, *, key_heads: int, capacity: int = 64) -> None:
        # conv: [N, kernel - 1, C] per lane; s0: [Hv, Dv, Dk] float32, shared
        if s0.ndim != 3:
            raise ValueError("s0 must be [Hv, Dv, Dk]")
        self.conv = conv
        self.s0 = mx.contiguous(s0.astype(F32))                           # [Hv, Dv, Dk]
        self.s0_t = mx.contiguous(self.s0.transpose(0, 2, 1))             # [Hv, Dk, Dv]
        self.value_heads, self.key_dim, self.value_dim = int(s0.shape[0]), int(s0.shape[2]), int(s0.shape[1])
        self.key_heads = int(key_heads)
        self.repeat = self.value_heads // self.key_heads
        lanes = int(conv.shape[0])
        self.log_g = mx.zeros((lanes, self.value_heads), dtype=F32)
        self.t = 0
        cap = max(1, int(capacity))
        self.k_hist = mx.zeros((lanes, self.key_heads, cap, self.key_dim), dtype=HIST)
        self.d_hist = mx.zeros((lanes, self.value_heads, cap, self.value_dim), dtype=HIST)
        self.lg_hist = mx.zeros((lanes, self.value_heads, cap), dtype=F32)
        self.lengths: mx.array | None = None
        self.left_padding = None
        self._pending: tuple[mx.array, mx.array, mx.array] | None = None

    # -- the cache protocol the model and the burst loop read -------------------
    @property
    def lanes(self) -> int:
        return int(self.conv.shape[0])

    @property
    def state(self) -> list[mx.array]:
        return [self.conv, self.log_g, self.k_hist, self.d_hist, self.lg_hist]

    def make_mask(self, N: int) -> Any:
        if self.lengths is not None:
            return mx.arange(N) < self.lengths[:, None]
        return None

    def prepare(self, lengths: Any = None, **kwargs: Any) -> None:
        if lengths is not None:
            self.lengths = mx.array(lengths)

    def finalize(self) -> None:
        self.lengths = None

    def advance(self, n: int) -> None:
        if self.lengths is not None:
            self.lengths = self.lengths - n

    def filter(self, keep: mx.array) -> None:
        if self._pending is not None:
            self._pending = tuple(a[keep] for a in self._pending)  # type: ignore[assignment]
        self.conv = self.conv[keep]
        self.log_g = self.log_g[keep]
        self.k_hist = self.k_hist[keep]
        self.d_hist = self.d_hist[keep]
        self.lg_hist = self.lg_hist[keep]
        if self.lengths is not None:
            self.lengths = self.lengths[keep]

    @property
    def nbytes(self) -> int:
        return sum(int(a.nbytes) for a in self.state)

    # -- the recurrence ----------------------------------------------------------
    def _grow(self) -> None:
        extra = self.growth
        n = self.lanes
        self.k_hist = mx.concatenate(
            [self.k_hist, mx.zeros((n, self.key_heads, extra, self.key_dim), dtype=HIST)], axis=2)
        self.d_hist = mx.concatenate(
            [self.d_hist, mx.zeros((n, self.value_heads, extra, self.value_dim), dtype=HIST)], axis=2)
        self.lg_hist = mx.concatenate(
            [self.lg_hist, mx.zeros((n, self.value_heads, extra), dtype=F32)], axis=2)

    def _shared(self, x: mx.array) -> mx.array:
        # x: [N, Hv, Dk] -> S0 x: [N, Hv, Dv]
        return (x.transpose(1, 0, 2) @ self.s0_t).transpose(1, 0, 2)

    def _history(self, x: mx.array, upto: int) -> mx.array:
        # sum_{i < upto} (G_t / G_i) (k_i . x) d_i for x per key head [N, Hk, Dk]
        keys = self.k_hist[:, :, :upto].astype(F32)
        deltas = self.d_hist[:, :, :upto].astype(F32)
        logs = self.lg_hist[:, :, :upto]
        dots = (keys @ x[..., None])[..., 0]                     # [N, Hk, t]
        if self.repeat > 1:
            dots = mx.repeat(dots, self.repeat, axis=1)         # [N, Hv, t]
        weights = mx.exp(self.log_g[..., None] - logs) * dots    # [N, Hv, t]
        return (weights[:, :, None, :] @ deltas)[:, :, 0, :]     # [N, Hv, Dv]

    use_kernel = True
    kernel_version = 2  # 2: one threadgroup per key head; 1: one per value head

    def _flush(self) -> None:
        """Write the previous step's entry into the history buffers.

        Deferred by one step on purpose: written before the next kernel reads the
        buffers, the update is the only user of the old buffer and MLX updates it
        in place; written right after the kernel that read it, the update had to
        copy the whole buffer (0.5 ms a layer at 128 lanes, 2026-09-22).
        """

        if self._pending is None:
            return
        if self.t >= int(self.k_hist.shape[2]):
            self._grow()
        k, delta, log_new = self._pending
        self._pending = None
        t = self.t
        self.k_hist[:, :, t, :] = k if k.dtype == HIST else k.astype(HIST)
        self.d_hist[:, :, t, :] = delta if delta.dtype == HIST else delta.astype(HIST)
        self.lg_hist[:, :, t] = log_new
        self.t = t + 1

    def step(self, q: mx.array, k: mx.array, v: mx.array, log_g: mx.array, beta: mx.array) -> mx.array:
        """One position for every lane. q, k: [N, Hk, Dk]; v: [N, Hv, Dv]; log_g, beta: [N, Hv]."""

        self._flush()
        usable = self.use_kernel and self.key_dim == 128 and self.value_dim == 128
        if usable and self.kernel_version == 2 and self.repeat * self.value_dim <= 1024:
            return self._step_kh(q, k, v, log_g, beta)
        kernel = _step_kernel() if usable else None
        if kernel is not None:
            n = self.lanes
            cap = int(self.k_hist.shape[2])
            # [Hv, 2N, Dk] @ [Hv, Dk, Dv]: S0 read once for every lane's key and query.
            kq = mx.concatenate([k, q], axis=1)                              # [N, 2Hk, Dk]
            kq = mx.repeat(kq.reshape(n, 2, self.key_heads, 1, self.key_dim), self.repeat, axis=3)
            kq = kq.reshape(n, 2, self.value_heads, self.key_dim).transpose(2, 0, 1, 3)
            s0kq = (kq.reshape(self.value_heads, 2 * n, self.key_dim).astype(F32) @ self.s0_t)
            s0kq = s0kq.reshape(self.value_heads, n, 2, self.value_dim).transpose(1, 0, 2, 3)
            y, delta, log_new = kernel(
                inputs=[q, k, v, log_g, beta, s0kq, self.log_g,
                        self.k_hist, self.d_hist, self.lg_hist, _t_array(self.t)],
                template=[("OutT", q.dtype), ("HistT", HIST), ("HK", self.key_heads),
                          ("HV", self.value_heads), ("DK", self.key_dim), ("DV", self.value_dim),
                          ("CAP", cap)],
                grid=(n * self.value_heads * self.value_dim, 1, 1),
                threadgroup=(self.value_dim, 1, 1),
                output_shapes=[(n, self.value_heads, self.value_dim),
                               (n, self.value_heads, self.value_dim), (n, self.value_heads)],
                output_dtypes=[q.dtype, HIST, F32],
            )
            self.log_g = log_new
            self._pending = (k, delta, log_new)
            return y
        qf, kf, vf = q.astype(F32), k.astype(F32), v.astype(F32)
        self.log_g = self.log_g + log_g
        decay = mx.exp(self.log_g)[..., None]                   # [N, Hv, 1]
        k_all = mx.repeat(kf, self.repeat, axis=1) if self.repeat > 1 else kf
        q_all = mx.repeat(qf, self.repeat, axis=1) if self.repeat > 1 else qf
        memory = decay * self._shared(k_all)
        output = decay * self._shared(q_all)
        if self.t:
            memory = memory + self._history(kf, self.t)
            output = output + self._history(qf, self.t)
        delta = (vf - memory) * beta[..., None]
        kq = (k_all * q_all).sum(axis=-1, keepdims=True)         # [N, Hv, 1]
        self._pending = (kf, delta, self.log_g)
        return output + kq * delta


def _step_kh(self: LaneGDNCache, q: mx.array, k: mx.array, v: mx.array, log_g: mx.array,
             beta: mx.array) -> mx.array:
    """``step`` through the key-head kernel (``_flush`` already done)."""

    n = self.lanes
    cap = int(self.k_hist.shape[2])
    # [Hv, 2N, Dk] @ [Hv, Dk, Dv] -> [Hv, N, 2, Dv], read in that order by the kernel.
    kq = mx.concatenate([k, q], axis=1)                              # [N, 2Hk, Dk]
    kq = mx.repeat(kq.reshape(n, 2, self.key_heads, 1, self.key_dim), self.repeat, axis=3)
    kq = kq.reshape(n, 2, self.value_heads, self.key_dim).transpose(2, 0, 1, 3)
    s0kq = kq.reshape(self.value_heads, 2 * n, self.key_dim).astype(F32) @ self.s0_t
    y, delta, log_new = _step_kernel_kh()(
        inputs=[q, k, v, log_g, beta, s0kq, self.log_g,
                self.k_hist, self.d_hist, self.lg_hist, _tn_array(self.t, n)],
        template=[("OutT", q.dtype), ("HistT", HIST), ("HK", self.key_heads), ("R", self.repeat),
                  ("HV", self.value_heads), ("DK", self.key_dim), ("DV", self.value_dim),
                  ("CAP", cap)],
        grid=(n * self.key_heads * self.repeat * self.value_dim, 1, 1),
        threadgroup=(self.repeat * self.value_dim, 1, 1),
        output_shapes=[(n, self.value_heads, self.value_dim),
                       (n, self.value_heads, self.value_dim), (n, self.value_heads)],
        output_dtypes=[q.dtype, HIST, F32],
    )
    self.log_g = log_new
    self._pending = (k, delta, log_new)
    return y


LaneGDNCache._step_kh = _step_kh  # type: ignore[attr-defined]


def lane_gdn_call(self: Any, inputs: mx.array, mask: Any, cache: LaneGDNCache) -> mx.array:
    """``GatedDeltaNet.__call__`` for a ``LaneGDNCache``: same projections, lane recurrence."""

    B, S, _ = inputs.shape
    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([cache.conv, qkv], axis=1)
    n_keep = self.conv_kernel_size - 1
    if cache.lengths is not None:
        ends = mx.clip(cache.lengths, 0, S)
        positions = (ends[:, None] + mx.arange(n_keep))[..., None]
        cache.conv = mx.take_along_axis(conv_input, positions, axis=1)
    else:
        cache.conv = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(self.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    log_g, beta = _gates(self.A_log, self.dt_bias, a, b)
    if mask is not None:
        log_g = mx.where(mask[..., None], log_g, 0.0)
        beta = mx.where(mask[..., None], beta, 0.0)

    outs = [cache.step(q[:, t], k[:, t], v[:, t], log_g[:, t], beta[:, t]) for t in range(S)]
    out = outs[0][:, None] if S == 1 else mx.stack(outs, axis=1)
    if out.dtype != inputs.dtype:
        out = out.astype(inputs.dtype)
    cache.advance(S)
    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1))


def install_lane_gdn(model: Any) -> int:
    """Route gated-delta layers through ``lane_gdn_call`` when the cache is a ``LaneGDNCache``."""

    language_model = getattr(model, "language_model", model)
    core = getattr(language_model, "model", language_model)
    layers = [layer.linear_attn for layer in core.layers if getattr(layer, "is_linear", False)]
    for cls in {type(layer) for layer in layers}:
        if getattr(cls, "_tensorfold_lane_gdn", False):
            continue
        stock = cls.__call__

        def patched(self: Any, inputs: mx.array, mask: Any = None, cache: Any = None,
                    _stock: Any = stock) -> mx.array:
            if isinstance(cache, LaneGDNCache):
                return lane_gdn_call(self, inputs, mask, cache)
            return _stock(self, inputs, mask, cache)

        cls.__call__ = patched
        cls._tensorfold_lane_gdn = True
    return len(layers)


__all__ = ["LaneGDNCache", "install_lane_gdn", "lane_gdn_call"]
