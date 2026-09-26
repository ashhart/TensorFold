"""GLM-5.3-Flash KDA (Kimi Delta Attention) decode rows in ONE Metal kernel per layer.

Ported from mlx-vlm PR #2105 (``mlx_vlm/models/glm5_next/fused_kda.py``, "glm5_next: fuse the KDA decode chain
into one Metal kernel", by avlp12; mlx-vlm is MIT licensed, Copyright (c) 2025 Prince Canuma). That kernel folds the
whole post-projection chain of one KDA decode step -- the causal conv1d window update, silu, the two L2 norms, the
safe forget gate, the sigmoid beta, the gated delta-rule state update and the gated RMSNorm, ~30 MLX dispatches --
into one launch, one threadgroup per head, the 128x128 fp32 state streamed through registers once, and proved it
bit-identical to mlx-vlm's eager ops. What TensorFold changes:

- **rows**: the kernel takes a verify window of R <= 16 consecutive rows and runs them in order inside the launch
  (the state stays in registers between rows). Every row's arithmetic is the same whatever R is, so a window gives
  each row the bits a one-row call gives it -- the exactness contract of TensorFold's drafted rounds. Rolling back to
  a prefix (``KDACache.keep``) re-runs the kept rows from the window's entry state and conv window through the same
  kernel, so the state after the kept prefix is bit-identical too.
- **the projections**: the input is TensorFold's stacked in-projection output (q | k | v | f_a | g_a | b) as it
  comes out of the matmul, no slices or copies; ``f_b_proj`` and ``g_b_proj`` (128 -> 8192, 4-bit, groups of 64)
  run inside the kernel with MLX's one-row ``qmv_quad`` arithmetic (the 4-bit form of #2105's opt-in 8-bit fold).
  One dispatch per KDA layer between the in-projection and ``o_proj``.
- **TensorFold's own rounding points**: the conv sums its taps in fp32 in tap order, silu and beta are MLX's precise
  sigmoid in bf16, the decays are ``exp(lower_bound * sigmoid(A * (a + dt_bias)))`` in fp32 with ``A = exp(A_log)``.
  This kernel IS the decode path's arithmetic (strategy C in the recipe book: the serial reference moves with
  it); the prefill path (> 16 rows) stays on MLX ops, as before.

Without Metal (tests on Linux) ``kda_rows_ops`` runs the same formulas one row at a time with MLX ops.
"""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

# MLX's sigmoid, transcribed (#2105): instantiated on the type the eager op used, with the precise exp.
_HEADER = r"""
template <typename U>
inline U mlx_sigmoid_precise(U x) {
  U e = static_cast<U>(metal::precise::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}
// `(x * x).sum(-1)` rounds the square before the add (no fma), as in #2105.
#pragma clang fp contract(off)
inline float sq_acc(float acc, float v) {
  return v * v + acc;
}
#pragma clang fp contract(on)
"""

_SOURCE = r"""
  // One threadgroup per head h: 32 lanes x TY rows of threads. Rows r = 0 .. R-1 of the window in order.
  const uint h    = threadgroup_position_in_grid.z;
  const uint lane = thread_position_in_threadgroup.x;
  const uint ty   = thread_position_in_threadgroup.y;
  const uint tid  = thread_index_in_threadgroup;
  constexpr int NT   = 32 * TY;
  constexpr int NDK  = D / 32;          // key elements per lane
  constexpr int NDV  = D / TY;          // value rows per thread
  constexpr int RBLK = D / 128;        // MLX's row reduce: 32 lanes x 4 reads a block, then the rest
  constexpr int REXTRA = D - RBLK * 128;
  constexpr uint W   = (uint)(H * D);   // q / k / v width
  constexpr uint C3  = 3u * W;          // conv channels
  constexpr uint FA  = C3;              // offsets in the stacked projection row
  constexpr uint GA  = C3 + (uint)D;
  constexpr uint BO  = C3 + 2u * (uint)D;
  const int R = int(P_shape[0]);
  const uint PS = (uint)P_shape[1];

  threadgroup float sq[D];
  threadgroup float sk[D];
  threadgroup float sv[D];
  threadgroup float sa[D];
  threadgroup float sg[D];
  threadgroup float sgate[D];
  threadgroup float sy[D];
  threadgroup float shr[3];

  device const float* si = ST + (size_t)h * D * D;
  float st[NDV][NDK];
  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) st[j][i] = si[(size_t)dv * D + NDK * lane + i];
  }
  const float a_h = A[h];
  const float lb = LB[0];
  const float eps = EPS[0];

  for (int r = 0; r < R; ++r) {
    device const bfloat* prow = P + (size_t)r * PS;
    // ---- f_b / g_b (128 -> H*D, 4-bit, groups of 64) for this head's D outputs each: MLX's one-row qmv_quad
    // (a quad of lanes per output, 32 inputs a lane, quad_sum), as kernels.qmv_quad_rows does.
    {
      constexpr int PER = D / 4;
      constexpr int KB = D / 2;
      constexpr int KG = D / 64;
      const uint q_id = tid / 4u, qlid = tid % 4u;
      for (uint t = q_id; t < 2u * (uint)D; t += (uint)(NT / 4)) {
        const uint proj = t / (uint)D;
        const uint d = t - proj * (uint)D;
        const uint row = h * (uint)D + d;
        device const bfloat* x = prow + (proj == 0u ? FA : GA) + qlid * (uint)PER;
        float xt[PER];
        float sum = 0.0f;
        for (int i = 0; i < PER; i += 4) {
          const bfloat a = x[i], b = x[i + 1], c = x[i + 2], e = x[i + 3];
          sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(e)));
          xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(e) / 4096.0f;
        }
        device const uint8_t* wb = (device const uint8_t*)(proj == 0u ? FBW : GBW) + (size_t)row * KB + qlid * (PER / 2);
        device const uint16_t* ws = (device const uint16_t*)wb;
        const uint gi = row * (uint)KG + qlid / (uint)(64 / PER);
        const float s = float(proj == 0u ? FBS[gi] : GBS[gi]);
        const float bb = float(proj == 0u ? FBB[gi] : GBB[gi]);
        float accum = 0.0f;
        for (int i = 0; i < PER / 4; i++)
          accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
                   xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
        float result = 0.0f;
        result += s * accum + sum * bb;
        const float v = quad_sum(result);
        if (qlid == 0u) {
          if (proj == 0u) sa[d] = float(bfloat(v));
          else            sgate[d] = float(bfloat(v));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // sa / sgate come from other threads' quads
    // ---- causal conv over [window ; rows], fp32 taps in order, then silu (bf16, precise sigmoid)
    for (uint idx = tid; idx < 3u * (uint)D; idx += NT) {
      const uint part = idx / (uint)D;
      const uint d = idx - part * (uint)D;
      const uint c = part * W + h * (uint)D + d;
      float acc = 0.0f;
      for (int j = 0; j < TAPS; ++j) {
        const int e = r + j;                       // position in [window (TAPS-1 rows) ; rows]
        const bfloat xv = e < TAPS - 1 ? CS[(size_t)e * C3 + c] : P[(size_t)(e - (TAPS - 1)) * PS + c];
        const float term = float(xv) * CW[(size_t)j * C3 + c];
        acc = j == 0 ? term : acc + term;
      }
      const bfloat xb = bfloat(acc);
      const bfloat sl = xb * mlx_sigmoid_precise<bfloat>(xb);
      if (part == 0u) sq[d] = float(sl);
      else if (part == 1u) sk[d] = float(sl);
      else sv[d] = float(sl);
    }
    // ---- decays and beta
    for (uint d = tid; d < (uint)D; d += NT) {
      const float av = float(bfloat(sa[d])) + DTB[h * (uint)D + d];
      sg[d] = metal::precise::exp(lb * mlx_sigmoid_precise<float>(a_h * av));
    }
    if (tid == 0u) shr[2] = float(mlx_sigmoid_precise<bfloat>(prow[BO + h]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- l2 norms of q and k (MLX's row-reduce order), q also * D^-1/2, back to bf16
    if (simdgroup_index_in_threadgroup == 0u) {
      float pq = 0.0f, pk = 0.0f;
      for (int blk = 0; blk < RBLK; ++blk) {
        const uint base = (uint)(blk * 128) + 4u * lane;
        for (int i = 0; i < 4; ++i) { pq = sq_acc(pq, sq[base + i]); pk = sq_acc(pk, sk[base + i]); }
      }
      for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA && i < 4; ++i) {
        const uint at = (uint)(RBLK * 128) + 4u * lane + (uint)i;
        pq = sq_acc(pq, sq[at]); pk = sq_acc(pk, sk[at]);
      }
      pq = simd_sum(pq);
      pk = simd_sum(pk);
      if (lane == 0u) {
        shr[0] = metal::precise::rsqrt(pq + 1.0e-6f);
        shr[1] = metal::precise::rsqrt(pk + 1.0e-6f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const float rq = shr[0], rk = shr[1];
      const float qscale = metal::precise::rsqrt(float(D));
      for (uint d = tid; d < (uint)D; d += NT) {
        sq[d] = float(bfloat((sq[d] * rq) * qscale));
        sk[d] = float(bfloat(sk[d] * rk));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- gated delta rule, one step (mlx-lm's kernel arithmetic: lane owns NDK key elements, simd_sum)
    {
      const float beta = shr[2];
      for (int j = 0; j < NDV; ++j) {
        const uint dv = ty + (uint)TY * (uint)j;
        float kv = 0.0f;
        for (int i = 0; i < NDK; ++i) {
          const uint s = NDK * lane + i;
          st[j][i] = st[j][i] * sg[s];
          kv += st[j][i] * sk[s];
        }
        kv = simd_sum(kv);
        const float delta = (sv[dv] - kv) * beta;
        float o = 0.0f;
        for (int i = 0; i < NDK; ++i) {
          const uint s = NDK * lane + i;
          st[j][i] = st[j][i] + sk[s] * delta;
          o += st[j][i] * sq[s];
        }
        o = simd_sum(o);
        if (lane == 0u) sy[dv] = float(bfloat(o));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- gated RMSNorm over the value axis (fp32), * sigmoid(gate), to bf16
    if (simdgroup_index_in_threadgroup == 0u) {
      float po = 0.0f;
      for (int blk = 0; blk < RBLK; ++blk) {
        const uint base = (uint)(blk * 128) + 4u * lane;
        for (int i = 0; i < 4; ++i) po = sq_acc(po, sy[base + i]);
      }
      for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA && i < 4; ++i)
        po = sq_acc(po, sy[(uint)(RBLK * 128) + 4u * lane + (uint)i]);
      po = simd_sum(po);
      if (lane == 0u) shr[0] = metal::precise::rsqrt(po / (float)D + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const float rn = shr[0];
      for (uint d = tid; d < (uint)D; d += NT) {
        float x = sy[d] * rn;
        x = ONW[d] * x;
        x = x * mlx_sigmoid_precise<float>(float(bfloat(sgate[d])));
        Y[(size_t)r * W + h * (uint)D + d] = bfloat(x);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // ---- the state after the last row, and the conv window: the last TAPS-1 rows of [window ; rows]
  device float* so = ST_OUT + (size_t)h * D * D;
  for (int j = 0; j < NDV; ++j) {
    const uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) so[(size_t)dv * D + NDK * lane + i] = st[j][i];
  }
  for (uint idx = tid; idx < 3u * (uint)D * (uint)(TAPS - 1); idx += NT) {
    const uint m = idx / (3u * (uint)D);
    const uint rem = idx - m * 3u * (uint)D;
    const uint part = rem / (uint)D;
    const uint d = rem - part * (uint)D;
    const uint c = part * W + h * (uint)D + d;
    const int e = R + int(m);
    CS_OUT[(size_t)m * C3 + c] = e < TAPS - 1 ? CS[(size_t)e * C3 + c] : P[(size_t)(e - (TAPS - 1)) * PS + c];
  }
"""

_kernel_obj: dict[str, Any] = {}
TY = 32


def sources() -> dict[str, str]:
    return {"kda_header": _HEADER, "kda_rows": _SOURCE}


def metal() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _kernel() -> Any:
    kernel = _kernel_obj.get("k")
    if kernel is None:
        digest = hashlib.sha256((_HEADER + _SOURCE).encode()).hexdigest()[:10]
        kernel = mx.fast.metal_kernel(
            name=f"tf_glm5_kda_rows_{digest}",
            input_names=["P", "CS", "CW", "FBW", "FBS", "FBB", "GBW", "GBS", "GBB", "A", "DTB", "ST", "ONW", "LB",
                         "EPS"],
            output_names=["Y", "ST_OUT", "CS_OUT"], header=_HEADER, source=_SOURCE)
        _kernel_obj["k"] = kernel
    return kernel


def fits(kda: Any) -> bool:
    """The kernel's shapes: head dim 128 (64 for the test checkpoint), 4-bit group-64 f_b / g_b with head-dim
    inputs, a stacked in-projection whose tail is f_a | g_a | b."""

    fb, gb = kda.f_b, kda.g_b
    return (kda.dim in (64, 128) and all(q.bits == 4 and q.group == 64 and q.ins == kda.dim for q in (fb, gb))
            and kda.cuts[2] == 3 * kda.width and kda.cuts[3] - kda.cuts[2] == kda.dim
            and kda.cuts[4] - kda.cuts[3] == kda.dim and kda.in_proj.outs - kda.cuts[4] == kda.heads)


def kda_rows(kda: Any, proj: mx.array, conv: mx.array, state: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """R rows of a KDA layer's decode step: proj [R, PS] (the stacked in-projection, bf16), conv window [taps-1, 3W]
    (bf16), state [1, H, D, D] (fp32) -> (y [R, H*D] bf16 for o_proj, the state after the last row, the new window)."""

    rows = int(proj.shape[0])
    if not metal():
        return kda_rows_ops(kda, proj, conv, state)
    h, d = kda.heads, kda.dim
    fb, gb = kda.f_b, kda.g_b
    y, st, cs = _kernel()(
        inputs=[proj, conv, kda.conv_w, fb.weight, fb.scales, fb.biases, gb.weight, gb.scales, gb.biases,
                kda.A_flat, kda.dt_bias_flat, state, kda.o_norm, kda.lb_array, kda.eps_array],
        template=[("H", h), ("D", d), ("TAPS", kda.taps), ("TY", TY)],
        grid=(32, TY, h), threadgroup=(32, TY, 1),
        output_shapes=[(rows, h * d), tuple(state.shape), tuple(conv.shape)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.bfloat16])
    return y, st, cs


def kda_rows_ops(kda: Any, proj: mx.array, conv: mx.array, state: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """The kernel's formulas with MLX ops, one row at a time (Linux / CPU): a row's result does not depend on how
    many rows share the call."""

    from mlx_lm.models import gated_delta as gd

    h, d, width, taps = kda.heads, kda.dim, kda.width, kda.taps
    c3 = 3 * width
    ci = mx.concatenate([conv, proj[:, :c3]])
    ys = []
    for r in range(int(proj.shape[0])):
        row = proj[r:r + 1]
        a = kda.f_b(row[:, c3:c3 + d])
        gate = kda.g_b(row[:, c3 + d:c3 + 2 * d])
        acc = ci[r:r + 1].astype(mx.float32) * kda.conv_w[0]
        for t in range(1, taps):
            acc = acc + ci[r + t:r + t + 1].astype(mx.float32) * kda.conv_w[t]
        xb = acc.astype(mx.bfloat16)
        co = xb * mx.sigmoid(xb)
        q, k, v = (co[:, i * width:(i + 1) * width].reshape(1, 1, h, d) for i in range(3))
        qf, kf = q.astype(mx.float32), k.astype(mx.float32)
        q = ((qf * mx.rsqrt((qf * qf).sum(-1, keepdims=True) + 1e-6)) * (d ** -0.5)).astype(mx.bfloat16)
        k = (kf * mx.rsqrt((kf * kf).sum(-1, keepdims=True) + 1e-6)).astype(mx.bfloat16)
        g = mx.exp(kda.cfg.linear_lower_bound
                   * mx.sigmoid(kda.A * (a.astype(mx.float32).reshape(1, 1, h, d) + kda.dt_bias)))
        beta = mx.sigmoid(row[:, c3 + 2 * d:]).reshape(1, 1, h)
        y, state = gd.gated_delta_ops(q, k, v, g, beta, state)
        yf = y.reshape(h, d).astype(mx.float32)
        o = yf * mx.rsqrt((yf * yf).mean(-1, keepdims=True) + kda.cfg.rms_norm_eps) * kda.o_norm
        o = o * mx.sigmoid(gate.reshape(h, d).astype(mx.float32))
        ys.append(o.astype(mx.bfloat16).reshape(1, width))
    rows = int(proj.shape[0])
    return mx.concatenate(ys), state, mx.contiguous(ci[rows:])
