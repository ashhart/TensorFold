"""The KDA decode step in one Metal kernel a layer, rows in order inside the launch (from mlx-vlm #2105)."""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

# MLX's sigmoid, transcribed (#2105): instantiated on the type the eager op used, with the precise exp.
_HEADER = r"""
// f_b / g_b for one output: MLX 0.32.2's one-row qmv_quad at 4 or 8 bits (the caller does the quad_sum)
template <int BITS, int PER>
inline float quad_dot(device const bfloat* x, device const uint8_t* wb, float s, float bb);
template <>
inline float quad_dot<4, 32>(device const bfloat* x, device const uint8_t* wb, float s, float bb) {
  constexpr int PER = 32;
  float xt[PER];
  float sum = 0.0f;
  for (int i = 0; i < PER; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], e = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(e)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(e) / 4096.0f;
  }
  device const uint16_t* ws = (device const uint16_t*)wb;
  float accum = 0.0f;
  for (int i = 0; i < PER / 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  float result = 0.0f;
  result += s * accum + sum * bb;
  return result;
}
template <>
inline float quad_dot<4, 16>(device const bfloat* x, device const uint8_t* wb, float s, float bb) {
  constexpr int PER = 16;
  float xt[PER];
  float sum = 0.0f;
  for (int i = 0; i < PER; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], e = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(e)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(e) / 4096.0f;
  }
  device const uint16_t* ws = (device const uint16_t*)wb;
  float accum = 0.0f;
  for (int i = 0; i < PER / 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  float result = 0.0f;
  result += s * accum + sum * bb;
  return result;
}
template <>
inline float quad_dot<8, 32>(device const bfloat* x, device const uint8_t* wb, float s, float bb) {
  constexpr int PER = 32;
  float xt[PER];
  float sum = 0.0f;
  for (int i = 0; i < PER; i++) { sum += float(x[i]); xt[i] = float(x[i]); }
  float accum = 0.0f;
  for (int i = 0; i < PER; i++) accum += xt[i] * wb[i];
  float result = 0.0f;
  result += s * accum + sum * bb;
  return result;
}
template <>
inline float quad_dot<8, 16>(device const bfloat* x, device const uint8_t* wb, float s, float bb) {
  constexpr int PER = 16;
  float xt[PER];
  float sum = 0.0f;
  for (int i = 0; i < PER; i++) { sum += float(x[i]); xt[i] = float(x[i]); }
  float accum = 0.0f;
  for (int i = 0; i < PER; i++) accum += xt[i] * wb[i];
  float result = 0.0f;
  result += s * accum + sum * bb;
  return result;
}
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
    // ---- f_b / g_b (128 -> H*D, FB- / GB-bit groups of 64): MLX's one-row qmv_quad, as kernels.qmv_quad_rows
    {
      constexpr int PER = D / 4;
      constexpr int KG = D / 64;
      constexpr int FKB = D * FB / 8;                 // bytes a weight row
      constexpr int GKB = D * GB / 8;
      const uint q_id = tid / 4u, qlid = tid % 4u;
      for (uint t = q_id; t < 2u * (uint)D; t += (uint)(NT / 4)) {
        const uint proj = t / (uint)D;
        const uint d = t - proj * (uint)D;
        const uint row = h * (uint)D + d;
        device const bfloat* x = prow + (proj == 0u ? FA : GA) + qlid * (uint)PER;
        const uint gi = row * (uint)KG + qlid / (uint)(64 / PER);
        const float s = float(proj == 0u ? FBS[gi] : GBS[gi]);
        const float bb = float(proj == 0u ? FBB[gi] : GBB[gi]);
        float result;
        if (proj == 0u) {
          device const uint8_t* wb = (device const uint8_t*)FBW + (size_t)row * FKB + qlid * (PER * FB / 8);
          result = quad_dot<FB, PER>(x, wb, s, bb);
        } else {
          device const uint8_t* wb = (device const uint8_t*)GBW + (size_t)row * GKB + qlid * (PER * GB / 8);
          result = quad_dot<GB, PER>(x, wb, s, bb);
        }
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
    """The kernel's shapes: head dim 64 or 128, 4- or 8-bit group-64 f_b / g_b, the stacked in-projection's tail."""

    fb, gb = kda.f_b, kda.g_b
    return (kda.dim in (64, 128) and all(q.bits in (4, 8) and q.group == 64 and q.ins == kda.dim for q in (fb, gb))
            and kda.cuts[2] == 3 * kda.width and kda.cuts[3] - kda.cuts[2] == kda.dim
            and kda.cuts[4] - kda.cuts[3] == kda.dim and kda.in_proj.outs - kda.cuts[4] == kda.heads)


def kda_rows(kda: Any, proj: mx.array, conv: mx.array, state: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """R rows of a KDA step from its conv window and fp32 state: (y for o_proj, the last row's state, the window)."""

    rows = int(proj.shape[0])
    if not metal():
        return kda_rows_ops(kda, proj, conv, state)
    h, d = kda.heads, kda.dim
    fb, gb = kda.f_b, kda.g_b
    y, st, cs = _kernel()(
        inputs=[proj, conv, kda.conv_w, fb.weight, fb.scales, fb.biases, gb.weight, gb.scales, gb.biases,
                kda.A_flat, kda.dt_bias_flat, state, kda.o_norm, kda.lb_array, kda.eps_array],
        template=[("H", h), ("D", d), ("TAPS", kda.taps), ("TY", TY), ("FB", fb.bits), ("GB", gb.bits)],
        grid=(32, TY, h), threadgroup=(32, TY, 1),
        output_shapes=[(rows, h * d), tuple(state.shape), tuple(conv.shape)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.bfloat16])
    return y, st, cs


def kda_rows_ops(kda: Any, proj: mx.array, conv: mx.array, state: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """The kernel's formulas in MLX ops, one row at a time (no Metal)."""

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
        xb = acc.astype(proj.dtype)
        co = xb * mx.sigmoid(xb)
        q, k, v = (co[:, i * width:(i + 1) * width].reshape(1, 1, h, d) for i in range(3))
        qf, kf = q.astype(mx.float32), k.astype(mx.float32)
        q = ((qf * mx.rsqrt((qf * qf).sum(-1, keepdims=True) + 1e-6)) * (d ** -0.5)).astype(proj.dtype)
        k = (kf * mx.rsqrt((kf * kf).sum(-1, keepdims=True) + 1e-6)).astype(proj.dtype)
        g = mx.exp(kda.cfg.linear_lower_bound
                   * mx.sigmoid(kda.A * (a.astype(mx.float32).reshape(1, 1, h, d) + kda.dt_bias)))
        beta = mx.sigmoid(row[:, c3 + 2 * d:]).reshape(1, 1, h)
        y, state = gd.gated_delta_ops(q, k, v, g, beta, state)
        yf = y.reshape(h, d).astype(mx.float32)
        o = yf * mx.rsqrt((yf * yf).mean(-1, keepdims=True) + kda.cfg.rms_norm_eps) * kda.o_norm
        o = o * mx.sigmoid(gate.reshape(h, d).astype(mx.float32))
        ys.append(o.astype(proj.dtype).reshape(1, width))
    rows = int(proj.shape[0])
    return mx.concatenate(ys), state, mx.contiguous(ci[rows:])
