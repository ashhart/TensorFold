"""The decode forward's glue in a few kernels: our own decoder for 1-32 rows.

Between its matmuls a verify forward ran ~1,400 small MLX kernels (norms,
residual adds, the recurrent layers' conv and gates, the lane matmul's group
sums), each ~3 us when the next depends on it: ~7 ms of a 45 ms forward. Here
each step between two matmuls is one kernel:

    norm_xs    residual add + RMSNorm + the next matmul's 64-group input sums
    gdn_pre    conv windows + conv + SiLU + q/k RMSNorm and scales + g + beta
    gdn_post   gated RMSNorm + the output projection's group sums
    mlp_act    SiLU(gate) * up + the down projection's group sums

Every value is a function of its own row only, so a row's bits never depend on
the window it rides in. The arithmetic follows mlx_lm's ops closely (fp32
math, bf16 where mlx_lm stores bf16) but is its own: every decode forward,
serial ones included, goes through these kernels, which is what keeps drafted
output byte identical to serial decoding. The group sums of gdn_post and
mlp_act are the lane matmul's own (sequential fp32 over each group's 64 bf16
inputs, as ``lane_qmm``'s XSUM kernel); norm_xs sums a group as four
16-element partials (see its source), and the projections it feeds always
take those sums, in serial decoding and in drafted rounds alike.
"""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

# norm_xs, live (2026-09-24): blocks of rounds alternating on 21k contexts, 60.02 -> 58.30 ms a round over 1,500
# rounds; drafted output byte-identical to serial (drafts off) under it. (Comments inside the source below are part
# of the lane decoder's version hash, which keys every stored snapshot: edit them only with the arithmetic.)
_NORM_XS = r"""
  // one threadgroup of K / 16 threads per row: thread t holds elements [16 t, 16 t + 16) in registers.
  // The row's sum of squares is each thread's sequential fma over its 16, then simd_sum, then the
  // simdgroups' sums in order; a 64-group's input sum (for the next lane matmul) is ((g0 + g1) + (g2 + g3))
  // over its 4 threads' sequential sums. Arithmetic since 2026-09-24 (it was 256 strided threads, two
  // barriers and 64-long sums read back from threadgroup memory): 11.5 -> 7 us a call at 16 rows.
  const uint t = thread_position_in_threadgroup.x;
  const uint m = threadgroup_position_in_grid.y;
  const int M = dims[0], MP = dims[1];
  constexpr int E = 16;
  constexpr int TPG = K / E;
  threadgroup float red[TPG / 32];
  if (int(m) >= M) {
    if ((t & 3) == 0) XS[(t >> 2) * MP + m] = 0.0f;
    return;
  }
  const int base = int(m) * K + int(t) * E;
  float hv[E];
  float ss = 0.0f;
  for (int i = 0; i < E; i++) {
    bfloat h = H[base + i];
    RESIDUAL_ADD
    hv[i] = float(h);
    ss = fma(hv[i], hv[i], ss);
  }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int i = 0; i < TPG / 32; i++) total += red[i];
  const float inv = metal::rsqrt(total / float(K) + eps[0]);
  float gs = 0.0f;
  for (int i = 0; i < E; i++) {
    const bfloat x = bfloat(float(Wt[int(t) * E + i]) * (hv[i] * inv));
    XO[base + i] = x;
    gs += float(x);
  }
  gs += simd_shuffle_xor(gs, 1);
  gs += simd_shuffle_xor(gs, 2);
  if ((t & 3) == 0) XS[(t >> 2) * MP + m] = gs;
"""

_GDN_PRE = r"""
  // one simdgroup per (row w, head): q heads [0, NK), k heads [NK, 2 NK), v heads [2 NK, 2 NK + NV)
  const uint lane = thread_index_in_simdgroup;
  const uint head = threadgroup_position_in_grid.y;
  const uint w = threadgroup_position_in_grid.z;
  constexpr int C = 2 * NK * DK + NV * DV;
  const bool isq = head < NK, isk = !isq && head < 2 * NK;
  const int c0 = isq ? int(head) * DK : (isk ? NK * DK + (int(head) - NK) * DK : 2 * NK * DK + (int(head) - 2 * NK) * DV);
  constexpr int PER = DK / 32;                          // channels per lane (DK == DV)
  float vals[PER];
  for (int j = 0; j < PER; j++) {
    const int c = c0 + int(lane) * PER + j;
    float acc = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int row = windows[w * TAPS + tap];          // into [conv state rows; window rows]
      const float xv = row < TAPS - 1 ? float(CS[row * C + c]) : float(QKV[(row - (TAPS - 1)) * C + c]);
      acc += float(CW[c * TAPS + tap]) * xv;
    }
    const float conv = float(bfloat(acc));
    const float sig = float(bfloat(1.0f / (1.0f + metal::exp(-conv))));
    vals[j] = float(bfloat(conv * sig));                // SiLU, bf16 like mlx_lm's two ops
  }
  if (isq || isk) {
    float ss = 0.0f;
    for (int j = 0; j < PER; j++) ss += vals[j] * vals[j];
    ss = simd_sum(ss);
    const float inv = metal::rsqrt(ss / float(DK) + 1e-6f);
    // mlx_lm: q = (DK^-0.5)^2 * rms_norm(q), k = DK^-0.5 * rms_norm(k), scales rounded to bf16
    const float scale = isq ? float(bfloat(1.0f / float(DK))) : float(bfloat(metal::rsqrt(float(DK))));
    for (int j = 0; j < PER; j++) {
      const bfloat out = bfloat(scale * float(bfloat(vals[j] * inv)));
      if (isq) Q[(w * NK + head) * DK + lane * PER + j] = out;
      else Kout[(w * NK + head - NK) * DK + lane * PER + j] = out;
    }
  } else {
    const int hv = int(head) - 2 * NK;
    for (int j = 0; j < PER; j++) Vout[(w * NV + hv) * DV + lane * PER + j] = bfloat(vals[j]);
    if (lane == 0) {
      // g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b) (mlx_lm's compute_g, sigmoid)
      const float s = float(bfloat(float(Ain[w * NV + hv]) + float(DT[hv])));
      const float sp = float(bfloat(metal::max(s, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(s)))));
      G[w * NV + hv] = metal::exp(-metal::exp(float(ALOG[hv])) * sp);
      BETA[w * NV + hv] = bfloat(1.0f / (1.0f + metal::exp(-float(Bin[w * NV + hv]))));
    }
  }
"""

_GDN_POST = r"""
  // one simdgroup per (row, v head): out = SiLU(z) * RMSNorm(y) * w, and the out projection's group sums
  const uint lane = thread_index_in_simdgroup;
  const uint hv = threadgroup_position_in_grid.y;
  const uint m = threadgroup_position_in_grid.z;
  const int M = dims[0], MP = dims[1];
  constexpr int PER = DV / 32;
  constexpr int GPH = DV / 64;                          // 64-groups per head
  threadgroup bfloat ob[DV];
  if (int(m) >= M) {
    if (lane < GPH) XS[(hv * GPH + lane) * MP + m] = 0.0f;
    return;
  }
  float yv[PER];
  float ss = 0.0f;
  for (int j = 0; j < PER; j++) {
    yv[j] = float(Y[(m * NV + hv) * DV + lane * PER + j]);
    ss += yv[j] * yv[j];
  }
  ss = simd_sum(ss);
  const float inv = metal::rsqrt(ss / float(DV) + eps[0]);
  for (int j = 0; j < PER; j++) {
    const int d = int(lane) * PER + j;
    const float x = float(bfloat(float(NW[d]) * (yv[j] * inv)));
    const float zf = float(Z[m * NV * DV + hv * DV + d]);
    const bfloat o = bfloat(zf / (1.0f + metal::exp(-zf)) * x);
    OUT[m * NV * DV + hv * DV + d] = o;
    ob[d] = o;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < GPH) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += float(ob[lane * 64 + i]);
    XS[(hv * GPH + lane) * MP + m] = acc;
  }
"""

_MLP_ACT = r"""
  // 64 threads per (row, 64-group): h = SiLU(gate) * up, and the down projection's group sums
  const uint t = thread_position_in_threadgroup.x;
  const uint g = threadgroup_position_in_grid.x;
  const uint m = threadgroup_position_in_grid.y;
  const int M = dims[0], MP = dims[1];
  threadgroup bfloat hb[64];
  if (int(m) >= M) {
    if (t == 0) XS[g * MP + m] = 0.0f;
    return;
  }
  const int e = int(m) * N + int(g) * 64 + int(t);
  const float gf = float(GATE[e]);
  const bfloat h = bfloat(gf / (1.0f + metal::exp(-gf)) * float(UP[e]));
  HOUT[e] = h;
  hb[t] = h;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += float(hb[i]);
    XS[g * MP + m] = acc;
  }
"""

_kernels: dict[str, Any] = {}


def _named(base: str, source: str) -> str:
    return f"{base}_{hashlib.sha256(source.encode()).hexdigest()[:16]}"


def _kernel(name: str) -> Any:
    if name not in _kernels:
        spec = {
            "norm": (_NORM_XS.replace("RESIDUAL_ADD", "h = bfloat(float(h) + float(R[base + i]));\n    HO[base + i] = h;"),
                     ["H", "R", "Wt", "eps", "dims"], ["HO", "XO", "XS"]),
            "norm_nores": (_NORM_XS.replace("RESIDUAL_ADD", ""), ["H", "Wt", "eps", "dims"], ["XO", "XS"]),

            "gdn_pre": (_GDN_PRE, ["QKV", "CS", "CW", "windows", "Ain", "Bin", "ALOG", "DT"], ["Q", "Kout", "Vout", "G", "BETA"]),
            "gdn_post": (_GDN_POST, ["Y", "Z", "NW", "eps", "dims"], ["OUT", "XS"]),
            "mlp_act": (_MLP_ACT, ["GATE", "UP", "dims"], ["HOUT", "XS"]),
        }[name]
        source, inputs, outputs = spec
        _kernels[name] = mx.fast.metal_kernel(name=_named("lane_glue_" + name, source), input_names=inputs,
                                              output_names=outputs, source=source)
    return _kernels[name]


_consts: dict[Any, mx.array] = {}


def _const(key: Any, make: Any) -> mx.array:
    if key not in _consts:
        _consts[key] = make()
    return _consts[key]


def _dims(m: int) -> mx.array:
    mp = 16 * ((m + 15) // 16)
    return _const(("dims", m), lambda: mx.array([m, mp], dtype=mx.int32))


def _eps(eps: float) -> mx.array:
    return _const(("eps", float(eps)), lambda: mx.array([float(eps)], dtype=mx.float32))


def remember(x: mx.array, xs: mx.array) -> mx.array:
    """Hand ``xs`` to the lane matmul as ``x``'s group sums (its XSUM kernel then does not run)."""

    from tensorfold.kernels import lane_qmm

    lane_qmm._xs_cache[id(x)] = (x, xs)
    while len(lane_qmm._xs_cache) > 4:
        lane_qmm._xs_cache.pop(next(iter(lane_qmm._xs_cache)))
    return x


def norm_xs(hidden: mx.array, residual: mx.array | None, weight: mx.array, eps: float
            ) -> tuple[mx.array, mx.array]:
    """(h, x): h = hidden + residual (hidden when residual is None), x = RMSNorm(h) * weight.

    ``x`` comes with its group sums remembered for the lane matmul. Shapes (1, M, K), bf16.
    """

    lead = hidden.shape[:-1]
    K = int(hidden.shape[-1])
    M = 1
    for d in lead:
        M *= int(d)
    MP = 16 * ((M + 15) // 16)
    if K % 512 or K > 16384:
        raise ValueError(f"norm_xs: the hidden size must be a multiple of 512 up to 16384, got {K}")
    tpg = K // 16                       # 16 elements a thread
    common = dict(grid=(tpg, MP, 1), threadgroup=(tpg, 1, 1))
    if residual is None:
        x, xs = _kernel("norm_nores")(
            inputs=[hidden.reshape(M, K), weight, _eps(eps), _dims(M)],
            template=[("K", K)],
            output_shapes=[(M, K), (K // 64, MP)], output_dtypes=[mx.bfloat16, mx.float32], **common)
        h = hidden
    else:
        h, x, xs = _kernel("norm")(
            inputs=[hidden.reshape(M, K), residual.reshape(M, K), weight, _eps(eps), _dims(M)],
            template=[("K", K)],
            output_shapes=[(M, K), (M, K), (K // 64, MP)], output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32],
            **common)
        h = h.reshape(*lead, K)
    x = x.reshape(*lead, K)
    return h, remember(x, xs)


def gdn_pre(qkv: mx.array, conv_state: mx.array, conv_weight: mx.array, windows: mx.array, a: mx.array,
            b: mx.array, a_log: mx.array, dt_bias: mx.array, *, nk: int, nv: int, dk: int, dv: int
            ) -> tuple[mx.array, ...]:
    """q, k [1, W, nk, dk], v [1, W, nv, dv], g [1, W, nv] fp32, beta [1, W, nv] for W window rows.

    ``windows`` [W, taps] indexes rows of [conv_state (taps - 1 rows); qkv rows]: row w's conv inputs.
    """

    W = int(qkv.shape[-2])
    C = int(qkv.shape[-1])
    taps = int(conv_weight.shape[1])
    if dk != dv or dk % 32:
        raise ValueError("gdn_pre: needs head_k_dim == head_v_dim, a multiple of 32")
    q, k, v, g, beta = _kernel("gdn_pre")(
        inputs=[qkv.reshape(W, C), conv_state.reshape(taps - 1, C), conv_weight.reshape(C, taps), windows,
                a.reshape(W, nv), b.reshape(W, nv), a_log, dt_bias],
        template=[("NK", nk), ("NV", nv), ("DK", dk), ("DV", dv), ("TAPS", taps)],
        grid=(32, 2 * nk + nv, W), threadgroup=(32, 1, 1),
        output_shapes=[(1, W, nk, dk), (1, W, nk, dk), (1, W, nv, dv), (1, W, nv), (1, W, nv)],
        output_dtypes=[qkv.dtype, qkv.dtype, qkv.dtype, mx.float32, qkv.dtype])
    return q, k, v, g, beta


def gdn_post(y: mx.array, z: mx.array, weight: mx.array, eps: float) -> mx.array:
    """SiLU(z) * RMSNorm(y) * weight per head: y [1, W, nv, dv], z [1, W, nv * dv] -> [1, W, nv * dv]."""

    _, W, nv, dv = (int(s) for s in y.shape)
    MP = 16 * ((W + 15) // 16)
    out, xs = _kernel("gdn_post")(
        inputs=[y, z.reshape(W, nv * dv), weight, _eps(eps), _dims(W)],
        template=[("NV", nv), ("DV", dv)],
        grid=(32, nv, MP), threadgroup=(32, 1, 1),
        output_shapes=[(1, W, nv * dv), (nv * dv // 64, MP)], output_dtypes=[y.dtype, mx.float32])
    return remember(out, xs)


def mlp_act(gate: mx.array, up: mx.array) -> mx.array:
    """SiLU(gate) * up, (1, W, N) bf16, with its group sums remembered for the down projection."""

    N = int(gate.shape[-1])
    W = int(gate.size // N)
    MP = 16 * ((W + 15) // 16)
    h, xs = _kernel("mlp_act")(
        inputs=[gate.reshape(W, N), up.reshape(W, N), _dims(W)], template=[("N", N)],
        grid=(64 * (N // 64), MP, 1), threadgroup=(64, 1, 1),
        output_shapes=[(W, N), (N // 64, MP)], output_dtypes=[gate.dtype, mx.float32])
    return remember(h.reshape(gate.shape), xs)


__all__ = ["gdn_post", "gdn_pre", "mlp_act", "norm_xs", "remember"]
