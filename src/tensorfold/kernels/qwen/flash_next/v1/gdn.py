"""Gated DeltaNet after its input projection: the conv, the delta-rule scan and the gated norm, rows in order."""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1.base import MAX_STREAMS, consts, count, ints, kernel, pick

_GDN_STEP = r"""
  // One threadgroup of 32 simdgroups per value head hv (key head hv / (NV / NK)); simdgroup s owns state rows
  // dv = 4 s .. 4 s + 3, lane l their columns dk = 4 l .. 4 l + 3 (the layout of mlx_lm's gated_delta kernel).
  // P rows are [qkv (C) | z (NV DV) | b (NV) | a (NV)]; the conv reads [conv state (TAPS - 1 rows); P rows].
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int hv = int(threadgroup_position_in_grid.x);
  const int hk = hv / (NV / NK);
  const int R = rows[0];
  constexpr int C = 2 * NK * DK + NV * DV;
  constexpr int PW = C + NV * DV + 2 * NV;
  constexpr int RPS = DV / 32;                              // state rows a simdgroup
  threadgroup float qs[DK], ks[DK], vs[DV], ys[DV];
  threadgroup float red[2][32];
  threadgroup float gates[2];
  // this head's conv channels: q (hk), k (hk), v (hv)
  int c = -1;
  if (int(t) < DK) c = hk * DK + int(t);
  else if (int(t) < 2 * DK) c = NK * DK + hk * DK + int(t) - DK;
  else if (int(t) < 2 * DK + DV) c = 2 * NK * DK + hv * DV + int(t) - 2 * DK;
  const bool writes_qk = (hv % (NV / NK)) == 0;
  float state[RPS][4];
  for (int j = 0; j < RPS; j++)
    for (int i = 0; i < 4; i++)
      state[j][i] = HAS_STATE ? SIN[(size_t(hv) * DV + sg * RPS + j) * DK + lane * 4 + i] : 0.0f;
  for (int r = 0; r < R; r++) {
    if (c >= 0) {
      float conv = 0.0f;
      for (int tap = 0; tap < TAPS; tap++) {
        const int at = r + tap;                               // into [conv state; P rows]
        const float xv = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[(at - (TAPS - 1)) * PW + c]);
        conv = fma(float(CW[c * TAPS + tap]), xv, conv);
      }
      const float act = bsilu(conv);                          // conv + SiLU in fp32, stored as bf16
      if (int(t) < DK) qs[t] = act;
      else if (int(t) < 2 * DK) ks[int(t) - DK] = act;
      else vs[int(t) - 2 * DK] = act;
      if (c < 2 * NK * DK ? writes_qk : true) {
        for (int j = 0; j < TAPS - 1; j++) {                  // the conv window after this row
          const int at = r + 1 + j;
          CSO[(r * (TAPS - 1) + j) * C + c] = at < TAPS - 1 ? CS[at * C + c] : P[(at - (TAPS - 1)) * PW + c];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < 2) {
      // q (sg 0) and k (sg 1): x / sqrt(sum(x^2) + 1e-6) in fp32 (the delta-rule kernel's in-kernel L2 norm),
      // q also times DK^-0.5; both stay fp32
      threadgroup float* x = sg == 0 ? qs : ks;
      float ss = 0.0f;
      for (int i = 0; i < DK / 32; i++) {
        const float v = x[lane * (DK / 32) + i];
        ss = fma(v, v, ss);
      }
      ss = simd_sum(ss);
      const float inv = metal::rsqrt(ss + 1e-6f) * (sg == 0 ? metal::rsqrt(float(DK)) : 1.0f);
      for (int i = 0; i < DK / 32; i++) x[lane * (DK / 32) + i] *= inv;
    } else if (sg == 2 && lane == 0) {
      // g = exp(-exp(A_log) * softplus(a + dt_bias)) in fp32, beta = sigmoid(b) as bf16
      const float b = float(P[r * PW + C + NV * DV + hv]);
      const float a = float(P[r * PW + C + NV * DV + NV + hv]);
      gates[0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
      gates[1] = bsig(b);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float g = gates[0], beta = gates[1];
    float kk[4], qq[4];
    for (int i = 0; i < 4; i++) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; }
    for (int j = 0; j < RPS; j++) {
      const int dv = int(sg) * RPS + j;
      float kv = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv += state[j][i] * kk[i];
      }
      kv = simd_sum(kv);
      const float delta = (vs[dv] - kv) * beta;
      float out = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out += state[j][i] * qq[i];
      }
      out = simd_sum(out);
      if (lane == 0) ys[dv] = float(bfloat(out));
      for (int i = 0; i < 4; i++) SO[((size_t(r) * NV + hv) * DV + dv) * DK + lane * 4 + i] = state[j][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float ss = 0.0f;
      for (int i = 0; i < DV / 32; i++) { const float v = ys[lane * (DV / 32) + i]; ss = fma(v, v, ss); }
      ss = simd_sum(ss);
      if (lane == 0) red[0][0] = metal::rsqrt(ss / float(DV) + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) < DV) {
      // sigmoid-gated RMSNorm: mx.fast.rms_norm's bf16(w * bf16(y * inv)), times sigmoid(z) in fp32, bf16 out
      const float y = float(bfloat(float(NW[t]) * float(bfloat(ys[t] * red[0][0]))));
      const float z = float(P[r * PW + C + hv * DV + int(t)]);
      OUT[r * NV * DV + hv * DV + int(t)] = bfloat(y * fsig(z));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""


def held(rows: int) -> int:
    """Rows the per-row state buffers are allocated with: a multiple of 8, so a round of a different row count
    reuses a cached buffer (MLX reuses one only of about the same size) instead of allocating 3 MB a row afresh.
    Rows past ``rows`` are never written or read."""

    return -(-rows // 8) * 8


def gdn_step(projected: mx.array, conv_state: mx.array, ssm_state: mx.array | None, conv_weight: mx.array,
             a_log: mx.array, dt_bias: mx.array, norm_weight: mx.array, eps: mx.array, *, nk: int, nv: int,
             dk: int, dv: int) -> tuple[mx.array, mx.array, mx.array]:
    """One Gated DeltaNet layer after its projection over R consecutive rows: (out [R, NV*DV] bf16, the conv and
    recurrent states after each row). ``ssm_state`` None starts from zeros."""

    rows = int(projected.shape[0])
    channels = 2 * nk * dk + nv * dv
    taps = int(conv_weight.shape[-1])
    if dk != 128 or dv != 128:
        raise ValueError("gdn_step: written for 128-dim heads")
    has_state = ssm_state is not None
    state = ssm_state if has_state else consts.get(("no state", nv, dv, dk))
    if state is None:                             # full size, so the input stays ``device`` (kernels.inputs)
        state = consts[("no state", nv, dv, dk)] = mx.zeros((nv, dv, dk), dtype=mx.float32)
    run = kernel("q4_gdn_step", _GDN_STEP,
                     ["P", "CS", "SIN", "CW", "ALOG", "DT", "NW", "eps", "rows"], ["OUT", "CSO", "SO"])
    out, conv_rows, ssm_rows = run(
        inputs=[projected, conv_state, state, conv_weight, a_log, dt_bias, norm_weight, eps, count(rows)],
        template=[("NK", nk), ("NV", nv), ("DK", dk), ("DV", dv), ("TAPS", taps), ("HAS_STATE", int(has_state))],
        grid=(nv * 1024, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(rows, nv * dv), (held(rows), taps - 1, channels), (held(rows), nv, dv, dk)],
        output_dtypes=[mx.bfloat16, conv_state.dtype, mx.float32])
    return out, conv_rows, ssm_rows


def _gdn_source(streams: int) -> str:
    src = _GDN_STEP
    swaps = [
        ("  const int R = rows[0];\n",
         "  const int sb = int(threadgroup_position_in_grid.y);\n"
         "  const int row0 = STARTS[sb];\n"
         "  const int R = STARTS[sb + 1] - row0;\n"
         f"  const device bfloat* CSb = {pick('CS', streams, 'sb')};\n"
         f"  const device float* SINb = {pick('SIN', streams, 'sb')};\n"),
        ("state[j][i] = HAS_STATE ? SIN[(size_t(hv) * DV + sg * RPS + j) * DK + lane * 4 + i] : 0.0f;",
         "state[j][i] = HAS_STATE ? SINb[(size_t(hv) * DV + sg * RPS + j) * DK + lane * 4 + i] : 0.0f;"),
        ("const float xv = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[(at - (TAPS - 1)) * PW + c]);",
         "const float xv = at < TAPS - 1 ? float(CSb[at * C + c]) : float(P[(row0 + at - (TAPS - 1)) * PW + c]);"),
        ("CSO[(r * (TAPS - 1) + j) * C + c] = at < TAPS - 1 ? CS[at * C + c] : P[(at - (TAPS - 1)) * PW + c];",
         "CSO[((row0 + r) * (TAPS - 1) + j) * C + c] = at < TAPS - 1 ? CSb[at * C + c] : "
         "P[(row0 + at - (TAPS - 1)) * PW + c];"),
        ("const float b = float(P[r * PW + C + NV * DV + hv]);",
         "const float b = float(P[(row0 + r) * PW + C + NV * DV + hv]);"),
        ("const float a = float(P[r * PW + C + NV * DV + NV + hv]);",
         "const float a = float(P[(row0 + r) * PW + C + NV * DV + NV + hv]);"),
        ("SO[((size_t(r) * NV + hv) * DV + dv) * DK + lane * 4 + i] = state[j][i];",
         "SO[((size_t(row0 + r) * NV + hv) * DV + dv) * DK + lane * 4 + i] = state[j][i];"),
        ("const float z = float(P[r * PW + C + hv * DV + int(t)]);",
         "const float z = float(P[(row0 + r) * PW + C + hv * DV + int(t)]);"),
        ("OUT[r * NV * DV + hv * DV + int(t)] = bfloat(y * fsig(z));",
         "OUT[(row0 + r) * NV * DV + hv * DV + int(t)] = bfloat(y * fsig(z));"),
    ]
    for old, new in swaps:
        if src.count(old) != 1:
            raise RuntimeError(f"gdn_step source changed; cannot derive the multi-stream variant at: {old!r}")
        src = src.replace(old, new)
    return src

def gdn_step_multi(projected: mx.array, conv_states: Sequence[mx.array], ssm_states: Sequence[mx.array],
                   rows: Sequence[int], conv_weight: mx.array, a_log: mx.array, dt_bias: mx.array,
                   norm_weight: mx.array, eps: mx.array, *, nk: int, nv: int, dk: int, dv: int
                   ) -> tuple[mx.array, mx.array, mx.array]:
    """``gdn_step`` over several streams' rows (stream b's rows[b] rows in order), each scanned from its own
    conv and recurrent state."""

    streams = len(rows)
    if not 1 <= streams <= MAX_STREAMS or len(conv_states) != streams or len(ssm_states) != streams:
        raise ValueError(f"gdn_step_multi: 1-{MAX_STREAMS} streams, a conv and a recurrent state each")
    if dk != 128 or dv != 128:
        raise ValueError("gdn_step_multi: written for 128-dim heads")
    total = int(projected.shape[0])
    if sum(int(n) for n in rows) != total or min(int(n) for n in rows) < 1:
        raise ValueError("gdn_step_multi: rows must cover the projected rows, one or more a stream")
    channels = 2 * nk * dk + nv * dv
    taps = int(conv_weight.shape[-1])
    starts = [0]
    for n in rows:
        starts.append(starts[-1] + int(n))
    names = (["P"] + [f"CS{b}" for b in range(streams)] + [f"SIN{b}" for b in range(streams)]
             + ["CW", "ALOG", "DT", "NW", "eps", "STARTS"])
    run = kernel(f"q4_gdn_step_multi{streams}", lambda: _gdn_source(streams), names, ["OUT", "CSO", "SO"])
    out, conv_rows, ssm_rows = run(
        inputs=[projected, *conv_states, *ssm_states, conv_weight, a_log, dt_bias, norm_weight, eps, ints(starts)],
        template=[("NK", nk), ("NV", nv), ("DK", dk), ("DV", dv), ("TAPS", taps), ("HAS_STATE", 1)],
        grid=(nv * 1024, streams, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(total, nv * dv), (held(total), taps - 1, channels), (held(total), nv, dv, dk)],
        output_dtypes=[mx.bfloat16, conv_states[0].dtype, mx.float32])
    return out, conv_rows, ssm_rows
