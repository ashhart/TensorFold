"""Qwen3.8 Flash Next decode in a few kernels a layer.

The reference forward (``model.py``) runs ~100 small MLX ops a layer, ~4,800 a
token; building and encoding them costs ~31 ms of CPU a token on the M3 Ultra
while the GPU needs ~5 ms for the weights. Here each step between two weight
reads is one kernel:

    hc_norm   write-back of the previous block into the four streams + their partial sums of squares
    hc_project  the hyper-connection's down and inject projections (split over the inputs) from the normed
              streams, then the up projection + sigmoid + the mean over streams of mix * normed
              (folding hc_norm into the split down projection made every threadgroup redo the norm: GPU 12.0 ->
              13.2 ms a step on the M3 Ultra, 2026-09-25)
    gdn_step  Gated DeltaNet after its input projection: conv window + conv + SiLU, q/k L2 norms,
              g and beta, the delta-rule state update and read-out, and the sigmoid-gated RMSNorm
    router    the MoE router's bf16 logits (a simdgroup an expert)
    route     softmax over the experts -> top k and their renormalized weights
    swiglu    SiLU(gate) * up for the shared expert (bf16 ops)
    expert_gateup   the routed experts' gate and up projections + SiLU(gate) * up, one pass over x
    expert_down     the routed experts' down projections + their weighted sum
    expert_group    each row's top-k experts and weights, and the distinct experts of all rows with the rows (and
                    slots) that picked each: rows of a window share experts (8 consecutive tokens pick ~40 distinct
                    of 80, 2026-09-25)
    grouped_gateup / grouped_down   the experts' projections, each distinct expert's weights read once for all
                    the rows that picked it; a row's sums are the ungrouped kernels', whatever the other rows
    attn_prep the attention's q/k norms and RoPE, from its stacked projection
    attn_gate attention output * sigmoid(gate)
    idx_scores / idx_select   the sparse attention's block choice past 2,048 keys: each row's block scores
              (sum over index heads of relu(q . pooled block) / sqrt(d), fp32), then its top blocks (radix select,
              lowest block id among ties) as the list of keys it reads, in position order, with its unfinished tail
    qmv       4-bit matvec for 1-8 rows with the same bits for a row at any row count (MLX 0.32's quantized
              matmul on an M3 Ultra sums a row differently when 2-4 rows ride together, 2026-09-25)

Every kernel takes R rows of consecutive tokens and treats each row on its own
in a fixed order, so a row's bits do not depend on how many rows ride with it.
The arithmetic is the checkpoint's training framework's (PyTorch/FLA): fp32
math and one rounding per op where a tensor is stored in bf16, and the delta
rule's in-kernel fp32 L2 norms. MLX's op-by-op version (``model.py``) rounds a
few more intermediates to bf16 (its bf16 sigmoid, conv and sums), so the two
agree to bf16 rounding, not bit for bit. Serial decoding goes through these
kernels too, so they define the reference TensorFold's drafted rounds must
reproduce.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import mlx.core as mx

MAX_ROWS = 16

# 4-bit affine, groups of 32: a group is 4 uint32 words, 8 values a word, low nibble first
_QDOT_HEADER = r"""
// sum_i w_i x_i over one 32-value group: w = scale * q + bias
inline float qgroup_dot(const device uint32_t* w, float scale, float bias, const thread float* x) {
  float dq = 0.0f, dx = 0.0f;
  for (int word = 0; word < 4; word++) {
    const uint32_t bits = w[word];
    for (int n = 0; n < 8; n++) {
      const float xv = x[word * 8 + n];
      dq = fma(float((bits >> (4 * n)) & 0xFu), xv, dq);
      dx += xv;
    }
  }
  return fma(scale, dq, bias * dx);
}
// Elementwise ops as the checkpoint's training framework does them on bf16 tensors: fp32 math, one rounding.
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
inline float bsilu(float x) { return float(bfloat(x / (1.0f + metal::exp(-x)))); }
inline float fsig(float x) { return 1.0f / (1.0f + metal::exp(-x)); }
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
// softplus in fp32 (threshold 20, as torch.nn.functional.softplus)
inline float fsoftplus(float x) { return x > 20.0f ? x : log1p_(metal::exp(x)); }


// Rank-k expert of a row inside one simdgroup: lane l holds logits l, l + 32, ...; rounds of (largest logit,
// lowest id); returns the id picked in round k and, through ``picked``, the logits of rounds 0..k.
template <int NE>
inline int simd_topk(const device float* logits, int k, uint lane, thread float* picked) {
  float v[NE / 32];
  for (int j = 0; j < NE / 32; j++) v[j] = logits[j * 32 + int(lane)];
  int id = 0;
  for (int round = 0; round <= k; round++) {
    float best = -INFINITY;
    int bid = NE;
    for (int j = 0; j < NE / 32; j++) {
      const int e = j * 32 + int(lane);
      if (v[j] > best || (v[j] == best && e < bid)) { best = v[j]; bid = e; }
    }
    for (int off = 16; off > 0; off /= 2) {
      const float ob = simd_shuffle_xor(best, off);
      const int oi = simd_shuffle_xor(bid, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
    }
    picked[round] = best;
    id = bid;
    if (int(lane) == bid % 32) v[bid / 32] = -INFINITY;
  }
  return id;
}
// simd_topk's rounds 0 .. TOPK-1 in one pass: ids[k] and logits picked[k] of each round
template <int NE, int TOPK>
inline void simd_topk_all(const device float* logits, uint lane, thread int* ids, thread float* picked) {
  float v[NE / 32];
  for (int j = 0; j < NE / 32; j++) v[j] = logits[j * 32 + int(lane)];
  for (int round = 0; round < TOPK; round++) {
    float best = -INFINITY;
    int bid = NE;
    for (int j = 0; j < NE / 32; j++) {
      const int e = j * 32 + int(lane);
      if (v[j] > best || (v[j] == best && e < bid)) { best = v[j]; bid = e; }
    }
    for (int off = 16; off > 0; off /= 2) {
      const float ob = simd_shuffle_xor(best, off);
      const int oi = simd_shuffle_xor(bid, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
    }
    picked[round] = best;
    ids[round] = bid;
    if (int(lane) == bid % 32) v[bid / 32] = -INFINITY;
  }
}
// MLX's 4-bit qmv inner loop (quantized.h): 16 inputs a lane, pre-divided by 1, 16, 256, 4096 so the masked
// nibbles need no shift; w = scale * q + bias gives scale * dot(q, x) + bias * sum(x). As in MLX, each run of 4
// inputs is summed in bf16 (its x[i] + x[i + 1] + ... on bfloat16_t) before the fp32 sum: with that, a row's
// result is bit for bit MLX's one-row quantized matmul.
inline float load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
// one lane's 16 inputs times its 8 bytes of one weight row (qdot16 with the weights already loaded)
inline float qdot16w(const thread uint16_t* ws, const thread float* xt, float scale, float bias, float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
inline float qdot16(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
"""

_WRITEBACK = "hv = float(bfloat(hv + float(bfloat(branch * float(INJ[r * S + s])))));"
# Branch written back into the streams, per model dim d of row r (``branch``, fp32 of a bf16 value).
_BRANCH_NONE = ""
_BRANCH_PLAIN = "const float branch = float(BR[r * D + d]);"
# MoE: the routed sum (bf16, from expert_down) + bf16(shared * bf16(sigmoid(shared gate))), a bf16 add
_BRANCH_MOE = r"""const float shared = float(bfloat(float(SH[r * D + d]) * bsig(float(SG[r]))));
      const float branch = float(bfloat(float(ROUTED[r * D + d]) + shared));"""

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

_ROUTER = r"""
  // one simdgroup per expert: lane l reads 8 consecutive inputs at a time, 256 apart; rows in order
  const uint lane = thread_index_in_simdgroup;
  const int e = int(threadgroup_position_in_grid.x) * (T / 32) + int(simdgroup_index_in_threadgroup);
  const int R = rows[0];
  if (e >= NE) return;
  const device bfloat* w = GW + size_t(e) * D;
  float acc[MAXR];
  for (int r = 0; r < MAXR; r++) acc[r] = 0.0f;
  for (int c = 8 * int(lane); c < D; c += 256) {
    float wv[8];
    for (int j = 0; j < 8; j++) wv[j] = float(w[c + j]);
    for (int r = 0; r < MAXR; r++) {
      if (r >= R) break;
      const device bfloat* xr = X + r * D + c;
      float a = acc[r];
      for (int j = 0; j < 8; j++) a = fma(float(xr[j]), wv[j], a);
      acc[r] = a;
    }
  }
  for (int r = 0; r < MAXR; r++) {
    if (r >= R) break;
    const float total = simd_sum(acc[r]);
    if (lane == 0) OUT[r * NE + e] = OUT_T(total);
  }
"""

_ROUTE = r"""
  // One threadgroup of NE threads per row, a thread an expert: softmax in fp32 (sums in simdgroup order), then
  // each expert's rank = experts with a larger probability, or an equal one and a lower id; ranks below TOPK are the
  // picks, in rank order; weights / their sum (in rank order) in fp32, bf16 out.
  const uint e = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.x);
  threadgroup float red[NE / 32];
  threadgroup float probs[NE];
  threadgroup float picked[TOPK];
  const float logit = float(L[r * NL + int(e)]);
  float m = simd_max(logit);
  if (lane == 0) red[g] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = red[0];
  for (int k = 1; k < NE / 32; k++) m = metal::max(m, red[k]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float x = metal::exp(logit - m);
  const float zs = simd_sum(x);
  if (lane == 0) red[g] = zs;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float z = 0.0f;
  for (int k = 0; k < NE / 32; k++) z += red[k];
  const float p = x / z;
  probs[e] = p;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int rank = 0;
  for (int j = 0; j < NE; j++) {
    const float q = probs[j];
    rank += (q > p || (q == p && j < int(e))) ? 1 : 0;
  }
  if (rank < TOPK) {
    EXPERTS[r * TOPK + rank] = uint32_t(e);
    picked[rank] = p;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (e == 0) {
    float total = 0.0f;
    for (int k = 0; k < TOPK; k++) total += picked[k];
    for (int k = 0; k < TOPK; k++) WEIGHTS[r * TOPK + k] = bfloat(picked[k] / total);
  }
"""

_SWIGLU = r"""
  // SiLU(gate) * up: bf16(bf16(silu(g)) * u); gate and up are N-wide runs of
  // their rows (stride S, offsets GO and UO: two arrays, or two parts of one stacked projection)
  const uint i = thread_position_in_grid.x;
  const int row = int(i) / N, col = int(i) % N;
  const float g = float(G[row * GSTRIDE + GO + col]);
  const float u = float(U[row * USTRIDE + UO + col]);
  OUT[i] = bfloat(bsilu(g) * u);
"""

_ATTN_PREP = r"""
  // one threadgroup of HD threads per (row, head): heads [0, NQ) are queries (from the stacked projection's
  // [q | gate] pairs), [NQ, NQ + NKV) keys, then NI indexer queries (IHD dims each, after the values). RMSNorm
  // with (1 + w) in fp32, bf16 out, then RoPE on the first RD dims (non-interleaved halves), angles in fp32 at the
  // row's position.
  const int d = int(thread_position_in_threadgroup.x);
  const int head = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const bool isq = head < NQ;
  const bool isi = head >= NQ + NKV;
  const int width = isi ? IHD : HD;
  const bool live = d < width;
  int src;
  if (isq) src = r * PW + head * 2 * HD + d;
  else if (!isi) src = r * PW + NQ * 2 * HD + (head - NQ) * HD + d;
  else src = r * PW + NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD + d;
  threadgroup float part[HD / 32];
  threadgroup float normed[HD];
  const float x = live ? float(P[src]) : 0.0f;
  float ss = simd_sum(x * x);
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int k = 0; k < width / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(width) + eps[0]);
  const float nw = live ? (isq ? QW[d] : (isi ? IW[d] : KW[d])) : 0.0f;
  normed[d] = float(bfloat((x * inv) * nw));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!live) return;
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    // as mx.fast.rope: inv_freq = exp2(-(i / half) * log2(base)), fast cos/sin of position * inv_freq
    const float freq = metal::exp2(-(float(i) / float(hr)) * LOG2BASE[0]);
    const float angle = float(POS[r]) * freq;
    const float c = metal::fast::cos(angle), s = metal::fast::sin(angle);
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  if (isq) Q[(r * NQ + head) * HD + d] = bfloat(out);
  else if (isi) IQ[(r * NI + head - NQ - NKV) * IHD + d] = bfloat(out);
  else Kout[(r * NKV + head - NQ) * HD + d] = bfloat(out);
"""

_ATTN_GATE = r"""
  // attention output [R, H, D] (rows of heads) times sigmoid(gate) (bf16 ops); gate from the [q | gate] pairs
  const uint i = thread_position_in_grid.x;
  const int r = int(i) / (NQ * HD), c = int(i) % (NQ * HD);
  const int head = c / HD, d = c % HD;
  const float g = float(P[r * PW + head * 2 * HD + HD + d]);
  OUT[i] = bfloat(float(A[i]) * bsig(g));
"""

_HC_NORM = r"""
  // Threadgroup (j, r): dims 256 j .. 256 j + 255 of row r in every stream: write the block's branch back into the
  // S streams (bf16 ops) and each stream's partial sum of squares over these dims (fp32, simdgroups in order).
  // Consumers take a stream's inverse RMS from its NT partials, added in j order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int j = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D;
  constexpr int NT = D / 256;
  threadgroup float part[8][S];
  const int d = j * 256 + int(t);
  float ss[S];
  BRANCH
  for (int s = 0; s < S; s++) {
    const int e = s * D + d;
    float hv = float(H[r * W + e]);
    WRITEBACK
    HN[r * W + e] = bfloat(hv);
    ss[s] = simd_sum(hv * hv);
  }
  if (lane == 0) for (int s = 0; s < S; s++) part[g][s] = ss[s];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < S) {
    float total = 0.0f;
    for (int k = 0; k < 8; k++) total += part[k][t];
    SSP[(r * NT + j) * S + t] = total;
  }
"""

# a stream's inverse RMS from hc_norm's partials (row r, stream s)
_RINV = r"""
inline float stream_rinv(const device float* ssp, int r, int s, int nt, int streams, int dims, float eps) {
  float total = 0.0f;
  for (int j = 0; j < nt; j++) total += ssp[(r * nt + j) * streams + s];
  return metal::rsqrt(total / float(dims) + eps);
}
"""

_HC_DOWN_SPLIT = r"""
  // Split-K matvec of the normed streams: threadgroup (i, k) takes rows 8 i .. 8 i + 7 (a simdgroup each)
  // over input groups 32 k .. 32 k + 31 (a lane each); partial sums PART[k][r][o] (fp32), summed in k order
  // by the consumer. The chunk's normed inputs bf16((h * rinv) * w) are staged once in threadgroup memory.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int R = rows[0];
  constexpr int W = S * D;
  constexpr int GROUPS = W / 32;
  constexpr int KS = GROUPS / 32;
  const int o = int(threadgroup_position_in_grid.x) * 8 + int(g);
  const int k = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);   // rows run in parallel threadgroups
  const int c0 = k * 32 * 32;                         // first input of the chunk
  threadgroup float xs[32 * 33];                      // group j at 33 j: a lane's reads hit distinct banks
  threadgroup float rinv[S];
  if (t < S) rinv[t] = stream_rinv(SSP, r, int(t), D / 256, S, D, eps[0]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(t); i < 32 * 32; i += 256) {
    const int e = c0 + i;
    xs[(i / 32) * 33 + i % 32] = float(bfloat((float(HN[r * W + e]) * rinv[e / D]) * NW[e]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (o < ND) {
    const int grp = k * 32 + int(lane);
    float x[32];
    for (int n = 0; n < 32; n++) x[n] = xs[lane * 33 + n];
    float acc = qgroup_dot(QW + (size_t(o) * GROUPS + grp) * 4, float(QS[o * GROUPS + grp]), float(QB[o * GROUPS + grp]), x);
    acc = simd_sum(acc);
    if (lane == 0) PART[(k * R + r) * ND + o] = acc;
  }
"""

_HC_UP2 = r"""
  // One threadgroup of 10 * 32 threads per 8 model dims. Prologue: the down projection's outputs from the
  // split-K partials (summed in chunk order) -> bf16 -> / S -> SiLU (bf16); threadgroup 0 also writes the
  // inject gates 2 sigmoid(inject / S). Then its 32 up rows {s * D + d0 + j}: thread t = 10 i + q takes group q of
  // row i; row sums in group order, sigmoid, times the normed stream, mean over streams.
  const uint t = thread_position_in_threadgroup.x;
  const int R = rows[0];
  const int r = int(threadgroup_position_in_grid.y);    // rows run in parallel threadgroups
  constexpr int W = S * D;
  constexpr int GPR = LOW / 32;
  constexpr int ROWS = S * 8;
  const int d0 = int(threadgroup_position_in_grid.x) * 8;
  threadgroup float act[GPR * 33];                    // group q at 33 q: distinct banks across a simdgroup
  threadgroup float part[ROWS][GPR];
  threadgroup float prod[ROWS];
  const int i = int(t) / GPR, q = int(t) % GPR;
  const int s = i / 8, d = d0 + i % 8;
  const int row = s * D + d;
  {
    for (int c = int(t); c < ND; c += ROWS * GPR) {
      float v = 0.0f;
      for (int k = 0; k < KS; k++) v += PART[(k * R + r) * ND + c];
      const float v4 = float(bfloat(float(bfloat(v)) / float(S)));
      if (c < LOW) act[(c / 32) * 33 + c % 32] = bsilu(v4);
      else if (threadgroup_position_in_grid.x == 0) INJOUT[r * S + (c - LOW)] = bfloat(2.0f * bsig(v4));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      float x[32];
      for (int n = 0; n < 32; n++) x[n] = act[q * 33 + n];
      part[i][q] = qgroup_dot(QW + (size_t(row) * GPR + q) * 4, float(QS[row * GPR + q]), float(QB[row * GPR + q]), x);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (q == 0) {
      float u = 0.0f;
      for (int qq = 0; qq < GPR; qq++) u += part[i][qq];
      const float rv = stream_rinv(SSP, r, s, D / 256, S, D, eps[0]);
      const float normed = float(bfloat((float(HN[r * W + row]) * rv) * NW[row]));
      prod[i] = float(bfloat(bsig(float(bfloat(u))) * normed));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 8) {
      float total = 0.0f;
      for (int ss = 0; ss < S; ss++) total += prod[ss * 8 + int(t)];
      MIXED[r * D + d0 + int(t)] = bfloat(total / float(S));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""

_EXPERT_GATEUP = r"""
  // Threadgroup (b, p): 2 simdgroups, rows 8 b + 4 g .. + 3 of slot p (p = r * SLOTS + k; slot k < TOPK is the
  // row's k-th expert by router logit, each simdgroup finding it itself), gate and up,
  // over K in steps of 512 (MLX's qmv_fast loop); then bf16(SiLU(bf16(gate)) * bf16(up)).
  // A slot past TOPK (SHARED = 1) is the shared expert, from its own matrices. Threadgroup (0, p)'s first
  // simdgroup also writes the slot's expert to PICK and, for the last routed slot (whose selection rounds give the
  // top-k logits), the weights exp(l_k - l_0) / their sum (fp32, bf16-rounded) to WTS: expert_down reads them.
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int p = int(threadgroup_position_in_grid.z);
  constexpr int SLOTS = TOPK + SHARED;
  const int r = p / SLOTS, slot = p % SLOTS;
  const bool shared = slot == TOPK;
  float picked[TOPK];
  const size_t e = shared ? 0 : size_t(simd_topk<NE>(LOGITS + r * NL, slot, lane, picked));
  if (!shared && threadgroup_position_in_grid.y == 0 && g == 0 && lane == 0) {
    PICK[r * TOPK + slot] = uint32_t(e);
    if (slot == TOPK - 1) {
      float total = 0.0f;
      float ex[TOPK];
      for (int kk = 0; kk < TOPK; kk++) { ex[kk] = metal::exp(picked[kk] - picked[0]); total += ex[kk]; }
      for (int kk = 0; kk < TOPK; kk++) WTS[r * TOPK + kk] = float(bfloat(ex[kk] / total));
    }
  }
  const int row0 = int(threadgroup_position_in_grid.y) * (SG * RPS) + int(g) * RPS;
  constexpr int KB = K / 2;                         // bytes a row
  constexpr int KG = K / 32;                        // groups a row
  const device uint32_t* GWp = shared ? SGW : GW;
  const device uint32_t* UWp = shared ? SUW : UW;
  const device bfloat* GSp = shared ? SGS : GS;
  const device bfloat* GBp = shared ? SGB : GB;
  const device bfloat* USp = shared ? SUS : US;
  const device bfloat* UBp = shared ? SUB : UB;
  const device uint8_t* gw = (const device uint8_t*)GWp + (e * N + row0) * KB + lane * 8;
  const device uint8_t* uw = (const device uint8_t*)UWp + (e * N + row0) * KB + lane * 8;
  const device bfloat* gs = GSp + (e * N + row0) * KG + lane / 2;
  const device bfloat* gb = GBp + (e * N + row0) * KG + lane / 2;
  const device bfloat* us = USp + (e * N + row0) * KG + lane / 2;
  const device bfloat* ub = UBp + (e * N + row0) * KG + lane / 2;
  const device bfloat* x = X + r * K + lane * 16;
  float xt[16];
  float ag[RPS], au[RPS];
  for (int row = 0; row < RPS; row++) { ag[row] = 0.0f; au[row] = 0.0f; }
  for (int k0 = 0; k0 < K; k0 += 512) {
    const float sum = load16(x, xt);
    for (int row = 0; row < RPS; row++) {
      ag[row] += qdot16(gw + row * KB, xt, float(gs[row * KG]), float(gb[row * KG]), sum);
      au[row] += qdot16(uw + row * KB, xt, float(us[row * KG]), float(ub[row * KG]), sum);
    }
    gw += 256; uw += 256; gs += 16; gb += 16; us += 16; ub += 16; x += 512;
  }
  for (int row = 0; row < RPS; row++) {
    const float gv = simd_sum(ag[row]), uv = simd_sum(au[row]);
    if (lane == 0) ACT[p * N + row0 + row] = bfloat(bsilu(float(bfloat(gv))) * float(bfloat(uv)));
  }
"""

_EXPERT_DOWN = r"""
  // Threadgroup (b, r): TOPK + SHARED simdgroups, simdgroup k takes slot k (the last: the shared expert; the
  // routed slots' experts and weights from expert_gateup's PICK and WTS) for model dims 8 b .. 8 b + 7: lane l reads 16-input chunks l and (l < NC - 32) 32 + l of each row (qmv's inner loop);
  // y = bf16(sum). routed = bf16(sum_k y_k w_k) (fp32, slots in order), shared = bf16(y_s * bf16(sigmoid(logit))),
  // out = bf16(routed + shared).
  const uint lane = thread_index_in_simdgroup;
  const int k = int(simdgroup_index_in_threadgroup);
  const int r = int(threadgroup_position_in_grid.z);
  const int d0 = int(threadgroup_position_in_grid.y) * 8;
  constexpr int SLOTS = TOPK + SHARED;
  constexpr int KB = NI / 2;
  constexpr int KG = NI / 32;
  constexpr int NC = NI / 16;                        // 16-input chunks a row
  threadgroup float ys[SLOTS][8];
  threadgroup float wts[TOPK];
  const bool shared = k == TOPK;
  // the routing expert_gateup found: slot k's expert, and the renormalized top-k weights
  const size_t e = shared ? 0 : size_t(PICK[r * TOPK + k]);
  if (k == 0 && int(lane) < TOPK) wts[lane] = WTS[r * TOPK + lane];
  const device uint32_t* DWp = shared ? SDW : DW;
  const device bfloat* DSp = shared ? SDS : DS;
  const device bfloat* DBp = shared ? SDB : DB;
  const device bfloat* x = ACT + (r * SLOTS + k) * NI;
  float xa[16], xb[16];
  const float sa = load16(x + lane * 16, xa);
  const bool second = int(lane) < NC - 32;
  const float sb = second ? load16(x + (32 + lane) * 16, xb) : 0.0f;
  for (int row = 0; row < 8; row++) {
    const size_t at = e * D + d0 + row;
    const device uint8_t* w = (const device uint8_t*)DWp + at * KB;
    float acc = qdot16(w + lane * 8, xa, float(DSp[at * KG + lane / 2]), float(DBp[at * KG + lane / 2]), sa);
    if (second)
      acc += qdot16(w + (32 + lane) * 8, xb, float(DSp[at * KG + (32 + lane) / 2]), float(DBp[at * KG + (32 + lane) / 2]), sb);
    acc = simd_sum(acc);
    if (lane == 0) ys[k][row] = float(bfloat(acc));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (k == 0 && lane < 8) {
    float routed = 0.0f;
    for (int kk = 0; kk < TOPK; kk++) routed = fma(ys[kk][lane], wts[kk], routed);
    float out = float(bfloat(routed));
    if (SHARED) out = float(bfloat(out + float(bfloat(ys[TOPK][lane] * bsig(float(bfloat(LOGITS[r * NL + NL - 1])))))));
    ROUTED[r * D + d0 + int(lane)] = bfloat(out);
  }
"""

_EXPERT_DOWN_Y = r"""
  // Threadgroup (b, z): SG simdgroups, each one (row, slot) pair (z SG + g in row-major order; slot TOPK: the shared
  // expert; routed slots' experts from expert_gateup's PICK) for model dims 8 b .. 8 b + 7: lane l reads 16-input
  // chunks l and (l < NC - 32) 32 + l of the activation; Y[row][slot][d] = bf16(sum), expert_down's sums. The
  // combine (weights, shared gate) is the next hc_norm's "grouped" write-back, expert_down's arithmetic.
  const uint lane = thread_index_in_simdgroup;
  const int R = rows[0];
  const int pair = int(threadgroup_position_in_grid.z) * SG + int(simdgroup_index_in_threadgroup);
  constexpr int SLOTS = TOPK + 1;
  if (pair >= R * SLOTS) return;
  const int r = pair / SLOTS, k = pair % SLOTS;
  const int d0 = int(threadgroup_position_in_grid.y) * 8;
  constexpr int KB = NI / 2;
  constexpr int KG = NI / 32;
  constexpr int NC = NI / 16;
  const bool shared = k == TOPK;
  const size_t e = shared ? 0 : size_t(PICK[r * TOPK + k]);
  const device uint32_t* DWp = shared ? SDW : DW;
  const device bfloat* DSp = shared ? SDS : DS;
  const device bfloat* DBp = shared ? SDB : DB;
  const device bfloat* x = ACT + (r * SLOTS + k) * NI;
  float xa[16], xb[16];
  const float sa = load16(x + lane * 16, xa);
  const bool second = int(lane) < NC - 32;
  const float sb = second ? load16(x + (32 + lane) * 16, xb) : 0.0f;
  for (int row = 0; row < 8; row++) {
    const size_t at = e * D + d0 + row;
    const device uint8_t* w = (const device uint8_t*)DWp + at * KB;
    float acc = qdot16(w + lane * 8, xa, float(DSp[at * KG + lane / 2]), float(DBp[at * KG + lane / 2]), sa);
    if (second)
      acc += qdot16(w + (32 + lane) * 8, xb, float(DSp[at * KG + (32 + lane) / 2]), float(DBp[at * KG + (32 + lane) / 2]), sb);
    acc = simd_sum(acc);
    if (lane == 0) Y[(r * SLOTS + k) * D + d0 + row] = bfloat(acc);
  }
"""

_QMV = r"""
  // MLX's qmv_fast inner loop: threadgroup b has SG simdgroups of RPS output rows each; lane l reads 16 inputs of
  // each 512-input step. The R input rows are taken in order inside, each with the same sums at any R (and MLX's
  // one-row sums). (Converting a word's nibbles once for all rows kept the bits but was slower at 4 rows on the
  // M3 Ultra, 124 -> 165 us, register pressure.)
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int row0 = int(threadgroup_position_in_grid.y) * (SG * RPS) + int(g) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 32;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 8;
  const device bfloat* sc = S + size_t(row0) * KG + lane / 2;
  const device bfloat* bi = B + size_t(row0) * KG + lane / 2;
  float acc[R][RPS];
  for (int r = 0; r < R; r++) for (int j = 0; j < RPS; j++) acc[r][j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[R][16], sum[R];
    for (int r = 0; r < R; r++) sum[r] = load16(X + r * K + k0 + lane * 16, xt[r]);
    for (int j = 0; j < RPS; j++) {
      uint16_t ws[4];
      const device uint16_t* wp = (const device uint16_t*)(w + j * KB);
      for (int i = 0; i < 4; i++) ws[i] = wp[i];
      const float s = float(sc[j * KG]), b = float(bi[j * KG]);
      for (int r = 0; r < R; r++) acc[r][j] += qdot16w(ws, xt[r], s, b, sum[r]);
    }
    w += 256; sc += 16; bi += 16;
  }
  for (int r = 0; r < R; r++)
    for (int j = 0; j < RPS; j++) {
      const float v = simd_sum(acc[r][j]);
      if (lane == 0) OUT[r * N + row0 + j] = bfloat(v);
    }
"""

_EXPERT_GROUP = r"""
  // One threadgroup of NE threads. Simdgroup r (rows in turn) ranks row r's experts by router logit (simd_topk:
  // largest first, lowest id among ties) and writes PICK[r][k] and the weights exp(l_k - l_0) / their sum (fp32,
  // bf16-rounded, as expert_down). Then thread e lists the (row, slot) pairs that picked expert e, in row order;
  // the distinct experts get places u in increasing id order: UIDS[u], UMEM[u][j] = row * 32 + slot (-1 after the
  // last), UCOUNT[0] = their number.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int R = rows[0];
  threadgroup int picks[MAXR][TOPK];
  threadgroup int offs[NE / 32];
  for (int r = int(g); r < R; r += NE / 32) {
    float picked[TOPK];
    int ids[TOPK];
    simd_topk_all<NE, TOPK>(LOGITS + r * NL, lane, ids, picked);
    if (lane == 0) {
      for (int k = 0; k < TOPK; k++) { picks[r][k] = ids[k]; PICK[r * TOPK + k] = uint32_t(ids[k]); }
      float ex[TOPK], total = 0.0f;
      for (int k = 0; k < TOPK; k++) { ex[k] = metal::exp(picked[k] - picked[0]); total += ex[k]; }
      for (int k = 0; k < TOPK; k++) WTS[r * TOPK + k] = float(bfloat(ex[k] / total));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int e = int(t);
  int members[MAXR];
  int count = 0;
  for (int r = 0; r < R; r++)
    for (int k = 0; k < TOPK; k++)
      if (picks[r][k] == e) members[count++] = r * 32 + k;
  const int used = count > 0 ? 1 : 0;
  const int before = simd_prefix_exclusive_sum(used);
  if (lane == 31) offs[g] = before + used;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int base = 0;
  for (int q = 0; q < int(g); q++) base += offs[q];
  if (used) {
    const int u = base + before;
    UIDS[u] = e;
    for (int j = 0; j < MAXR; j++) UMEM[u * MAXR + j] = j < count ? members[j] : -1;
  }
  if (t == NE - 1) UCOUNT[0] = base + before + used;
"""

_GROUPED_GATEUP = r"""
  // Threadgroup (b, u): MAXM simdgroups, simdgroup m takes the m-th row that picked distinct expert u (u = MAXU:
  // the shared expert, every row, slot TOPK) and computes its gate and up rows RPS b .. RPS b + RPS - 1 with
  // expert_gateup's loop (its bits). The simdgroups read the same weight rows: memory serves them once.
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  const int u = int(threadgroup_position_in_grid.z);
  const int R = rows[0];
  constexpr int SLOTS = TOPK + 1;
  const bool shared = u == MAXU;
  if (!shared && u >= UCOUNT[0]) return;
  const int member = shared ? (m < R ? m * 32 + TOPK : -1) : UMEM[u * MAXR + m];
  if (member < 0) return;
  const size_t e = shared ? 0 : size_t(UIDS[u]);
  const int r = member / 32, slot = member % 32;
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 32;
  const device uint32_t* GWp = shared ? SGW : GW;
  const device uint32_t* UWp = shared ? SUW : UW;
  const device bfloat* GSp = shared ? SGS : GS;
  const device bfloat* GBp = shared ? SGB : GB;
  const device bfloat* USp = shared ? SUS : US;
  const device bfloat* UBp = shared ? SUB : UB;
  const device uint8_t* gw = (const device uint8_t*)GWp + (e * N + row0) * KB + lane * 8;
  const device uint8_t* uw = (const device uint8_t*)UWp + (e * N + row0) * KB + lane * 8;
  const device bfloat* gs = GSp + (e * N + row0) * KG + lane / 2;
  const device bfloat* gb = GBp + (e * N + row0) * KG + lane / 2;
  const device bfloat* us = USp + (e * N + row0) * KG + lane / 2;
  const device bfloat* ub = UBp + (e * N + row0) * KG + lane / 2;
  const device bfloat* x = X + r * K + lane * 16;
  float xt[16];
  float ag[RPS], au[RPS];
  for (int row = 0; row < RPS; row++) { ag[row] = 0.0f; au[row] = 0.0f; }
  for (int k0 = 0; k0 < K; k0 += 512) {
    const float sum = load16(x, xt);
    for (int row = 0; row < RPS; row++) {
      ag[row] += qdot16(gw + row * KB, xt, float(gs[row * KG]), float(gb[row * KG]), sum);
      au[row] += qdot16(uw + row * KB, xt, float(us[row * KG]), float(ub[row * KG]), sum);
    }
    gw += 256; uw += 256; gs += 16; gb += 16; us += 16; ub += 16; x += 512;
  }
  for (int row = 0; row < RPS; row++) {
    const float gv = simd_sum(ag[row]), uv = simd_sum(au[row]);
    if (lane == 0) ACT[(r * SLOTS + slot) * N + row0 + row] = bfloat(bsilu(float(bfloat(gv))) * float(bfloat(uv)));
  }
"""

_GROUPED_DOWN = r"""
  // Threadgroup (b, u): MAXM simdgroups, simdgroup m takes the m-th row that picked distinct expert u (MAXU: the
  // shared expert) for model dims 8 b .. 8 b + 7: lane l reads 16-input chunks l and (l < NC - 32) 32 + l of the
  // row's activation; Y[row][slot][d] = bf16(sum), expert_down's sums. Weight rows read once for all members.
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  const int u = int(threadgroup_position_in_grid.z);
  const int R = rows[0];
  constexpr int SLOTS = TOPK + 1;
  const bool shared = u == MAXU;
  if (!shared && u >= UCOUNT[0]) return;
  const int member = shared ? (m < R ? m * 32 + TOPK : -1) : UMEM[u * MAXR + m];
  if (member < 0) return;
  const size_t e = shared ? 0 : size_t(UIDS[u]);
  const int r = member / 32, slot = member % 32;
  const int d0 = int(threadgroup_position_in_grid.y) * 8;
  constexpr int KB = NI / 2;
  constexpr int KG = NI / 32;
  constexpr int NC = NI / 16;
  const device uint32_t* DWp = shared ? SDW : DW;
  const device bfloat* DSp = shared ? SDS : DS;
  const device bfloat* DBp = shared ? SDB : DB;
  const bool second = int(lane) < NC - 32;
  const device bfloat* x = ACT + (r * SLOTS + slot) * NI;
  float xa[16], xb[16];
  const float sa = load16(x + lane * 16, xa);
  const float sb = second ? load16(x + (32 + lane) * 16, xb) : 0.0f;
  for (int row = 0; row < 8; row++) {
    const size_t at = e * D + d0 + row;
    const device uint8_t* w = (const device uint8_t*)DWp + at * KB;
    float acc = qdot16(w + lane * 8, xa, float(DSp[at * KG + lane / 2]), float(DBp[at * KG + lane / 2]), sa);
    if (second)
      acc += qdot16(w + (32 + lane) * 8, xb, float(DSp[at * KG + (32 + lane) / 2]), float(DBp[at * KG + (32 + lane) / 2]), sb);
    acc = simd_sum(acc);
    if (lane == 0) Y[(r * SLOTS + slot) * D + d0 + row] = bfloat(acc);
  }
"""

# The grouped MoE's combine, in the next hc_norm's write-back: expert_down's arithmetic on the slots' outputs
_BRANCH_GROUPED = r"""float routed = 0.0f;
      for (int k = 0; k < TOPK; k++) routed = fma(float(Y[(r * (TOPK + 1) + k) * D + d]), WTS[r * TOPK + k], routed);
      const float shared = float(bfloat(float(Y[(r * (TOPK + 1) + TOPK) * D + d]) * bsig(float(bfloat(LG[r * NL + NL - 1])))));
      const float branch = float(bfloat(float(bfloat(routed)) + shared));"""

_QMV_ROWS = r"""
  // Threadgroup b: R simdgroups, simdgroup r computes output rows RPS b .. RPS b + RPS - 1 for input row r with
  // MLX's one-row qmv_fast loop (its bits); the R simdgroups read the same weight rows, so memory serves them once.
  const uint lane = thread_index_in_simdgroup;
  const int r = int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 32;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 8;
  const device bfloat* sc = S + size_t(row0) * KG + lane / 2;
  const device bfloat* bi = B + size_t(row0) * KG + lane / 2;
  const device bfloat* x = X + r * K + lane * 16;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    const float sum = load16(x, xt);
    for (int j = 0; j < RPS; j++)
      acc[j] += qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 256; sc += 16; bi += 16; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[r * N + row0 + j] = bfloat(v);
  }
"""

_IDX_POOL = r"""
  // Threadgroup j (DI threads): block START + j's pooled indexer key: the mean of its 4 raw keys (fp32 in order,
  // bf16), RMSNorm with (1 + w) (fp32, bf16), RoPE (RD dims, non-interleaved halves) at the block's first position.
  const int d = int(thread_position_in_threadgroup.x);
  const int j = int(threadgroup_position_in_grid.y);
  const int b = START[0] + j;
  threadgroup float part[DI / 32];
  threadgroup float normed[DI];
  const device bfloat* src = RAW + size_t(4 * b) * DI + d;
  float m = float(src[0]);
  for (int k = 1; k < 4; k++) m += float(src[k * DI]);
  const float x = float(bfloat(m * 0.25f));
  float ss = simd_sum(x * x);
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int k = 0; k < DI / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(DI) + eps[0]);
  normed[d] = float(bfloat((x * inv) * W[d]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    const float freq = metal::exp2(-(float(i) / float(hr)) * LOG2BASE[0]);
    const float angle = float(4 * b) * freq;
    const float c = metal::fast::cos(angle), s = metal::fast::sin(angle);
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  OUT[j * DI + d] = bfloat(out);
"""

_IDX_SCORES = r"""
  // A simdgroup a block, rows in grid y: block b's score for row r is the sum over the HI indexer heads (in order)
  // of relu(q . pooled b) (fp32: a lane's DI / 32 dims in order, then simd_sum), over sqrt(DI). Only rows past TOP
  // complete blocks, and only their complete blocks, are scored (nothing else is read).
  const uint lane = thread_index_in_simdgroup;
  const int b = int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup);
  const int r = int(threadgroup_position_in_grid.y);
  const int complete = COMPLETE[r];
  if (complete <= TOP || b >= complete) return;
  constexpr int PER = DI / 32;
  const device bfloat* pb = POOLED + size_t(b) * DI + lane * PER;
  float p[PER];
  for (int i = 0; i < PER; i++) p[i] = float(pb[i]);
  float s = 0.0f;
  for (int h = 0; h < HI; h++) {
    const device bfloat* qh = Q + (r * HI + h) * DI + lane * PER;
    float dot = 0.0f;
    for (int i = 0; i < PER; i++) dot = fma(float(qh[i]), p[i], dot);
    s += metal::max(simd_sum(dot), 0.0f);
  }
  if (lane == 0) SC[size_t(r) * POOLED_shape[0] + b] = s / metal::precise::sqrt(float(DI));
"""

_IDX_SELECT = r"""
  // One threadgroup (1024 threads) a row past TOP complete blocks: its TOP best blocks by score (radix select over
  // order-preserving keys, 8 bits a pass; among scores equal to the cut, the lowest block ids), written as the keys
  // they cover (4 a block) in position order, then the row's tail keys [4 complete, ENDS).
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.x);
  const int nb = COMPLETE[r];
  if (nb <= TOP) return;
  const int ends = ENDS[r];
  const int stride = SC_shape[1];
  const device float* sc = SC + size_t(r) * stride;
  device int* keys = KEYS + size_t(r) * KW;
  threadgroup atomic_uint hist[256];
  threadgroup uint cut_t, need_t;
  threadgroup int tot_a[32], tot_e[32];
  uint prefix = 0u, mask = 0u, need = TOP;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int b = int(t); b < nb; b += 1024) {
      const uint k = tf_key(sc[b]);
      if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      uint above = 0u;
      int bin = 255;
      for (; bin > 0; bin--) {
        const uint n = atomic_load_explicit(&hist[bin], memory_order_relaxed);
        if (above + n >= need) break;
        above += n;
      }
      cut_t = prefix | (uint(bin) << shift);
      need_t = need - above;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix = cut_t;
    need = need_t;
    mask |= 255u << shift;
  }
  // `prefix` is the cut score's key: every block above it is taken, and the first `need` equal to it
  const int chunk = (nb + 1023) / 1024;
  const int lo = min(nb, int(t) * chunk), hi = min(nb, lo + chunk);
  int n_above = 0, n_equal = 0;
  for (int b = lo; b < hi; b++) {
    const uint k = tf_key(sc[b]);
    n_above += k > prefix ? 1 : 0;
    n_equal += k == prefix ? 1 : 0;
  }
  int pa = simd_prefix_exclusive_sum(n_above), pe = simd_prefix_exclusive_sum(n_equal);
  if (lane == 31) { tot_a[sg] = pa + n_above; tot_e[sg] = pe + n_equal; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const int a = tot_a[lane], e = tot_e[lane];
    tot_a[lane] = simd_prefix_exclusive_sum(a);
    tot_e[lane] = simd_prefix_exclusive_sum(e);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  pa += tot_a[sg];
  pe += tot_e[sg];
  int out = pa + min(pe, int(need));
  for (int b = lo; b < hi; b++) {
    const uint k = tf_key(sc[b]);
    bool take = k > prefix;
    if (k == prefix) { take = pe < int(need); pe++; }
    if (take) {
      for (int j = 0; j < 4; j++) keys[out * 4 + j] = 4 * b + j;
      out++;
    }
  }
  if (t == 0)
    for (int k = 4 * nb; k < ends; k++) keys[4 * TOP + (k - 4 * nb)] = k;
"""

_ATTN_PARTS = r"""
  // Threadgroup (h, r, p): query head h of row r over part p of the row's key list (SPARSE[r]: the NK[r] ids
  // IDS[r]; else keys 0 .. NK[r] - 1), entries [p n / P, (p + 1) n / P); 8 simdgroups, simdgroup g taking every
  // 8th entry from the part's start, a lane D / 32 dims. fp32: scores q . k with q pre-scaled, an online softmax
  // per simdgroup, the simdgroups combined in order into the part's (max, sum, output).
  constexpr int PER = D / 32;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int h = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const int part = int(threadgroup_position_in_grid.z);
  const int kvh = h / (H / KVH);
  const int n = NK[r];
  const int lo = int((long(part) * n) / P), hi = int((long(part + 1) * n) / P);
  const bool sparse = SPARSE[r] != 0;
  const size_t cap = size_t(Kc_shape[2]);
  const device bfloat* kb = Kc + size_t(kvh) * cap * D + lane * PER;
  const device bfloat* vb = Vc + size_t(kvh) * cap * D + lane * PER;
  const auto ids = IDS + size_t(r) * IDS_shape[1];      // device, or constant when MLX binds a small array so
  const device bfloat* qp = Q + (size_t(r) * H + h) * D + lane * PER;
  float q[PER], o[PER];
  for (int i = 0; i < PER; i++) { q[i] = SCALE[0] * float(qp[i]); o[i] = 0.0f; }
  float m = -INFINITY, l = 0.0f;
  for (int j = lo + int(g); j < hi; j += 8) {
    const size_t key = size_t(sparse ? ids[j] : j) * D;
    float sc = 0.0f;
    for (int i = 0; i < PER; i++) sc = fma(q[i], float(kb[key + i]), sc);
    sc = simd_sum(sc);
    const float mn = metal::max(m, sc);
    const float f = metal::exp(m - mn), e = metal::exp(sc - mn);
    l = fma(l, f, e);
    for (int i = 0; i < PER; i++) o[i] = fma(e, float(vb[key + i]), o[i] * f);
    m = mn;
  }
  threadgroup float ms[8], ls[8];
  threadgroup float tile[8][D];
  if (lane == 0) { ms[g] = m; ls[g] = l; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float top = -INFINITY;
  for (int k = 0; k < 8; k++) top = metal::max(top, ms[k]);
  const float mine = m == -INFINITY ? 0.0f : metal::exp(m - top);
  for (int i = 0; i < PER; i++) tile[g][lane * PER + i] = o[i] * mine;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const size_t at = (size_t(r) * H + h) * P + part;
  for (int d = int(thread_position_in_threadgroup.x); d < D; d += 256) {
    float acc = 0.0f;
    for (int k = 0; k < 8; k++) acc += tile[k][d];
    PO[at * D + d] = acc;
  }
  if (thread_position_in_threadgroup.x == 0) {
    float total = 0.0f;
    for (int k = 0; k < 8; k++) total += ms[k] == -INFINITY ? 0.0f : ls[k] * metal::exp(ms[k] - top);
    PM[at * 2] = top;
    PM[at * 2 + 1] = total;
  }
"""

_ATTN_MERGE = r"""
  // Threadgroup (h, r), a thread a dim: the P parts of head h of row r combined in part order.
  const int d = int(thread_position_in_threadgroup.x);
  const int h = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const size_t at = (size_t(r) * H + h) * P;
  float top = -INFINITY;
  for (int k = 0; k < P; k++) top = metal::max(top, PM[(at + k) * 2]);
  float total = 0.0f, acc = 0.0f;
  for (int k = 0; k < P; k++) {
    const float mk = PM[(at + k) * 2];
    const float w = mk == -INFINITY ? 0.0f : metal::exp(mk - top);
    total = fma(PM[(at + k) * 2 + 1], w, total);
    acc = fma(PO[(at + k) * D + d], w, acc);
  }
  OUT[(size_t(r) * H + h) * D + d] = bfloat(acc / total);
"""

_PLE_LOOKUP = r"""
  // Thread (d, h, r): dim d of head h of row r. Row id IDS[r][h] lies in one of 8 table groups (row starts GSTART);
  // its 4-bit value q, scale and bias give bf16(bf16(scale * q) + bias) (mx.dequantize on bf16 scales).
  const int d = int(thread_position_in_grid.x);
  const int h = int(thread_position_in_grid.y);
  const int r = int(thread_position_in_grid.z);
  const uint id = IDS[r * H + h];
  int g = 0;
  for (int j = 1; j < 8; j++) g += id >= GSTART[j] ? 1 : 0;
  const size_t row = size_t(id - GSTART[g]);
  const device uint32_t* W; const device bfloat* SC; const device bfloat* BI;
  switch (g) {
    case 0: W = W0; SC = S0; BI = B0; break;
    case 1: W = W1; SC = S1; BI = B1; break;
    case 2: W = W2; SC = S2; BI = B2; break;
    case 3: W = W3; SC = S3; BI = B3; break;
    case 4: W = W4; SC = S4; BI = B4; break;
    case 5: W = W5; SC = S5; BI = B5; break;
    case 6: W = W6; SC = S6; BI = B6; break;
    default: W = W7; SC = S7; BI = B7; break;
  }
  const uint word = W[row * (DIMS / 8) + d / 8];
  const bfloat q = bfloat(float((word >> (4 * (d % 8))) & 0xFu));
  const bfloat sc = SC[row * (DIMS / 32) + d / 32], bi = BI[row * (DIMS / 32) + d / 32];
  OUT[(r * H + h) * DIMS + d] = sc * q + bi;
"""

_EMBED_ROWS = r"""
  // Thread (d, r): dim d of token row r (quantized embedding, mx.dequantize's bf16(bf16(scale * q) + bias)),
  // written to each of the TILE copies of the row (the residual streams start as copies of the embedding).
  const int d = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  const size_t row = size_t(IDS[r]);
  const uint word = W[row * (DIMS / 8) + d / 8];
  const bfloat q = bfloat(float((word >> (4 * (d % 8))) & 0xFu));
  const bfloat v = SC[row * (DIMS / 32) + d / 32] * q + BI[row * (DIMS / 32) + d / 32];
  for (int t = 0; t < TILE; t++) OUT[(r * TILE + t) * DIMS + d] = v;
"""

_RMS_ROWS = r"""
  // One threadgroup of 1024 threads per row (per group of G features when G < W): bf16((x * rinv) * scale), the
  // sum of squares in fp32 (each thread's features in order, then the simdgroups in order).
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.y);
  const int grp = int(threadgroup_position_in_grid.x);
  const size_t base = size_t(r) * W + size_t(grp) * G;
  threadgroup float part[32];
  float ss = 0.0f;
  for (int i = int(t); i < G; i += 1024) { const float v = float(X[base + i]); ss = fma(v, v, ss); }
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int k = 0; k < 32; k++) total += part[k];
  const float rinv = metal::rsqrt(total / float(G) + eps[0]);
  for (int i = int(t); i < G; i += 1024)
    OUT[base + i] = bfloat((float(X[base + i]) * rinv) * SCALE[(grp * G + i) % SW]);
"""

_kernels: dict[str, Any] = {}


def _named(base: str, source: str) -> str:
    return f"{base}_{hashlib.sha256(source.encode()).hexdigest()[:16]}"


def _kernel(name: str, source: Any, inputs: list[str], outputs: list[str], header: str = _QDOT_HEADER) -> Any:
    """The compiled kernel for ``name`` (one source per name; ``source`` may be a callable building it, called
    only the first time). Looked up by name alone: hashing the source on every call cost ~2 ms of host time a
    Flash Next step (496 lookups, 2026-09-25); the hash still names the Metal function."""

    kernel = _kernels.get(name)
    if kernel is None:
        text = source() if callable(source) else source
        kernel = mx.fast.metal_kernel(name=_named(name, header + text), input_names=inputs, output_names=outputs,
                                      source=text, header=header)
        _kernels[name] = kernel
    return kernel


_counts: dict[int, mx.array] = {}
_consts: dict[Any, mx.array] = {}


def _rows(count: int) -> mx.array:
    value = _counts.get(count)
    if value is None:
        value = mx.array([count], dtype=mx.int32)
        _counts[count] = value
    return value


class QWeights:
    """A 4-bit group-32 quantized matrix [N, K] as three arrays (words, scales, biases)."""

    def __init__(self, weight: mx.array, scales: mx.array, biases: mx.array) -> None:
        self.weight, self.scales, self.biases = weight, scales, biases
        self.rows = int(weight.shape[0])
        self.cols = int(weight.shape[1]) * 8

    @classmethod
    def of(cls, *linears: Any) -> "QWeights":
        """One matrix from quantized linears' rows, stacked in order."""

        for linear in linears:
            if getattr(linear, "bits", 4) != 4 or getattr(linear, "group_size", 32) != 32:
                raise ValueError("expected 4-bit weights in groups of 32")
        if len(linears) == 1:
            one = linears[0]
            return cls(one.weight, one.scales, one.biases)
        return cls(mx.concatenate([l.weight for l in linears]), mx.concatenate([l.scales for l in linears]),
                   mx.concatenate([l.biases for l in linears]))


def gdn_step(projected: mx.array, conv_state: mx.array, ssm_state: mx.array | None, conv_weight: mx.array,
             a_log: mx.array, dt_bias: mx.array, norm_weight: mx.array, eps: mx.array, *, nk: int, nv: int,
             dk: int, dv: int) -> tuple[mx.array, mx.array, mx.array]:
    """One Gated DeltaNet layer after its input projection, for R consecutive rows.

    projected [R, C + NV*DV + 2 NV] ([qkv | z | b | a]), conv_state [TAPS - 1, C], ssm_state [NV, DV, DK] fp32
    (or None: zeros) -> (out [R, NV*DV] bf16, conv states after each row [R, TAPS - 1, C], ssm states after each
    row [R, NV, DV, DK] fp32).
    """

    rows = int(projected.shape[0])
    channels = 2 * nk * dk + nv * dv
    taps = int(conv_weight.shape[-1])
    if dk != 128 or dv != 128:
        raise ValueError("gdn_step: written for 128-dim heads")
    has_state = ssm_state is not None
    state = ssm_state if has_state else mx.zeros((1,), dtype=mx.float32)
    kernel = _kernel("q4_gdn_step", _GDN_STEP,
                     ["P", "CS", "SIN", "CW", "ALOG", "DT", "NW", "eps", "rows"], ["OUT", "CSO", "SO"])
    out, conv_rows, ssm_rows = kernel(
        inputs=[projected, conv_state, state, conv_weight, a_log, dt_bias, norm_weight, eps, _rows(rows)],
        template=[("NK", nk), ("NV", nv), ("DK", dk), ("DV", dv), ("TAPS", taps), ("HAS_STATE", int(has_state))],
        grid=(nv * 1024, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(rows, nv * dv), (rows, taps - 1, channels), (rows, nv, dv, dk)],
        output_dtypes=[mx.bfloat16, conv_state.dtype, mx.float32])
    return out, conv_rows, ssm_rows


def router(x: mx.array, gate_weight: mx.array, *, threads: int = 256, dtype: Any = mx.float32) -> mx.array:
    """x [R, D] @ gate_weight.T -> logits [R, E] (fp32 accumulation, stored as ``dtype``), each row with the
    same bits at any R. fp32 logits keep what bf16 rounds away: over 512 experts, bf16 logits tie at the
    top-10 cut in about a third of the layers of a token (2026-09-25), and the tie decided the expert."""

    rows, dims = x.shape
    experts = int(gate_weight.shape[0])
    out_t = "float" if dtype == mx.float32 else "bfloat"
    kernel = _kernel(f"q4_router_{out_t}", lambda: _ROUTER.replace("OUT_T", out_t), ["X", "GW", "rows"], ["OUT"])
    return kernel(inputs=[x, gate_weight, _rows(rows)],
                  template=[("D", dims), ("NE", experts), ("T", threads), ("MAXR", MAX_ROWS)],
                  grid=(-(-experts // (threads // 32)) * threads, 1, 1), threadgroup=(threads, 1, 1),
                  output_shapes=[(rows, experts)], output_dtypes=[dtype])[0]


def route(logits: mx.array, top_k: int, experts: int | None = None) -> tuple[mx.array, mx.array]:
    """Softmax over the first ``experts`` logits, top k by probability (ties: lower id), weights renormalized:
    ([R, k] ids, bf16)."""

    rows, width = logits.shape
    experts = width if experts is None else experts
    kernel = _kernel("q4_route", _ROUTE, ["L"], ["EXPERTS", "WEIGHTS"])
    return tuple(kernel(inputs=[logits], template=[("NE", experts), ("NL", width), ("TOPK", top_k)],
                        grid=(experts * rows, 1, 1), threadgroup=(experts, 1, 1),
                        output_shapes=[(rows, top_k), (rows, top_k)], output_dtypes=[mx.uint32, mx.bfloat16]))


def swiglu(gate: mx.array, up: mx.array, *, width: int | None = None, gate_at: int = 0, up_at: int = 0
           ) -> mx.array:
    """SiLU(gate) * up, [..., N] bf16. Two arrays of the same shape, or (width given) one stacked array twice
    with the gate and up runs at columns gate_at and up_at of its rows."""

    n = int(width if width is not None else gate.shape[-1])
    rows = int(gate.size // gate.shape[-1])
    kernel = _kernel("q4_swiglu", _SWIGLU, ["G", "U"], ["OUT"])
    lead = gate.shape[:-1]
    return kernel(inputs=[gate, up],
                  template=[("N", n), ("GSTRIDE", int(gate.shape[-1])), ("USTRIDE", int(up.shape[-1])),
                            ("GO", gate_at), ("UO", up_at)],
                  grid=(rows * n, 1, 1), threadgroup=(min(256, rows * n), 1, 1),
                  output_shapes=[(*lead, n)], output_dtypes=[mx.bfloat16])[0]


def attn_prep(projected: mx.array, positions: mx.array, q_norm: mx.array, k_norm: mx.array, index_norm: mx.array,
              eps: mx.array, *, q_heads: int, kv_heads: int, head_dim: int, index_heads: int, index_dim: int,
              rotary_dim: int, base: float) -> tuple[mx.array, mx.array, mx.array]:
    """Normed, rotated queries [R, NQ, HD], keys [R, NKV, HD] and indexer queries [R, NI, IHD] from the stacked
    projection [R, PW] ([q | gate] pairs, keys, values, indexer queries, indexer key); positions [R] int32; norms
    take (1 + w) scales (fp32)."""

    rows, width = projected.shape
    kernel = _kernel("q4_attn_prep", _ATTN_PREP, ["P", "POS", "QW", "KW", "IW", "eps", "LOG2BASE"],
                     ["Q", "Kout", "IQ"])
    return tuple(kernel(inputs=[projected, positions, q_norm, k_norm, index_norm, eps, _log2(base)],
                        template=[("NQ", q_heads), ("NKV", kv_heads), ("HD", head_dim), ("RD", rotary_dim),
                                  ("PW", width), ("NI", index_heads), ("IHD", index_dim)],
                        grid=(head_dim, q_heads + kv_heads + index_heads, rows), threadgroup=(head_dim, 1, 1),
                        output_shapes=[(rows, q_heads, head_dim), (rows, kv_heads, head_dim),
                                       (rows, index_heads, index_dim)],
                        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16]))


def _log2(base: float) -> mx.array:
    value = _consts.get(("log2", base))
    if value is None:
        value = _consts[("log2", base)] = mx.array([math.log2(base)], dtype=mx.float32)
    return value


def attn_gate(attended: mx.array, projected: mx.array, *, q_heads: int, head_dim: int) -> mx.array:
    """attended [R, NQ, HD] * sigmoid(gate) -> [R, NQ * HD] bf16."""

    rows = int(attended.shape[0])
    width = int(projected.shape[-1])
    kernel = _kernel("q4_attn_gate", _ATTN_GATE, ["A", "P"], ["OUT"])
    return kernel(inputs=[attended, projected], template=[("NQ", q_heads), ("HD", head_dim), ("PW", width)],
                  grid=(rows * q_heads * head_dim, 1, 1), threadgroup=(256, 1, 1),
                  output_shapes=[(rows, q_heads * head_dim)], output_dtypes=[mx.bfloat16])[0]


def hc_norm(h: mx.array, *, streams: int, write_back: str = "none", branch: tuple[mx.array, ...] = (),
            inject: mx.array | None = None) -> tuple[mx.array, mx.array]:
    """Streams h [R, S*D] with the previous block written back -> (h_new [R, S*D], partial sums of squares
    [R, D / 256, S] fp32 for hc_project)."""

    rows, wide = h.shape
    dims = wide // streams
    if dims % 256:
        raise ValueError("hc_norm: D must be a multiple of 256")
    names, inputs = ["H"], [h]
    if write_back == "none":
        make = lambda: _HC_NORM.replace("BRANCH", "").replace("WRITEBACK", "")          # noqa: E731
    elif write_back == "plain":
        if inject is None:
            raise ValueError("hc_norm: write-back needs the previous inject gates")
        make = lambda: _HC_NORM.replace("WRITEBACK", _WRITEBACK).replace("BRANCH", _BRANCH_PLAIN)  # noqa: E731
        names += ["INJ", "BR"]
        inputs += [inject, branch[0]]
    elif write_back == "grouped":
        if inject is None:
            raise ValueError("hc_norm: write-back needs the previous inject gates")
        y, weights, logits = branch
        make = lambda: _HC_NORM.replace("WRITEBACK", _WRITEBACK).replace("BRANCH", _BRANCH_GROUPED)  # noqa: E731
        names += ["INJ", "Y", "WTS", "LG"]
        inputs += [inject, y, weights, logits]
        extra = [("TOPK", int(weights.shape[-1])), ("NL", int(logits.shape[-1]))]
    else:
        raise ValueError(f"hc_norm: unknown write-back {write_back!r}")
    kernel = _kernel(f"q4_hc_norm_{write_back}", make, names, ["HN", "SSP"])
    template = [("S", streams), ("D", dims)] + (extra if write_back == "grouped" else [])
    return tuple(kernel(inputs=inputs, template=template, grid=(dims, rows, 1),
                        threadgroup=(256, 1, 1), output_shapes=[(rows, wide), (rows, dims // 256, streams)],
                        output_dtypes=[mx.bfloat16, mx.float32]))


def hc_project(h_new: mx.array, ssp: mx.array, down: QWeights, up: QWeights, norm_scale: mx.array, *,
               eps: mx.array, streams: int, low: int) -> tuple[mx.array, mx.array]:
    """The hyper-connection after hc_norm: (mixed [R, D], inject gates [R, S] (unused without inject rows))."""

    rows, wide = h_new.shape
    dims = wide // streams
    groups = wide // 32
    splits = groups // 32
    if groups % 32:
        raise ValueError("hc_project: S * D must be a multiple of 1024")
    down_kernel = _kernel("q4_hc_down_split", _HC_DOWN_SPLIT, ["HN", "SSP", "NW", "QW", "QS", "QB", "eps", "rows"],
                          ["PART"], header=_QDOT_HEADER + _RINV)
    part = down_kernel(inputs=[h_new, ssp, norm_scale, down.weight, down.scales, down.biases, eps, _rows(rows)],
                       template=[("S", streams), ("D", dims), ("ND", down.rows)],
                       grid=(-(-down.rows // 8) * 256, splits, rows), threadgroup=(256, 1, 1),
                       output_shapes=[(splits, rows, down.rows)], output_dtypes=[mx.float32])[0]
    per_row = low // 32
    threads = streams * 8 * per_row
    up_kernel = _kernel("q4_hc_up2", _HC_UP2, ["HN", "SSP", "PART", "QW", "QS", "QB", "NW", "eps", "rows"],
                        ["MIXED", "INJOUT"], header=_QDOT_HEADER + _RINV)
    mixed, inject = up_kernel(
        inputs=[h_new, ssp, part, up.weight, up.scales, up.biases, norm_scale, eps, _rows(rows)],
        template=[("S", streams), ("D", dims), ("LOW", low), ("ND", down.rows), ("KS", splits)],
        grid=(dims // 8 * threads, rows, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(rows, dims), (rows, streams)], output_dtypes=[mx.bfloat16, mx.bfloat16])
    return mixed, inject


def expert_gateup(x: mx.array, logits: mx.array, top_k: int, experts: int, gate: Any, up: Any,
                  shared: tuple[Any, Any] | None = None, *, rows_per_simdgroup: int = 4, simdgroups: int = 2
                  ) -> mx.array:
    """SiLU(gate_e x) * up_e x for each row's top-k experts by router logit (fp32 [R, NL], the first ``experts``
    are experts), + the shared expert as a last slot: x [R, K] -> ([R, k (+1), N] bf16, the routing for
    expert_down: picks [R, k] uint32, weights [R, k] fp32)."""

    rows, dims = x.shape
    width = int(gate.weight.shape[1])
    if dims % 512 or width % 8:
        raise ValueError("expert_gateup: needs K % 512 == 0 and N % 8 == 0")
    extra = 1 if shared is not None else 0
    sg, su = shared if shared is not None else (gate, up)
    kernel = _kernel("q4_expert_gateup", _EXPERT_GATEUP,
                     ["X", "LOGITS", "GW", "GS", "GB", "UW", "US", "UB", "SGW", "SGS", "SGB", "SUW", "SUS", "SUB"],
                     ["ACT", "PICK", "WTS"])
    return tuple(kernel(inputs=[x, logits, gate.weight, gate.scales, gate.biases, up.weight, up.scales, up.biases,
                                sg.weight, sg.scales, sg.biases, su.weight, su.scales, su.biases],
                        template=[("K", dims), ("N", width), ("TOPK", top_k), ("SHARED", extra), ("NE", experts),
                                  ("NL", int(logits.shape[-1])), ("RPS", rows_per_simdgroup), ("SG", simdgroups)],
                        grid=(32 * simdgroups, width // (rows_per_simdgroup * simdgroups), rows * (top_k + extra)),
                        threadgroup=(32 * simdgroups, 1, 1),
                        output_shapes=[(rows, top_k + extra, width), (rows, top_k), (rows, top_k)],
                        output_dtypes=[mx.bfloat16, mx.uint32, mx.float32]))


def expert_down(act: mx.array, picks: mx.array, weights: mx.array, logits: mx.array, top_k: int, experts: int,
                down: Any, shared: Any | None = None) -> mx.array:
    """sum_k w_k * bf16(down_e act_k) over each row's top-k experts (picks and weights from expert_gateup),
    + bf16(shared * sigmoid(the last logit)) when act has a shared slot: act [R, k (+1), NI] -> [R, D] bf16."""

    rows, slots, width = act.shape
    extra = slots - top_k
    dims = int(down.weight.shape[1])
    if width % 16 or width // 16 > 64 or dims % 8:
        raise ValueError("expert_down: needs NI % 16 == 0, NI <= 1024 and D % 8 == 0")
    sd = shared if shared is not None else down
    kernel = _kernel("q4_expert_down", _EXPERT_DOWN,
                     ["ACT", "PICK", "WTS", "LOGITS", "DW", "DS", "DB", "SDW", "SDS", "SDB"], ["ROUTED"])
    return kernel(inputs=[act, picks, weights, logits, down.weight, down.scales, down.biases, sd.weight, sd.scales,
                          sd.biases],
                  template=[("NI", width), ("D", dims), ("TOPK", top_k), ("SHARED", extra), ("NE", experts),
                            ("NL", int(logits.shape[-1]))],
                  grid=(32 * slots, dims // 8, rows), threadgroup=(32 * slots, 1, 1),
                  output_shapes=[(rows, dims)], output_dtypes=[mx.bfloat16])[0]

def expert_down_y(act: mx.array, picks: mx.array, down: Any, shared: Any, *, simdgroups: int = 2) -> mx.array:
    """bf16(down_e act) for every (row, slot) (slot k < k_top: the row's k-th expert from ``picks``; the last: the
    shared expert): act [R, k + 1, NI] -> [R, k + 1, D]; combined by hc_norm's "grouped" write-back."""

    rows, slots, width = act.shape
    top_k = int(picks.shape[-1])
    dims = int(down.weight.shape[1])
    kernel = _kernel("q4_expert_down_y", _EXPERT_DOWN_Y,
                     ["ACT", "PICK", "DW", "DS", "DB", "SDW", "SDS", "SDB", "rows"], ["Y"])
    return kernel(inputs=[act, picks, down.weight, down.scales, down.biases, shared.weight, shared.scales,
                          shared.biases, _rows(rows)],
                  template=[("NI", width), ("D", dims), ("TOPK", top_k), ("SG", simdgroups)],
                  grid=(32 * simdgroups, dims // 8, -(-rows * slots // simdgroups)),
                  threadgroup=(32 * simdgroups, 1, 1),
                  output_shapes=[(rows, slots, dims)], output_dtypes=[mx.bfloat16])[0]


def qmv(x: mx.array, weights: Any, *, rows_per_simdgroup: int = 4, simdgroups: int = 2) -> mx.array:
    """x [R, K] @ W.T for a 4-bit group-32 matrix (a quantized linear or QWeights) -> [R, N] bf16, R <= 8;
    a row's bits do not depend on R."""

    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    rows, dims = x2.shape
    n = int(weights.weight.shape[0])
    block = rows_per_simdgroup * simdgroups
    if dims % 512 or n % block or rows > 8:
        raise ValueError(f"qmv: needs K % 512 == 0, N % {block} == 0 and at most 8 rows (K {dims}, N {n}, R {rows})")
    kernel = _kernel("q4_qmv", _QMV, ["X", "W", "S", "B"], ["OUT"])
    out = kernel(inputs=[x2, weights.weight, weights.scales, weights.biases],
                 template=[("K", dims), ("N", n), ("R", rows), ("RPS", rows_per_simdgroup), ("SG", simdgroups)],
                 grid=(32 * simdgroups, n // block, 1), threadgroup=(32 * simdgroups, 1, 1),
                 output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(*shape[:-1], n)


def expert_group(logits: mx.array, top_k: int, experts: int, *, max_rows: int = 16
                 ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Each row's top-k experts by logit and their weights, and the distinct experts with their (row, slot)
    members: (picks [R, k] uint32, weights [R, k] fp32, count [1], ids [R*k], members [R*k, max_rows])."""

    rows, width = logits.shape
    if rows > max_rows or experts % 32:
        raise ValueError(f"expert_group: at most {max_rows} rows, experts a multiple of 32")
    most = rows * top_k
    kernel = _kernel("q4_expert_group", _EXPERT_GROUP, ["LOGITS", "rows"], ["PICK", "WTS", "UCOUNT", "UIDS", "UMEM"])
    return tuple(kernel(inputs=[logits, _rows(rows)],
                        template=[("NE", experts), ("NL", width), ("TOPK", top_k), ("MAXR", max_rows)],
                        grid=(experts, 1, 1), threadgroup=(experts, 1, 1),
                        output_shapes=[(rows, top_k), (rows, top_k), (1,), (most,), (most, max_rows)],
                        output_dtypes=[mx.uint32, mx.float32, mx.int32, mx.int32, mx.int32]))


def grouped_gateup(x: mx.array, group: tuple[mx.array, ...], gate: Any, up: Any, shared: tuple[Any, Any], *,
                   max_rows: int = 16, rows_per_simdgroup: int = 4) -> mx.array:
    """SiLU(gate x) * up x for every (row, slot) of the group (slot k: the row's k-th expert; the last: the
    shared expert), a threadgroup per distinct expert with a simdgroup per row that picked it: x [R, K] ->
    [R, k + 1, N] bf16, each row expert_gateup's bits."""

    rows, dims = x.shape
    picks, _, count, ids, members = group
    top_k = int(picks.shape[-1])
    width = int(gate.weight.shape[1])
    most = int(ids.shape[0])
    sg, su = shared
    kernel = _kernel("q4_grouped_gateup", _GROUPED_GATEUP,
                     ["X", "UCOUNT", "UIDS", "UMEM", "GW", "GS", "GB", "UW", "US", "UB", "SGW", "SGS", "SGB",
                      "SUW", "SUS", "SUB", "rows"], ["ACT"])
    return kernel(inputs=[x, count, ids, members, gate.weight, gate.scales, gate.biases, up.weight, up.scales,
                          up.biases, sg.weight, sg.scales, sg.biases, su.weight, su.scales, su.biases, _rows(rows)],
                  template=[("K", dims), ("N", width), ("TOPK", top_k), ("MAXU", most), ("MAXR", max_rows),
                            ("RPS", rows_per_simdgroup)],
                  grid=(32 * rows, width // rows_per_simdgroup, most + 1), threadgroup=(32 * rows, 1, 1),
                  output_shapes=[(rows, top_k + 1, width)], output_dtypes=[mx.bfloat16])[0]


def grouped_down(act: mx.array, group: tuple[mx.array, ...], down: Any, shared: Any, *, max_rows: int = 16
                 ) -> mx.array:
    """bf16(down_e act) for every (row, slot) of the group, a threadgroup per distinct expert with a simdgroup per
    row that picked it: [R, k + 1, NI] -> [R, k + 1, D]; combined by hc_norm's "grouped" write-back."""

    rows, slots, width = act.shape
    picks, _, count, ids, members = group
    top_k = int(picks.shape[-1])
    dims = int(down.weight.shape[1])
    most = int(ids.shape[0])
    kernel = _kernel("q4_grouped_down", _GROUPED_DOWN,
                     ["ACT", "UCOUNT", "UIDS", "UMEM", "DW", "DS", "DB", "SDW", "SDS", "SDB", "rows"], ["Y"])
    return kernel(inputs=[act, count, ids, members, down.weight, down.scales, down.biases, shared.weight,
                          shared.scales, shared.biases, _rows(rows)],
                  template=[("NI", width), ("D", dims), ("TOPK", top_k), ("MAXU", most), ("MAXR", max_rows)],
                  grid=(32 * rows, dims // 8, most + 1), threadgroup=(32 * rows, 1, 1),
                  output_shapes=[(rows, slots, dims)], output_dtypes=[mx.bfloat16])[0]


def qmv_rows(x: mx.array, weights: Any, *, rows_per_simdgroup: int = 4) -> mx.array:
    """qmv with a simdgroup per input row (R <= 32): each row MLX's one-row bits; the rows share weight reads."""

    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    rows, dims = x2.shape
    n = int(weights.weight.shape[0])
    if dims % 512 or n % rows_per_simdgroup or rows > 32:
        raise ValueError(f"qmv_rows: needs K % 512 == 0, N % {rows_per_simdgroup} == 0, at most 32 rows")
    kernel = _kernel("q4_qmv_rows", _QMV_ROWS, ["X", "W", "S", "B"], ["OUT"])
    out = kernel(inputs=[x2, weights.weight, weights.scales, weights.biases],
                 template=[("K", dims), ("N", n), ("RPS", rows_per_simdgroup)],
                 grid=(32 * rows, n // rows_per_simdgroup, 1), threadgroup=(32 * rows, 1, 1),
                 output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(*shape[:-1], n)



_SELECT_HEADER = r"""
inline uint tf_key(float v) { uint b = as_type<uint>(v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
"""


def index_pool(raw: mx.array, start: int, stop: int, norm: mx.array, eps: mx.array, *, rotary_dim: int,
               base: float) -> mx.array:
    """Pooled indexer keys [stop - start, DI] of blocks [start, stop) from raw keys [keys, DI] (4 keys a block)."""

    dims = int(raw.shape[-1])
    kernel = _kernel("q4_idx_pool", _IDX_POOL, ["RAW", "START", "W", "eps", "LOG2BASE"], ["OUT"])
    return kernel(inputs=[raw, mx.array([start], dtype=mx.int32), norm, eps, _log2(base)],
                  template=[("DI", dims), ("RD", rotary_dim)],
                  grid=(dims, stop - start, 1), threadgroup=(dims, 1, 1),
                  output_shapes=[(stop - start, dims)], output_dtypes=[mx.bfloat16])[0]


def index_select(q: mx.array, pooled: mx.array, complete: list[int], ends: list[int], *, top: int) -> mx.array:
    """Key ids [R, 4 top + 3] of each row past ``top`` complete blocks: its best ``top`` blocks' keys in position
    order, then its tail keys [4 complete, ends) (a row reads 4 top + ends - 4 complete of them). q [R, HI, DI]
    (normed, rotated indexer queries), pooled [NB, DI] (the complete blocks, NB >= every row's complete count)."""

    rows, heads, dims = q.shape
    nb = int(pooled.shape[0])
    counts = mx.array([int(c) for c in complete], dtype=mx.int32)
    score = _kernel("q4_idx_scores", _IDX_SCORES, ["Q", "POOLED", "COMPLETE"], ["SC"])
    sc = score(inputs=[q, pooled, counts], template=[("HI", heads), ("DI", dims), ("TOP", top)],
               grid=(-(-nb // 8) * 256, rows, 1), threadgroup=(256, 1, 1),
               output_shapes=[(rows, nb)], output_dtypes=[mx.float32])[0]
    width = 4 * top + 3
    select = _kernel("q4_idx_select", _IDX_SELECT, ["SC", "COMPLETE", "ENDS"], ["KEYS"], header=_SELECT_HEADER)
    return select(inputs=[sc, counts, mx.array([int(e) for e in ends], dtype=mx.int32)],
                  template=[("TOP", top), ("KW", width)],
                  grid=(1024 * rows, 1, 1), threadgroup=(1024, 1, 1),
                  output_shapes=[(rows, width)], output_dtypes=[mx.int32])[0]


def attention_rows(q: mx.array, keys: mx.array, values: mx.array, counts: list[int], ids: mx.array | None,
                   sparse: list[bool], scale: float, *, parts: int = 16) -> mx.array:
    """Attention of each row's query heads q [R, H, D] over its own keys: a sparse row the first counts[r] ids of
    ids[r], a dense row keys 0 .. counts[r] - 1, from the cache buffers keys / values [1, KVH, cap, D]. Each
    row's list is cut into ``parts`` pieces by its own length (threadgroups in parallel), then merged in order,
    so a row's result does not depend on the other rows: [R, H, D] bf16."""

    rows, heads, dims = q.shape
    kv_heads = int(keys.shape[1])
    if ids is None:
        ids = _consts.get(("no ids", rows))
        if ids is None:
            ids = _consts[("no ids", rows)] = mx.zeros((rows, 1), dtype=mx.int32)
    scale_arr = _consts.get(("scale", scale))
    if scale_arr is None:
        scale_arr = _consts[("scale", scale)] = mx.array([scale], dtype=mx.float32)
    first = _kernel("q4_attn_parts", _ATTN_PARTS, ["Q", "Kc", "Vc", "IDS", "NK", "SPARSE", "SCALE"], ["PO", "PM"])
    po, pm = first(inputs=[q, keys, values, ids, mx.array([int(c) for c in counts], dtype=mx.int32),
                           mx.array([int(bool(x)) for x in sparse], dtype=mx.int32), scale_arr],
                   template=[("H", heads), ("KVH", kv_heads), ("D", dims), ("P", parts)],
                   grid=(256 * heads, rows, parts), threadgroup=(256, 1, 1),
                   output_shapes=[(rows, heads, parts, dims), (rows, heads, parts, 2)],
                   output_dtypes=[mx.float32, mx.float32])
    merge = _kernel("q4_attn_merge", _ATTN_MERGE, ["PO", "PM"], ["OUT"])
    return merge(inputs=[po, pm], template=[("H", heads), ("D", dims), ("P", parts)],
                 grid=(dims, heads, rows), threadgroup=(dims, 1, 1),
                 output_shapes=[(rows, heads, dims)], output_dtypes=[mx.bfloat16])[0]


class PleTables:
    """An n-gram embedding's shards as 8 concatenated groups (the shards keep views into them), for ple_lookup."""

    groups = 8

    def __init__(self, emb: Any) -> None:
        shards = emb.shards
        per = -(-len(shards) // self.groups)
        self.weights, self.scales, self.biases, starts = [], [], [], [0]
        for g in range(self.groups):
            part = shards[g * per:(g + 1) * per]
            w = mx.concatenate([sh.weight for sh in part])
            sc = mx.concatenate([sh.scales for sh in part])
            bi = mx.concatenate([sh.biases for sh in part])
            mx.eval(w, sc, bi)
            at = 0
            for sh in part:
                n = int(sh.weight.shape[0])
                sh.weight, sh.scales, sh.biases = w[at:at + n], sc[at:at + n], bi[at:at + n]
                mx.eval(sh.weight, sh.scales, sh.biases)
                at += n
            self.weights.append(w)
            self.scales.append(sc)
            self.biases.append(bi)
            starts.append(starts[-1] + at)
            mx.clear_cache()
        self.starts = mx.array(starts[:-1], dtype=mx.uint32)
        self.dims = int(emb.dims)
        mx.eval(self.starts)


def ple_lookup(ids: Any, tables: PleTables) -> mx.array:
    """Dequantized rows [R, H * DIMS] bf16 for global n-gram row ids [R, H] (the shards' concatenated order)."""

    import numpy as np

    ids = np.asarray(ids).reshape(-1, np.asarray(ids).shape[-1])
    rows, heads = ids.shape
    names = ["IDS", "GSTART"] + [f"{k}{g}" for g in range(8) for k in ("W", "S", "B")]
    kernel = _kernel("q4_ple_lookup", _PLE_LOOKUP, names, ["OUT"])
    arrays = [mx.array(ids.astype(np.uint32)), tables.starts]
    for g in range(8):
        arrays += [tables.weights[g], tables.scales[g], tables.biases[g]]
    return kernel(inputs=arrays, template=[("H", heads), ("DIMS", tables.dims)],
                  grid=(tables.dims, heads, rows), threadgroup=(tables.dims, 1, 1),
                  output_shapes=[(rows, heads * tables.dims)], output_dtypes=[mx.bfloat16])[0]


def embed_rows(ids: Any, embedding: Any, *, tile: int = 1) -> mx.array:
    """Token rows of a 4-bit quantized embedding (mx.dequantize's bits), each written ``tile`` times: ids [R] ->
    [R, tile * DIMS] bf16."""

    import numpy as np

    if not isinstance(ids, mx.array):
        ids = mx.array(np.asarray(ids, dtype=np.uint32).reshape(-1))
    rows = int(ids.size)
    dims = int(embedding.weight.shape[1]) * 8
    kernel = _kernel("q4_embed_rows", _EMBED_ROWS, ["IDS", "W", "SC", "BI"], ["OUT"])
    return kernel(inputs=[ids.reshape(-1).astype(mx.uint32), embedding.weight, embedding.scales, embedding.biases],
                  template=[("DIMS", dims), ("TILE", tile)], grid=(dims, rows, 1), threadgroup=(min(dims, 256), 1, 1),
                  output_shapes=[(rows, tile * dims)], output_dtypes=[mx.bfloat16])[0]


def rms_norm_rows(x: mx.array, scale: mx.array, eps: mx.array, *, group: int | None = None) -> mx.array:
    """CenteredRMSNorm's (1 + w) RMSNorm over each row (or each run of ``group`` features) of x [R, W] -> bf16."""

    rows, width = x.shape
    g = int(group or width)
    kernel = _kernel("q4_rms_rows", _RMS_ROWS, ["X", "SCALE", "eps"], ["OUT"])
    return kernel(inputs=[x, scale, eps], template=[("W", width), ("G", g), ("SW", int(scale.shape[-1]))],
                  grid=(1024 * (width // g), rows, 1), threadgroup=(1024, 1, 1),
                  output_shapes=[(rows, width)], output_dtypes=[mx.bfloat16])[0]
