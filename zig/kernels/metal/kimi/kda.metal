// Kimi Delta Attention for a round: per (segment, head) the conv, gates and delta rule run row by row in order.
#include "kimi_common.h"

constant constexpr uint D = 128;
constant constexpr uint TAPS = 4;

struct KdaArgs {
  uint heads, head0, proj_stride, gate_stride, out_stride, log_rows;
  float lower_bound, eps;
};

// A segment: one stream's rows of this round; kept rows of its last logged window are replayed first.
struct KdaSeg {
  uint first, rows, kept, flags, slot;
};

constant constexpr uint SEG_COMMIT = 1;

// One layer's state arenas, slot-major: S [heads][D][D] fp32, conv window [3][3W] bf16, the logged window's rows.
struct KdaState {
  device float* S;
  device bfloat* conv;
  device float* log_a;
  device float* log_k;
  device float* log_u;
  device bfloat* log_x;
};

struct KdaWeights {
  device const float* conv_q;
  device const float* conv_k;
  device const float* conv_v;
  device const float* A_log;
  device const float* dt_bias;
  device const float* o_norm;
};

// The delta rule on one thread's 16 values x 4 keys: decay, then S += k u (u = beta (v - S^T k)).
inline void k3_kda_update(thread float (&st)[16][4], thread const float* a, thread const float* kn,
                          thread const float* u) {
  for (int j = 0; j < 16; ++j)
    for (int i = 0; i < 4; ++i) st[j][i] = fma(kn[i], u[j], st[j][i] * a[i]);
}

kernel void k3_kda(device const bfloat* proj [[buffer(0)]], device const bfloat* fgate [[buffer(1)]],
                   device const bfloat* braw [[buffer(2)]], device const bfloat* g2 [[buffer(3)]],
                   device bfloat* y [[buffer(4)]], device const KdaSeg* segs [[buffer(5)]],
                   constant KdaState& base [[buffer(6)]], constant KdaWeights& w [[buffer(7)]],
                   constant KdaArgs& a [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                   uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tq[D], tk[D], tv[D], ta[D], part[8];
  threadgroup float beta;
  const uint hl = tg.x, h = a.head0 + hl, W = a.heads * D;
  const KdaSeg seg = segs[tg.y];
  const uint lr = a.log_rows, sz = a.heads * D;
  const KdaState s = {base.S + seg.slot * sz * D, base.conv + seg.slot * 9 * W, base.log_a + seg.slot * lr * sz,
                      base.log_k + seg.slot * lr * sz, base.log_u + seg.slot * lr * sz, base.log_x + seg.slot * lr * 3 * W};
  device float* S = s.S + hl * D * D;
  float st[16][4];
  for (int j = 0; j < 16; ++j)
    for (int i = 0; i < 4; ++i) st[j][i] = S[(16 * sg + j) * D + 4 * lane + i];
  uint ch[2];
  float cw[2][TAPS], win[2][TAPS - 1];
  const uint nch = t < 128 ? 2 : 1;
  for (uint c = 0; c < nch; ++c) {
    const uint lc = t + 256 * c, part3 = lc / D, d = lc % D;
    ch[c] = part3 * W + hl * D + d;
    device const float* cwp = (part3 == 0 ? w.conv_q : part3 == 1 ? w.conv_k : w.conv_v) + (h * D + d) * TAPS;
    for (uint j = 0; j < TAPS; ++j) cw[c][j] = cwp[j];
    for (uint j = 0; j < TAPS - 1; ++j) win[c][j] = float(s.conv[j * 3 * W + ch[c]]);
  }
  for (uint r = 0; r < seg.kept; ++r) {
    float ka[4], kk[4], ku[16];
    for (int i = 0; i < 4; ++i) {
      ka[i] = s.log_a[(r * a.heads + hl) * D + 4 * lane + i];
      kk[i] = s.log_k[(r * a.heads + hl) * D + 4 * lane + i];
    }
    for (int j = 0; j < 16; ++j) ku[j] = s.log_u[(r * a.heads + hl) * D + 16 * sg + j];
    k3_kda_update(st, ka, kk, ku);
    for (uint c = 0; c < nch; ++c) {
      win[c][0] = win[c][1], win[c][1] = win[c][2];
      win[c][2] = float(s.log_x[r * 3 * W + ch[c]]);
    }
  }
  const float A = precise::exp(w.A_log[h]);
  for (uint r = 0; r <= seg.rows; ++r) {
    if ((r == 0 && seg.kept > 0) || (r == seg.rows && (seg.flags & SEG_COMMIT) != 0)) {
      for (int j = 0; j < 16; ++j)
        for (int i = 0; i < 4; ++i) S[(16 * sg + j) * D + 4 * lane + i] = st[j][i];
      for (uint c = 0; c < nch; ++c)
        for (uint j = 0; j < TAPS - 1; ++j) s.conv[j * 3 * W + ch[c]] = bfloat(win[c][j]);
    }
    if (r == seg.rows) break;
    const uint row = seg.first + r;
    const bool log = (seg.flags & SEG_COMMIT) == 0;
    for (uint c = 0; c < nch; ++c) {
      const float x = float(proj[row * a.proj_stride + ch[c]]);
      float acc = win[c][0] * cw[c][0];
      acc = fma(win[c][1], cw[c][1], acc);
      acc = fma(win[c][2], cw[c][2], acc);
      acc = fma(x, cw[c][3], acc);
      const float act = float(bfloat(acc * k3_sigmoid(acc)));
      const uint lc = t + 256 * c;
      (lc < D ? tq : lc < 2 * D ? tk : tv)[lc % D] = act;
      win[c][0] = win[c][1], win[c][1] = win[c][2], win[c][2] = x;
      if (log) s.log_x[r * 3 * W + ch[c]] = bfloat(x);
    }
    if (t < D) {
      const float g = float(fgate[row * a.gate_stride + hl * D + t]) + w.dt_bias[h * D + t];
      ta[t] = precise::exp(a.lower_bound * k3_sigmoid(A * g));
    }
    if (t == 0) beta = k3_sigmoid(float(braw[row * a.heads + hl]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qn[4], kn[4], av[4], ss_q = 0.0f, ss_k = 0.0f;
    for (int i = 0; i < 4; ++i) {
      qn[i] = tq[4 * lane + i], kn[i] = tk[4 * lane + i], av[i] = ta[4 * lane + i];
      ss_q = fma(qn[i], qn[i], ss_q);
      ss_k = fma(kn[i], kn[i], ss_k);
    }
    const float rq = precise::sqrt(simd_sum(ss_q) + 1e-6f), rk = precise::sqrt(simd_sum(ss_k) + 1e-6f);
    for (int i = 0; i < 4; ++i) qn[i] = qn[i] / rq * 0.08838834764831845f, kn[i] = kn[i] / rk;
    float u[16], o[16];
    for (int j = 0; j < 16; ++j) {
      float p = 0.0f;
      for (int i = 0; i < 4; ++i) p = fma(st[j][i] * av[i], kn[i], p);
      u[j] = (tv[16 * sg + j] - simd_sum(p)) * beta;
    }
    k3_kda_update(st, av, kn, u);
    for (int j = 0; j < 16; ++j) {
      float p = 0.0f;
      for (int i = 0; i < 4; ++i) p = fma(st[j][i], qn[i], p);
      o[j] = float(bfloat(simd_sum(p)));
    }
    if (log) {
      if (sg == 0)
        for (int i = 0; i < 4; ++i) {
          s.log_a[(r * a.heads + hl) * D + 4 * lane + i] = av[i];
          s.log_k[(r * a.heads + hl) * D + 4 * lane + i] = kn[i];
        }
      if (lane == 0)
        for (int j = 0; j < 16; ++j) s.log_u[(r * a.heads + hl) * D + 16 * sg + j] = u[j];
    }
    float ss = 0.0f;
    for (int j = 0; j < 16; ++j) ss = fma(o[j], o[j], ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.0f;
    for (uint g = 0; g < 8; ++g) var += part[g];
    const float rstd = 1.0f / precise::sqrt(var / float(D) + a.eps);
    if (lane < 16) {
      const uint v = 16 * sg + lane;
      const float gate = k3_sigmoid(float(g2[row * a.gate_stride + hl * D + v]));
      y[row * a.out_stride + hl * D + v] = bfloat(o[lane] * rstd * w.o_norm[v] * gate);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
