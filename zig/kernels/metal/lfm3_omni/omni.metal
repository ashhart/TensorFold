// LFM3-Omni's Metal ops for its four Liquid stacks, rounding to bf16 wherever the bf16 reference rounds.
#include "om_common.h"

// ---- norms

struct NormArgs {
  int D, ld, ldy;
  float eps;
  int mode, ldw, has_r, has_pw;
};

// RMSNorm a group a row, X = bf16(X + R) first (R normed by PW if given); mode 0 rounds x * rsqrt, mode 1 once.
template <uint NT>
inline void rmsnorm_body(device bf* X, device const bf* R, device const bf* Wb, device bf* Y, device const bf* PW, constant NormArgs& a,
                         uint row, uint tid, threadgroup float* red) {
  device const bf* W = Wb + (long)row * a.ldw;
  device bf* x = X + (long)row * a.ld;
  device bf* y = Y + (long)row * a.ldy;
  device const bf* r = R + (long)row * a.ld;
  float inv_r = 0.0f;
  if (a.has_r && a.has_pw) {
    float sr = 0.0f;
    for (int i = tid; i < a.D; i += NT) sr += f(r[i]) * f(r[i]);
    inv_r = precise::rsqrt(group_sum<NT>(sr, red, tid) / float(a.D) + a.eps);
  }
  float ss = 0.0f;
  for (int i = tid; i < a.D; i += NT) {
    float v = f(x[i]);
    if (a.has_r) {
      const float add = a.has_pw ? rb(f(PW[i]) * rb(f(r[i]) * inv_r)) : f(r[i]);
      v = rb(v + add);
      x[i] = b(v);
    }
    ss += v * v;
  }
  const float inv = precise::rsqrt(group_sum<NT>(ss, red, tid) / float(a.D) + a.eps);
  for (int i = tid; i < a.D; i += NT) {
    const float v = f(x[i]) * inv;
    y[i] = a.mode == 0 ? b(f(W[i]) * rb(v)) : b(v * f(W[i]));
  }
}

#define OM_RMSNORM(NAME, NT)                                                                                                       \
  [[max_total_threads_per_threadgroup(NT)]] kernel void NAME(                                                                       \
      device bf* X [[buffer(0)]], device const bf* R [[buffer(1)]], device const bf* Wb [[buffer(2)]], device bf* Y [[buffer(3)]],   \
      device const bf* PW [[buffer(4)]], constant NormArgs& a [[buffer(5)]], uint row [[threadgroup_position_in_grid]],             \
      uint tid [[thread_index_in_threadgroup]]) {                                                                                  \
    threadgroup float red[NT / 32];                                                                                                \
    rmsnorm_body<NT>(X, R, Wb, Y, PW, a, row, tid, red);                                                                           \
  }
OM_RMSNORM(om_rmsnorm, 256)
OM_RMSNORM(om_rmsnorm_1k, 1024)

struct HeadNormArgs {
  int T, H, hd, ld;
  float eps;
};

// Per-head RMSNorm (mode 0 rounding) in place: one simdgroup a (row, head).
kernel void om_head_norm(device bf* X [[buffer(0)]], device const bf* W [[buffer(1)]], constant HeadNormArgs& a [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
  const int gw = gid >> 5, lane = gid & 31;
  if (gw >= a.T * a.H) return;
  device bf* x = X + (long)(gw / a.H) * a.ld + (gw % a.H) * a.hd;
  float ss = 0.0f;
  for (int i = lane; i < a.hd; i += 32) ss += f(x[i]) * f(x[i]);
  ss = simd_sum(ss);
  const float inv = precise::rsqrt(ss / float(a.hd) + a.eps);
  for (int i = lane; i < a.hd; i += 32) x[i] = b(f(W[i]) * rb(f(x[i]) * inv));
}

struct LayerNormArgs {
  int D;
  float eps;
};

// LayerNorm with bias over rows of D in fp32, one rounding (a group a row).
[[max_total_threads_per_threadgroup(256)]] kernel void om_layernorm(device const bf* X [[buffer(0)]], device const bf* W [[buffer(1)]],
                                                                 device const bf* Bb [[buffer(2)]], device bf* Y [[buffer(3)]],
                                                                 constant LayerNormArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                                                                 uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float red[8];
  device const bf* x = X + (long)row * a.D;
  float s = 0.0f;
  for (int i = tid; i < a.D; i += 256) s += f(x[i]);
  const float mean = group_sum<256>(s, red, tid) / float(a.D);
  float v = 0.0f;
  for (int i = tid; i < a.D; i += 256) {
    const float d = f(x[i]) - mean;
    v += d * d;
  }
  const float rstd = precise::rsqrt(group_sum<256>(v, red, tid) / float(a.D) + a.eps);
  for (int i = tid; i < a.D; i += 256) Y[(long)row * a.D + i] = b((f(x[i]) - mean) * rstd * f(W[i]) + f(Bb[i]));
}

// ---- rotary, caches, positions

struct RopeArgs {
  int T, H, hd, ld, pos0, mode;
};

// Rotate-half RoPE in place at pos0 + t; mode 0 fp32 cos/sin, 1 bf16 cos/sin, 2 every product and sum rounded.
kernel void om_rope(device bf* X [[buffer(0)]], device const float* INV [[buffer(1)]], constant RopeArgs& a [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
  const int half_ = a.hd >> 1;
  const long i = gid;
  if (i >= (long)a.T * a.H * half_) return;
  const int p = int(i % half_), h = int((i / half_) % a.H), t = int(i / ((long)half_ * a.H));
  device bf* x = X + (long)t * a.ld + h * a.hd;
  const float fr = float(a.pos0 + t) * INV[p];
  float c = precise::cos(fr), s = precise::sin(fr);
  if (a.mode != 0) {
    c = rb(c);
    s = rb(s);
  }
  const float xa = f(x[p]), z = f(x[p + half_]);
  if (a.mode == 2) {
    x[p] = b(rb(xa * c) + rb(-z * s));
    x[p + half_] = b(rb(z * c) + rb(xa * s));
  } else {
    x[p] = b(xa * c - z * s);
    x[p + half_] = b(z * c + xa * s);
  }
}

struct CopyRowsArgs {
  int T, n, lds, ldd;
};

kernel void om_copy_rows(device const bf* S [[buffer(0)]], device bf* D [[buffer(1)]], constant CopyRowsArgs& a [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.T * a.n) return;
  const int t = int(i / a.n), c = int(i % a.n);
  D[(long)t * a.ldd + c] = S[(long)t * a.lds + c];
}

// n 32-bit words from the argument bytes (small host writes in order with the dispatches).
kernel void om_set_words(device uint* D [[buffer(0)]], constant uint* V [[buffer(1)]], constant uint& n [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid < n) D[gid] = V[gid];
}

kernel void om_zero_words(device uint* D [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) D[gid] = 0;
}

kernel void om_copy_words(device const uint* S [[buffer(0)]], device uint* D [[buffer(1)]], constant uint& n [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
  if (gid < n) D[gid] = S[gid];
}

// ---- Liquid's gated short conv

struct ConvArgs {
  int ld, ldy, T, D, K;
};

// y = C * conv(B * x) over T rows of B | C | x, a thread a channel, STATE [D, K-1] updated.
kernel void om_short_conv(device const bf* BCx [[buffer(0)]], device const bf* Wc [[buffer(1)]], device bf* STATE [[buffer(2)]],
                          device bf* Y [[buffer(3)]], constant ConvArgs& a [[buffer(4)]], uint d [[thread_position_in_grid]]) {
  if (int(d) >= a.D) return;
  const int K = a.K, D = a.D;
  float hist[8];
  float w[8];
  for (int j = 0; j < K; ++j) w[j] = f(Wc[d * K + j]);
  for (int j = 0; j < K - 1; ++j) hist[j] = f(STATE[d * (K - 1) + j]);
  for (int t = 0; t < a.T; ++t) {
    device const bf* row = BCx + (long)t * a.ld;
    const float bx = rb(f(row[d]) * f(row[2 * D + d]));
    float acc = 0.0f;
    for (int j = 0; j < K - 1; ++j) acc += w[j] * hist[j];
    acc += w[K - 1] * bx;
    for (int j = 0; j + 1 < K - 1; ++j) hist[j] = hist[j + 1];
    if (K > 1) hist[K - 2] = bx;
    Y[(long)t * a.ldy + d] = b(f(row[D + d]) * rb(acc));
  }
  for (int j = 0; j < K - 1; ++j) STATE[d * (K - 1) + j] = b(hist[j]);
}

// As om_short_conv for many rows, a thread a (row, channel); the state is left alone (om_conv_state follows).
kernel void om_short_conv_rows(device const bf* BCx [[buffer(0)]], device const bf* Wc [[buffer(1)]], device const bf* STATE [[buffer(2)]],
                               device bf* Y [[buffer(3)]], constant ConvArgs& a [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.T * a.D) return;
  const int K = a.K, D = a.D;
  const int t = int(i / D), d = int(i % D);
  float acc = 0.0f;
  for (int j = 0; j < K; ++j) {
    const int src = t - (K - 1) + j;
    float v;
    if (src >= 0) {
      device const bf* row = BCx + (long)src * a.ld;
      v = rb(f(row[d]) * f(row[2 * D + d]));
    } else {
      v = f(STATE[d * (K - 1) + (K - 1 + src)]);
    }
    acc += f(Wc[d * K + j]) * v;
  }
  Y[(long)t * a.ldy + d] = b(f(BCx[(long)t * a.ld + D + d]) * rb(acc));
}

kernel void om_conv_state(device const bf* BCx [[buffer(0)]], device bf* STATE [[buffer(1)]], constant ConvArgs& a [[buffer(2)]],
                          uint d [[thread_position_in_grid]]) {
  if (int(d) >= a.D) return;
  const int K = a.K, D = a.D, T = a.T;
  for (int j = 0; j < K - 1; ++j) {
    const int src = T - (K - 1) + j;
    if (src >= 0) {
      device const bf* row = BCx + (long)src * a.ld;
      STATE[d * (K - 1) + j] = b(f(row[d]) * f(row[2 * D + d]));
    } else {
      STATE[d * (K - 1) + j] = STATE[d * (K - 1) + j + T];
    }
  }
}

// ---- attention

struct AttnArgs {
  int ldq, ldkv, ldo, T, H, Hkv, hd, pos0, kv_len, window, causal;
  float scale;
  int has_gate, ldg;
};

// Prompt attention: 16 rows of a head a group over 32-key tiles, online softmax, then LFM3's gate G if given.
#define AT_Q 16
#define AT_K 32
[[max_total_threads_per_threadgroup(128)]] kernel void om_attention_tiles(device const bf* Q [[buffer(0)]], device const bf* Kc [[buffer(1)]],
                                                                       device const bf* Vc [[buffer(2)]], device bf* O [[buffer(3)]],
                                                                       device const bf* G [[buffer(4)]], constant AttnArgs& a [[buffer(5)]],
                                                                       uint2 tg [[threadgroup_position_in_grid]],
                                                                       uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float qs[AT_Q * 128];
  threadgroup float ss[AT_Q * AT_K];
  threadgroup bf ks[AT_K * 130];
  threadgroup bf vs[AT_K * 130];
  const int hd = a.hd, hdp = hd + 2;
  const int h = tg.y, kh = h / (a.H / a.Hkv);
  const int r0 = tg.x * AT_Q;
  const int warp = tid >> 5, lane = tid & 31;
  const int dpl = hd >> 5;
  for (int i = tid; i < AT_Q * hd; i += 128) {
    const int r = i / hd, c = i % hd;
    qs[i] = r0 + r < a.T ? f(Q[(long)(r0 + r) * a.ldq + h * hd + c]) : 0.0f;
  }
  const int last = min(a.T, r0 + AT_Q) - 1;
  int hi = a.causal ? a.pos0 + last + 1 : a.kv_len;
  hi = min(hi, a.kv_len);
  const int lo = (a.causal && a.window > 0) ? max(0, a.pos0 + r0 + 1 - a.window) : 0;
  float m[4], l[4], acc[4][4];
  for (int x = 0; x < 4; ++x) {
    m[x] = -INFINITY;
    l[x] = 0.0f;
    for (int c = 0; c < 4; ++c) acc[x][c] = 0.0f;
  }
  for (int k0 = lo; k0 < hi; k0 += AT_K) {
    const int nk = min(AT_K, hi - k0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < nk * hd; i += 128) {
      const int k = i / hd, c = i % hd;
      const long g = (long)(k0 + k) * a.ldkv + kh * hd + c;
      ks[k * hdp + c] = Kc[g];
      vs[k * hdp + c] = Vc[g];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < AT_Q * AT_K; i += 128) {
      const int r = i / AT_K, k = i % AT_K;
      const int pos = a.pos0 + r0 + r, kp = k0 + k;
      float s = -INFINITY;
      if (k < nk && r0 + r < a.T && (!a.causal || kp <= pos) && (!(a.causal && a.window > 0) || kp > pos - a.window)) {
        s = 0.0f;
        threadgroup const float* qq = qs + r * hd;
        threadgroup const bf* kr = ks + k * hdp;
        for (int c = 0; c < hd; c += 2) s += qq[c] * f(kr[c]) + qq[c + 1] * f(kr[c + 1]);
        s *= a.scale;
      }
      ss[i] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int x = 0; x < 4; ++x) {
      const int r = warp * 4 + x;
      threadgroup float* sr = ss + r * AT_K;
      const float mx = simd_max(sr[lane]);
      const float mn = max(m[x], mx);
      if (mn == -INFINITY) continue;
      const float scl = precise::exp(m[x] - mn);
      const float p0 = precise::exp(sr[lane] - mn);
      sr[lane] = p0;
      const float ps = simd_sum(p0);
      l[x] = l[x] * scl + ps;
      m[x] = mn;
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (int c = 0; c < 4; ++c) acc[x][c] *= scl;
      // only the row's visible keys: a masked key's zero weight times a non-finite value is NaN
      const int pos = a.pos0 + r0 + r;
      const int k_lo = (a.causal && a.window > 0) ? max(0, pos - a.window + 1 - k0) : 0;
      const int k_hi = a.causal ? min(nk, pos - k0 + 1) : nk;
      for (int k = k_lo; k < k_hi; ++k) {
        const float p = sr[k];
        threadgroup const bf* vr = vs + k * hdp;
        for (int c = 0; c < 4; ++c)
          if (c < dpl) acc[x][c] += p * f(vr[lane + 32 * c]);
      }
    }
  }
  for (int x = 0; x < 4; ++x) {
    const int r = r0 + warp * 4 + x;
    if (r >= a.T) continue;
    const float inv = 1.0f / l[x];
    const float g = a.has_gate ? rb(sigm(f(G[(long)r * a.ldg + h]))) : 1.0f;
    for (int c = 0; c < 4; ++c)
      if (c < dpl) {
        const bf o = b(acc[x][c] * inv);
        O[(long)r * a.ldo + h * hd + lane + 32 * c] = a.has_gate ? b(f(o) * g) : o;
      }
  }
}

// ---- elementwise

struct SwigluArgs {
  int T, F, inter;
};

// y = bf16(bf16(silu(a)) * c) from rows of 2F: a | c halves, or (inter) pairs a_j, c_j side by side.
kernel void om_swiglu(device const bf* AC [[buffer(0)]], device bf* Y [[buffer(1)]], constant SwigluArgs& a [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.T * a.F) return;
  const long t = i / a.F;
  const int c = int(i % a.F);
  const long ia = a.inter ? t * 2 * a.F + 2 * c : t * 2 * a.F + c;
  const long ic = a.inter ? ia + 1 : ia + a.F;
  const float x = f(AC[ia]);
  Y[i] = b(rb(x / (1.0f + precise::exp(-x))) * f(AC[ic]));
}

struct BiasActArgs {
  int rows, cols, per_row, act, has_bias;
};

// Y = act(bf16(A + bias)) from an fp32 product; bias per column (per_row 0) or per row (1); act 1 SiLU, 2 exact GELU.
kernel void om_bias_act(device const float* A [[buffer(0)]], device const bf* Bias [[buffer(1)]], device bf* Y [[buffer(2)]],
                        constant BiasActArgs& a [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.rows * a.cols) return;
  const int r = int(i / a.cols), c = int(i % a.cols);
  float v = A[i] + (a.has_bias ? f(Bias[a.per_row ? r : c]) : 0.0f);
  v = rb(v);
  if (a.act == 1) v = v / (1.0f + precise::exp(-v));
  else if (a.act == 2) v = 0.5f * v * (1.0f + erf_f(v * 0.70710678118654752f));
  Y[i] = b(v);
}

struct EmbedArgs {
  int T, D, NQ, size;
};

// Y[t] = TABLE[IDS[t]]; ids < 0 leave the row alone.
kernel void om_embed(device const bf* TABLE [[buffer(0)]], device const int* IDS [[buffer(1)]], device bf* Y [[buffer(2)]],
                     constant EmbedArgs& a [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.T * a.D) return;
  const int t = int(i / a.D), c = int(i % a.D);
  const int id = IDS[t];
  if (id >= 0) Y[i] = TABLE[(long)id * a.D + c];
}

// A speech frame's input row: the fp32 sum over the codebooks of TABLE[code + q * size], one rounding.
kernel void om_embed_codes(device const bf* TABLE [[buffer(0)]], device const int* CODES [[buffer(1)]], device bf* Y [[buffer(2)]],
                           constant EmbedArgs& a [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.T * a.D) return;
  const int t = int(i / a.D), c = int(i % a.D);
  float s = 0.0f;
  for (int q = 0; q < a.NQ; ++q) s += f(TABLE[((long)CODES[t * a.NQ + q] + (long)q * a.size) * a.D + c]);
  Y[i] = b(s);
}

// The depth decoder's slots: X[q] = bf16(DIN[q] + (MASKED[q] ? MASK : TABLES[q][CODES[q]])).
kernel void om_depth_in(device const bf* DIN [[buffer(0)]], device const bf* MASK [[buffer(1)]], device const bf* TABLES [[buffer(2)]],
                        device const int* CODES [[buffer(3)]], device const int* MASKED [[buffer(4)]], device bf* X [[buffer(5)]],
                        constant EmbedArgs& a [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  const int i = gid;
  if (i >= a.NQ * a.D) return;
  const int q = i / a.D, c = i % a.D;
  const float e = MASKED[q] ? f(MASK[c]) : f(TABLES[((long)q * a.size + CODES[q]) * a.D + c]);
  X[i] = b(f(DIN[i]) + e);
}

struct Conv2dArgs {
  int CIN, H, W, C, H2, W2, silu;
};

// Stride-2 3x3 conv (padding 1, bias, rounded) over [C, H, W]; CIN 1 reads plane 0, else depthwise; then SiLU.
kernel void om_conv2d_s2(device const bf* X [[buffer(0)]], device const bf* Wt [[buffer(1)]], device const bf* Bias [[buffer(2)]],
                         device bf* Y [[buffer(3)]], constant Conv2dArgs& a [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.C * a.H2 * a.W2) return;
  const int c = int(i / ((long)a.H2 * a.W2)), h = int((i / a.W2) % a.H2), w = int(i % a.W2);
  device const bf* x = X + (long)(a.CIN == 1 ? 0 : c) * a.H * a.W;
  float acc = 0.0f;
  for (int kh = 0; kh < 3; ++kh) {
    const int ih = 2 * h - 1 + kh;
    if (ih < 0 || ih >= a.H) continue;
    for (int kw = 0; kw < 3; ++kw) {
      const int iw = 2 * w - 1 + kw;
      if (iw < 0 || iw >= a.W) continue;
      acc += f(Wt[c * 9 + kh * 3 + kw]) * f(x[(long)ih * a.W + iw]);
    }
  }
  float v = rb(acc + f(Bias[c]));
  if (a.silu) v = rb(v / (1.0f + precise::exp(-v)));
  Y[i] = b(v);
}

kernel void om_f32_bf16(device const float* X [[buffer(0)]], device bf* Y [[buffer(1)]], constant uint& n [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid < n) Y[gid] = b(X[gid]);
}

kernel void om_bf16_f32(device const bf* X [[buffer(0)]], device float* Y [[buffer(1)]], constant uint& n [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid < n) Y[gid] = f(X[gid]);
}

struct FsqArgs {
  int n, NQ;
};

// The codec's finite scalar quantizer read back: CODES [n, NQ] to latents [n, NQ * 4], levels 8 7 6 6.
kernel void om_fsq_decode(device const int* CODES [[buffer(0)]], device bf* L [[buffer(1)]], constant FsqArgs& a [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
  const int i = gid;
  if (i >= a.n * a.NQ) return;
  const int t = i / a.NQ, q = i % a.NQ;
  const int levels[4] = {8, 7, 6, 6};
  const int basis[4] = {1, 8, 56, 336};
  const int idx = CODES[t * a.NQ + q];
  for (int j = 0; j < 4; ++j) {
    const int hf = levels[j] / 2;
    const int digit = (idx / basis[j]) % levels[j] - hf;
    L[(long)t * a.NQ * 4 + q * 4 + j] = b(float(digit) / float(hf));
  }
}

struct RepeatArgs {
  int n, D, rep;
};

kernel void om_repeat_rows(device const bf* X [[buffer(0)]], device bf* Y [[buffer(1)]], constant RepeatArgs& a [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.n * a.rep * a.D) return;
  const long r = i / a.D;
  Y[i] = X[(r / a.rep) * a.D + i % a.D];
}

struct PolarArgs {
  int T, NB;
};

// The codec head's rows (log-magnitude | phase) to a real | imaginary spectrum in fp32.
kernel void om_polar(device const bf* X [[buffer(0)]], device float* S [[buffer(1)]], constant PolarArgs& a [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.T * a.NB) return;
  const long t = i / a.NB;
  const int k = int(i % a.NB);
  const float m = precise::exp(f(X[t * 2 * a.NB + k])), ph = f(X[t * 2 * a.NB + a.NB + k]);
  S[t * 2 * a.NB + k] = m * precise::cos(ph);
  S[t * 2 * a.NB + a.NB + k] = m * precise::sin(ph);
}
