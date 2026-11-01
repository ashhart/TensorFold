// Gemma 4 E4B glue on Intel Xe2 (OpenCL C): norms, per-layer inputs, GeGLU, attention; x: fp32 residual of bf16 values.

// h: fp16 GEMM input [T, K] with group sums (AHI, ALO); rounding follows transformers' bf16 (rbf); GEMM outputs fp16.
#include "glue.cl"

#define D4 2560
#define FF4 10240
#define PLE 256
#define NL4 42
#define PLE_ALL 10752
#define SQ_2560 50.5f
#define KEYS 128
#ifndef HD
#define HD 256   // head size of the attention kernels in this build: 256 (sliding layers) or 512 (global)
#endif

static inline float gelu_tanh(float x) { return 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))); }

// 1 / sqrt(a), rounded to nearest in all but rare cases: rsqrt, then one correction from the residual.
static inline float rsqrt_rn(float a) {
  const float y = rsqrt(a);
  return fma(0.5f * y, fma(-y * y, a, 1.0f), y);
}

// mlx_lm's gelu_approx as MLX's bf16 ops (layer_input.zig gelu, weak scalars rounded to bf16; subnormals not flushed).
static inline float gelu_bf(float x) {
  const float cube = rbf(x * x * x);
  const float inner = rbf(x + rbf(0.044677734375f * cube));
  const float th = rbf(tanh(rbf(0.796875f * inner)));
  return rbf(rbf(0.5f * x) * rbf(1.0f + th));
}

static float wg_sum512(float v, local float *red) {
  const float s = sub_group_reduce_add(v);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (int i = 0; i < 32; i++) t += red[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

// x = bf16(bf16(q·s + b) · 50.5): the scaled embedding of the tied table (MLX 4/64, words [V, D/8]).
KERNEL(256) void g4_embed(global const int *IDS, global const uint *W, global const ushort *S, global const ushort *B, global float *X) {
  const int t = get_group_id(0);
  const long id = IDS[t];
  for (int e = get_local_id(0); e < D4; e += 256) {
    const uint w = W[id * (D4 / 8) + e / 8];
    const float q = (float)((w >> (4 * (e % 8))) & 0xFu);
    const float s = bf2f(S[id * (D4 / GS) + e / GS]), b = bf2f(B[id * (D4 / GS) + e / GS]);
    X[(long)t * D4 + e] = rbf(rbf(q * s + b) * SQ_2560);
  }
}

// The per-layer table's row of a token: bf16(bf16(q·s + b) · 16) [T, 10752] (MLX 4/64).
KERNEL(256) void g4_ple_tok(global const int *IDS, global const uint *W, global const ushort *S, global const ushort *B, global ushort *TOK) {
  const int t = get_group_id(0);
  const long id = IDS[t];
  for (int e = get_local_id(0); e < PLE_ALL; e += 256) {
    const uint w = W[id * (PLE_ALL / 8) + e / 8];
    const float q = (float)((w >> (4 * (e % 8))) & 0xFu);
    const float s = bf2f(S[id * (PLE_ALL / GS) + e / GS]), b = bf2f(B[id * (PLE_ALL / GS) + e / GS]);
    TOK[(long)t * PLE_ALL + e] = f2bf(rbf(q * s + b) * 16.0f);
  }
}

// pli = bf16(bf16(n + tok) · 0.70703125), n = bf16(w · bf16(v · inv)), v = bf16(bf16(proj) · 0.019775390625); (T, 42).
KERNEL(16) void g4_ple_combine(global const half *PROJ, global const ushort *TOK, global const ushort *WN, global ushort *PLI, float eps) {
  const int t = get_group_id(0), l = get_group_id(1), lane = get_sub_group_local_id();
  const long base = (long)t * PLE_ALL + l * PLE;
  float v[16], ss = 0.0f;
  for (int i = 0; i < 16; i++) {
    v[i] = rbf(rbf((float)PROJ[base + i * 16 + lane]) * 0.019775390625f);
    ss += v[i] * v[i];
  }
  const float inv = rsqrt_rn(sub_group_reduce_add(ss) / PLE + eps);
  for (int i = 0; i < 16; i++) {
    const float n = rbf(bf2f(WN[i * 16 + lane]) * rbf(v[i] * inv));
    PLI[base + i * 16 + lane] = f2bf(rbf(n + bf2f(TOK[base + i * 16 + lane])) * 0.70703125f);
  }
}

// h = fp16(x) [T, 2560] and its group sums: x as an unnormalized GEMM input (the per-layer embedding's projection).
KERNEL(256) void g4_x2h(global const float *X, global half *H, global half *AHI, global half *ALO) {
  local half hl[D4];
  const int t = get_group_id(0);
  for (int e = get_local_id(0); e < D4; e += 256) {
    const half v = convert_half_rte(X[(long)t * D4 + e]);
    H[(long)t * D4 + e] = v;
    hl[e] = v;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  rowsums(hl, D4, AHI, ALO, t, 64);
}

// Residual: x = bf16(x + bf16(rmsnorm(R)·WPOST)), x = bf16(x·ls); h = fp16(x) if RAW else fp16(bf16(rmsnorm(x)·WPRE)).
KERNEL(512) void g4_resid(global float *X, global const half *R, global const ushort *WPOST, global const ushort *LS, global const ushort *WPRE,
                          global half *H, global half *AHI, global half *ALO, float eps, int raw) {
  local float red[32];
  local half hl[D4];
  const int t = get_group_id(0), lid = get_local_id(0);
  float rr[5], xv[5];
  float inv_r = 0.0f;
  if (R) {
    float ss = 0.0f;
    for (int i = 0; i < 5; i++) {
      rr[i] = rbf((float)R[(long)t * D4 + lid + i * 512]);
      ss += rr[i] * rr[i];
    }
    const float mean = wg_sum512(ss, red) / D4 + eps;
    inv_r = LS ? rsqrt_rn(mean) : rsqrt(mean); // with LS, the per-layer block: layer_input.zig apply()'s norm
  }
  const float ls = LS ? bf2f(LS[0]) : 1.0f;
  float ss2 = 0.0f;
  for (int i = 0; i < 5; i++) {
    const int e = lid + i * 512;
    float x = X[(long)t * D4 + e];
    if (R) x = rbf(x + (LS ? rbf(bf2f(WPOST[e]) * rbf(rr[i] * inv_r)) : rbf(rr[i] * inv_r * bf2f(WPOST[e]))));
    if (LS) x = rbf(x * ls);
    X[(long)t * D4 + e] = x;
    xv[i] = x;
    ss2 += x * x;
  }
  if (!raw && !WPRE) return;
  const float inv_x = raw ? 1.0f : rsqrt(wg_sum512(ss2, red) / D4 + eps);
  for (int i = 0; i < 5; i++) {
    const int e = lid + i * 512;
    const half v = raw ? convert_half_rte(xv[i]) : convert_half_rte(rbf(xv[i] * inv_x * bf2f(WPRE[e])));
    H[(long)t * D4 + e] = v;
    hl[e] = v;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  rowsums(hl, D4, AHI, ALO, t, 64);
}

// GeGLU: act = fp16(bf16(bf16(gelu_tanh(g)) · u)) [T, 10240] + group sums; GU fp16 [T, 20480] = gate | up; (T, 10).
KERNEL(256) void g4_geglu(global const half *GU, global half *H, global half *AHI, global half *ALO) {
  local half hl[1024];
  const int t = get_group_id(0), blk = get_group_id(1), lid = get_local_id(0);
  {
    const int e = lid * 4, c = blk * 1024 + e;
    const half4 gv = vload4(0, GU + (long)t * (2 * FF4) + c), uv = vload4(0, GU + (long)t * (2 * FF4) + FF4 + c);
    half4 o;
    for (int i = 0; i < 4; i++) o[i] = convert_half_rte(rbf(rbf(gelu_tanh(rbf((float)gv[i]))) * rbf((float)uv[i])));
    vstore4(o, 0, H + (long)t * FF4 + c);
    vstore4(o, 0, hl + e);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = get_sub_group_local_id();
  for (int g = get_sub_group_id(); g < 16; g += 16) {
    float s = 0.0f;
    for (int k = 0; k < 4; k++) s += (float)hl[g * 64 + lane * 4 + k];
    s = sub_group_reduce_add(s) * (1.0f / 64);
    if (lane == 0) {
      const half hi = convert_half_rte(s * ASC);
      AHI[(long)t * (FF4 / 64) + blk * 16 + g] = hi;
      ALO[(long)t * (FF4 / 64) + blk * 16 + g] = convert_half_rte(s * ASC - (float)hi);
    }
  }
}

// bf16(gelu_tanh(x)) for every bf16 x, from the code g4_geglu runs: the table the GEMM's GeGLU epilogue reads.
KERNEL(256) void g4_geglu_tab(global ushort *T) {
  const int i = get_global_id(0);
  T[i] = f2bf(gelu_tanh(bf2f((ushort)i)));
}

// The group sums of the GEMM-fused GeGLU's activation H [T, 10240], summed as g4_geglu does; (T, 10).
KERNEL(256) void g4_geglu_sums(global const half *H, global half *AHI, global half *ALO) {
  const int t = get_group_id(0), blk = get_group_id(1), lid = get_local_id(0);
  const half4 hv = vload4(0, H + (long)t * FF4 + blk * 1024 + lid * 4);
  float s = 0.0f;
  for (int k = 0; k < 4; k++) s += (float)hv[k];
  s = sub_group_reduce_add(s) * (1.0f / 64);
  if (get_sub_group_local_id() == 0) {
    const int g = get_sub_group_id();
    const half hi = convert_half_rte(s * ASC);
    AHI[(long)t * (FF4 / 64) + blk * 16 + g] = hi;
    ALO[(long)t * (FF4 / 64) + blk * 16 + g] = convert_half_rte(s * ASC - (float)hi);
  }
}

// Per-layer gate: h = fp16(bf16(gelu_bf(gate) · pli_l)) [T, 256] and its sums; GO bf16 in fp16, PLI bf16 [T,42,256].
KERNEL(64) void g4_ple_mid(global const half *GO, global const ushort *PLI, int layer, global half *H, global half *AHI, global half *ALO) {
  local half hl[PLE];
  const int t = get_group_id(0), lid = get_local_id(0);
  for (int j = 0; j < 4; j++) {
    const int e = lid * 4 + j;
    const half v = convert_half_rte(rbf(gelu_bf(rbf((float)GO[(long)t * PLE + e])) * bf2f(PLI[(long)t * PLE_ALL + layer * PLE + e])));
    H[(long)t * PLE + e] = v;
    hl[e] = v;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  rowsums(hl, PLE, AHI, ALO, t, 64);
}

// Attention front end: P fp16 [T, ldp] = q|k|v (8, 2, 2 heads of hd); RMS, RoPE on q, k; K, V to the cache slot.
KERNEL(256) void g4_attn_prep(global const half *P, int ldp, global const ushort *QN, global const ushort *KN, global const float *INVF,
                              global ushort *Q, global ushort *KC, global ushort *VC, int p0, float eps, int ring, int own) {
  const int t = get_group_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  if (sg >= (own ? 12 : 8)) return;
  const int hd = HD, n = HD / 16;
  const float pos = (float)(p0 + t);
  float v[HD / 16], ss = 0.0f;
  for (int i = 0; i < n; i++) {
    v[i] = rbf((float)P[(long)t * ldp + sg * hd + i * 16 + lane]);
    ss += v[i] * v[i];
  }
  const float inv = rsqrt(sub_group_reduce_add(ss) / hd + eps);
  if (sg < 10) {
    global const ushort *w = sg < 8 ? QN : KN;
    for (int i = 0; i < n; i++) v[i] = rbf(v[i] * inv * bf2f(w[i * 16 + lane]));
    for (int i = 0; i < n / 2; i++) {
      const float ang = pos * INVF[i * 16 + lane], c = rbf(cos(ang)), s = rbf(sin(ang));
      const float x1 = v[i], x2 = v[i + n / 2];
      v[i] = rbf(rbf(x1 * c) + rbf(-x2 * s));
      v[i + n / 2] = rbf(rbf(x2 * c) + rbf(x1 * s));
    }
  } else {
    for (int i = 0; i < n; i++) v[i] = rbf(v[i] * inv);
  }
  const long slot = ring ? (p0 + t) % ring : (p0 + t);
  global ushort *dst = sg < 8 ? Q + ((long)t * 8 + sg) * hd : (sg < 10 ? KC : VC) + (slot * 2 + (sg < 10 ? sg - 8 : sg - 10)) * hd;
  for (int i = 0; i < n; i++) dst[i * 16 + lane] = f2bf(v[i]);
}


// Folds a row's tile partials (g4_attn_dp_part) in order, 64 columns of o a group; writes fp16(bf16(o / l)) and sums.
KERNEL(64) void g4_attn_merge(global const float *PM, global const float *PL, global const float *PO, int p0, int window, int ntl,
                              global half *H, global half *AHI, global half *ALO) {
  const int hd = HD;
  local half hl[64];
  const int head = get_group_id(0), t = get_group_id(1), z = get_group_id(2), lid = get_local_id(0);
  const int col = z * 64 + lid;
  const int pos = p0 + t, lo = window ? max(0, pos - window + 1) : 0;
  const int nt = pos / KEYS - lo / KEYS + 1;
  const long base = ((long)t * 8 + head) * ntl;
  float m = PM[base], l = PL[base], acc = PO[base * hd + col];
  for (int i0 = 1; i0 < nt; i0 += 8) {
    const int n = min(8, nt - i0);
    float mi[8], li[8], oi[8];
#pragma unroll
    for (int u = 0; u < 8; u++)
      if (u < n) {
        mi[u] = PM[base + i0 + u];
        li[u] = PL[base + i0 + u];
        oi[u] = PO[(base + i0 + u) * hd + col];
      }
#pragma unroll
    for (int u = 0; u < 8; u++)
      if (u < n) {
        const float mm = fmax(m, mi[u]), a = exp(m - mm), b = exp(mi[u] - mm);
        l = l * a + li[u] * b;
        acc = acc * a + oi[u] * b;
        m = mm;
      }
  }
  const half v = convert_half_rte(rbf(acc / l));
  hl[lid] = v;
  H[(long)t * 8 * hd + head * hd + col] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_sub_group_id() == 0) {
    const int lane = get_sub_group_local_id();
    float s = 0.0f;
    for (int k = 0; k < 4; k++) s += (float)hl[lane * 4 + k];
    s = sub_group_reduce_add(s) * (1.0f / 64);
    if (lane == 0) {
      const half hi = convert_half_rte(s * ASC);
      AHI[(long)t * (8 * hd / 64) + head * (hd / 64) + z] = hi;
      ALO[(long)t * (8 * hd / 64) + head * (hd / 64) + z] = convert_half_rte(s * ASC - (float)hi);
    }
  }
}

// The same fold with a group of 128 columns: fewer, longer groups, better when a row has few tiles.
KERNEL(128) void g4_attn_merge_w(global const float *PM, global const float *PL, global const float *PO, int p0, int window, int ntl,
                               global half *H, global half *AHI, global half *ALO) {
  const int hd = HD;
  local half hl[HD];
  const int head = get_group_id(0), t = get_group_id(1), lid = get_local_id(0);
  const int pos = p0 + t, lo = window ? max(0, pos - window + 1) : 0, nt = pos / KEYS - lo / KEYS + 1;
  const long b0 = ((long)t * 8 + head) * ntl;
  float acc[HD / 128], m = PM[b0], l = PL[b0];
  for (int j = 0; j < hd / KEYS; j++) acc[j] = PO[b0 * hd + lid + j * KEYS];
  for (int i = 1; i < nt; i++) {
    const long idx = b0 + i;
    const float mi = PM[idx], li = PL[idx], mm = fmax(m, mi), a = exp(m - mm), b = exp(mi - mm);
    l = l * a + li * b;
    for (int j = 0; j < hd / KEYS; j++) acc[j] = acc[j] * a + PO[idx * hd + lid + j * KEYS] * b;
    m = mm;
  }
  for (int j = 0; j < hd / KEYS; j++) {
    const half v = convert_half_rte(rbf(acc[j] / l));
    hl[lid + j * KEYS] = v;
    H[(long)t * 8 * hd + head * hd + lid + j * KEYS] = v;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = get_sub_group_local_id();
  for (int g = get_sub_group_id(); g < hd / 64; g += 8) {
    float s = 0.0f;
    for (int k = 0; k < 4; k++) s += (float)hl[g * 64 + lane * 4 + k];
    s = sub_group_reduce_add(s) * (1.0f / 64);
    if (lane == 0) {
      const half hi = convert_half_rte(s * ASC);
      AHI[(long)t * (8 * hd / 64) + head * (hd / 64) + g] = hi;
      ALO[(long)t * (8 * hd / 64) + head * (hd / 64) + g] = convert_half_rte(s * ASC - (float)hi);
    }
  }
}

// out = a · Wᵀ, K = 256 (per-layer projection), fp32 sums in a fixed order (group-sum pitch too short for 2D blocks).
kernel void g4_gemv256(global const half *A, global const uint *W, global const half *S, global const half *BG, global half *OUT, int M, int N) {
  const int n = get_global_id(0), m = get_global_id(1);
  if (n >= N || m >= M) return;
  float acc = 0.0f;
  for (int g = 0; g < 4; g++) {
    const float s = (float)S[(long)g * N + n], b = 4.0f * (float)BG[(long)g * N + n];
    for (int w = 0; w < 8; w++) {
      const uint word = W[(long)(g * 8 + w) * N + n];
      for (int j = 0; j < 8; j++) {
        const float q = (float)((word >> (4 * (j / 2) + 16 * (j & 1))) & 0xFu);
        acc = fma((float)A[(long)m * 256 + g * 64 + w * 8 + j], fma(q, s, b), acc);
      }
    }
  }
  OUT[(long)m * N + n] = convert_half_rte(acc);
}

// Top-K of fp16 logits rows [R, 262144] (K <= 64), exact, ties to lower id: stage 1, grid (256, R), stage 2 (R) merges.
static int better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

// group-wide maximum of (v, i) by (value, lower id), the winner to all work-items; red: local float[16] and int[16]
static void wg_best(float *v, int *i, local float *rv, local int *ri) {
  float bv = sub_group_reduce_max(*v);
  int bi = sub_group_reduce_min(*v == bv ? *i : 0x7fffffff);
  if (get_sub_group_local_id() == 0) {
    rv[get_sub_group_id()] = bv;
    ri[get_sub_group_id()] = bi;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bv = rv[0];
  bi = ri[0];
  for (int s = 1; s < 16; s++)
    if (better(rv[s], ri[s], bv, bi)) {
      bv = rv[s];
      bi = ri[s];
    }
  barrier(CLK_LOCAL_MEM_FENCE);
  *v = bv;
  *i = bi;
}

KERNEL(256) void g4_topk_part(global const half *L, int V, int K, global float *PV, global int *PI) {
  local float rv[16];
  local int ri[16];
  const int lid = get_local_id(0), seg = get_group_id(0), row = get_group_id(1);
  const int per = V / 256;
  float v[4];
  int id[4];
  for (int j = 0; j < 4; j++) {
    id[j] = seg * per + lid + 256 * j;
    v[j] = (float)L[(long)row * V + id[j]];
  }
  for (int k = 0; k < K; k++) {
    float mv = v[0];
    int mi = id[0], mj = 0;
    for (int j = 1; j < 4; j++)
      if (better(v[j], id[j], mv, mi)) {
        mv = v[j];
        mi = id[j];
        mj = j;
      }
    float bv = mv;
    int bi = mi;
    wg_best(&bv, &bi, rv, ri);
    if (mi == bi) v[mj] = -INFINITY;
    if (lid == 0) {
      PV[((long)row * 256 + seg) * K + k] = bv;
      PI[((long)row * 256 + seg) * K + k] = bi;
    }
  }
}

KERNEL(256) void g4_topk_merge(global const float *PV, global const int *PI, int K, global float *OV, global int *OI) {
  local float rv[16];
  local int ri[16];
  const int lid = get_local_id(0), row = get_group_id(0);
  int ptr = 0;
  for (int k = 0; k < K; k++) {
    float v = ptr < K ? PV[((long)row * 256 + lid) * K + ptr] : -INFINITY;
    int i = ptr < K ? PI[((long)row * 256 + lid) * K + ptr] : 0x7fffffff;
    float bv = v;
    int bi = i;
    wg_best(&bv, &bi, rv, ri);
    if (i == bi && v == bv) ptr++;
    if (lid == 0) {
      OV[row * K + k] = bv;
      OI[row * K + k] = bi;
    }
  }
}

// Dense DPAS projection OUT = A · Wᵀ, A [M, K], W [N, K] fp16, 16 slices of K added in order: _f fp32, _b bf16-rounded.
#define DENSE_KERNEL(NAME, TYPE, STORE)                                                                                                 \
  KERNEL(256) void NAME(global const half *A, global const half *W, global TYPE *OUT, int M, int N, int K) {                           \
    local float red[16 * 16 * 16];                                                                                                      \
    const int KS = K / 16;                                                                                                              \
    const int m0 = get_group_id(0) * 16, n0 = get_group_id(1) * 16, sg = get_sub_group_id(), lane = get_sub_group_local_id();          \
    float8 acc0 = 0.0f, acc1 = 0.0f;                                                                                                    \
    global const uint *wr = (global const uint *)(W + (long)(n0 + lane) * K);                                                           \
    for (int k = sg * KS; k < (sg + 1) * KS; k += 16) {                                                                                 \
      short8 a0, a1;                                                                                                                    \
      intel_sub_group_2d_block_read_16b_8r16x1c((global void *)A, K * 2, M, K * 2, (int2)(k, m0), (private ushort *)&a0);              \
      intel_sub_group_2d_block_read_16b_8r16x1c((global void *)A, K * 2, M, K * 2, (int2)(k, m0 + 8), (private ushort *)&a1);          \
      const int8 bb = as_int8(vload8(0, wr + k / 2));                                                                                   \
      acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bb, acc0);                                                                      \
      acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bb, acc1);                                                                      \
    }                                                                                                                                   \
    for (int r = 0; r < 8; r++) {                                                                                                       \
      red[(sg * 16 + r) * 16 + lane] = acc0[r];                                                                                         \
      red[(sg * 16 + 8 + r) * 16 + lane] = acc1[r];                                                                                     \
    }                                                                                                                                   \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                                                       \
    float v = 0.0f;                                                                                                                     \
    for (int s = 0; s < 16; s++) v += red[(s * 16 + sg) * 16 + lane];                                                                   \
    if (m0 + sg < M) OUT[(long)(m0 + sg) * N + n0 + lane] = STORE(v);                                                                   \
  }
#define STORE_H(v) convert_half_rte(v)
#define STORE_B(v) convert_half_rte(rbf(v))
DENSE_KERNEL(g4_dense, half, STORE_H)
DENSE_KERNEL(g4_dense_b, half, STORE_B)

// 8-bit MLX affine weights (groups of 64): OUT = A · Wᵀ fp16; W uchar [N, K], SC / BI bf16 [N, K/64], AHI / ALO sums.

// A group is four DPAS steps on the integer codes (exact in fp16), then acc += total · scale + bias · groupsum.

// The 16 subgroups take groups sg, sg + 16, ... and add totals in subgroup order: a row's bits depend on K only.
#define W8_KERNEL(NAME, TILES)                                                                                                          \
  KERNEL(256) void NAME(global const half *A, global const uchar *W, global const ushort *SC, global const ushort *BI, global half *OUT,  \
                        global const half *AHI, global const half *ALO, int M, int N, int K) {                                           \
    local float red[16 * 16 * 16];                                                                                                       \
    const int G = K / 64;                                                                                                                \
    const int mb = get_group_id(0) * 16 * TILES, n0 = get_group_id(1) * 16, sg = get_sub_group_id(), lane = get_sub_group_local_id();     \
    const int n = n0 + lane;                                                                                                             \
    float8 acc[TILES][2];                                                                                                                \
    for (int u = 0; u < TILES; u++) acc[u][0] = acc[u][1] = 0.0f;                                                                        \
    global const uchar *wr = W + (long)n * K;                                                                                            \
    for (int g = sg; g < G; g += 16) {                                                                                                   \
      float8 t[TILES][2];                                                                                                                \
      for (int u = 0; u < TILES; u++) t[u][0] = t[u][1] = 0.0f;                                                                          \
      for (int j = 0; j < 4; j++) {                                                                                                      \
        const int k = g * 64 + j * 16;                                                                                                   \
        const int8 bb = as_int8(convert_half16(vload16(0, wr + k)));                                                                     \
        for (int u = 0; u < TILES; u++) {                                                                                                \
          short8 a0, a1;                                                                                                                 \
          intel_sub_group_2d_block_read_16b_8r16x1c((global void *)A, K * 2, M, K * 2, (int2)(k, mb + 16 * u), (private ushort *)&a0);  \
          intel_sub_group_2d_block_read_16b_8r16x1c((global void *)A, K * 2, M, K * 2, (int2)(k, mb + 16 * u + 8), (private ushort *)&a1); \
          t[u][0] = intel_sub_group_f16_f16_matrix_mad_k16(a0, bb, t[u][0]);                                                             \
          t[u][1] = intel_sub_group_f16_f16_matrix_mad_k16(a1, bb, t[u][1]);                                                             \
        }                                                                                                                                \
      }                                                                                                                                  \
      const float s = as_float((uint)SC[(long)n * G + g] << 16), b = as_float((uint)BI[(long)n * G + g] << 16);                          \
      for (int u = 0; u < TILES; u++)                                                                                                    \
        for (int r = 0; r < 8; r++) {                                                                                                    \
          const int ra = mb + 16 * u + r, rb = ra + 8;                                                                                   \
          const float sa = ra < M ? ((float)AHI[(long)ra * G + g] + (float)ALO[(long)ra * G + g]) * 0.25f : 0.0f;                       \
          const float sb = rb < M ? ((float)AHI[(long)rb * G + g] + (float)ALO[(long)rb * G + g]) * 0.25f : 0.0f;                       \
          acc[u][0][r] += t[u][0][r] * s + b * sa;                                                                                       \
          acc[u][1][r] += t[u][1][r] * s + b * sb;                                                                                       \
        }                                                                                                                                \
    }                                                                                                                                    \
    for (int u = 0; u < TILES; u++) {                                                                                                    \
      if (u > 0) barrier(CLK_LOCAL_MEM_FENCE);                                                                                           \
      for (int r = 0; r < 8; r++) {                                                                                                      \
        red[(sg * 16 + r) * 16 + lane] = acc[u][0][r];                                                                                   \
        red[(sg * 16 + 8 + r) * 16 + lane] = acc[u][1][r];                                                                               \
      }                                                                                                                                  \
      barrier(CLK_LOCAL_MEM_FENCE);                                                                                                      \
      float v = 0.0f;                                                                                                                    \
      for (int q = 0; q < 16; q++) v += red[(q * 16 + sg) * 16 + lane];                                                                  \
      if (mb + 16 * u + sg < M) OUT[(long)(mb + 16 * u + sg) * N + n0 + lane] = convert_half_rte(v);                                     \
    }                                                                                                                                    \
  }
W8_KERNEL(g4_w8, 1)

// ---- Attention on DPAS (bf16 inputs, fp32 sums), 128 keys a tile; same arithmetic for decode and prompt ----

// A group: 16 subgroups, one kv head, 4 tokens; 16 slots (token, head) as DPAS columns; S = K·Qᵀ, P (bf16), O = P·V.
#define NSG 16
#define DTS (HD / 16 / NSG)
#define SLOT(kk) (ring ? ((kk) % ring) : (kk))

// Tile `tile` for tokens t0 .. t0 + 3: per slot m (OM), l (OL) in local memory, o[half][j] in registers.
static void dp_tile(global const ushort *Q, global const ushort *KC, global const ushort *VC, int t0, int T, int p0, int kvh, int ring, int krows,
                    int window, int tile, local ushort *Pl, local float *MX, local float *SM, local float *OM, local float *OL, float8 o[2][DTS]) {
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int token = t0 + lane / 4, hq = lane % 4;
  const bool qok = token < T;
  const int qpos = p0 + token, lo = window ? max(0, qpos - window + 1) : 0;
  const int ts = tile * KEYS, kb = ts + sg * 8;
  const int maxpos = p0 + min(t0 + 3, T - 1);
  global const uint *qrow = (global const uint *)(Q + ((long)(qok ? token : 0) * 8 + kvh * 4 + hq) * HD);
  float8 s = 0.0f;
  for (int ds = 0; ds < HD / 16; ds++) {
    ushort8 ka;
    intel_sub_group_2d_block_read_16b_8r16x1c((global void *)(KC + kvh * HD), HD * 2, krows, 2 * HD * 2, (int2)(ds * 16, SLOT(kb)), (private ushort *)&ka);
    const int8 qb = qok ? as_int8(vload8(0, qrow + ds * 8)) : (int8)(0);
    s = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(ka), qb, s);
  }
  float mx = -INFINITY;
  for (int r = 0; r < 8; r++) {
    const int kk = kb + r;
    const float x = (qok && kk >= lo && kk <= qpos) ? s[r] : -INFINITY;
    s[r] = x;
    mx = fmax(mx, x);
  }
  MX[sg * 16 + lane] = mx;
  barrier(CLK_LOCAL_MEM_FENCE);
  float m = -INFINITY;
  for (int i = 0; i < NSG; i++) m = fmax(m, MX[i * 16 + lane]);
  float sum = 0.0f;
  for (int r = 0; r < 8; r++) {
    const float p = s[r] == -INFINITY ? 0.0f : exp(s[r] - m);
    Pl[lane * KEYS + sg * 8 + r] = f2bf(p);
    sum += p;
  }
  SM[sg * 16 + lane] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg == 0) {
    float l = 0.0f;
    for (int i = 0; i < NSG; i++) l += SM[i * 16 + lane];
    OM[lane] = m;
    OL[lane] = l;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int h = 0; h < 2; h++)
    for (int j = 0; j < DTS; j++) o[h][j] = 0.0f;
  for (int c = 0; c < KEYS / 16; c++) {
    if (ts + c * 16 > maxpos) break;
    short8 a0, a1;
    for (int mm = 0; mm < 8; mm++) {
      a0[mm] = (short)Pl[mm * KEYS + c * 16 + lane];
      a1[mm] = (short)Pl[(8 + mm) * KEYS + c * 16 + lane];
    }
    for (int j = 0; j < DTS; j++) {
      uint8 vw;
      intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)(VC + kvh * HD), HD * 2, krows, 2 * HD * 2, (int2)((sg * DTS + j) * 16, SLOT(ts + c * 16)), (private uint *)&vw);
      o[0][j] = intel_sub_group_bf16_bf16_matrix_mad_k16(a0, as_int8(vw), o[0][j]);
      o[1][j] = intel_sub_group_bf16_bf16_matrix_mad_k16(a1, as_int8(vw), o[1][j]);
    }
  }
}

static int first_tile(int qpos, int window) { return (window ? max(0, qpos - window + 1) : 0) / KEYS; }

// Decode and verification windows: one group a (tile, kv head, block of 4 tokens); partial (m, l, o) for g4_attn_merge.
KERNEL(256) void g4_attn_dp_part(global const ushort *Q, global const ushort *KC, global const ushort *VC, global float *PM, global float *PL, global float *PO,
                                 int p0, int T, int window, int ring, int krows, int ntl) {
  local ushort Pl[16 * KEYS];
  local float MX[NSG * 16], SM[NSG * 16], OM[16], OL[16];
  const int kvh = get_group_id(1), t0 = get_group_id(2) * 4, lane = get_sub_group_local_id(), sg = get_sub_group_id();
  const int min_first = first_tile(p0 + t0, window), max_last = (p0 + min(t0 + 3, T - 1)) / KEYS;
  const int tile = min_first + get_group_id(0);
  if (tile > max_last) return;
  float8 o[2][DTS];
  dp_tile(Q, KC, VC, t0, T, p0, kvh, ring, krows, window, tile, Pl, MX, SM, OM, OL, o);
  // slot s = lane: token t0 + s / 4, head 4 kvh + s % 4
  {
    const int token = t0 + lane / 4, head = kvh * 4 + lane % 4, qpos = p0 + token;
    if (sg == 0 && token < T && tile >= first_tile(qpos, window) && tile <= qpos / KEYS) {
      const long idx = ((long)token * 8 + head) * ntl + (tile - first_tile(qpos, window));
      PM[idx] = OM[lane];
      PL[idx] = OL[lane];
    }
  }
  for (int h = 0; h < 2; h++)
    for (int i = 0; i < 8; i++) {
      const int slot = 8 * h + i, token = t0 + slot / 4, head = kvh * 4 + slot % 4, qpos = p0 + token;
      if (token >= T || tile < first_tile(qpos, window) || tile > qpos / KEYS) continue;
      const long idx = ((long)token * 8 + head) * ntl + (tile - first_tile(qpos, window));
      for (int j = 0; j < DTS; j++) PO[idx * HD + (sg * DTS + j) * 16 + lane] = o[h][j][i];
    }
}

#include "attn_rows_g4.cl"

// The group sums of an attention output H [T, 8 HD] (g4_attn_merge's, from memory): grid (8 heads, T).
KERNEL(128) void g4_attn_sums(global const half *H, global half *AHI, global half *ALO) {
  const int head = get_group_id(0), t = get_group_id(1), lane = get_sub_group_local_id();
  for (int g = get_sub_group_id(); g < HD / 64; g += 8) {
    float s = 0.0f;
    for (int k = 0; k < 4; k++) s += (float)H[(long)t * 8 * HD + head * HD + g * 64 + lane * 4 + k];
    s = sub_group_reduce_add(s) * (1.0f / 64);
    if (lane == 0) {
      const half hi = convert_half_rte(s * ASC);
      AHI[(long)t * (8 * HD / 64) + head * (HD / 64) + g] = hi;
      ALO[(long)t * (8 * HD / 64) + head * (HD / 64) + g] = convert_half_rte(s * ASC - (float)hi);
    }
  }
}
