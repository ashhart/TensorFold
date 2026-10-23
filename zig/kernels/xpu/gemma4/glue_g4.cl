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

// pli = bf16((rmsnorm256(bf16(proj · 2560^-1/2)) + tok) · 2^-1/2) per (row, layer) slice of 256; grid (T, 42).
KERNEL(16) void g4_ple_combine(global const half *PROJ, global const ushort *TOK, global const ushort *WN, global ushort *PLI, float eps) {
  const int t = get_group_id(0), l = get_group_id(1), lane = get_sub_group_local_id();
  const long base = (long)t * PLE_ALL + l * PLE;
  float v[16], ss = 0.0f;
  for (int i = 0; i < 16; i++) {
    v[i] = rbf(rbf((float)PROJ[base + i * 16 + lane]) * 0.019764235376052370f);
    ss += v[i] * v[i];
  }
  const float inv = rsqrt(sub_group_reduce_add(ss) / PLE + eps);
  for (int i = 0; i < 16; i++) {
    const float n = rbf(v[i] * inv * bf2f(WN[i * 16 + lane]));
    PLI[base + i * 16 + lane] = f2bf(rbf(n + bf2f(TOK[base + i * 16 + lane])) * 0.70710678118654752f);
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
    inv_r = rsqrt(wg_sum512(ss, red) / D4 + eps);
  }
  const float ls = LS ? bf2f(LS[0]) : 1.0f;
  float ss2 = 0.0f;
  for (int i = 0; i < 5; i++) {
    const int e = lid + i * 512;
    float x = X[(long)t * D4 + e];
    if (R) x = rbf(x + rbf(rr[i] * inv_r * bf2f(WPOST[e])));
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

// Per-layer gate: h = fp16(bf16(bf16(gelu_tanh(gate)) · pli_l)) [T, 256] and its sums; GO fp16 [T, 256], PLI bf16 [T, 42, 256].
KERNEL(64) void g4_ple_mid(global const half *GO, global const ushort *PLI, int layer, global half *H, global half *AHI, global half *ALO) {
  local half hl[PLE];
  const int t = get_group_id(0), lid = get_local_id(0);
  for (int j = 0; j < 4; j++) {
    const int e = lid * 4 + j;
    const half v = convert_half_rte(rbf(rbf(gelu_tanh(rbf((float)GO[(long)t * PLE + e]))) * bf2f(PLI[(long)t * PLE_ALL + layer * PLE + e])));
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


// The tile's values: this subgroup's keys k0 + sg, +8, ... in order, UNR at a time so their loads overlap; each accumulator still
// takes the keys one after the other, so the bits are those of a plain loop.
#if HD == 512
#define UNR 2
#else
#define UNR 4
#endif
#define PV_LOOP(ACC)                                                                                                       \
  {                                                                                                                        \
    int k = k0 + sg;                                                                                                       \
    for (; k + 8 * (UNR - 1) <= k1; k += 8 * UNR) {                                                                        \
      float vv[UNR][HD / 16];                                                                                              \
      for (int u = 0; u < UNR; u++) {                                                                                      \
        global const ushort *vr = VC + ((ring ? (ts + k + 8 * u) % ring : (ts + k + 8 * u)) * 2L + kvh) * hd;             \
        for (int j = 0; j < nd; j++) vv[u][j] = bf2f(vr[j * 16 + lane]);                                                   \
      }                                                                                                                    \
      for (int u = 0; u < UNR; u++) {                                                                                      \
        const float pk = sc[k + 8 * u];                                                                                    \
        for (int j = 0; j < nd; j++) ACC[j] = fma(pk, vv[u][j], ACC[j]);                                                   \
      }                                                                                                                    \
    }                                                                                                                      \
    for (; k <= k1; k += 8) {                                                                                              \
      const float pk = sc[k];                                                                                              \
      global const ushort *vr = VC + ((ring ? (ts + k) % ring : (ts + k)) * 2L + kvh) * hd;                                \
      for (int j = 0; j < nd; j++) ACC[j] = fma(pk, bf2f(vr[j * 16 + lane]), ACC[j]);                                      \
    }                                                                                                                      \
  }

// Attention over one tile of 128 keys (absolute positions 128j .. 128j + 127) for one (row, head): m, l and the unnormalized
// o of the tile in fp32, scores = q·k (scale 1), keys limited to [pos - window + 1, pos]. Grid (tiles, 8, T); tile index
// counted from the row's first tile. The cache of the source layer: ring slots pos % ring, or positions.
// Scores: one key a work-item. Values: the 8 subgroups each take 16 keys in order and every lane hd / 16 dimensions;
// the 8 partial sums are added in subgroup order.
KERNEL(128) void g4_attn_part(global const ushort *Q, global const ushort *KC, global const ushort *VC, global float *PM, global float *PL,
                              global float *PO, int p0, int window, int ring, int ntl) {
  const int hd = HD;
  local float ql[HD], sc[KEYS], red[8], part[8 * HD];
  const int tl = get_group_id(0), head = get_group_id(1), t = get_group_id(2), lid = get_local_id(0);
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int pos = p0 + t, lo = window ? max(0, pos - window + 1) : 0;
  const int tile = lo / KEYS + tl, ts = tile * KEYS;
  if (ts > pos) return;
  const int kvh = head / 4;
  for (int d = lid; d < hd; d += KEYS) ql[d] = bf2f(Q[((long)t * 8 + head) * hd + d]);
  barrier(CLK_LOCAL_MEM_FENCE);
  const int kk = ts + lid;
  const bool valid = kk >= lo && kk <= pos;
  float s = -INFINITY;
  if (valid) {
    global const ushort *kr = KC + ((ring ? kk % ring : kk) * 2L + kvh) * hd;
    float a = 0.0f;
    for (int d = 0; d < hd; d += 8) {
      const float8 kf = as_float8(convert_uint8(vload8(0, kr + d)) << 16), qf = vload8(0, ql + d);
      a = fma(qf.s0, kf.s0, a);
      a = fma(qf.s1, kf.s1, a);
      a = fma(qf.s2, kf.s2, a);
      a = fma(qf.s3, kf.s3, a);
      a = fma(qf.s4, kf.s4, a);
      a = fma(qf.s5, kf.s5, a);
      a = fma(qf.s6, kf.s6, a);
      a = fma(qf.s7, kf.s7, a);
    }
    s = a;
  }
  float m = sub_group_reduce_max(s);
  if (lane == 0) red[sg] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  m = red[0];
  for (int i = 1; i < 8; i++) m = fmax(m, red[i]);
  barrier(CLK_LOCAL_MEM_FENCE);
  const float p = valid ? exp(s - m) : 0.0f;
  sc[lid] = p;
  float l = sub_group_reduce_add(p);
  if (lane == 0) red[sg] = l;
  barrier(CLK_LOCAL_MEM_FENCE);
  l = 0.0f;
  for (int i = 0; i < 8; i++) l += red[i];
  const long idx = ((long)t * 8 + head) * ntl + tl;
  if (lid == 0) {
    PM[idx] = m;
    PL[idx] = l;
  }
  const int k0 = max(lo - ts, 0), k1 = min(pos - ts, KEYS - 1);
  const int nd = HD / 16;
  float acc[HD / 16];
  for (int j = 0; j < nd; j++) acc[j] = 0.0f;
  PV_LOOP(acc)
  for (int j = 0; j < nd; j++) part[sg * HD + j * 16 + lane] = acc[j];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int d = lid; d < hd; d += KEYS) {
    float o = 0.0f;
    for (int i = 0; i < 8; i++) o += part[i * HD + d];
    PO[idx * hd + d] = o;
  }
}

// Folds a row's tile partials in order (m, l, o) and writes the head's output fp16(bf16(o / l)) [T, 8 hd] with group sums.
KERNEL(128) void g4_attn_merge(global const float *PM, global const float *PL, global const float *PO, int p0, int window, int ntl,
                               global half *H, global half *AHI, global half *ALO) {
  const int hd = HD;
  local half hl[HD];
  const int head = get_group_id(0), t = get_group_id(1), lid = get_local_id(0);
  const int pos = p0 + t, lo = window ? max(0, pos - window + 1) : 0;
  const int nt = pos / KEYS - lo / KEYS + 1;
  float acc[HD / 128];
  for (int j = 0; j < HD / 128; j++) acc[j] = 0.0f;
  float m = 0.0f, l = 0.0f;
  for (int i = 0; i < nt; i++) {
    const long idx = ((long)t * 8 + head) * ntl + i;
    const float mi = PM[idx], li = PL[idx];
    if (i == 0) {
      m = mi;
      l = li;
      for (int j = 0; j < hd / KEYS; j++) acc[j] = PO[idx * hd + lid + j * KEYS];
    } else {
      const float mm = fmax(m, mi), a = exp(m - mm), b = exp(mi - mm);
      l = l * a + li * b;
      for (int j = 0; j < hd / KEYS; j++) acc[j] = acc[j] * a + PO[idx * hd + lid + j * KEYS] * b;
      m = mm;
    }
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

// Attention of a prompt row: g4_attn_part over every tile of the row in order, each tile's (m, l, o) folded into the running
// state at once, then g4_attn_merge's output. The same arithmetic as part + merge, so the same bits, without the partials in
// memory. Grid (8, T).
KERNEL(128) void g4_attn_rows(global const ushort *Q, global const ushort *KC, global const ushort *VC, int p0, int window, int ring,
                              global half *H, global half *AHI, global half *ALO) {
  const int hd = HD;
  local float ql[HD], sc[KEYS], red[8], part[8 * HD];
  local half hl[HD];
  const int head = get_group_id(0), t = get_group_id(1), lid = get_local_id(0);
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int pos = p0 + t, lo = window ? max(0, pos - window + 1) : 0;
  const int first = lo / KEYS, nt = pos / KEYS - first + 1;
  const int kvh = head / 4, nd = HD / 16;
  for (int d = lid; d < hd; d += KEYS) ql[d] = bf2f(Q[((long)t * 8 + head) * hd + d]);
  barrier(CLK_LOCAL_MEM_FENCE);
  float acc[HD / 128];
  for (int j = 0; j < HD / 128; j++) acc[j] = 0.0f;
  float mrun = 0.0f, lrun = 0.0f;
  for (int tl = 0; tl < nt; tl++) {
    const int ts = (first + tl) * KEYS;
    const int kk = ts + lid;
    const bool valid = kk >= lo && kk <= pos;
    float s = -INFINITY;
    if (valid) {
      global const ushort *kr = KC + ((ring ? kk % ring : kk) * 2L + kvh) * hd;
      float a = 0.0f;
      for (int d = 0; d < hd; d += 8) {
        const float8 kf = as_float8(convert_uint8(vload8(0, kr + d)) << 16), qf = vload8(0, ql + d);
        a = fma(qf.s0, kf.s0, a);
        a = fma(qf.s1, kf.s1, a);
        a = fma(qf.s2, kf.s2, a);
        a = fma(qf.s3, kf.s3, a);
        a = fma(qf.s4, kf.s4, a);
        a = fma(qf.s5, kf.s5, a);
        a = fma(qf.s6, kf.s6, a);
        a = fma(qf.s7, kf.s7, a);
      }
      s = a;
    }
    float m = sub_group_reduce_max(s);
    if (lane == 0) red[sg] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    m = red[0];
    for (int i = 1; i < 8; i++) m = fmax(m, red[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    const float p = valid ? exp(s - m) : 0.0f;
    sc[lid] = p;
    float l = sub_group_reduce_add(p);
    if (lane == 0) red[sg] = l;
    barrier(CLK_LOCAL_MEM_FENCE);
    l = 0.0f;
    for (int i = 0; i < 8; i++) l += red[i];
    const int k0 = max(lo - ts, 0), k1 = min(pos - ts, KEYS - 1);
    float pv[HD / 16];
    for (int j = 0; j < nd; j++) pv[j] = 0.0f;
    PV_LOOP(pv)
    for (int j = 0; j < nd; j++) part[sg * HD + j * 16 + lane] = pv[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tl == 0) {
      mrun = m;
      lrun = l;
      for (int j = 0; j < HD / 128; j++) {
        float o = 0.0f;
        for (int i = 0; i < 8; i++) o += part[i * HD + lid + j * KEYS];
        acc[j] = o;
      }
    } else {
      const float mm = fmax(mrun, m), a = exp(mrun - mm), b = exp(m - mm);
      lrun = lrun * a + l * b;
      for (int j = 0; j < HD / 128; j++) {
        float o = 0.0f;
        for (int i = 0; i < 8; i++) o += part[i * HD + lid + j * KEYS];
        acc[j] = acc[j] * a + o * b;
      }
      mrun = mm;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int j = 0; j < HD / 128; j++) {
    const half v = convert_half_rte(rbf(acc[j] / lrun));
    hl[lid + j * KEYS] = v;
    H[(long)t * 8 * hd + head * hd + lid + j * KEYS] = v;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int g = sg; g < hd / 64; g += 8) {
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

// Dense projection by DPAS (a bf16 weight converted to fp16 at load: exact for these weights): OUT = A · Wᵀ in fp16, fp32 sums.
// A fp16 [M, K], W fp16 [N, K], K a multiple of 256. Group (rows / 16, N / 16) of 16 subgroups, each summing a slice of K / 16 in
// order; the 16 slices are added in order, so a row's bits depend on neither the other rows nor the tile it sits in.
KERNEL(256) void g4_dense(global const half *A, global const half *W, global half *OUT, int M, int N, int K) {
  local float red[16 * 16 * 16];
  const int KS = K / 16;
  const int m0 = get_group_id(0) * 16, n0 = get_group_id(1) * 16, sg = get_sub_group_id(), lane = get_sub_group_local_id();
  float8 acc0 = 0.0f, acc1 = 0.0f;
  global const uint *wr = (global const uint *)(W + (long)(n0 + lane) * K);
  for (int k = sg * KS; k < (sg + 1) * KS; k += 16) {
    short8 a0, a1;
    intel_sub_group_2d_block_read_16b_8r16x1c((global void *)A, K * 2, M, K * 2, (int2)(k, m0), (private ushort *)&a0);
    intel_sub_group_2d_block_read_16b_8r16x1c((global void *)A, K * 2, M, K * 2, (int2)(k, m0 + 8), (private ushort *)&a1);
    const int8 bb = as_int8(vload8(0, wr + k / 2));
    acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bb, acc0);
    acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bb, acc1);
  }
  for (int r = 0; r < 8; r++) {
    red[(sg * 16 + r) * 16 + lane] = acc0[r];
    red[(sg * 16 + 8 + r) * 16 + lane] = acc1[r];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float v = 0.0f;
  for (int s = 0; s < 16; s++) v += red[(s * 16 + sg) * 16 + lane];
  if (m0 + sg < M) OUT[(long)(m0 + sg) * N + n0 + lane] = convert_half_rte(v);
}
