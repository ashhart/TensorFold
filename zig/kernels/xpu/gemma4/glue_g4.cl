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
                              global ushort *Q, global ushort *KC, global ushort *VC, int p0, float eps, int hd, int ring, int own) {
  const int t = get_group_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  if (sg >= (own ? 12 : 8)) return;
  const int n = hd / 16;
  const float pos = (float)(p0 + t);
  float v[32], ss = 0.0f;
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

// Attention over one tile of 128 keys (absolute positions 128j .. 128j + 127) for one (row, head): m, l and the unnormalized
// o of the tile in fp32, scores = q·k (scale 1), keys limited to [pos - window + 1, pos]. Grid (tiles, 8, T); tile index
// counted from the row's first tile. The cache of the source layer: ring slots pos % ring, or positions.
KERNEL(128) void g4_attn_part(global const ushort *Q, global const ushort *KC, global const ushort *VC, global float *PM, global float *PL,
                              global float *PO, int p0, int hd, int window, int ring, int ntl) {
  local float ql[512], sc[KEYS], red[8];
  const int tl = get_group_id(0), head = get_group_id(1), t = get_group_id(2), lid = get_local_id(0);
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
      const ushort8 kv = vload8(0, kr + d);
      for (int j = 0; j < 8; j++) a = fma(ql[d + j], bf2f(kv[j]), a);
    }
    s = a;
  }
  // max
  float m = sub_group_reduce_max(s);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  m = red[0];
  for (int i = 1; i < 8; i++) m = fmax(m, red[i]);
  barrier(CLK_LOCAL_MEM_FENCE);
  const float p = valid ? exp(s - m) : 0.0f;
  sc[lid] = p;
  float l = sub_group_reduce_add(p);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = l;
  barrier(CLK_LOCAL_MEM_FENCE);
  l = 0.0f;
  for (int i = 0; i < 8; i++) l += red[i];
  const long idx = ((long)t * 8 + head) * ntl + tl;
  if (lid == 0) {
    PM[idx] = m;
    PL[idx] = l;
  }
  const int k0 = max(lo - ts, 0), k1 = min(pos - ts, KEYS - 1);
  for (int d = lid; d < hd; d += KEYS) {
    float acc = 0.0f;
    for (int k = k0; k <= k1; k++) acc = fma(sc[k], bf2f(VC[((ring ? (ts + k) % ring : (ts + k)) * 2L + kvh) * hd + d]), acc);
    PO[idx * hd + d] = acc;
  }
}

// Folds a row's tile partials in order (m, l, o) and writes the head's output fp16(bf16(o / l)) [T, 8 hd] with group sums.
KERNEL(128) void g4_attn_merge(global const float *PM, global const float *PL, global const float *PO, int p0, int hd, int window, int ntl,
                               global half *H, global half *AHI, global half *ALO) {
  local half hl[512];
  const int head = get_group_id(0), t = get_group_id(1), lid = get_local_id(0);
  const int pos = p0 + t, lo = window ? max(0, pos - window + 1) : 0;
  const int nt = pos / KEYS - lo / KEYS + 1;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
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
