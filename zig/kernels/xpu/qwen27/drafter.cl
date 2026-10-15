// DFlash2 drafter for Qwen3.8-27B on Intel Xe2: 5 layers, a block of L rows attending its context window and itself.
// group means stored × ASC (and B · group / ASC): the low half stays out of fp16 subnormals
#ifndef ASC
#define ASC 256.0f
#endif
#pragma OPENCL FP_CONTRACT OFF

#define GS 64
#define D 5120
#define HD 128
#define NH 32
#define NKV 8
#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) {
  return intel_convert_bfloat16_as_ushort(f); // hardware conversion (round to nearest even)
}
static inline float rbf(float f) { return bf2f(f2bf(f)); }

// Row sums of a fp16 row held in local memory (K values): one subgroup per group of 64.
static void rowsums(local const half *hl, int K, global half *AHI, global half *ALO, long row) {
  const int lane = get_sub_group_local_id(), nsg = get_num_sub_groups();
  for (int g = get_sub_group_id(); g < K / GS; g += nsg) {
    float s = 0.0f;
    for (int k = 0; k < 4; k++) s += (float)hl[g * GS + lane * 4 + k];
    s = sub_group_reduce_add(s) * (1.0f / GS);
    if (lane == 0) {
      const half hi = convert_half_rte(s * ASC);
      AHI[row * (K / GS) + g] = hi;
      ALO[row * (K / GS) + g] = convert_half_rte(s * ASC - (float)hi);
    }
  }
}

// Two-tap grouped dynamic causal convolution in the block, y[r] = bf16(x[r]*k0 + x[r-1]*k1); RES adds to the residual.
KERNEL(512) void dconv(global const half *X, global const half *DYN, int DS, global const ushort *BASE, int branch,
                       global float *RES, int has_res, global half *H, global half *AHI, global half *ALO, int L) {
  local half hl[512];
  const int r = get_group_id(0), blk = get_group_id(1), c = blk * 512 + get_local_id(0), g = c / 16;
  const float x = (float)X[(long)r * D + c], prev = r % L > 0 ? (float)X[(long)(r - 1) * D + c] : 0.0f;
  const float d0 = (float)DYN[(long)r * DS + (branch * 2) * 320 + g], d1 = (float)DYN[(long)r * DS + (branch * 2 + 1) * 320 + g];
  const float k0 = rbf(bf2f(BASE[(branch * 2) * D + c]) + d0), k1 = rbf(bf2f(BASE[(branch * 2 + 1) * D + c]) + d1);
  const float y = rbf(x * k0 + prev * k1);
  if (has_res) {
    RES[(long)r * D + c] = rbf(RES[(long)r * D + c] + y);
    return;
  }
  const half o = convert_half_rte(y);
  H[(long)r * D + c] = o;
  hl[get_local_id(0)] = o;
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
  if (sg < 8) {
    float s = 0.0f;
    for (int k = 0; k < 4; k++) s += (float)hl[sg * GS + lane * 4 + k];
    s = sub_group_reduce_add(s) * (1.0f / GS);
    if (lane == 0) {
      const half hi = convert_half_rte(s * ASC);
      AHI[(long)r * (D / GS) + blk * 8 + sg] = hi;
      ALO[(long)r * (D / GS) + blk * 8 + sg] = convert_half_rte(s * ASC - (float)hi);
    }
  }
}

// q, k: per-head RMS norm with weight, then rotate-half RoPE, all in bf16; v is copied; K, V go to the ring caches.
KERNEL(16) void dprep(global const half *P, int PS, int qoff, global const ushort *QN, global const ushort *KN,
                      global const float *INVF, global ushort *Q, global ushort *KC, global ushort *VC, int CAP, int pos0,
                      float eps, global const int4 *ROWS, global const ulong *PTRS) {
  const int r = get_group_id(0), lane = get_sub_group_local_id();
  const int hh = get_group_id(1) + (qoff < 0 ? NH : 0);
  const int kv0 = qoff < 0 ? 0 : NH * HD;            // column of k in P
  int at = pos0 + r;
  if (ROWS) {                                         // several sequences: the row's position and its caches
    const int4 ri = ROWS[r];
    at = ri.x;
    KC = (global ushort *)PTRS[ri.w * 2];
    VC = (global ushort *)PTRS[ri.w * 2 + 1];
  }
  const float pos = (float)at;
  if (hh >= NH + NKV) {                              // v
    const int h = hh - NH - NKV;
    for (int i = 0; i < 8; i++)
      VC[((long)h * CAP + at % CAP) * HD + i * 16 + lane] = f2bf((float)P[(long)r * PS + kv0 + NKV * HD + h * HD + i * 16 + lane]);
    return;
  }
  const bool isq = hh < NH;
  const long src = (long)r * PS + (isq ? hh * HD : kv0 + (hh - NH) * HD);
  float v[8], ss = 0.0f;
  for (int i = 0; i < 8; i++) {
    v[i] = (float)P[src + i * 16 + lane];
    ss += v[i] * v[i];
  }
  const float inv = rsqrt(sub_group_reduce_add(ss) / HD + eps);
  for (int i = 0; i < 8; i++) v[i] = rbf(v[i] * inv * bf2f((isq ? QN : KN)[i * 16 + lane]));
  for (int i = 0; i < 4; i++) {                      // halves: dims 0-63 (v[0..3]) and 64-127 (v[4..7])
    const float ang = pos * INVF[i * 16 + lane], c = cos(ang), s = sin(ang);
    const float a = v[i], b = v[i + 4];
    v[i] = a * c - b * s;
    v[i + 4] = b * c + a * s;
  }
  global ushort *dst = isq ? Q + ((long)r * NH + hh) * HD : KC + ((long)(hh - NH) * CAP + at % CAP) * HD;
  for (int i = 0; i < 8; i++) dst[i * 16 + lane] = f2bf(v[i]);
}

// Block attention (K, V in a ring, CAP > WIN + L): a row sees its context window and every block row (not causal).
KERNEL(16) void dattn(global const ushort *Q, global const ushort *KC, global const ushort *VC, int CAP, int c0,
                      int pos0, int L, int WIN, float scale, global half *O, global const int4 *ROWS,
                      global const ulong *PTRS) {
  const int r = get_group_id(0), h = get_group_id(1), lane = get_sub_group_local_id();
  int p = pos0 + r, bs = pos0;
  if (ROWS) {                                         // several blocks: the row's position, context start, block start
    const int4 ri = ROWS[r];
    p = ri.x;
    c0 = ri.y;
    bs = ri.z;
    KC = (global const ushort *)PTRS[ri.w * 2];
    VC = (global const ushort *)PTRS[ri.w * 2 + 1];
  }
  float q[4][8], o[4][8], m[4], l[4];
  for (int a = 0; a < 4; a++) {
    for (int i = 0; i < 8; i++) {
      q[a][i] = bf2f(Q[((long)r * NH + h * 4 + a) * HD + i * 16 + lane]);
      o[a][i] = 0.0f;
    }
    m[a] = -INFINITY;
    l[a] = 0.0f;
  }
  const int first = max(c0, p - WIN);
  for (int j = first; j < bs + L; j++) {
    float kv[8];
    const long jr = (long)h * CAP + j % CAP;          // the cache is a ring of CAP positions
    for (int i = 0; i < 8; i++) kv[i] = bf2f(KC[jr * HD + i * 16 + lane]);
    float sc[4];
    for (int a = 0; a < 4; a++) {
      float d = 0.0f;
      for (int i = 0; i < 8; i++) d += q[a][i] * kv[i];
      sc[a] = sub_group_reduce_add(d) * scale;
    }
    for (int i = 0; i < 8; i++) kv[i] = bf2f(VC[jr * HD + i * 16 + lane]);
    for (int a = 0; a < 4; a++) {
      const float mn = fmax(m[a], sc[a]);
      const float al = exp(m[a] - mn), p = exp(sc[a] - mn);
      l[a] = l[a] * al + p;
      for (int i = 0; i < 8; i++) o[a][i] = o[a][i] * al + p * kv[i];
      m[a] = mn;
    }
  }
  for (int a = 0; a < 4; a++)
    for (int i = 0; i < 8; i++) O[(long)r * NH * HD + (h * 4 + a) * HD + i * 16 + lane] = convert_half_rte(rbf(o[a][i] / l[a]));
}

// Row sums for a fp16 matrix already written (o_proj's input after dattn): grid L, groups of 256.
KERNEL(256) void rowsums_rows(global const half *H, int K, global half *AHI, global half *ALO) {
  local half hl[25600];
  const int r = get_group_id(0);
  for (int e = get_local_id(0); e < K; e += 256) hl[e] = H[(long)r * K + e];
  barrier(CLK_LOCAL_MEM_FENCE);
  rowsums(hl, K, AHI, ALO, r);
}

// Target taps bf16 [n, 25 600] to fp16 with row sums (fc's input). Grid n, groups of 256.
KERNEL(256) void taps_in(global const ushort *TAPS, global half *H, global half *AHI, global half *ALO) {
  local half hl[25600];
  const int r = get_group_id(0);
  for (int e = get_local_id(0); e < 25600; e += 256) {
    const half v = convert_half_rte(bf2f(TAPS[(long)r * 25600 + e]));
    H[(long)r * 25600 + e] = v;
    hl[e] = v;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  rowsums(hl, 25600, AHI, ALO, r);
}

// fp16 [n, D] to the fp32 residual stream (rounded to bf16): the drafter's x after fc's norm or the embedding.
KERNEL(256) void to_res(global const half *H, global float *X) {
  const long i = get_global_id(0);
  X[i] = rbf((float)H[i]);
}

// Top 16 of each row of logits in two passes: 256 groups per row keep their top 16, the host merges; ties: lower index.
KERNEL(16) void topk_part(global const half *L, int N, int NS, global float *PV, global int *PI) {
  const int gid = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id();
  const int per = (N + 255) / 256, a = gid * per, b = min(N, a + per);
  float bv[16];
  int bi[16];
  for (int i = 0; i < 16; i++) { bv[i] = -INFINITY; bi[i] = 0x7fffffff; }
  if (lane == 0)
    for (int j = a; j < b; j++) {
      const float v = (float)L[(long)row * NS + j];
      if (v <= bv[15]) continue;
      int k = 15;
      while (k > 0 && v > bv[k - 1]) { bv[k] = bv[k - 1]; bi[k] = bi[k - 1]; k--; }
      bv[k] = v;
      bi[k] = j;
    }
  if (lane == 0)
    for (int i = 0; i < 16; i++) {
      PV[((long)row * 256 + gid) * 16 + i] = bv[i];
      PI[((long)row * 256 + gid) * 16 + i] = bi[i];
    }
}

// Affine 4-bit quantization of a bf16 weight in groups of 64 along K (MLX layout), one work-item per (row, group).
kernel void quantize4(global const ushort *W, int N, int K, global uint *MW, global ushort *MS, global ushort *MB) {
  const int n = get_global_id(0), g = get_global_id(1);
  if (n >= N || g >= K / GS) return;
  float lo = INFINITY, hi = -INFINITY;
  for (int i = 0; i < GS; i++) {
    const float v = bf2f(W[(long)n * K + g * GS + i]);
    lo = fmin(lo, v);
    hi = fmax(hi, v);
  }
  const float scale = rbf(fmax((hi - lo) / 15.0f, 1e-8f)), bias = rbf(lo);
  MS[(long)n * (K / GS) + g] = f2bf(scale);
  MB[(long)n * (K / GS) + g] = f2bf(bias);
  for (int w = 0; w < GS / 8; w++) {
    uint word = 0;
    for (int j = 0; j < 8; j++) {
      const float v = bf2f(W[(long)n * K + g * GS + w * 8 + j]);
      const int q = clamp((int)rint((v - bias) / scale), 0, 15);
      word |= (uint)q << (4 * j);
    }
    MW[(long)n * (K / 8) + g * (GS / 8) + w] = word;
  }
}
