// Qwen3.8-27B glue on Intel Xe2 (OpenCL C); x: fp32 residual [T, 5120], h: fp16 GEMM input [T, K].

// group means stored × ASC (and B · group / ASC): the low half stays out of fp16 subnormals
#ifndef ASC
#define ASC 256.0f
#endif

// AHI, ALO: fp16 hi/lo row sums of h per group of GSZ (64 or 128); subgroups of 16, fixed reduction order.
#pragma OPENCL FP_CONTRACT OFF

#define GS 64
#include "shape.h"
#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) {                   // round to nearest even
  return intel_convert_bfloat16_as_ushort(f); // hardware conversion (round to nearest even)
}
static inline float rbf(float f) { return bf2f(f2bf(f)); }
static inline float silu(float x) { return x / (1.0f + exp(-x)); }
static inline float sigmoid_f(float x) { return 1.0f / (1.0f + exp(-x)); }

// Sum over a work-group of 256 in fixed order: subgroups, then the 16 partial sums in order.
static float wg_sum256(float v, local float *red) {
  const float s = sub_group_reduce_add(v);
  const int sg = get_sub_group_id();
  if (get_sub_group_local_id() == 0) red[sg] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (int i = 0; i < 16; i++) t += red[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

static void quant_row(local const half *hl, int K, global char *A8, global float *SA, int MS, global half *AHI,
                      global half *ALO, int t);
static void out_rows(local const half *hl, int K, global half *H, int t, int rot, int gs, global char *A8,
                     global float *SA, int MS, global half *AHI, global half *ALO);

// Row sums of an fp16 row in local memory, written to [row, K / gs]: one subgroup per group of gs, fixed order.
static void rowsums(local const half *hl, int K, global half *AHI, global half *ALO, long row, int gs) {
  const int lane = get_sub_group_local_id(), nsg = get_num_sub_groups(), per = gs / 16;
  for (int g = get_sub_group_id(); g < K / gs; g += nsg) {
    float s = 0.0f;
    for (int k = 0; k < per; k++) s += (float)hl[g * gs + lane * per + k];
    s = sub_group_reduce_add(s) * (1.0f / gs);
    if (lane == 0) {
      const half hi = convert_half_rte(s * ASC);
      AHI[row * (K / gs) + g] = hi;
      ALO[row * (K / gs) + g] = convert_half_rte(s * ASC - (float)hi);
    }
  }
}

// Rotated weights: each 128-block of a GEMM input is Hadamard-rotated (normalized) in registers, fp32 butterflies.
static half8 rot128_reg(half8 x) {
  const int lane = get_sub_group_local_id();
  float v[8];
  for (int j = 0; j < 8; j++) v[j] = (float)x[j];
  for (int m = 1; m < 8; m <<= 1)
    for (int j = 0; j < 8; j++)
      if (!(j & m)) {
        const float a = v[j], b = v[j | m];
        v[j] = a + b;
        v[j | m] = a - b;
      }
  for (int m = 1; m < 16; m <<= 1)
    for (int j = 0; j < 8; j++) {
      const float o = sub_group_shuffle_xor(v[j], m);
      v[j] = (lane & m) ? o - v[j] : v[j] + o;
    }
  half8 r;
  for (int j = 0; j < 8; j++) r[j] = convert_half_rte(v[j] * 0.08838834764831845f);
  return r;
}

// Embedding: x = bf16(q·s + b) (MLX affine 4-bit, group 64; words [V, D/8], nibble j = element 8w + j).
KERNEL(256) void embed(global const int *IDS, global const uint *W, global const ushort *S, global const ushort *B,
                       global float *X, int D) {
  const int t = get_group_id(0);
  const long id = IDS[t];
  for (int e = get_local_id(0); e < D; e += 256) {
    const uint w = W[id * (D / 8) + e / 8];
    const float q = (float)((w >> (4 * (e % 8))) & 0xFu);
    const float s = bf2f(S[id * (D / GS) + e / GS]), b = bf2f(B[id * (D / GS) + e / GS]);
    X[(long)t * D + e] = rbf(q * s + b);
  }
}

// Dense embedding: x = bf16(E[id]) (fp16 table [V, D], GPTQ checkpoint).
KERNEL(256) void embed_dense(global const int *IDS, global const half *E, global float *X, int D) {
  const int t = get_group_id(0);
  const long id = IDS[t];
  for (int e = get_local_id(0); e < D; e += 256) X[(long)t * D + e] = rbf((float)E[id * D + e]);
}

// Small dense fp16 projection (GDN gates a, b), N = 96, K = 5 120, by DPAS; a row's bits do not depend on T.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
KERNEL(256) void dense_rows(global const half *H, global const half *W, global half *P, int T, int ldp, int c0) {
  local float red[16 * 16 * 16];
  const int K = Q27_D, KS = K / 16;
  const int m0 = get_group_id(0) * 16, n0 = get_group_id(1) * 16, sg = get_sub_group_id(), lane = get_sub_group_local_id();
  float8 acc0 = 0.0f, acc1 = 0.0f;
  global const uint *wr = (global const uint *)(W + (long)(n0 + lane) * K);
  for (int k = sg * KS; k < (sg + 1) * KS; k += 16) {
    short8 a0, a1;
    intel_sub_group_2d_block_read_16b_8r16x1c((global void *)H, K * 2, T, K * 2, (int2)(k, m0), (private ushort *)&a0);
    intel_sub_group_2d_block_read_16b_8r16x1c((global void *)H, K * 2, T, K * 2, (int2)(k, m0 + 8), (private ushort *)&a1);
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
  if (m0 + sg < T) P[(long)(m0 + sg) * ldp + c0 + n0 + lane] = convert_half_rte(v);
}

// x ← bf16(x + r) (if R), h = fp16(bf16(rms(x)·w)) and its row sums; one group of 512 per row, fixed order.
KERNEL(128) void add_rmsnorm(global float *X, global const half *R, global const ushort *WN, global half *H,
                              global half *AHI, global half *ALO, int D, float eps, int has_r, int gs,
                              global char *A8, global float *SA, int MS, int rot) {
  // 128 work-items per row, 4 consecutive elements per read; D a multiple of 512, at most 5 120.
  local float red[8];
  local half hl[Q27_D];
  const int t = get_group_id(0), lid = get_local_id(0);
  float4 xv[Q27_D512];
  float ss = 0.0f;
  for (int i = 0; i < D / 512; i++) {
    const int e = (i * 128 + lid) * 4;
    float4 x = vload4(0, X + (long)t * D + e);
    if (has_r) {
      const float4 r = convert_float4(vload4(0, R + (long)t * D + e));
      x = (float4)(rbf(x.s0 + r.s0), rbf(x.s1 + r.s1), rbf(x.s2 + r.s2), rbf(x.s3 + r.s3));
      vstore4(x, 0, X + (long)t * D + e);
    }
    xv[i] = x;
    ss += x.s0 * x.s0 + x.s1 * x.s1 + x.s2 * x.s2 + x.s3 * x.s3;
  }
  const float s = sub_group_reduce_add(ss);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float tot = 0.0f;
  for (int i = 0; i < 8; i++) tot += red[i];
  const float inv = rsqrt(tot / D + eps);
  for (int i = 0; i < D / 512; i++) {
    const int e = (i * 128 + lid) * 4;
    const ushort4 w = vload4(0, WN + e);
    const half4 v = (half4)(convert_half_rte(rbf(xv[i].s0 * inv * bf2f(w.s0))), convert_half_rte(rbf(xv[i].s1 * inv * bf2f(w.s1))),
                            convert_half_rte(rbf(xv[i].s2 * inv * bf2f(w.s2))), convert_half_rte(rbf(xv[i].s3 * inv * bf2f(w.s3))));
    if (!rot) vstore4(v, 0, H + (long)t * D + e);
    vstore4(v, 0, hl + e);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  out_rows(hl, D, H, t, rot, gs, A8, SA, MS, AHI, ALO);
}

// Same computation, 512 work-items per row: for decode and verification (16 rows at most).
KERNEL(512) void add_rmsnorm_dec(global float *X, global const half *R, global const ushort *WN, global half *H,
                                  global half *AHI, global half *ALO, int D, float eps, int has_r, int gs,
                                  global char *A8, global float *SA, int MS, int rot) {
  local float red[32];
  local half hl[Q27_D];
  const int t = get_group_id(0), lid = get_local_id(0);
  float xv[Q27_D512], ss = 0.0f;
  for (int i = 0; i < Q27_D512; i++) {
    const int e = lid + i * 512;
    float x = e < D ? X[(long)t * D + e] : 0.0f;
    if (has_r && e < D) {
      x = rbf(x + (float)R[(long)t * D + e]);
      X[(long)t * D + e] = x;
    }
    xv[i] = x;
    ss += x * x;
  }
  const float s = sub_group_reduce_add(ss);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float tot = 0.0f;
  for (int i = 0; i < 32; i++) tot += red[i];
  const float inv = rsqrt(tot / D + eps);
  for (int i = 0; i < Q27_D512; i++) {
    const int e = lid + i * 512;
    if (e >= D) continue;
    const half v = convert_half_rte(rbf(xv[i] * inv * bf2f(WN[e])));
    if (!rot) H[(long)t * D + e] = v;
    hl[e] = v;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  out_rows(hl, D, H, t, rot, gs, A8, SA, MS, AHI, ALO);
}

// GDN, before the recurrence. P: output of the gdn.in GEMM, fp16 [T, 16 640] = qkv | z | a | b | padding.

// Causal convolution of kernel 4 over qkv with the state, SiLU, bf16 rounding; q, k normalized per head.
#define NQK Q27_NQK                                     // 16 heads × 128
#define NCONV Q27_NCONV
// Grid (T, 40): one group of 256 per block of 256 channels (two heads of 128); block 0 also computes g and β.
KERNEL(256) void gdn_pre(global const half *P, global const half *CIN, global half *COUT, global const ushort *CW,
                         global const ushort *ALOG, global const ushort *DT, global ushort *Q, global ushort *K,
                         global ushort *V, global float *G, global float *BETA, int T, global const int *TREE,
                         global const ulong *MT) {
  // MT (if non-null): several streams, each a chain of rows reading its input state and writing its post-window state.
  local float cv[256];
  const int t = get_group_id(0), blk = get_group_id(1), lid = get_local_id(0), sg = get_sub_group_id();
  const int lane = get_sub_group_local_id(), c = blk * 256 + lid;
  float s = 0.0f;
  if (MT) {
    CIN = (global const half *)MT[t * 8 + 2];
    COUT = (global half *)MT[t * 8 + 3];
  }
  const int wf = MT ? (int)MT[t * 8 + 6] : 0, ww = MT ? (int)MT[t * 8 + 7] : T;
  const int dep = TREE ? TREE[t * 17] & 255 : MT ? t - wf : t;
  for (int j = 0; j < 4; j++) {
    const int dd = dep - 3 + j;                       // depth of the input (negative: the input state)
    const int row = dd < 0 ? 0 : TREE ? TREE[t * 17 + 1 + dd] : MT ? wf + dd : dd;
    const float x = dd < 0 ? (float)CIN[(3 + dd) * NCONV + c] : (float)P[(long)row * Q27_GDN_IN + c];
    s += x * bf2f(CW[c * 4 + j]);
  }
  cv[lid] = rbf(silu(s));
  if (MT && TREE) {
  } else if (MT) {                                     // the window as a block of ww rows
    if (dep >= ww - 3) COUT[(dep - (ww - 3)) * NCONV + c] = P[(long)t * Q27_GDN_IN + c];
    if (ww < 3 && dep == 0)
      for (int r = 0; r < 3 - ww; r++) COUT[r * NCONV + c] = CIN[(r + ww) * NCONV + c];
  } else {
    if (t >= T - 3) COUT[(t - (T - 3)) * NCONV + c] = P[(long)t * Q27_GDN_IN + c];
    if (T < 3 && t == 0)                                // the state keeps part of the old one
      for (int r = 0; r < 3 - T; r++) COUT[r * NCONV + c] = CIN[(r + T) * NCONV + c];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (c < NQK * 2) {                                    // heads q (0-15) and k (16-31): two per block
    if (sg < 2) {
      const int hh = blk * 2 + sg;
      float v[8], ss = 0.0f;
      for (int i = 0; i < 8; i++) {
        v[i] = cv[sg * 128 + i * 16 + lane];
        ss += v[i] * v[i];
      }
      const float inv = rsqrt(sub_group_reduce_add(ss) / 128.0f + 1e-6f);
      const float scale = hh < Q27_GK ? 1.0f / 128.0f : rsqrt(128.0f);
      global ushort *dst = hh < Q27_GK ? Q + ((long)t * Q27_GK + hh) * 128 : K + ((long)t * Q27_GK + hh - Q27_GK) * 128;
      for (int i = 0; i < 8; i++) dst[i * 16 + lane] = f2bf(v[i] * inv * scale);
    }
  } else
    V[(long)t * Q27_VD + c - NQK * 2] = f2bf(cv[lid]);
  if (blk == 0 && lid < Q27_GV) {
    const float a = (float)P[(long)t * Q27_GDN_IN + Q27_GA + lid], b = (float)P[(long)t * Q27_GDN_IN + Q27_GB + lid];
    const float sp = a + bf2f(DT[lid]);
    const float soft = sp > 20.0f ? sp : log1p(exp(sp));
    BETA[(long)t * Q27_GV + lid] = sigmoid_f(b);
    G[(long)t * Q27_GV + lid] = exp(-exp(bf2f(ALOG[lid])) * soft);
  }
}

// One decode row of a GDN layer (no window, no streams): gdn_pre, gdn_step and gated_norm of v head h in one
// work-group, with the same bits as the three kernels. Each work-group computes the convolution of its own channels:
// the q and k of its K head and the v of its head (the first V head of a K head writes the q, k state). Sub-group sg
// then holds rows 8 sg ... 8 sg + 7 of the head's state, as gdn_step. S: the state, read and written in place.
// Grid (Q27_GV); gs: AHI, ALO groups, 16 to 128.
KERNEL(256) void gdn_one(global const half *P, global const half *CIN, global half *COUT, global const ushort *CW,
                         global const ushort *ALOG, global const ushort *DT, global float *S, global const ushort *WN,
                         global half *H, global half *AHI, global half *ALO, float eps, int gs) {
  local float cq[128], ck[128], cv[128], yb[128], gb[2];
  local half hl[128];
  const int h = get_group_id(0), kh = h / Q27_GVK, lid = get_local_id(0), sg = get_sub_group_id();
  const int lane = get_sub_group_local_id();
  for (int idx = lid; idx < 384; idx += 256) {          // the convolution, as gdn_pre for T = 1
    const int part = idx / 128, i = idx % 128;
    const int c = part == 0 ? kh * 128 + i : part == 1 ? NQK + kh * 128 + i : 2 * NQK + h * 128 + i;
    float s = 0.0f;
    for (int j = 0; j < 4; j++) s += (j < 3 ? (float)CIN[j * NCONV + c] : (float)P[c]) * bf2f(CW[c * 4 + j]);
    const float o = rbf(silu(s));
    if (part == 0) cq[i] = o; else if (part == 1) ck[i] = o; else cv[i] = o;
    if (part == 2 || h % Q27_GVK == 0) {                 // the state keeps the last 3 inputs
      COUT[c] = CIN[NCONV + c];
      COUT[NCONV + c] = CIN[2 * NCONV + c];
      COUT[2 * NCONV + c] = P[c];
    }
  }
  if (lid == 0) {
    const float a = (float)P[Q27_GA + h], b = (float)P[Q27_GB + h];
    const float sp = a + bf2f(DT[h]);
    const float soft = sp > 20.0f ? sp : log1p(exp(sp));
    gb[0] = exp(-exp(bf2f(ALOG[h])) * soft);
    gb[1] = sigmoid_f(b);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg < 2) {                                         // l2 norm of q (sub-group 0) and k (1), stored as bf16
    local float *src = sg == 0 ? cq : ck;
    float v[8], ss = 0.0f;
    for (int i = 0; i < 8; i++) {
      v[i] = src[i * 16 + lane];
      ss += v[i] * v[i];
    }
    const float inv = rsqrt(sub_group_reduce_add(ss) / 128.0f + 1e-6f);
    const float scale = sg == 0 ? 1.0f / 128.0f : rsqrt(128.0f);
    for (int i = 0; i < 8; i++) src[i * 16 + lane] = rbf(v[i] * inv * scale);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {                                                     // the step, as gdn_step: rows 8 sg + r
    const int vb = sg;
    float s[8][8], q[8], k[8];
    for (int r = 0; r < 8; r++)
      for (int i = 0; i < 8; i++) s[r][i] = S[((long)h * 128 + vb * 8 + r) * 128 + i * 16 + lane];
    for (int i = 0; i < 8; i++) {
      q[i] = cq[i * 16 + lane];
      k[i] = ck[i * 16 + lane];
    }
    const float g = gb[0], beta = gb[1];
    for (int r = 0; r < 8; r++) {
      float mem = 0.0f;
      for (int i = 0; i < 8; i++) {
        s[r][i] *= g;
        mem += s[r][i] * k[i];
      }
      mem = sub_group_reduce_add(mem);
      const float u = (cv[vb * 8 + r] - mem) * beta;
      float y = 0.0f;
      for (int i = 0; i < 8; i++) {
        s[r][i] += k[i] * u;
        y += s[r][i] * q[i];
      }
      y = sub_group_reduce_add(y);
      if (lane == 0) yb[vb * 8 + r] = rbf(y);
    }
    for (int r = 0; r < 8; r++)
      for (int i = 0; i < 8; i++) S[((long)h * 128 + vb * 8 + r) * 128 + i * 16 + lane] = s[r][i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg == 0) {                                        // gated_norm of the head
    float v[8], ss = 0.0f;
    for (int i = 0; i < 8; i++) {
      v[i] = yb[i * 16 + lane];
      ss += v[i] * v[i];
    }
    const float inv = rsqrt(sub_group_reduce_add(ss) / 128.0f + eps);
    for (int i = 0; i < 8; i++) {
      const int e = i * 16 + lane;
      const float yn = v[i] * inv * bf2f(WN[e]);
      const half o = convert_half_rte(rbf(silu((float)P[Q27_NCONV + h * 128 + e]) * yn));
      H[h * 128 + e] = o;
      hl[e] = o;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int per = gs / 16;                              // rowsums of the head's groups, as out_rows
  for (int g = sg; g < 128 / gs; g += get_num_sub_groups()) {
    float s = 0.0f;
    for (int k = 0; k < per; k++) s += (float)hl[g * gs + lane * per + k];
    s = sub_group_reduce_add(s) * (1.0f / gs);
    if (lane == 0) {
      const half hi = convert_half_rte(s * ASC);
      AHI[h * (128 / gs) + g] = hi;
      ALO[h * (128 / gs) + g] = convert_half_rte(s * ASC - (float)hi);
    }
  }
}

// Same computation, one row per group of 256 (all the token's channels): better for long prompt chunks.
KERNEL(256) void gdn_pre_rows(global const half *P, global const half *CIN, global half *COUT, global const ushort *CW,
                              global const ushort *ALOG, global const ushort *DT, global ushort *Q, global ushort *K,
                              global ushort *V, global float *G, global float *BETA, int T) {
  // Grid (⌈T/8⌉, 5): 8 rows and 16 heads per group; the window slides through registers, same bits as row by row.
  const int t0 = get_group_id(0) * 8, sg = get_sub_group_id(), lane = get_sub_group_local_id(), lid = get_local_id(0);
  const int head = get_group_id(1) * 16 + sg;
  // subgroup block reads and writes: element i of the lane is channel 128 head + 16 i + lane
#define ROW8(r) as_half8((r) < 3 ? intel_sub_group_block_read_us8((global const ushort *)(CIN + (r) * NCONV + head * 128)) \
                                 : intel_sub_group_block_read_us8((global const ushort *)(P + (long)((r) - 3) * Q27_GDN_IN + head * 128)))
  float w[8][4], x[8][3];
  for (int i = 0; i < 8; i++) {
    const int c = head * 128 + i * 16 + lane;
    for (int j = 0; j < 4; j++) w[i][j] = bf2f(CW[c * 4 + j]);
  }
  for (int j = 0; j < 3; j++) {
    const half8 xr = ROW8(t0 + j);
    for (int i = 0; i < 8; i++) x[i][j] = (float)xr[i];
  }
  for (int t = t0; t < min(t0 + 8, T); t++) {
    float v[8], ss = 0.0f;
    const half8 x3r = ROW8(t + 3);
    for (int i = 0; i < 8; i++) {
      const int c = head * 128 + i * 16 + lane;
      const float x3 = (float)x3r[i];
      float s = 0.0f;
      s += x[i][0] * w[i][0];
      s += x[i][1] * w[i][1];
      s += x[i][2] * w[i][2];
      s += x3 * w[i][3];
      x[i][0] = x[i][1];
      x[i][1] = x[i][2];
      x[i][2] = x3;
      v[i] = rbf(silu(s));
      ss += v[i] * v[i];
      if (t >= T - 3) COUT[(t - (T - 3)) * NCONV + c] = P[(long)t * Q27_GDN_IN + c];
      if (T < 3 && t == 0)
        for (int q = 0; q < 3 - T; q++) COUT[q * NCONV + c] = CIN[(q + T) * NCONV + c];
    }
    ushort8 o;
    if (head < 2 * Q27_GK) {
      const float inv = rsqrt(sub_group_reduce_add(ss) / 128.0f + 1e-6f);
      const float scale = head < Q27_GK ? 1.0f / 128.0f : rsqrt(128.0f);
      global ushort *dst = head < Q27_GK ? Q + ((long)t * Q27_GK + head) * 128 : K + ((long)t * Q27_GK + head - Q27_GK) * 128;
      for (int i = 0; i < 8; i++) o[i] = f2bf(v[i] * inv * scale);
      intel_sub_group_block_write_us8(dst, o);
    } else {
      for (int i = 0; i < 8; i++) o[i] = f2bf(v[i]);
      intel_sub_group_block_write_us8(V + (long)t * Q27_VD + (head - 2 * Q27_GK) * 128, o);
    }
    if (get_group_id(1) == 0 && lid < Q27_GV) {
      const float a = (float)P[(long)t * Q27_GDN_IN + Q27_GA + lid], b = (float)P[(long)t * Q27_GDN_IN + Q27_GB + lid];
      const float sp = a + bf2f(DT[lid]);
      const float soft = sp > 20.0f ? sp : log1p(exp(sp));
      BETA[(long)t * Q27_GV + lid] = sigmoid_f(b);
      G[(long)t * Q27_GV + lid] = exp(-exp(bf2f(ALOG[lid])) * soft);
    }
  }
}
#undef ROW8

// After the recurrence: out = fp16(bf16(SiLU(z)·rms(y)·w)) per head of 128 and its row sums, or (A8) its quantization.
KERNEL(256) void gated_norm(global const ushort *Y, global const half *P, global const ushort *WN, global half *H,
                            global half *AHI, global half *ALO, float eps, int gs, int rot, global char *A8,
                            global float *SA, int MS) {
  local half hl[Q27_VD];
  const int t = get_group_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const ushort8 wn = intel_sub_group_block_read_us8(WN);   // element i of the lane: 16 i + lane (block reads)
  for (int hh = sg; hh < Q27_GV; hh += 16) {
    const ushort8 yv = intel_sub_group_block_read_us8(Y + ((long)t * Q27_GV + hh) * 128);
    const half8 zv = as_half8(intel_sub_group_block_read_us8((global const ushort *)(P + (long)t * Q27_GDN_IN + Q27_NCONV + hh * 128)));
    float v[8], ss = 0.0f;
    for (int i = 0; i < 8; i++) {
      v[i] = bf2f(yv[i]);
      ss += v[i] * v[i];
    }
    const float inv = rsqrt(sub_group_reduce_add(ss) / 128.0f + eps);
    half8 o;
    for (int i = 0; i < 8; i++) {
      const float yn = v[i] * inv * bf2f(wn[i]);
      o[i] = convert_half_rte(rbf(silu((float)zv[i]) * yn));
      hl[hh * 128 + i * 16 + lane] = o[i];
    }
    if (!rot) intel_sub_group_block_write_us8((global ushort *)(H + (long)t * Q27_VD + hh * 128), as_ushort8(o));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  out_rows(hl, Q27_VD, H, t, rot, gs, A8, SA, MS, AHI, ALO);
}

#include "glue_attn.cl"
#include "glue_quant.cl"
