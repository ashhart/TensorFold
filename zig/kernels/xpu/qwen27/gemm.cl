// Weight-only int4 GEMM in OpenCL C for Xe2, compiled per weight format (-DGSZ=64 affine MLX, -DGSZ=128 symmetric GPTQ)

// group means stored × ASC (and B · group / ASC): the low half stays out of fp16 subnormals
#ifndef ASC
#define ASC 256.0f
#endif

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef GSZ
#define GSZ 64
#endif
#define SUBS (GSZ / 64)   // 64-K blocks per group

#if GSZ == 64
// --- fp16 dequantization (g64 only): w = fma(q, s, b) in fp16, a single rounding; the reference for the bits.
inline int pair(uint w, int p, half s, half b) {
  half2 h = as_half2(((w >> (4 * p)) & 0x000F000Fu) | 0x64006400u);  // 1024 + q, exact
  h = h - (half2)(1024.0h);                                             // q, exact
  return as_int(fma(h, (half2)(s), (half2)(b)));
}

#define W4_KERNEL(NAME, SG_M, SG_N, WG_M, WG_N)                                                                     \
  __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * WG_M * WG_N, 1, 1)))      \
  kernel void NAME(global const half *A, global const uint *W, global const half *S, global const half *B,          \
                   global half *C, int M, int N, int K) {                                                            \
    const int sg = get_sub_group_id();                                                                               \
    const int lane = get_sub_group_local_id();                                                                       \
    const int m0 = get_group_id(0) * (WG_M * SG_M) + (sg % WG_M) * SG_M;                                             \
    const int n0 = get_group_id(1) * (WG_N * SG_N) + (sg / WG_M) * SG_N;                                             \
    float8 acc[SG_M / 8][SG_N / 16];                                                                                 \
    for (int i = 0; i < SG_M / 8; i++)                                                                               \
      for (int j = 0; j < SG_N / 16; j++) acc[i][j] = 0.0f;                                                          \
    for (int g = 0; g < K / 64; g++) {                                                                               \
      uint w[SG_N / 16][8];                                                                                          \
      half s[SG_N / 16], b[SG_N / 16];                                                                               \
      for (int j = 0; j < SG_N / 16; j++) {                                                                          \
        intel_sub_group_2d_block_read_32b_8r16x1c((global void *)W, N * 4, K / 8, N * 4, (int2)(n0 + 16 * j, g * 8), w[j]); \
        s[j] = S[g * N + n0 + 16 * j + lane];                                                                        \
        b[j] = B[g * N + n0 + 16 * j + lane];                                                                        \
      }                                                                                                              \
      for (int t = 0; t < 4; t++) {                                                                                  \
        short8 a[SG_M / 8];                                                                                          \
        for (int i = 0; i < SG_M / 8; i++)                                                                           \
          intel_sub_group_2d_block_read_16b_8r16x1c((global void *)A, K * 2, M, K * 2, (int2)(g * 64 + t * 16, m0 + 8 * i), \
                                                    (private ushort *)&a[i]);                                        \
        for (int j = 0; j < SG_N / 16; j++) {                                                                        \
          const uint w0 = w[j][2 * t], w1 = w[j][2 * t + 1];                                                         \
          int8 bb = (int8)(pair(w0, 0, s[j], b[j]), pair(w0, 1, s[j], b[j]), pair(w0, 2, s[j], b[j]),               \
                           pair(w0, 3, s[j], b[j]), pair(w1, 0, s[j], b[j]), pair(w1, 1, s[j], b[j]),               \
                           pair(w1, 2, s[j], b[j]), pair(w1, 3, s[j], b[j]));                                        \
          for (int i = 0; i < SG_M / 8; i++) acc[i][j] = intel_sub_group_f16_f16_matrix_mad_k16(a[i], bb, acc[i][j]); \
        }                                                                                                            \
      }                                                                                                              \
    }                                                                                                                \
    for (int i = 0; i < SG_M / 8; i++)                                                                               \
      for (int j = 0; j < SG_N / 16; j++) {                                                                          \
        half8 h = convert_half8_rte(acc[i][j]);                                                                      \
        intel_sub_group_2d_block_write_16b_8r16x1c(C, N * 2, M, N * 2, (int2)(n0 + 16 * j, m0 + 8 * i), (private ushort *)&h); \
      }                                                                                                              \
  }

W4_KERNEL(w4_decode, 16, 32, 1, 4)
W4_KERNEL(w4_prompt, 32, 64, 4, 2)
#endif

// --- Scaling after DPAS: sum a*(q*s + b) = s * sum a*q + b * sum a, with q fed to the DPAS as a subnormal fp16 q*2^-24

// Asum[g * Mp + m] = sum over group g of A[m, k] in fp32, increasing k; Mp is M rounded up to a multiple of 8.
kernel void rowsum(global const half *A, global float *Asum, int M, int Mp, int K) {
  const int m = get_global_id(0), g = get_global_id(1);
  if (m >= Mp) return;
  float s = 0.0f;
  if (m < M)
    for (int k = 0; k < GSZ; k++) s += (float)A[(long)m * K + g * GSZ + k];
  Asum[g * Mp + m] = s;
}

// Common body: one group g, DPAS partial sum, then acc += s*part + b*sum a; R rows per A read, PF prefetch.
#define GROUP_STEP(SG_M, SG_N, R, READ_A, PF, g, acc) GROUP_STEP_B(SG_M, SG_N, R, READ_A, PF, 1, g, acc)
// BIAS = 0: without the b·Σa term (s4dpb family, bias by DPAS at the end of the kernel).
#define GROUP_STEP_B(SG_M, SG_N, R, READ_A, PF, BIAS, g, acc)                                                                \
  {                                                                                                                  \
    float s[SG_N / 16], b[SG_N / 16];                                                                                \
    float8 part[SG_M / 8][SG_N / 16];                                                                                \
    for (int u = 0; u < SUBS; u++) {                                                                                 \
      const int kb = (g) * SUBS + u;                                                                                 \
      if (PF > 0 && kb + PF < K / 64) {                                                                              \
        for (int j = 0; j < SG_N / 16; j++)                                                                          \
          intel_sub_group_2d_block_prefetch_32b_8r16x1c((global void *)W, N * 4, K / 8, N * 4, (int2)(n0 + 16 * j, (kb + PF) * 8)); \
        for (int r = 0; r < SG_M / R; r++)                                                                           \
          for (int h = 0; h < 2; h++)                                                                                \
            intel_sub_group_2d_block_prefetch_16b_8r16x2c((global void *)A, K * 2, M, K * 2, (int2)((kb + PF) * 64 + 32 * h, m0 + R * r)); \
      }                                                                                                              \
      uint w[SG_N / 16][8];                                                                                           \
      for (int j = 0; j < SG_N / 16; j++) {                                                                          \
        intel_sub_group_2d_block_read_32b_8r16x1c((global void *)W, N * 4, K / 8, N * 4, (int2)(n0 + 16 * j, kb * 8), w[j]); \
        if (u == 0) {                                                                                                \
          s[j] = (float)S[(g) * N + n0 + 16 * j + lane] * 16777216.0f;                                               \
          if (BIAS) b[j] = (float)B[(g) * N + n0 + 16 * j + lane];                                                   \
        }                                                                                                            \
      }                                                                                                              \
      if (u == 0)                                                                                                    \
        for (int i = 0; i < SG_M / 8; i++)                                                                           \
          for (int j = 0; j < SG_N / 16; j++) part[i][j] = 0.0f;                                                     \
      for (int t2 = 0; t2 < 2; t2++) {                                                                               \
        ushort ab[SG_M / R][2 * R];                                                                                  \
        for (int r = 0; r < SG_M / R; r++)                                                                           \
          READ_A((global void *)A, K * 2, M, K * 2, (int2)(kb * 64 + t2 * 32, m0 + R * r), ab[r]);                  \
        for (int h = 0; h < 2; h++) {                                                                                \
          const int t = 2 * t2 + h;                                                                                  \
          for (int j = 0; j < SG_N / 16; j++) {                                                                      \
            const uint w0 = w[j][2 * t], w1 = w[j][2 * t + 1];                                                       \
            int8 bb = as_int8((uint8)(w0, w0 >> 4, w0 >> 8, w0 >> 12, w1, w1 >> 4, w1 >> 8, w1 >> 12) & 0x000F000Fu); \
            for (int i = 0; i < SG_M / 8; i++) {                                                                     \
              const short8 a = as_short8(vload8(0, &ab[(8 * i) / R][h * R + (8 * i) % R]));                          \
              part[i][j] = intel_sub_group_f16_f16_matrix_mad_k16(a, bb, part[i][j]);                                \
            }                                                                                                        \
          }                                                                                                          \
        }                                                                                                            \
      }                                                                                                              \
    }                                                                                                                \
    for (int i = 0; i < SG_M / 8; i++) {                                                                             \
      for (int j = 0; j < SG_N / 16; j++) acc[i][j] = fma(part[i][j], (float8)(s[j]), acc[i][j]);                   \
      if (BIAS) {                                                                                                    \
        const float8 as = vload8(0, Asum + (g) * Mp + m0 + 8 * i);                                                   \
        for (int j = 0; j < SG_N / 16; j++) acc[i][j] = fma(as, (float8)(b[j]), acc[i][j]);                         \
      }                                                                                                              \
    }                                                                                                                \
  }

// Order of the prompt work-groups: packets of GM row tiles, all columns of a packet first, so its A rows stay in L2.
#define GM 8
#define TILE_ORDER(SWIZZLE)                                                                                          \
  int mt = get_group_id(0), nt = get_group_id(1);                                                                    \
  if (SWIZZLE) {                                                                                                     \
    const int nm = get_num_groups(0), nn = get_num_groups(1);                                                        \
    const int pid = get_group_id(1) * nm + get_group_id(0);                                                          \
    const int first = pid / (GM * nn) * GM, size = min(nm - first, GM);                                              \
    mt = first + (pid % (GM * nn)) % size;                                                                           \
    nt = (pid % (GM * nn)) / size;                                                                                   \
  }

#define STORE_HALF(acc)                                                                                              \
  for (int i = 0; i < SG_M / 8; i++)                                                                                 \
    for (int j = 0; j < SG_N / 16; j++) {                                                                            \
      half8 h = convert_half8_rte(acc[i][j]);                                                                        \
      intel_sub_group_2d_block_write_16b_8r16x1c(C, N * 2, M, N * 2, (int2)(n0 + 16 * j, m0 + 8 * i), (private ushort *)&h); \
    }

#define KERNEL_HEAD(NAME, SG_M_, SG_N_, WG_M, WG_N, OUT_T)                                                           \
  __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * WG_M * WG_N, 1, 1)))      \
  kernel void NAME(global const half *A, global const uint *W, global const half *S, global const half *B,          \
                   global OUT_T *C, int M, int N, int K, global const float *Asum, int Mp)

// Without slices (wide shapes): groups in order, first to last.
#define W4N_KERNEL(NAME, SG_M_, SG_N_, WG_M, WG_N, R, READ_A, PF, SWIZZLE)                                           \
  KERNEL_HEAD(NAME, SG_M_, SG_N_, WG_M, WG_N, half) {                                                                \
    enum { SG_M = SG_M_, SG_N = SG_N_ };                                                                             \
    TILE_ORDER(SWIZZLE)                                                                                              \
    const int sg = get_sub_group_id(), lane = get_sub_group_local_id();                                             \
    const int m0 = mt * (WG_M * SG_M) + (sg % WG_M) * SG_M;                                                          \
    const int n0 = nt * (WG_N * SG_N) + (sg / WG_M) * SG_N;                                                          \
    float8 acc[SG_M / 8][SG_N / 16];                                                                                 \
    for (int i = 0; i < SG_M / 8; i++)                                                                               \
      for (int j = 0; j < SG_N / 16; j++) acc[i][j] = 0.0f;                                                          \
    for (int g = 0; g < K / GSZ; g++) GROUP_STEP(SG_M, SG_N, R, READ_A, PF, g, acc)                                  \
    STORE_HALF(acc)                                                                                                  \
  }

// With slices (narrow shapes): K in fixed slices summed from 0, total = ((t0 + t1) + t2) + t3, same bits either way.
#define SLICES 4
#define W4S3_KERNEL(NAME, SG_M_, SG_N_, WG_M, WG_N, R, READ_A, PF, SWIZZLE)                                          \
  KERNEL_HEAD(NAME, SG_M_, SG_N_, WG_M, WG_N, half) {                                                                \
    enum { SG_M = SG_M_, SG_N = SG_N_ };                                                                             \
    TILE_ORDER(SWIZZLE)                                                                                              \
    const int sg = get_sub_group_id(), lane = get_sub_group_local_id();                                             \
    const int m0 = mt * (WG_M * SG_M) + (sg % WG_M) * SG_M;                                                          \
    const int n0 = nt * (WG_N * SG_N) + (sg / WG_M) * SG_N;                                                          \
    const int per = K / GSZ / SLICES;                                                                                \
    float8 total[SG_M / 8][SG_N / 16];                                                                               \
    for (int i = 0; i < SG_M / 8; i++)                                                                               \
      for (int j = 0; j < SG_N / 16; j++) total[i][j] = 0.0f;                                                        \
    for (int sl = 0; sl < SLICES; sl++) {                                                                            \
      float8 acc[SG_M / 8][SG_N / 16];                                                                               \
      for (int i = 0; i < SG_M / 8; i++)                                                                             \
        for (int j = 0; j < SG_N / 16; j++) acc[i][j] = 0.0f;                                                        \
      for (int g = sl * per; g < (sl + 1) * per; g++) GROUP_STEP(SG_M, SG_N, R, READ_A, PF, g, acc)                  \
      for (int i = 0; i < SG_M / 8; i++)                                                                             \
        for (int j = 0; j < SG_N / 16; j++) total[i][j] += acc[i][j];                                                \
    }                                                                                                                \
    STORE_HALF(total)                                                                                                \
  }

// One slice (get_group_id(2)) per work-group, in fp32 in P[slice][Mp][N]; then w4s3_reduce.
#define W4S3_SPLIT_KERNEL(NAME, SG_M_, SG_N_, WG_M, WG_N, R, READ_A)                                                 \
  KERNEL_HEAD(NAME, SG_M_, SG_N_, WG_M, WG_N, float) {                                                               \
    enum { SG_M = SG_M_, SG_N = SG_N_ };                                                                             \
    const int sg = get_sub_group_id(), lane = get_sub_group_local_id();                                             \
    const int m0 = get_group_id(0) * (WG_M * SG_M) + (sg % WG_M) * SG_M;                                             \
    const int n0 = get_group_id(1) * (WG_N * SG_N) + (sg / WG_M) * SG_N;                                             \
    const int per = K / GSZ / SLICES, sl = get_group_id(2);                                                          \
    float8 acc[SG_M / 8][SG_N / 16];                                                                                 \
    for (int i = 0; i < SG_M / 8; i++)                                                                               \
      for (int j = 0; j < SG_N / 16; j++) acc[i][j] = 0.0f;                                                          \
    for (int g = sl * per; g < (sl + 1) * per; g++) GROUP_STEP(SG_M, SG_N, R, READ_A, 2, g, acc)                     \
    for (int i = 0; i < SG_M / 8; i++)                                                                               \
      for (int j = 0; j < SG_N / 16; j++) {                                                                          \
        float8 f = acc[i][j];                                                                                        \
        intel_sub_group_2d_block_write_32b_8r16x1c(C + (long)sl * Mp * N, N * 4, Mp, N * 4,                          \
                                                   (int2)(n0 + 16 * j, m0 + 8 * i), (private uint *)&f);             \
      }                                                                                                              \
  }

// C = fp16(((P0 + P1) + P2) + P3), the order of W4S3_KERNEL.
kernel void w4s3_reduce(global const float *P, global half *C, int M, int Mp, int N) {
  const int n = get_global_id(0), m = get_global_id(1);
  if (m >= M) return;
  float t = 0.0f;
  for (int s = 0; s < SLICES; s++) t += P[((long)s * Mp + m) * N + n];
  C[(long)m * N + n] = convert_half_rte(t);
}

#define READ32 intel_sub_group_2d_block_read_16b_32r16x2c
#define READ16 intel_sub_group_2d_block_read_16b_16r16x2c

// Wide shapes (large N): without slices.
W4N_KERNEL(w4n_decode, 16, 32, 1, 4, 16, READ16, 4, 0)
W4N_KERNEL(w4n_prompt, 32, 16, 2, 8, 32, READ32, 1, 1)
// Decode variants (same bit family as w4n_decode), swept by zig/dec.zig.
W4N_KERNEL(w4n_d16, 16, 16, 1, 4, 16, READ16, 4, 0)
W4N_KERNEL(w4n_d64, 16, 64, 1, 4, 16, READ16, 4, 0)
W4N_KERNEL(w4n_d32w8, 16, 32, 1, 8, 16, READ16, 4, 0)
W4N_KERNEL(w4n_d32pf8, 16, 32, 1, 4, 16, READ16, 8, 0)
W4N_KERNEL(w4n_d32pf2, 16, 32, 1, 4, 16, READ16, 2, 0)
W4N_KERNEL(w4n_v32, 32, 32, 1, 4, 32, READ32, 2, 0)    // draft verification: 32 rows per subgroup
// Narrow shapes (small N, large K): with slices; decode is split into 4 launches + w4s3_reduce.
W4S3_KERNEL(w4s3_prompt, 32, 16, 2, 8, 32, READ32, 1, 1)
W4S3_SPLIT_KERNEL(w4s3_dsplit, 16, 32, 1, 4, 16, READ16)

// --- Bias by DPAS (dpb family): sum_g Asum[g, m]*b[g, n] as a small GEMM of depth K / GSZ, Asum split in hi/lo fp16.
kernel void rowsum_hl(global const half *A, global half *Ahi, global half *Alo, int M, int Mp, int K, int G) {
  const int m = get_global_id(0), g = get_global_id(1);
  if (m >= Mp) return;
  float s = 0.0f;
  if (m < M)
    for (int k = 0; k < GSZ; k++) s += (float)A[(long)m * K + g * GSZ + k];
  s *= 1.0f / GSZ;
  const half hi = convert_half_rte(s * ASC);
  Ahi[(long)m * G + g] = hi;
  Alo[(long)m * G + g] = convert_half_rte(s * ASC - (float)hi);
}

// --- s4dpb family: fixed K slices summed from 0, then the bias by DPAS on the total; same bits for any partitioning.
#define BIAS_TILE(tot, i, j)                                                                                         \
  for (int st = 0; st < (G + 15) / 16; st++) {                                                                       \
    ushort8 ah, al;                                                                                                  \
    uint8 bb;                                                                                                        \
    intel_sub_group_2d_block_read_16b_8r16x1c((global void *)Ahi, G * 2, M, G * 2, (int2)(16 * st, m0 + 8 * (i)), (private ushort *)&ah); \
    intel_sub_group_2d_block_read_16b_8r16x1c((global void *)Alo, G * 2, M, G * 2, (int2)(16 * st, m0 + 8 * (i)), (private ushort *)&al); \
    intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)Bg, N * 2, G, N * 2, (int2)(n0 + 16 * (j), 16 * st), (private uint *)&bb); \
    tot = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(ah), as_int8(bb), tot);                                  \
    tot = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(al), as_int8(bb), tot);                                  \
  }

#define KERNEL_HEAD_K(NAME, LOCAL)                                                                                   \
  __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(LOCAL, 1, 1)))                 \
  kernel void NAME(global const half *A, global const uint *W, global const half *S, global const ushort *Bg,       \
                   global half *C, int M, int N, int K, global const ushort *Ahi, global const ushort *Alo, int Mp, \
                   int G)

// Slices in registers, one after the other (wide shapes).
#define W4K_KERNEL(NAME, SG_M_, SG_N_, WG_N, R, READ_A, PF) W4K_KERNEL_SL(NAME, SG_M_, SG_N_, WG_N, R, READ_A, PF, SLICES)
// SL = 1: dpb family (no slices), the wide-shape one; same bits as w4g_dpb (nGEN).
#define W4K_KERNEL_SL(NAME, SG_M_, SG_N_, WG_N, R, READ_A, PF, SL)                                                   \
  KERNEL_HEAD_K(NAME, 16 * WG_N) {                                                                                   \
    enum { SG_M = SG_M_, SG_N = SG_N_ };                                                                             \
    const global half *B = 0;                                                                                        \
    const global float *Asum = 0;                                                                                    \
    const int sg = get_sub_group_id(), lane = get_sub_group_local_id();                                             \
    const int m0 = get_group_id(0) * SG_M, n0 = (get_group_id(1) * WG_N + sg) * SG_N;                               \
    const int per = G / (SL);                                                                                        \
    float8 total[SG_M / 8][SG_N / 16];                                                                               \
    for (int i = 0; i < SG_M / 8; i++)                                                                               \
      for (int j = 0; j < SG_N / 16; j++) total[i][j] = 0.0f;                                                        \
    for (int sl = 0; sl < (SL); sl++) {                                                                              \
      float8 acc[SG_M / 8][SG_N / 16];                                                                               \
      for (int i = 0; i < SG_M / 8; i++)                                                                             \
        for (int j = 0; j < SG_N / 16; j++) acc[i][j] = 0.0f;                                                        \
      for (int g = sl * per; g < (sl + 1) * per; g++) GROUP_STEP_B(SG_M, SG_N, R, READ_A, PF, 0, g, acc)             \
      for (int i = 0; i < SG_M / 8; i++)                                                                             \
        for (int j = 0; j < SG_N / 16; j++) total[i][j] += acc[i][j];                                                \
    }                                                                                                                \
    for (int i = 0; i < SG_M / 8; i++)                                                                               \
      for (int j = 0; j < SG_N / 16; j++) BIAS_TILE(total[i][j], i, j)                                               \
    STORE_HALF(total)                                                                                                \
  }

// One slice per subgroup (4 subgroups on one tile), reduced in order in local memory; narrow shapes and verification.
#define W4KS_KERNEL(NAME, SG_M_, SG_N_, R, READ_A)                                                                   \
  KERNEL_HEAD_K(NAME, 64) {                                                                                          \
    enum { SG_M = SG_M_, SG_N = SG_N_, TI = SG_M_ / 8, TJ = SG_N_ / 16 };                                            \
    const global half *B = 0;                                                                                        \
    const global float *Asum = 0;                                                                                    \
    local float8 buf[SLICES][TI][TJ][16];                                                                            \
    const int sg = get_sub_group_id(), lane = get_sub_group_local_id();                                             \
    const int m0 = get_group_id(0) * SG_M, n0 = get_group_id(1) * SG_N;                                             \
    const int per = G / SLICES;                                                                                      \
    float8 acc[TI][TJ];                                                                                              \
    for (int i = 0; i < TI; i++)                                                                                     \
      for (int j = 0; j < TJ; j++) acc[i][j] = 0.0f;                                                                 \
    for (int g = sg * per; g < (sg + 1) * per; g++) GROUP_STEP_B(SG_M, SG_N, R, READ_A, 2, 0, g, acc)                \
    for (int i = 0; i < TI; i++)                                                                                     \
      for (int j = 0; j < TJ; j++) buf[sg][i][j][lane] = acc[i][j];                                                  \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                                    \
    for (int t = sg; t < TI * TJ; t += SLICES) {                                                                     \
      const int i = t / TJ, j = t % TJ;                                                                              \
      if (m0 + 8 * i >= M) break;                                                                                    \
      float8 tot = 0.0f;                                                                                             \
      for (int s = 0; s < SLICES; s++) tot += buf[s][i][j][lane];                                                    \
      BIAS_TILE(tot, i, j)                                                                                           \
      half8 h = convert_half8_rte(tot);                                                                              \
      intel_sub_group_2d_block_write_16b_8r16x1c(C, N * 2, M, N * 2, (int2)(n0 + 16 * j, m0 + 8 * i), (private ushort *)&h); \
    }                                                                                                                \
  }

W4KS_KERNEL(w4k_dsplit, 16, 32, 16, READ16)
W4KS_KERNEL(w4k_s16x16, 16, 16, 16, READ16)
W4KS_KERNEL(w4k_s32x16, 32, 16, 32, READ32)
W4KS_KERNEL(w4k_s32x32, 32, 32, 32, READ32)
W4KS_KERNEL(w4k_s48x16, 48, 16, 16, READ16)
W4KS_KERNEL(w4k_s64x16, 64, 16, 32, READ32)

W4K_KERNEL(w4k_decode, 16, 32, 4, 16, READ16, 2)

// dpb family (wide shapes: gdn.in, attn.qkv, gate|up, head), original quartets: decode and verification.
W4K_KERNEL_SL(w4k1_decode, 16, 32, 4, 16, READ16, 2, 1)
W4K_KERNEL_SL(w4k1_d16, 16, 16, 4, 16, READ16, 2, 1)
W4K_KERNEL_SL(w4k1_d64, 16, 64, 4, 16, READ16, 2, 1)
W4K_KERNEL_SL(w4k1_w8, 16, 32, 8, 16, READ16, 2, 1)
W4K_KERNEL_SL(w4k1_v32, 32, 32, 4, 32, READ32, 2, 1)
