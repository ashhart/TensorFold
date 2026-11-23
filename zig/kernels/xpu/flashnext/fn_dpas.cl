// Qwen3.8 Flash Next row-invariant Intel Xe2 BF16 XMX DPAS kernels (packed 16-col tile layout, 0-ULP across R=1..2048).

#define ONES_BF16 ((int8)(0x3f803f80))

static inline float dpas32_1(short a0, short a1, int8 b0, int8 b1) {
  const float d = intel_sub_group_bf16_bf16_matrix_mad_k16(a0, b0, 0.0f);
  return intel_sub_group_bf16_bf16_matrix_mad_k16(a1, b1, d);
}

static inline float8 dpas32_8(short8 a0, short8 a1, int8 b0, int8 b1) {
  const float8 d = intel_sub_group_bf16_bf16_matrix_mad_k16(a0, b0, (float8)(0.0f));
  return intel_sub_group_bf16_bf16_matrix_mad_k16(a1, b1, d);
}

static inline void unpack_q4_dpas(uint w0, uint w1, uint w2, uint w3, int8 *b0, int8 *b1) {
  int8 lo, hi;
  #pragma unroll
  for (int p = 0; p < 4; p++) {
    const int sh = p * 8;
    lo[p] = (int)((uint)to_bf((float)((w0 >> sh) & 15u)) | ((uint)to_bf((float)((w0 >> (sh + 4)) & 15u)) << 16));
    lo[p + 4] = (int)((uint)to_bf((float)((w1 >> sh) & 15u)) | ((uint)to_bf((float)((w1 >> (sh + 4)) & 15u)) << 16));
    hi[p] = (int)((uint)to_bf((float)((w2 >> sh) & 15u)) | ((uint)to_bf((float)((w2 >> (sh + 4)) & 15u)) << 16));
    hi[p + 4] = (int)((uint)to_bf((float)((w3 >> sh) & 15u)) | ((uint)to_bf((float)((w3 >> (sh + 4)) & 15u)) << 16));
  }
  *b0 = lo;
  *b1 = hi;
}

static inline void load_q4_tile(global const uint *W, global const ushort *S, global const ushort *B,
                                long nb, int groups, int g, int lane, int8 *b0, int8 *b1, float *sc, float *bi) {
  const long tg = nb * (long)groups + g;
  global const uint *wp = W + tg * 64 + lane;
  unpack_q4_dpas(wp[0], wp[16], wp[32], wp[48], b0, b1);
  const long sb = tg * 16 + lane;
  *sc = bf(S[sb]);
  *bi = bf(B[sb]);
}

// 1-row Split-K Xe2 DPAS GEMV: WG = 64 (4 subgroups). Grid: ((ceil(N/16) + (ksplit==1?3:0)) / (ksplit==1?4:1)).
#define FN_QMV1_DPAS_BODY(STORE_STMT)                                                                           \
  local float kpart[4][16];                                                                                     \
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();                                           \
  const int groups = K / 32;                                                                                    \
  const bool split4 = (groups >= 16) && ((groups & 3) == 0);                                                    \
  const int nb = split4 ? (int)get_group_id(0) : ((int)get_group_id(0) * 4 + sg);                               \
  const int col = nb * 16 + lane;                                                                               \
  const int nb_max = (N + 15) / 16;                                                                             \
  if (!split4) {                                                                                                \
    if (nb >= nb_max) return;                                                                                   \
    float acc = 0.0f;                                                                                           \
    for (int g = 0; g < groups; g++) {                                                                          \
      const short a_lo = (short)X[g * 32 + lane];                                                               \
      const short a_hi = (short)X[g * 32 + 16 + lane];                                                          \
      const float xs = dpas32_1(a_lo, \
          a_hi, ONES_BF16, ONES_BF16);                    \
      int8 b0, b1; float sc, bi;                                                                                \
      load_q4_tile(W, S, B, nb, groups, g, lane, &b0, &b1, &sc, &bi);                                           \
      const float d = dpas32_1(a_lo, \
          a_hi, b0, b1);                                  \
      acc = fma(xs, bi, fma(d, sc, acc));                                                                       \
    }                                                                                                           \
    if (col < N) { const int row_i = 0; const float val = acc; const long idx = col; STORE_STMT; }              \
    return;                                                                                                     \
  }                                                                                                             \
  const int gp = groups >> 2, g0 = sg * gp, g1 = g0 + gp;                                                       \
  float qacc = 0.0f;                                                                                            \
  for (int g = g0; g < g1; g++) {                                                                               \
    const short a_lo = (short)X[g * 32 + lane];                                                                 \
    const short a_hi = (short)X[g * 32 + 16 + lane];                                                            \
    const float xs = dpas32_1(a_lo, \
        a_hi, ONES_BF16, ONES_BF16);                      \
    int8 b0, b1; float sc, bi;                                                                                  \
    load_q4_tile(W, S, B, nb, groups, g, lane, &b0, &b1, &sc, &bi);                                             \
    const float d = dpas32_1(a_lo, \
        a_hi, b0, b1);                                    \
    qacc = fma(xs, bi, fma(d, sc, qacc));                                                                       \
  }                                                                                                             \
  kpart[sg][lane] = qacc;                                                                                       \
  barrier(CLK_LOCAL_MEM_FENCE);                                                                                 \
  if (sg == 0 && col < N) {                                                                                     \
    const int row_i = 0;                                                                                        \
    const float val = (((kpart[0][lane]  + kpart[1][lane]) + kpart[2][lane] ) + kpart[3][lane]);                    \
    const long idx = col;                                                                                       \
    STORE_STMT;                                                                                                 \
  }

// 2..16-row Split-K Xe2 DPAS GEMV/GEMM: WG = 64 (4 subgroups). Processes up to 16 rows in 1 weight pass!
#define FN_QMV16_DPAS_BODY(STORE_STMT)                                                                          \
  local float kpart0[4][8][16];                                                                                 \
  local float kpart1[4][8][16];                                                                                 \
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();                                           \
  const int r0 = (int)get_group_id(1) * 16;                                                                     \
  if (r0 >= R) return;                                                                                          \
  const int m = min(16, R - r0);                                                                                \
  const int groups = K / 32;                                                                                    \
  const bool split4 = (groups >= 16) && ((groups & 3) == 0);                                                    \
  const int nb = split4 ? (int)get_group_id(0) : ((int)get_group_id(0) * 4 + sg);                               \
  const int col = nb * 16 + lane;                                                                               \
  const int nb_max = (N + 15) / 16;                                                                             \
  if (!split4 && nb >= nb_max) return;                                                                          \
  const int gp = split4 ? (groups >> 2) : groups;                                                               \
  const int g0 = split4 ? (sg * gp) : 0, g1 = g0 + gp;                                                          \
  float8 q0 = 0.0f, q1 = 0.0f;                                                                                  \
  for (int g = g0; g < g1; g++) {                                                                               \
    int8 b0, b1; float sc, bi;                                                                                  \
    load_q4_tile(W, S, B, nb, groups, g, lane, &b0, &b1, &sc, &bi);                                             \
    short8 a0_lo = 0, a0_hi = 0;                                                                                \
    for (int r = 0; r < 8; r++) {                                                                               \
      if (r < m) {                                                                                              \
        global const ushort *xr = X + (long)(r0 + r) * K + g * 32;                                              \
        a0_lo[r] = (short)xr[lane];                                                                             \
        a0_hi[r] = (short)xr[16 + lane];                                                                        \
      }                                                                                                         \
    }                                                                                                           \
    const float8 xs0 = dpas32_8(a0_lo, \
        a0_hi, ONES_BF16, ONES_BF16);                    \
    const float8 d0 = dpas32_8(a0_lo, \
        a0_hi, b0, b1);                                  \
    for (int r = 0; r < 8; r++) q0[r] = fma(xs0[r], bi, fma(d0[r], sc, q0[r]));                                 \
    if (m > 8) {                                                                                                \
      short8 a1_lo = 0, a1_hi = 0;                                                                              \
      for (int r = 0; r < 8; r++) {                                                                             \
        if (8 + r < m) {                                                                                        \
          global const ushort *xr = X + (long)(r0 + 8 + r) * K + g * 32;                                        \
          a1_lo[r] = (short)xr[lane];                                                                           \
          a1_hi[r] = (short)xr[16 + lane];                                                                      \
        }                                                                                                       \
      }                                                                                                         \
      const float8 xs1 = dpas32_8(a1_lo, \
          a1_hi, ONES_BF16, ONES_BF16);                  \
      const float8 d1 = dpas32_8(a1_lo, \
          a1_hi, b0, b1);                                \
      for (int r = 0; r < 8; r++) q1[r] = fma(xs1[r], bi, fma(d1[r], sc, q1[r]));                               \
    }                                                                                                           \
  }                                                                                                             \
  if (!split4) {                                                                                                \
    if (col < N) {                                                                                              \
      for (int r = 0; r < min(m, 8); r++) { const int row_i = r0 + r; const float val = q0[r]; const long idx = (long)row_i * N + col; STORE_STMT; } \
      for (int r = 0; r < m - 8; r++) { const int row_i = r0 + 8 + r; const float val = q1[r]; const long idx = (long)row_i * N + col; STORE_STMT; } \
    }                                                                                                           \
    return;                                                                                                     \
  }                                                                                                             \
  for (int r = 0; r < 8; r++) { kpart0[sg][r][lane] = q0[r]; kpart1[sg][r][lane] = q1[r]; }                     \
  barrier(CLK_LOCAL_MEM_FENCE);                                                                                 \
  if (sg == 0 && col < N) {                                                                                     \
    for (int r = 0; r < min(m, 8); r++) {                                                                       \
      const int row_i = r0 + r;                                                                                 \
      const float val = (((kpart0[0][r][lane]  + kpart0[1][r][lane]) + kpart0[2][r][lane] ) + kpart0[3][r][lane]);  \
      const long idx = (long)row_i * N + col;                                                                   \
      STORE_STMT;                                                                                               \
    }                                                                                                           \
    for (int r = 0; r < m - 8; r++) {                                                                           \
      const int row_i = r0 + 8 + r;                                                                             \
      const float val = (((kpart1[0][r][lane]  + kpart1[1][r][lane]) + kpart1[2][r][lane] ) + kpart1[3][r][lane]);  \
      const long idx = (long)row_i * N + col;                                                                   \
      STORE_STMT;                                                                                               \
    }                                                                                                           \
  }

// Wide Prompt Xe2 DPAS GEMM (R > 16): WG = 128 (2 M-sgs x 4 N-sgs = [32x64]), K=64 SLM staging.
#define FN_QMM_TILE_DPAS_BODY(STORE_STMT)                                                                       \
  local ushort xl[32][64];                                                                                      \
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id(), lid = get_local_id(0);                    \
  const int sg_m = sg >> 2, sg_n = sg & 3;                                                                      \
  const int m0 = (int)get_group_id(1) * 32;                                                                     \
  const int nb = (int)get_group_id(0) * 4 + sg_n;                                                               \
  const int col = nb * 16 + lane;                                                                               \
  const int nb_max = (N + 15) / 16;                                                                             \
  const long safe_nb = (nb < nb_max) ? (long)nb : 0;                                                            \
  const bool nb_ok = (nb < nb_max);                                                                             \
  const int groups = K / 32;                                                                                    \
  const bool split4 = (groups >= 16) && ((groups & 3) == 0);                                                    \
  const int q_cnt = split4 ? 4 : 1;                                                                             \
  const int gp = groups / q_cnt;                                                                                \
  float8 acc0 = 0.0f, acc1 = 0.0f;                                                                                          \
  for (int q = 0; q < q_cnt; q++) {                                                                             \
    float8 q0 = 0.0f, q1 = 0.0f;                                                                                \
    const int g_end = (q + 1) * gp;                                                                             \
    for (int g = q * gp; g < g_end; g += 2) {                                                                   \
      const bool has2 = (g + 1 < g_end);                                                                        \
      const int load_w = has2 ? 64 : 32;                                                                        \
      for (int e = lid; e < 32 * (load_w / 4); e += 128) {                                                      \
        const int rr = e / (load_w / 4), cc = (e % (load_w / 4)) * 4;                                           \
        const ushort4 v = (m0 + rr < R) ? vload4(0, X + (long)(m0 + rr) * K + g * 32 + cc) : (ushort4)(0);      \
        xl[rr][cc + 0] = v.s0; xl[rr][cc + 1] = v.s1; xl[rr][cc + 2] = v.s2; xl[rr][cc + 3] = v.s3;             \
      }                                                                                                         \
      barrier(CLK_LOCAL_MEM_FENCE);                                                                             \
      for (int sub = 0; sub < (has2 ? 2 : 1); sub++) {                                                          \
        int8 b0 = 0, b1 = 0; float sc = 0.0f, bi = 0.0f;                                                        \
        if (nb_ok) load_q4_tile(W, S, B, safe_nb, groups, g + sub, lane, &b0, &b1, &sc, &bi);                   \
        short8 a0_0, a0_1, a1_0, a1_1;                                                                          \
        const int kbase = sub * 32;                                                                             \
        for (int r = 0; r < 8; r++) {                                                                           \
          a0_0[r] = (short)xl[sg_m * 16 + r][kbase + lane];                                                     \
          a0_1[r] = (short)xl[sg_m * 16 + r][kbase + 16 + lane];                                                \
          a1_0[r] = (short)xl[sg_m * 16 + 8 + r][kbase + lane];                                                 \
          a1_1[r] = (short)xl[sg_m * 16 + 8 + r][kbase + 16 + lane];                                            \
        }                                                                                                       \
        const float8 xs0 = dpas32_8(a0_0, \
            a0_1, ONES_BF16, ONES_BF16);                  \
        const float8 xs1 = dpas32_8(a1_0, \
            a1_1, ONES_BF16, ONES_BF16);                  \
        const float8 d0 = dpas32_8(a0_0, \
            a0_1, b0, b1);                                \
        const float8 d1 = dpas32_8(a1_0, \
            a1_1, b0, b1);                                \
        for (int r = 0; r < 8; r++) {                                                                           \
          q0[r] = fma(xs0[r], bi, fma(d0[r], sc, q0[r]));                                                       \
          q1[r] = fma(xs1[r], bi, fma(d1[r], sc, q1[r]));                                                       \
        }                                                                                                       \
      }                                                                                                         \
      barrier(CLK_LOCAL_MEM_FENCE);                                                                             \
    }                                                                                                           \
    acc0 = (q == 0) ? q0 : (acc0 + q0);                                                                         \
    acc1 = (q == 0) ? q1 : (acc1 + q1);                                                                         \
  }                                                                                                             \
  if (col < N) {                                                                                                \
    for (int r = 0; r < 8; r++) {                                                                               \
      const int row0 = m0 + sg_m * 16 + r;                                                                      \
      if (row0 < R) { const int row_i = row0; const float val = acc0[r]; const long idx = (long)row0 * N + col; STORE_STMT; } \
      const int row1 = row0 + 8;                                                                                \
      if (row1 < R) { const int row_i = row1; const float val = acc1[r]; const long idx = (long)row1 * N + col; STORE_STMT; } \
    }                                                                                                           \
  }

KERNEL(64) void fn_qmv1_dpas_bf16(global const uint *W, global const ushort *S, global const ushort *B,
                                  global const ushort *X, global ushort *Y, int K, int N) {
  FN_QMV1_DPAS_BODY({ (void)row_i; Y[idx] = to_bf(val); })
}

KERNEL(64) void fn_qmv1_dpas_f32(global const uint *W, global const ushort *S, global const ushort *B,
                                 global const ushort *X, global float *Y, int K, int N) {
  FN_QMV1_DPAS_BODY({ (void)row_i; Y[idx] = val; })
}

KERNEL(64) void fn_qmv16_dpas_bf16(global const uint *W, global const ushort *S, global const ushort *B,
                                   global const ushort *X, global ushort *Y, int K, int N, int R) {
  FN_QMV16_DPAS_BODY({ (void)row_i; Y[idx] = to_bf(val); })
}

KERNEL(64) void fn_qmv16_dpas_f32(global const uint *W, global const ushort *S, global const ushort *B,
                                  global const ushort *X, global float *Y, int K, int N, int R) {
  FN_QMV16_DPAS_BODY({ (void)row_i; Y[idx] = val; })
}

KERNEL(128) void fn_qmm_tile_dpas_bf16(global const uint *W, global const ushort *S, global const ushort *B,
                                       global const ushort *X, global ushort *Y, int K, int N, int R) {
  FN_QMM_TILE_DPAS_BODY({ (void)row_i; Y[idx] = to_bf(val); })
}

KERNEL(128) void fn_qmm_tile_dpas_f32(global const uint *W, global const ushort *S, global const ushort *B,
                                      global const ushort *X, global float *Y, int K, int N, int R) {
  FN_QMM_TILE_DPAS_BODY({ (void)row_i; Y[idx] = val; })
}

// Hyper-Connection Down (K=10240 = 320 groups, N=320 or 324) for 1..16 rows (grid: ((N+15)/16, R)).
KERNEL(64) void fn_hc_down_1(global const ushort *X, global const uint *W, global const ushort *S,
                             global const ushort *B, global ushort *ACT, global ushort *INJ, int N, int has_inj) {
  local float kpart[4][16];
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int nb = (int)get_group_id(0), row_i = (int)get_group_id(1);
  const int col = nb * 16 + lane;
  global const ushort *xr = X + (long)row_i * 10240;
  float qacc = 0.0f;
  for (int g = sg * 80; g < (sg + 1) * 80; g++) {
    const short a_lo = (short)xr[g * 32 + lane];
    const short a_hi = (short)xr[g * 32 + 16 + lane];
    const float xs = dpas32_1(a_lo, a_hi, ONES_BF16, ONES_BF16);
    int8 b0, b1; float sc, bi;
    load_q4_tile(W, S, B, nb, 320, g, lane, &b0, &b1, &sc, &bi);
    const float d = dpas32_1(a_lo, a_hi, b0, b1);
    qacc = fma(xs, bi, fma(d, sc, qacc));
  }
  kpart[sg][lane] = qacc;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg == 0 && col < N) {
    const float sum = (((kpart[0][lane] + kpart[1][lane]) + kpart[2][lane]) + kpart[3][lane]);
    const float y = rbf(rbf(sum) * 0.25f);
    if (col < 320) ACT[(long)row_i * 320 + col] = to_bf(bsilu(y));
    else if (has_inj) INJ[(long)row_i * 4 + (col - 320)] = to_bf(2.0f * bsig(y));
  }
}

// Wide HC Down (R > 16): WG = 128 ([32 rows x 64 cols], coalesced SLM staging).
KERNEL(128) void fn_hc_down_wide(global const ushort *X, global const uint *W, global const ushort *S,
                                 global const ushort *B, global ushort *ACT, global ushort *INJ,
                                 int R, int N, int has_inj) {
  const int K = 10240;
  FN_QMM_TILE_DPAS_BODY({
    (void)idx;
    const float y = rbf(rbf(val) * 0.25f);
    if (col < 320) ACT[(long)row_i * 320 + col] = to_bf(bsilu(y));
    else if (has_inj) INJ[(long)row_i * 4 + (col - 320)] = to_bf(2.0f * bsig(y));
  })
}

// 1..16-row Hyper-Connection UpMix on Xe2 DPAS (0 SLM, minimal GRF). Grid: (40, R).
KERNEL(64) void fn_hc_upmix_1(global const ushort *ACT, global const ushort *NORM, global const uint *W,
                              global const ushort *S, global const ushort *B, global ushort *MIXED) {
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int nb0 = (int)get_group_id(0) * 4 + sg;
  const int row_i = (int)get_group_id(1);
  if (nb0 >= 160) return;
  const int col = nb0 * 16 + lane;
  global const ushort *ar = ACT + (long)row_i * 320;
  global const ushort *nr = NORM + (long)row_i * 10240;
  short a_lo[10], a_hi[10];
  float xs[10];
  for (int g = 0; g < 10; g++) {
    a_lo[g] = (short)ar[g * 32 + lane];
    a_hi[g] = (short)ar[g * 32 + 16 + lane];
    xs[g] = dpas32_1(a_lo[g], a_hi[g], ONES_BF16, ONES_BF16);
  }
  float total = 0.0f;
  for (int s = 0; s < 4; s++) {
    const long nb = (long)s * 160 + nb0;
    float acc = 0.0f;
    for (int g = 0; g < 10; g++) {
      int8 b0, b1; float sc, bi;
      load_q4_tile(W, S, B, nb, 10, g, lane, &b0, &b1, &sc, &bi);
      const float d = dpas32_1(a_lo[g], a_hi[g], b0, b1);
      acc = fma(xs[g], bi, fma(d, sc, acc));
    }
    total += rbf(bsig(rbf(acc)) * bf(nr[(long)s * 2560 + col]));
  }
  MIXED[(long)row_i * 2560 + col] = to_bf(total * 0.25f);
}

// Wide HC UpMix on Xe2 DPAS (R > 16): WG = 128 ([32 rows x 64 cols], stages [32, 320] ACT in 20 KiB SLM once).
KERNEL(128) void fn_hc_upmix_dpas(global const ushort *ACT, global const ushort *NORM, global const uint *W,
                                  global const ushort *S, global const ushort *B, global ushort *MIXED, int R) {
  local ushort al[32][320];
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id(), lid = get_local_id(0);
  const int sg_m = sg >> 2, sg_n = sg & 3;
  const int m0 = (int)get_group_id(1) * 32;
  const int nb0 = (int)get_group_id(0) * 4 + sg_n;
  const int col = nb0 * 16 + lane;
  for (int e = lid; e < 32 * 80; e += 128) {
    const int rr = e / 80, cc = (e % 80) * 4;
    const ushort4 v = (m0 + rr < R) ? vload4(0, ACT + (long)(m0 + rr) * 320 + cc) : (ushort4)(0);
    al[rr][cc + 0] = v.s0; al[rr][cc + 1] = v.s1; al[rr][cc + 2] = v.s2; al[rr][cc + 3] = v.s3;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float8 tot0 = 0.0f, tot1 = 0.0f;
  for (int s = 0; s < 4; s++) {
    float8 acc0 = 0.0f, acc1 = 0.0f;
    for (int g = 0; g < 10; g++) {
      short8 a0_0, a0_1, a1_0, a1_1;
      for (int r = 0; r < 8; r++) {
        a0_0[r] = (short)al[sg_m * 16 + r][g * 32 + lane];
        a0_1[r] = (short)al[sg_m * 16 + r][g * 32 + 16 + lane];
        a1_0[r] = (short)al[sg_m * 16 + 8 + r][g * 32 + lane];
        a1_1[r] = (short)al[sg_m * 16 + 8 + r][g * 32 + 16 + lane];
      }
      const float8 xs0 = dpas32_8(a0_0, a0_1, ONES_BF16, ONES_BF16);
      const float8 xs1 = dpas32_8(a1_0, a1_1, ONES_BF16, ONES_BF16);
      int8 b0, b1; float sc, bi;
      load_q4_tile(W, S, B, (long)s * 160 + nb0, 10, g, lane, &b0, &b1, &sc, &bi);
      const float8 d0 = dpas32_8(a0_0, a0_1, b0, b1);
      const float8 d1 = dpas32_8(a1_0, a1_1, b0, b1);
      for (int r = 0; r < 8; r++) {
        acc0[r] = fma(xs0[r], bi, fma(d0[r], sc, acc0[r]));
        acc1[r] = fma(xs1[r], bi, fma(d1[r], sc, acc1[r]));
      }
    }
    for (int r = 0; r < 8; r++) {
      const int row0 = m0 + sg_m * 16 + r;
      if (row0 < R) {
        const float nv = bf(NORM[(long)row0 * 10240 + (long)s * 2560 + col]);
        tot0[r] += rbf(bsig(rbf(acc0[r])) * nv);
      }
      const int row1 = row0 + 8;
      if (row1 < R) {
        const float nv = bf(NORM[(long)row1 * 10240 + (long)s * 2560 + col]);
        tot1[r] += rbf(bsig(rbf(acc1[r])) * nv);
      }
    }
  }
  for (int r = 0; r < 8; r++) {
    const int row0 = m0 + sg_m * 16 + r;
    if (row0 < R) MIXED[(long)row0 * 2560 + col] = to_bf(tot0[r] * 0.25f);
    const int row1 = row0 + 8;
    if (row1 < R) MIXED[(long)row1 * 2560 + col] = to_bf(tot1[r] * 0.25f);
  }
}

// Router Logits on Xe2 BF16 DPAS for 1..16 rows (grid: (33, R), 128 B SLM).
KERNEL(64) void fn_router_dpas_1(global const ushort *X, global const int *W, global float *OUT) {
  local float kpart[4][16];
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int nb = (int)get_group_id(0), row_i = (int)get_group_id(1);
  if (nb >= 33) return;
  const int col = nb * 16 + lane;
  global const ushort *xr = X + (long)row_i * 2560;
  float qacc = 0.0f;
  for (int kb = sg * 40; kb < (sg + 1) * 40; kb++) {
    global const int *wp = W + ((long)nb * 160 + kb) * 128 + lane;
    int8 b;
    for (int p = 0; p < 8; p++) b[p] = wp[p * 16];
    const short a = (short)xr[kb * 16 + lane];
    qacc = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, qacc);
  }
  kpart[sg][lane] = qacc;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg == 0 && col < 513) {
    OUT[(long)row_i * 513 + col] = (((kpart[0][lane] + kpart[1][lane]) + kpart[2][lane]) + kpart[3][lane]);
  }
}

KERNEL(128) void fn_router_dpas_mm(global const ushort *X, global const int *W, global float *OUT, int R) {
  local ushort xl[32][64];
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id(), lid = get_local_id(0);
  const int sg_m = sg >> 2, sg_n = sg & 3;
  const int m0 = (int)get_group_id(1) * 32;
  const int nb = (int)get_group_id(0) * 4 + sg_n;
  const int col = nb * 16 + lane;
  const bool nb_ok = (nb < 33);
  const long safe_nb = nb_ok ? (long)nb : 0;
  float8 acc0 = 0.0f, acc1 = 0.0f;
  for (int q = 0; q < 4; q++) {
    float8 q0 = 0.0f, q1 = 0.0f;
    for (int kb = q * 40; kb < (q + 1) * 40; kb += 4) {
      for (int e = lid; e < 32 * 16; e += 128) {
        const int rr = e >> 4, cc = (e & 15) * 4;
        const ushort4 v = (m0 + rr < R) ? vload4(0, X + (long)(m0 + rr) * 2560 + kb * 16 + cc) : (ushort4)(0);
        xl[rr][cc + 0] = v.s0; xl[rr][cc + 1] = v.s1; xl[rr][cc + 2] = v.s2; xl[rr][cc + 3] = v.s3;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int sub = 0; sub < 4; sub++) {
        int8 b = 0;
        if (nb_ok) {
          global const int *wp = W + (safe_nb * 160 + kb + sub) * 128 + lane;
          for (int p = 0; p < 8; p++) b[p] = wp[p * 16];
        }
        short8 a0, a1;
        for (int r = 0; r < 8; r++) {
          a0[r] = (short)xl[sg_m * 16 + r][sub * 16 + lane];
          a1[r] = (short)xl[sg_m * 16 + 8 + r][sub * 16 + lane];
        }
        q0 = intel_sub_group_bf16_bf16_matrix_mad_k16(a0, b, q0);
        q1 = intel_sub_group_bf16_bf16_matrix_mad_k16(a1, b, q1);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    acc0 = (q == 0) ? q0 : (acc0 + q0);
    acc1 = (q == 0) ? q1 : (acc1 + q1);
  }
  if (col < 513) {
    for (int r = 0; r < 8; r++) {
      if (m0 + sg_m * 16 + r < R) OUT[(long)(m0 + sg_m * 16 + r) * 513 + col] = acc0[r];
      if (m0 + sg_m * 16 + 8 + r < R) OUT[(long)(m0 + sg_m * 16 + 8 + r) * 513 + col] = acc1[r];
    }
  }
}
