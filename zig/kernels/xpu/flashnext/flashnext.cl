// Qwen3.8 Flash Next (qwen4_exp) kernels for Intel Xe2 (Level Zero / OpenCL C), supporting TP={1,2,4}. Arithmetic and

#pragma OPENCL FP_CONTRACT OFF
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SG 16
#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf(ushort h) { return as_float((uint)h << 16); }
static inline ushort to_bf(float f) { return intel_convert_bfloat16_as_ushort(f); }
static inline float rbf(float f) { return bf(to_bf(f)); }
static inline float tf_silu(float x) { return x / (1.0f + exp(-x)); }
static inline float tf_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float bsilu(float x) { return rbf(tf_silu(x)); }
static inline float bsig(float x) { return rbf(tf_sigmoid(x)); }
static inline float softplus_f(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

// Deterministic work-group sum over N threads (N a multiple of 16, up to 1024).
static inline float wg_sum(float v, local float *red, int nsg) {
  const float s = sub_group_reduce_add(v);
  const int sg = get_sub_group_id();
  if (get_sub_group_local_id() == 0) red[sg] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float tot = 0.0f;
  for (int i = 0; i < nsg; i++) tot += red[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  return tot;
}

// 1. Embeddings, RMSNorm, Stream Add

// MLX affine 4-bit group-32 embedding copied into `copies` streams: out [rows, copies, D] bf16.
KERNEL(256) void fn_embed_q4(global const int *IDS, global const uint *W, global const ushort *S, global const ushort *B,
                             global ushort *OUT, int D, int copies) {
  const int r = get_group_id(0);
  const long id = (long)IDS[r];
  const int words = D / 8, groups = D / 32;
  for (int d = get_local_id(0); d < D; d += 256) {
    const uint w = W[id * words + d / 8];
    const float q = (float)((w >> (4 * (d & 7))) & 0xFu);
    const int g = d / 32;
    const ushort val = to_bf(q * bf(S[id * groups + g]) + bf(B[id * groups + g]));
    for (int s = 0; s < copies; s++) {
      OUT[((long)r * copies + s) * D + d] = val;
    }
  }
}

KERNEL(256) void fn_embed_bf16(global const int *IDS, global const ushort *W, global ushort *OUT, int D, int copies) {
  const int r = get_group_id(0);
  const long id = (long)IDS[r];
  for (int d = get_local_id(0); d < D; d += 256) {
    const ushort val = W[id * D + d];
    for (int s = 0; s < copies; s++) {
      OUT[((long)r * copies + s) * D + d] = val;
    }
  }
}

// Whole-row RMSNorm (used by MTP pre_fc_norm_embedding [D=2560] and pre_fc_norm_hidden [wide=10240]).
KERNEL(256) void fn_rmsnorm_rows(global const ushort *X, int x_stride, global const float *W, global ushort *OUT,
                                 float eps, int K) {
  local float red[16];
  const int r = get_group_id(0), lid = get_local_id(0);
  const global ushort *xr = X + (long)r * x_stride;
  float ss = 0.0f;
  for (int i = lid; i < K; i += 256) {
    const float v = bf(xr[i]);
    ss = fma(v, v, ss);
  }
  const float tot = wg_sum(ss, red, 16);
  const float rinv = rsqrt(tot / (float)K + eps);
  global ushort *yr = OUT + (long)r * K;
  for (int i = lid; i < K; i += 256) {
    yr[i] = to_bf(bf(xr[i]) * rinv * W[i]);
  }
}

// MTP stream add: OUT[r, s, d] = bf16(E[r, d] + HS[r, s, d]), grid (rows, (D + 255) / 256).
KERNEL(256) void fn_add_streams(global const ushort *E, global const ushort *HS, global ushort *OUT, int D, int S) {
  const int r = get_group_id(0), d = get_group_id(1) * 256 + get_local_id(0);
  if (d >= D) return;
  const float ev = bf(E[(long)r * D + d]);
  for (int s = 0; s < S; s++) {
    const long at = ((long)r * S + s) * D + d;
    OUT[at] = to_bf(ev + bf(HS[at]));
  }
}

// 2. Hyper-Connections (Writeback + Normalize, Down + ReduceAct, Up + Mix)

// Fused hc_writeback + normalize for D=2560, S=4 (WIDE=10240): grid (rows). has_br == 0: mode 0 (no branch added, H
KERNEL(256) void fn_hc_writeback_norm(global ushort *H, global const float *BR, global const ushort *INJ,
                                      global const float *SCALE, global ushort *NORM, float eps, int has_br) {
  local float red[10][16][4];
  local float rinv[4];
  const int r = get_group_id(0), lid = get_local_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int D = 2560;
  float hv[10][4];
  for (int c = 0; c < 10; c++) {
    const int d = c * 256 + lid;
    const float branch = has_br ? rbf(BR[(long)r * D + d]) : 0.0f;
    for (int s = 0; s < 4; s++) {
      const long at = (long)r * (4 * D) + (long)s * D + d;
      float v = bf(H[at]);
      if (has_br) {
        v = rbf(fma(branch, bf(INJ[(long)r * 4 + s]), v));
        H[at] = to_bf(v);
      }
      hv[c][s] = v;
      const float ss = sub_group_reduce_add(v * v);
      if (lane == 0) red[c][sg][s] = ss;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 4) {
    float all = 0.0f;
    for (int c = 0; c < 10; c++) {
      float chunk_s = 0.0f;
      for (int w = 0; w < 16; w++) chunk_s += red[c][w][lid];
      all += chunk_s;
    }
    rinv[lid] = rsqrt(all / (float)D + eps);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int c = 0; c < 10; c++) {
    const int d = c * 256 + lid;
    for (int s = 0; s < 4; s++) {
      const int col = s * D + d;
      NORM[(long)r * (4 * D) + col] = to_bf((hv[c][s] * rinv[s]) * SCALE[col]);
    }
  }
}

// Writeback only (without NORM), used before final head or PLE when needed: grid (rows, 10).
KERNEL(256) void fn_hc_writeback_only(global ushort *H, global const float *BR, global const ushort *INJ, int has_br) {
  if (!has_br) return;
  const int r = get_group_id(0), c = get_group_id(1), lid = get_local_id(0);
  const int D = 2560, d = c * 256 + lid;
  const float branch = rbf(BR[(long)r * D + d]);
  for (int s = 0; s < 4; s++) {
    const long at = (long)r * (4 * D) + (long)s * D + d;
    const float v = rbf(fma(branch, bf(INJ[(long)r * 4 + s]), bf(H[at])));
    H[at] = to_bf(v);
  }
}

// Helper: 4-bit group-32 dot of up to 16 rows at once for one output row `row_idx` of [N, K]. Each 16-lane subgroup
static inline void q4g32_dot16_range(global const uint *W, global const ushort *S, global const ushort *B,
                                     global const ushort *X, int x_stride, int m, long row_idx, int K,
                                     int wi_begin, int wi_end, float acc[16]) {
  const int lane = get_sub_group_local_id();
  const int words = K / 8, groups = K / 32;
  global const uint *wr = W + row_idx * words;
  global const ushort *sr = S + row_idx * groups;
  global const ushort *br = B + row_idx * groups;
  for (int r = 0; r < m; r++) acc[r] = 0.0f;
  for (int wi = wi_begin + lane; wi < wi_end; wi += SG) {
    const uint pk = wr[wi];
    const int g = wi >> 2;
    const float sc = bf(sr[g]), bi = bf(br[g]);
    const float w0 = fma((float)(pk & 15u), sc, bi);
    const float w1 = fma((float)((pk >> 4) & 15u), sc, bi);
    const float w2 = fma((float)((pk >> 8) & 15u), sc, bi);
    const float w3 = fma((float)((pk >> 12) & 15u), sc, bi);
    const float w4 = fma((float)((pk >> 16) & 15u), sc, bi);
    const float w5 = fma((float)((pk >> 20) & 15u), sc, bi);
    const float w6 = fma((float)((pk >> 24) & 15u), sc, bi);
    const float w7 = fma((float)((pk >> 28) & 15u), sc, bi);
    for (int r = 0; r < m; r++) {
      const ushort8 xv8 = vload8(0, X + (long)r * x_stride + wi * 8);
      acc[r] = fma(bf(xv8.s0), w0, acc[r]);
      acc[r] = fma(bf(xv8.s1), w1, acc[r]);
      acc[r] = fma(bf(xv8.s2), w2, acc[r]);
      acc[r] = fma(bf(xv8.s3), w3, acc[r]);
      acc[r] = fma(bf(xv8.s4), w4, acc[r]);
      acc[r] = fma(bf(xv8.s5), w5, acc[r]);
      acc[r] = fma(bf(xv8.s6), w6, acc[r]);
      acc[r] = fma(bf(xv8.s7), w7, acc[r]);
    }
  }
  for (int r = 0; r < m; r++) acc[r] = sub_group_reduce_add(acc[r]);
}

static inline void q4g32_dot16(global const uint *W, global const ushort *S, global const ushort *B,
                               global const ushort *X, int x_stride, int m, long row_idx, int K, float acc[16]) {
  q4g32_dot16_range(W, S, B, X, x_stride, m, row_idx, K, 0, K / 8, acc);
}

// Hyper-connection down projection + SiLU/inject activation (4-way Split-K for full EU occupancy!): NORM [R, 10240] x
KERNEL(64) void fn_hc_down_reduce(global const ushort *NORM, global const uint *W, global const ushort *S,
                                  global const ushort *B, global ushort *ACT, global ushort *INJ,
                                  int R, int ndn, int has_inj) {
  local float kpart[4][16];
  const int col = get_group_id(0);
  const int r0 = get_group_id(1) * 16;
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  if (col >= ndn || r0 >= R) return;
  const int m = min(16, R - r0);
  float acc[16];
  q4g32_dot16_range(W, S, B, NORM + (long)r0 * 10240, 10240, m, col, 10240, sg * 320, (sg + 1) * 320, acc);
  if (lane == 0) {
    for (int r = 0; r < m; r++) kpart[sg][r] = acc[r];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg == 0 && lane < m) {
    const float sum = kpart[0][lane] + kpart[1][lane] + kpart[2][lane] + kpart[3][lane];
    const float y = rbf(rbf(sum) * 0.25f);
    if (col < 320) {
      ACT[(long)(r0 + lane) * 320 + col] = to_bf(bsilu(y));
    } else if (has_inj) {
      INJ[(long)(r0 + lane) * 4 + (col - 320)] = to_bf(2.0f * bsig(y));
    }
  }
}

// Hyper-connection up projection + stream sigmoid gating + 4-stream average: ACT [R, 320] x W [10240, 320], gated by
KERNEL(256) void fn_hc_upmix(global const ushort *ACT, global const ushort *NORM, global const uint *W,
                             global const ushort *S, global const ushort *B, global ushort *MIXED, int R) {
  local ushort act_l[16 * 320];
  const int col = get_group_id(0) * 16 + get_sub_group_id();
  const int r0 = get_group_id(1) * 16;
  const int lid = get_local_id(0), lane = get_sub_group_local_id();
  if (r0 >= R) return;
  const int m = min(16, R - r0);
  for (int e = lid; e < m * 320; e += 256) {
    act_l[e] = ACT[(long)r0 * 320 + e];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (col >= 2560) return;

  float total[16];
  for (int r = 0; r < m; r++) total[r] = 0.0f;
  for (int s = 0; s < 4; s++) {
    const long wrow = (long)s * 2560 + col;
    global const uint *wr = W + wrow * 40;
    global const ushort *sr = S + wrow * 10;
    global const ushort *br = B + wrow * 10;
    float acc[16];
    for (int r = 0; r < m; r++) acc[r] = 0.0f;
    for (int wi = lane; wi < 40; wi += 16) {
      const uint pk = wr[wi];
      const int g = wi >> 2;
      const float sc = bf(sr[g]), bi = bf(br[g]);
      const float w0 = fma((float)(pk & 15u), sc, bi);
      const float w1 = fma((float)((pk >> 4) & 15u), sc, bi);
      const float w2 = fma((float)((pk >> 8) & 15u), sc, bi);
      const float w3 = fma((float)((pk >> 12) & 15u), sc, bi);
      const float w4 = fma((float)((pk >> 16) & 15u), sc, bi);
      const float w5 = fma((float)((pk >> 20) & 15u), sc, bi);
      const float w6 = fma((float)((pk >> 24) & 15u), sc, bi);
      const float w7 = fma((float)((pk >> 28) & 15u), sc, bi);
      for (int r = 0; r < m; r++) {
        const int base = r * 320 + wi * 8;
        acc[r] = fma(bf(act_l[base + 0]), w0, acc[r]);
        acc[r] = fma(bf(act_l[base + 1]), w1, acc[r]);
        acc[r] = fma(bf(act_l[base + 2]), w2, acc[r]);
        acc[r] = fma(bf(act_l[base + 3]), w3, acc[r]);
        acc[r] = fma(bf(act_l[base + 4]), w4, acc[r]);
        acc[r] = fma(bf(act_l[base + 5]), w5, acc[r]);
        acc[r] = fma(bf(act_l[base + 6]), w6, acc[r]);
        acc[r] = fma(bf(act_l[base + 7]), w7, acc[r]);
      }
    }
    for (int r = 0; r < m; r++) {
      const float u = rbf(sub_group_reduce_add(acc[r]));
      if (lane == 0) {
        const float nv = bf(NORM[(long)(r0 + r) * 10240 + wrow]);
        total[r] += rbf(bsig(u) * nv);
      }
    }
  }
  if (lane == 0) {
    for (int r = 0; r < m; r++) {
      MIXED[(long)(r0 + r) * 2560 + col] = to_bf(total[r] * 0.25f);
    }
  }
}

// 3. MLX q4g32 Dense Projections (Multi-row Subgroup GEMV & Xe2 BF16 DPAS GEMM)

// Decode / Verify (1..16 rows in a single weight pass!): grid ((N + 15) / 16, (R + 15) / 16), WG = 256.
KERNEL(256) void fn_qmv4_g32_bf16(global const uint *W, global const ushort *S, global const ushort *B,
                                  global const ushort *X, global ushort *Y, int K, int N, int R) {
  const int col = get_group_id(0) * 16 + get_sub_group_id();
  const int r0 = get_group_id(1) * 16;
  if (col >= N || r0 >= R) return;
  const int m = min(16, R - r0);
  float acc[16];
  q4g32_dot16(W, S, B, X + (long)r0 * K, K, m, col, K, acc);
  if (get_sub_group_local_id() == 0) {
    for (int r = 0; r < m; r++) Y[(long)(r0 + r) * N + col] = to_bf(acc[r]);
  }
}

KERNEL(256) void fn_qmv4_g32_f32(global const uint *W, global const ushort *S, global const ushort *B,
                                 global const ushort *X, global float *Y, int K, int N, int R) {
  const int col = get_group_id(0) * 16 + get_sub_group_id();
  const int r0 = get_group_id(1) * 16;
  if (col >= N || r0 >= R) return;
  const int m = min(16, R - r0);
  float acc[16];
  q4g32_dot16(W, S, B, X + (long)r0 * K, K, m, col, K, acc);
  if (get_sub_group_local_id() == 0) {
    for (int r = 0; r < m; r++) Y[(long)(r0 + r) * N + col] = acc[r];
  }
}

// Prompt GEMM on Intel Xe2 BF16 DPAS: Workgroup = 128 threads = 8 subgroups: 2 M-subgroups x 4 N-subgroups = [32 rows
#define FN_DPAS_GEMM_BODY(STORE_STMT)                                                                            \
  local ushort xl[32][32];                                                                                       \
  local float xsl[32];                                                                                           \
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id(), lid = get_local_id(0);                     \
  const int sg_m = sg >> 2, sg_n = sg & 3;                                                                       \
  const int m0 = get_group_id(1) * 32, n0 = get_group_id(0) * 64;                                                \
  const int col = n0 + sg_n * 16 + lane;                                                                         \
  const bool col_ok = col < N;                                                                                   \
  const int words = K / 8, groups = K / 32;                                                                      \
  const long safe_col = col_ok ? (long)col : 0;                                                                  \
  global const uint *wc = W + safe_col * words;                                                                  \
  global const ushort *sc_p = S + safe_col * groups;                                                             \
  global const ushort *bi_p = B + safe_col * groups;                                                             \
  float8 acc0 = 0.0f, acc1 = 0.0f;                                                                               \
  for (int g = 0; g < groups; g++) {                                                                             \
    for (int e = lid; e < 32 * 8; e += 128) {                                                                    \
      const int rr = e >> 3, cc = (e & 7) * 4;                                                                   \
      ushort4 v = (m0 + rr < R) ? vload4(0, X + (long)(m0 + rr) * K + g * 32 + cc) : (ushort4)(0);               \
      xl[rr][cc + 0] = v.s0; xl[rr][cc + 1] = v.s1; xl[rr][cc + 2] = v.s2; xl[rr][cc + 3] = v.s3;                \
    }                                                                                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                                \
    if (sg < 2) {                                                                                                \
      for (int rr = sg * 16 + lane; rr < 32; rr += 32) {                                                         \
        float s = 0.0f;                                                                                          \
        for (int k = 0; k < 32; k++) s += bf(xl[rr][k]);                                                         \
        xsl[rr] = s;                                                                                             \
      }                                                                                                          \
    }                                                                                                            \
    const uint4 w4 = col_ok ? vload4(0, wc + g * 4) : (uint4)(0);                                                \
    const float sc = col_ok ? bf(sc_p[g]) : 0.0f;                                                                \
    const float bi = col_ok ? bf(bi_p[g]) : 0.0f;                                                                \
    int8 b0, b1;                                                                                                 \
    for (int p = 0; p < 8; p++) {                                                                                \
      const uint w_lo = (p < 4) ? w4.s0 : w4.s1;                                                                 \
      const int sh = (p & 3) * 8;                                                                                \
      const ushort q_ev = to_bf((float)((w_lo >> sh) & 15u));                                                    \
      const ushort q_od = to_bf((float)((w_lo >> (sh + 4)) & 15u));                                              \
      b0[p] = (int)((uint)q_ev | ((uint)q_od << 16));                                                            \
      const uint w_hi = (p < 4) ? w4.s2 : w4.s3;                                                                 \
      const ushort q2_ev = to_bf((float)((w_hi >> sh) & 15u));                                                   \
      const ushort q2_od = to_bf((float)((w_hi >> (sh + 4)) & 15u));                                             \
      b1[p] = (int)((uint)q2_ev | ((uint)q2_od << 16));                                                          \
    }                                                                                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                                \
    short8 a0_0, a0_1, a1_0, a1_1;                                                                               \
    for (int r = 0; r < 8; r++) {                                                                                \
      a0_0[r] = (short)xl[sg_m * 16 + r][lane];                                                                  \
      a0_1[r] = (short)xl[sg_m * 16 + r][16 + lane];                                                             \
      a1_0[r] = (short)xl[sg_m * 16 + 8 + r][lane];                                                              \
      a1_1[r] = (short)xl[sg_m * 16 + 8 + r][16 + lane];                                                         \
    }                                                                                                            \
    float8 d0 = 0.0f, d1 = 0.0f;                                                                                 \
    d0 = intel_sub_group_bf16_bf16_matrix_mad_k16(a0_0, b0, d0);                                                 \
    d0 = intel_sub_group_bf16_bf16_matrix_mad_k16(a0_1, b1, d0);                                                 \
    d1 = intel_sub_group_bf16_bf16_matrix_mad_k16(a1_0, b0, d1);                                                 \
    d1 = intel_sub_group_bf16_bf16_matrix_mad_k16(a1_1, b1, d1);                                                 \
    for (int r = 0; r < 8; r++) {                                                                                \
      acc0[r] = fma(xsl[sg_m * 16 + r], bi, fma(d0[r], sc, acc0[r]));                                            \
      acc1[r] = fma(xsl[sg_m * 16 + 8 + r], bi, fma(d1[r], sc, acc1[r]));                                        \
    }                                                                                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                                \
  }                                                                                                              \
  if (col_ok) {                                                                                                  \
    for (int r = 0; r < 8; r++) {                                                                                \
      const int row0 = m0 + sg_m * 16 + r;                                                                       \
      if (row0 < R) { const int row_i = row0; const float val = acc0[r]; const long idx = (long)row0 * N + col; STORE_STMT; } \
      const int row1 = row0 + 8;                                                                                 \
      if (row1 < R) { const int row_i = row1; const float val = acc1[r]; const long idx = (long)row1 * N + col; STORE_STMT; } \
    }                                                                                                            \
  }

KERNEL(128) void fn_qmm4_g32_dpas_bf16(global const uint *W, global const ushort *S, global const ushort *B,
                                       global const ushort *X, global ushort *Y, int K, int N, int R) {
  FN_DPAS_GEMM_BODY({ (void)row_i; Y[idx] = to_bf(val); })
}

KERNEL(128) void fn_qmm4_g32_dpas_f32(global const uint *W, global const ushort *S, global const ushort *B,
                                      global const ushort *X, global float *Y, int K, int N, int R) {
  FN_DPAS_GEMM_BODY({ (void)row_i; Y[idx] = val; })
}

KERNEL(128) void fn_hc_down_dpas(global const ushort *X, global const uint *W, global const ushort *S,
                                 global const ushort *B, global ushort *ACT, global ushort *INJ,
                                 int R, int N, int has_inj) {
  const int K = 10240;
  FN_DPAS_GEMM_BODY({
    (void)idx;
    const float y = rbf(rbf(val) * 0.25f);
    if (col < 320) ACT[(long)row_i * 320 + col] = to_bf(bsilu(y));
    else if (has_inj) INJ[(long)row_i * 4 + (col - 320)] = to_bf(2.0f * bsig(y));
  })
}

// 4. Layer 1 Per-Layer N-Gram Embedding (PLE) Gate, Dilated Conv, Shift Window

#include "fn_mixers.cl"
#include "fn_moe_head.cl"
