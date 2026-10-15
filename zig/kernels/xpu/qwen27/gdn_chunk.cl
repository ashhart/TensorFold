// Prompt GDN in chunked form (C = 64) on Xe2 XMX: bf16 and fp16 DPAS, fp32 state; gdn_prep, gdn_state, gdn_out.
#pragma OPENCL FP_CONTRACT OFF
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define CK 64
#define D 128
#define DPAS(a, b, c) intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(a), as_int8(b), c)

static inline ushort bf(float f) {                     // fp32 → bf16 round to nearest even, hardware conversion
  return intel_convert_bfloat16_as_ushort(f);
}
// fp32 operands against an exact bf16 one enter as three bf16 terms h + m + l (about 24 bits).
#define SPLIT3(f, H, M, L)                                                                                             \
  {                                                                                                                    \
    const float f_ = (f);                                                                                              \
    H = bf(f_);                                                                                                        \
    const float r_ = f_ - as_float((uint)(H) << 16);                                                                   \
    M = bf(r_);                                                                                                        \
    L = bf(r_ - as_float((uint)(M) << 16));                                                                            \
  }
// Both operands split: the six products down to 2^-16 of the leading one, smallest first.
#define DPH(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a), as_int8(b), c)
// Two fp32 operands: hi + lo fp16 terms after a 2^e row scale that puts the row's largest magnitude in [2^14, 2^15).
static inline int rexp(float m) { return m > 0.0f ? clamp(14 - ilogb(m), -100, 100) : 0; }
#define SPLIT2(f, e, H, L)                                                                                             \
  {                                                                                                                    \
    const float f_ = ldexp((f), (e));                                                                                  \
    const half h_ = convert_half_rte(f_);                                                                              \
    H = as_ushort(h_);                                                                                                 \
    L = as_ushort(convert_half_rte(f_ - (float)h_));                                                                   \
  }
// Both operands split: lo·hi, hi·lo, hi·hi (lo·lo is below 2^-22 of the leading term).
#define DPH3(a, b, c) DPH(a[0], b[0], DPH(a[0], b[1], DPH(a[1], b[0], c)))
// The largest magnitude of row r over the subgroup's lanes and the n tiles x[0..n).
#define ROWMAX(x, n, r, m)                                                                                             \
  {                                                                                                                    \
    float m_ = 0.0f;                                                                                                   \
    for (int j_ = 0; j_ < (n); j_++) m_ = fmax(m_, fabs(x[j_][r]));                                                    \
    m = sub_group_reduce_max(m_);                                                                                      \
  }

#define KERNEL_WG(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel
#define RD16(base, wbytes, h, pitch, x, y, dst) \
  intel_sub_group_2d_block_read_16b_8r16x1c((global void *)(base), wbytes, h, pitch, (int2)(x, y), (private ushort *)&(dst))
#define RDT(base, wbytes, h, pitch, x, y, dst) \
  intel_sub_group_2d_block_read_transpose_32b_16r8x1c((global void *)(base), wbytes, h, pitch, (int2)(x, y), (private uint *)&(dst))
#define RDV(base, wbytes, h, pitch, x, y, dst) \
  intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)(base), wbytes, h, pitch, (int2)(x, y), (private uint *)&(dst))

KERNEL_WG(64) void gdn_prep(global const ushort *K, global const ushort *V, global const float *G,
                            global const float *BETA, global float *LA, global ushort *WB, global float *U0, int HK,
                            int HV, int NC, int skip) {
  // T = (I + A)^-1 takes the place of A once the inversion is done, saving local memory
  local float as_[CK * CK], la[CK], be[CK], co[CK];
  local float *ts = as_;
  const int c = get_group_id(0), h = get_group_id(1), lane = get_sub_group_local_id(), kh = h / (HV / HK);
  const int sg = get_sub_group_id(), T = NC * CK;
  const global ushort *Kh = K + kh * D, *Vh = V + h * D;
  // Cumulative decay within the chunk (scan in packets of 16, fixed order; subgroup 0).
  float carry = 0.0f;
  for (int b = 0; b < CK / 16 && sg == 0; b++) {
    const int t = c * CK + b * 16 + lane;
    // subnormal or zero g (very strong gate): log g bounded, otherwise −∞ − (−∞) = NaN below; e^−87 < 1e−37
    const float s = sub_group_scan_inclusive_add(fmax(log(G[(long)t * HV + h]), -87.0f)) + carry;
    const float bt = BETA[(long)t * HV + h];
    la[b * 16 + lane] = s;
    be[b * 16 + lane] = bt;
    co[b * 16 + lane] = bt * exp(s);
    LA[((long)h * NC + c) * CK + b * 16 + lane] = s;
    carry = sub_group_broadcast(s, 15);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // A[t, i] = beta_t e^(la_t - la_i) (k_t . k_i), i < t; row tiles {sg, 7 - sg} balance the triangular load.
  for (int w = 0; w < 2 && !(skip & 1); w++) {
    const int ti = w == 0 ? sg : CK / 8 - 1 - sg;
    float8 acc[CK / 16];
    for (int tj = 0; tj < CK / 16; tj++) acc[tj] = 0.0f;
    for (int kk = 0; kk < D / 16; kk++) {
      ushort8 a;
      RD16(Kh, D * 2, T, HK * D * 2, kk * 16, c * CK + ti * 8, a);
      for (int tj = 0; tj <= ti / 2; tj++) {
        uint8 b;
        RDT(Kh, D * 2, T, HK * D * 2, kk * 8, c * CK + tj * 16, b);
        acc[tj] = DPAS(a, b, acc[tj]);
      }
    }
    for (int tj = 0; tj < CK / 16; tj++) {
      const int i = tj * 16 + lane;
      for (int r = 0; r < 8; r++) {
        const int t = ti * 8 + r;
        as_[t * CK + i] = i < t ? be[t] * ((skip & 8) ? 1.0f : exp(la[t] - la[i])) * acc[tj][r] : 0.0f;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // T = (I + A)^-1 by 16 x 16 blocks (unit lower triangular), fixed summation order; L and D are read as broadcasts.
  local float dsm[CK / 16][16 * 16];
  for (int bi = sg; bi == sg && !(skip & 2); bi++) {        // diagonal block bi = subgroup
    float d[16];
#pragma unroll
    for (int r = 0; r < 16; r++) {
      float x = r == lane ? 1.0f : 0.0f;
#pragma unroll
      for (int k = 0; k < r; k++) x -= as_[(bi * 16 + r) * CK + bi * 16 + k] * d[k];
      d[r] = lane <= r ? x : 0.0f;
    }
#pragma unroll
    for (int r = 0; r < 16; r++) dsm[bi][r * 16 + lane] = d[r];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float tcol[CK / 16][16];                           // T_kj, the lane's column, k = bj..3
  for (int bj = sg; bj == sg && !(skip & 2); bj++) {        // column block bj = subgroup
#pragma unroll
    for (int r = 0; r < 16; r++) tcol[bj][r] = dsm[bj][r * 16 + lane];
    for (int bi = bj + 1; bi < CK / 16; bi++) {
      float x[16];
#pragma unroll
      for (int r = 0; r < 16; r++) x[r] = 0.0f;
      for (int bk = bj; bk < bi; bk++)
#pragma unroll
        for (int k = 0; k < 16; k++) {
          const float tk = tcol[bk][k];
#pragma unroll
          for (int r = 0; r < 16; r++) x[r] += as_[(bi * 16 + r) * CK + bk * 16 + k] * tk;
        }
#pragma unroll
      for (int r = 0; r < 16; r++) {
        float y = 0.0f;
#pragma unroll
        for (int k = 0; k <= r; k++) y -= dsm[bi][r * 16 + k] * x[k];
        tcol[bi][r] = y;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);                      // A is no longer read: T is written in its place
  for (int bj = sg; bj == sg && !(skip & 2); bj++) {
    for (int bi = 0; bi < CK / 16; bi++)
#pragma unroll
      for (int r = 0; r < 16; r++) ts[(bi * 16 + r) * CK + bj * 16 + lane] = bi < bj ? 0.0f : tcol[bi][r];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // W = (T.diag(beta e^la)).K then U0 = (T.diag(beta)).V; T's column scaling precedes the bf16 rounding of operand A.
  for (int pass = 0; pass < 2 && !(skip & 4); pass++) {
    const local float *sc = pass == 0 ? co : be;
    for (int w = 0; w < 2; w++) {                    // row tiles {sg, 7 − sg}, all columns
      const int ti = w == 0 ? sg : CK / 8 - 1 - sg;
      ushort8 a[CK / 16][3];
      for (int q = 0; q <= ti / 2; q++) {
        const float f = sc[q * 16 + lane];
        for (int r = 0; r < 8; r++) SPLIT3(ts[(ti * 8 + r) * CK + q * 16 + lane] * f, a[q][0][r], a[q][1][r], a[q][2][r]);
      }
      float8 acc[D / 16];
      for (int tj = 0; tj < D / 16; tj++) acc[tj] = 0.0f;
      for (int q = 0; q <= ti / 2; q++)              // T[t][i] = 0 if i > t
        for (int tj = 0; tj < D / 16; tj++) {
          uint8 b;
          if (pass == 0)
            RDV(Kh, D * 2, T, HK * D * 2, tj * 16, c * CK + q * 16, b);
          else
            RDV(Vh, D * 2, T, HV * D * 2, tj * 16, c * CK + q * 16, b);
          acc[tj] = DPAS(a[q][2], b, DPAS(a[q][1], b, DPAS(a[q][0], b, acc[tj])));
        }
      const int row = (c * HV + h) * CK + ti * 8;
      int ew[8];
      for (int r = 0; r < 8 && pass == 0; r++) {
        float m;
        ROWMAX(acc, D / 16, r, m);
        ew[r] = rexp(m);
        if (lane == 0) ((global float *)(WB + 2L * NC * HV * CK * D))[row + r] = (float)ew[r];
      }
      for (int tj = 0; tj < D / 16; tj++) {
        if (pass == 0) {
          ushort8 o[2];
          for (int r = 0; r < 8; r++) SPLIT2(acc[tj][r], ew[r], o[0][r], o[1][r]);
          for (int p = 0; p < 2; p++)
            intel_sub_group_2d_block_write_16b_8r16x1c((global void *)(WB + (long)p * NC * HV * CK * D), D * 2,
                                                       NC * HV * CK, D * 2, (int2)(tj * 16, row), (private ushort *)&o[p]);
        } else
          intel_sub_group_2d_block_write_32b_8r16x1c((global void *)U0, D * 4, NC * HV * CK, D * 4,
                                                     (int2)(tj * 16, row), (private uint *)&acc[tj]);
      }
    }
  }
}

// State recurrence: one subgroup per block of 8 v rows of S, S in registers, the next chunk's tiles prefetched (2D).
#define RV 8
KERNEL_WG(16 * (D / RV)) void gdn_state(global const ushort *K, global const ushort *WB, global const float *U0,
                                        global const float *LA, global const float *S0, global float *SF,
                                        global ushort *HS, global ushort *VN, int HK, int HV, int NC) {
  const int h = get_group_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id(), kh = h / (HV / HK);
  const int v0 = sg * RV, T = NC * CK;
  const global ushort *Kh = K + kh * D;
  // W, HS and VN: hi and lo fp16 planes, then their rows' exponents (float) in the third plane's place
  const global float *WE = (const global float *)(WB + 2L * NC * HV * CK * D);
  global float *HE = (global float *)(HS + 2L * NC * HV * D * D), *VE = (global float *)(VN + 2L * NC * HV * D * CK);
  float8 s[D / 16];
  for (int tj = 0; tj < D / 16; tj++)
    intel_sub_group_2d_block_read_32b_8r16x1c((global void *)S0, D * 4, HV * D, D * 4, (int2)(tj * 16, h * D + v0),
                                              (private uint *)&s[tj]);
  for (int c = 0; c < NC; c++) {
    const int row = (c * HV + h) * D + v0;           // v row of HS and VN
    const int trow = (c * HV + h) * CK;              // t row of WB and U0
    if (c + 1 < NC) {                                // next chunk: W, K
      const int nrow = ((c + 1) * HV + h) * CK;
      for (int tt = 0; tt < CK / 16; tt++) {
        intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void *)WB, D * 2, NC * HV * CK, D * 2, (int2)(0, nrow + tt * 16));
        intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void *)WB, D * 2, NC * HV * CK, D * 2, (int2)(64, nrow + tt * 16));
        intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void *)Kh, D * 2, T, HK * D * 2, (int2)(0, (c + 1) * CK + tt * 16));
        intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void *)Kh, D * 2, T, HK * D * 2, (int2)(64, (c + 1) * CK + tt * 16));
      }
    }
    int es[8];
    for (int r = 0; r < 8; r++) {
      float m;
      ROWMAX(s, D / 16, r, m);
      es[r] = rexp(m);
      if (lane == 0) HE[row + r] = (float)es[r];
    }
    for (int tj = 0; tj < D / 16; tj++) {
      ushort8 o[2];
      for (int r = 0; r < 8; r++) SPLIT2(s[tj][r], es[r], o[0][r], o[1][r]);
      for (int p = 0; p < 2; p++)
        intel_sub_group_2d_block_write_16b_8r16x1c((global void *)(HS + (long)p * NC * HV * D * D), D * 2, NC * HV * D,
                                                   D * 2, (int2)(tj * 16, row), (private ushort *)&o[p]);
    }
    // Uᵀ[v][t] = U0[t][v] − Σ_k S[v][k] W[t][k].
    float8 u[CK / 16];
    for (int tt = 0; tt < CK / 16; tt++) u[tt] = 0.0f;
    for (int kk = 0; kk < D / 16; kk++) {
      ushort8 a[2];
      for (int r = 0; r < 8; r++) SPLIT2(s[kk][r], es[r], a[0][r], a[1][r]);
      for (int tt = 0; tt < CK / 16; tt++) {
        uint8 b[2];
        for (int p = 0; p < 2; p++)
          RDT(WB + (long)p * NC * HV * CK * D, D * 2, NC * HV * CK, D * 2, kk * 8, trow + tt * 16, b[p]);
        u[tt] = DPH3(a, b, u[tt]);
      }
    }
    float8 ut[CK / 16];
    for (int tt = 0; tt < CK / 16; tt++) {
      const int ew = (int)WE[trow + tt * 16 + lane];
      float8 u0;
      RDT(U0, D * 4, NC * HV * CK, D * 4, v0, trow + tt * 16, u0);
      for (int r = 0; r < 8; r++) ut[tt][r] = u0[r] - ldexp(u[tt][r], -(es[r] + ew));
    }
    int ev[8];
    for (int r = 0; r < 8; r++) {
      float m;
      ROWMAX(ut, CK / 16, r, m);
      ev[r] = rexp(m);
      if (lane == 0) VE[row + r] = (float)ev[r];
    }
    const float lc = LA[((long)h * NC + c) * CK + CK - 1];
    ushort8 a[CK / 16][3];                           // Uᵀ·diag(e^(la_C − la)), A operand of the update
    for (int tt = 0; tt < CK / 16; tt++) {
      const float d = exp(lc - LA[((long)h * NC + c) * CK + tt * 16 + lane]);
      ushort8 o[2];
      for (int r = 0; r < 8; r++) {
        SPLIT2(ut[tt][r], ev[r], o[0][r], o[1][r]);
        SPLIT3(ut[tt][r] * d, a[tt][0][r], a[tt][1][r], a[tt][2][r]);
      }
      for (int p = 0; p < 2; p++)
        intel_sub_group_2d_block_write_16b_8r16x1c((global void *)(VN + (long)p * NC * HV * D * CK), CK * 2, NC * HV * D,
                                                   CK * 2, (int2)(tt * 16, row), (private ushort *)&o[p]);
    }
    // S ← e^la_C·S + Σ_t a[v][t] K[t][k].
    const float dc = exp(lc);
    for (int tj = 0; tj < D / 16; tj++) {
      s[tj] *= dc;
      for (int tt = 0; tt < CK / 16; tt++) {
        uint8 b;
        RDV(Kh, D * 2, T, HK * D * 2, tj * 16, c * CK + tt * 16, b);
        s[tj] = DPAS(a[tt][2], b, DPAS(a[tt][1], b, DPAS(a[tt][0], b, s[tj])));
      }
    }
  }
  for (int tj = 0; tj < D / 16; tj++)
    intel_sub_group_2d_block_write_32b_8r16x1c((global void *)SF, D * 4, HV * D, D * 4, (int2)(tj * 16, h * D + v0),
                                               (private uint *)&s[tj]);
}

// Output: P = M . Q.K^T and Q go to local memory in fp16 with row exponents, then Y = e^la.Q.S^T + P.U.
KERNEL_WG(128) void gdn_out(global const ushort *Q, global const ushort *K, global const float *LA,
                            global const ushort *HS, global const ushort *VN, global ushort *Y, int HK, int HV,
                            int NC) {
  local ushort pa[2][(CK / 8) * (CK / 16) * 128], qa[(CK / 8) * (D / 16) * 128];
  local float la[CK], qe[CK], pe[CK];
  const int c = get_group_id(0), h = get_group_id(1), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int kh = h / (HV / HK), T = NC * CK;
  const global ushort *Qh = Q + kh * D, *Kh = K + kh * D;
  if (sg < CK / 16) la[sg * 16 + lane] = LA[((long)h * NC + c) * CK + sg * 16 + lane];
  {
    const int ti = sg;                               // rows t = 8 sg .. 8 sg + 7
    ushort8 aq[D / 16];
    float8 qf[D / 16];
    for (int kk = 0; kk < D / 16; kk++) {
      RD16(Qh, D * 2, T, HK * D * 2, kk * 16, c * CK + ti * 8, aq[kk]);
      for (int r = 0; r < 8; r++) qf[kk][r] = as_float((uint)aq[kk][r] << 16);
    }
    int eq[8];
    for (int r = 0; r < 8; r++) {
      float m;
      ROWMAX(qf, D / 16, r, m);
      eq[r] = rexp(m);
      if (lane == 0) qe[ti * 8 + r] = (float)eq[r];
    }
    for (int kk = 0; kk < D / 16; kk++) {            // a bf16 q scaled by 2^e is exact in fp16 down to 2^-24
      ushort8 hq;
      for (int r = 0; r < 8; r++) hq[r] = as_ushort(convert_half_rte(ldexp(qf[kk][r], eq[r])));
      intel_sub_group_block_write_us8(qa + (ti * (D / 16) + kk) * 128, hq);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float8 pf[CK / 16];
    for (int tj = 0; tj < CK / 16; tj++) {
      float8 acc = 0.0f;
      for (int kk = 0; kk < D / 16 && tj <= ti / 2; kk++) {
        uint8 b;
        RDT(Kh, D * 2, T, HK * D * 2, kk * 8, c * CK + tj * 16, b);
        acc = DPAS(aq[kk], b, acc);
      }
      const int i = tj * 16 + lane;
      for (int r = 0; r < 8; r++) {
        const int t = ti * 8 + r;
        pf[tj][r] = i <= t ? exp(la[t] - la[i]) * acc[r] : 0.0f;
      }
    }
    int ep[8];
    for (int r = 0; r < 8; r++) {
      float m;
      ROWMAX(pf, CK / 16, r, m);
      ep[r] = rexp(m);
      if (lane == 0) pe[ti * 8 + r] = (float)ep[r];
    }
    for (int tj = 0; tj < CK / 16; tj++) {
      ushort8 o[2];
      for (int r = 0; r < 8; r++) SPLIT2(pf[tj][r], ep[r], o[0][r], o[1][r]);
      for (int p = 0; p < 2; p++) intel_sub_group_block_write_us8(pa[p] + (ti * (CK / 16) + tj) * 128, o[p]);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int n = sg;                                  // columns v = 16 n .. 16 n + 15
  const int vrow = (c * HV + h) * D + n * 16;
  float8 acc[CK / 8];
  for (int ti = 0; ti < CK / 8; ti++) acc[ti] = 0.0f;
  for (int kk = 0; kk < D / 16; kk++) {              // Q·Sᵀ: B = (S[v][2p], S[v][2p+1]), transposed read of HS
    uint8 b[2];
    for (int p = 0; p < 2; p++) RDT(HS + (long)p * NC * HV * D * D, D * 2, NC * HV * D, D * 2, kk * 8, vrow, b[p]);
    for (int ti = 0; ti < CK / 8; ti++) {
      const ushort8 qv = intel_sub_group_block_read_us8(qa + (ti * (D / 16) + kk) * 128);
      acc[ti] = DPH(qv, b[0], DPH(qv, b[1], acc[ti]));
    }
  }
  const global float *HE = (const global float *)(HS + 2L * NC * HV * D * D);
  const global float *VE = (const global float *)(VN + 2L * NC * HV * D * CK);
  const int eh = (int)HE[vrow + lane], ev = (int)VE[vrow + lane];
  for (int ti = 0; ti < CK / 8; ti++)
    for (int r = 0; r < 8; r++) acc[ti][r] = ldexp(acc[ti][r], -((int)qe[ti * 8 + r] + eh)) * exp(la[ti * 8 + r]);
  float8 pu[CK / 8];
  for (int ti = 0; ti < CK / 8; ti++) pu[ti] = 0.0f;
  for (int kk = 0; kk < CK / 16; kk++) {             // P·U: B = (U[2p][v], U[2p+1][v]), transposed read of VN
    uint8 b[2];
    for (int p = 0; p < 2; p++) RDT(VN + (long)p * NC * HV * D * CK, CK * 2, NC * HV * D, CK * 2, kk * 8, vrow, b[p]);
    for (int ti = kk * 2; ti < CK / 8; ti++) {
      ushort8 pv[2];
      for (int p = 0; p < 2; p++) pv[p] = intel_sub_group_block_read_us8(pa[p] + (ti * (CK / 16) + kk) * 128);
      pu[ti] = DPH3(pv, b, pu[ti]);
    }
  }
  for (int ti = 0; ti < CK / 8; ti++)
    for (int r = 0; r < 8; r++) acc[ti][r] += ldexp(pu[ti][r], -((int)pe[ti * 8 + r] + ev));
  for (int ti = 0; ti < CK / 8; ti++) {
    ushort8 o;
    for (int r = 0; r < 8; r++) o[r] = bf(acc[ti][r]);
    intel_sub_group_2d_block_write_16b_8r16x1c((global void *)Y, HV * D * 2, T, HV * D * 2,
                                               (int2)(h * D + n * 16, c * CK + ti * 8), (private ushort *)&o);
  }
}
