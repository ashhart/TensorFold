// Qwen3.8 Flash Next MoE top-k, parallel expert compaction, Xe2 DPAS MoE projections, and logits kernels.

// Exact top-10 of 512 routed experts (ties: lowest id) + slot 10 shared-expert sigmoid gate. Grid: (R), WG = 128.
KERNEL(128) void fn_route_topk(global const float *L, global int *PICK, global float *WTS) {
  local float sv[8];
  local int si[8];
  const int r = get_group_id(0), t = get_local_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  global const float *row = L + (long)r * 513;
  float v[4];
  for (int j = 0; j < 4; j++) v[j] = row[t + j * 128];
  float top = 0.0f, total = 0.0f, mine = 0.0f;
  int my_pick = 0;
  for (int k = 0; k < 10; k++) {
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int j = 0; j < 4; j++) {
      const int idx = t + j * 128;
      if (v[j] > bv || (v[j] == bv && idx < bi)) {
        bv = v[j];
        bi = idx;
      }
    }
    const float sg_max = sub_group_reduce_max(bv);
    const int sg_id = sub_group_reduce_min(bv == sg_max ? bi : 0x7fffffff);
    if (lane == 0) {
      sv[sg] = sg_max;
      si[sg] = sg_id;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float win_v = sv[0];
    int win_i = si[0];
    for (int w = 1; w < 8; w++) {
      if (sv[w] > win_v || (sv[w] == win_v && si[w] < win_i)) {
        win_v = sv[w];
        win_i = si[w];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (k == 0) top = win_v;
    for (int j = 0; j < 4; j++) {
      if (t + j * 128 == win_i) v[j] = -INFINITY;
    }
    const float ex = exp(win_v - top);
    total += ex;
    if (t == k) {
      mine = ex;
      my_pick = win_i;
    }
  }
  if (t < 10) {
    PICK[(long)r * 11 + t] = my_pick;
    WTS[(long)r * 11 + t] = rbf(mine / total);
  }
  if (t == 0) {
    PICK[(long)r * 11 + 10] = 512;
    WTS[(long)r * 11 + 10] = rbf(tf_sigmoid(rbf(row[512])));
  }
}

// Parallel expert compaction for R = 1..2048 into 32-pair Xe2 DPAS tiles while preserving ascending slot order per row.
KERNEL(256) void fn_moe_compact(global const int *PICK, global int *ACTIVE, global int *EXP_GRP, global int *N_ACT,
                                global int *ROW_LIST, int R, int first_e, int local_e) {
  local int e_cnt[512];
  local int e_cur[512];
  local int g_off[512];
  const int tid = get_local_id(0);
  if (R == 1) {
    if (tid == 0) {
      int rc = 0;
      for (int s = 0; s < 10; s++) {
        const int le = PICK[s] - first_e;
        if (le >= 0 && le < local_e) {
          ACTIVE[rc] = (s << 10) | le;
          EXP_GRP[rc] = (le << 22) | (1 << 16) | rc;
          ROW_LIST[rc] = rc;
          rc++;
        }
      }
      ROW_LIST[rc] = -1;
      N_ACT[0] = rc;
      N_ACT[1] = rc;
    }
    return;
  }
  for (int le = tid; le < local_e; le += 256) e_cnt[le] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int r = tid; r < R; r += 256) {
    global const int *pr = PICK + (long)r * 11;
    for (int s = 0; s < 10; s++) {
      const int le = pr[s] - first_e;
      if (le >= 0 && le < local_e) atomic_inc(&e_cnt[le]);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (tid == 0) {
    int p_acc = 0, g_acc = 0;
    for (int le = 0; le < local_e; le++) {
      const int c = e_cnt[le];
      e_cur[le] = p_acc;
      g_off[le] = g_acc;
      p_acc += c;
      g_acc += (c + 31) >> 5;
    }
    N_ACT[0] = g_acc;
    N_ACT[1] = p_acc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int le = tid; le < local_e; le += 256) {
    const int c = e_cnt[le];
    if (c == 0) continue;
    const int p0 = e_cur[le], g0 = g_off[le];
    for (int g = 0; g * 32 < c; g++) {
      const int m = min(32, c - g * 32);
      EXP_GRP[g0 + g] = (le << 22) | (m << 16) | (p0 + g * 32);
    }
  }

  for (int r = tid; r < R; r += 256) {
    global const int *pr = PICK + (long)r * 11;
    int rc = 0;
    for (int s = 0; s < 10; s++) {
      const int le = pr[s] - first_e;
      if (le >= 0 && le < local_e) {
        const int pair_idx = atomic_inc(&e_cur[le]);
        ACTIVE[pair_idx] = (r << 14) | (s << 10) | le;
        ROW_LIST[(long)r * 11 + rc++] = pair_idx;
      }
    }
    ROW_LIST[(long)r * 11 + rc] = -1;
  }
}

// Per-pair Routed Gate+Up for R = 1..16 (WG = 64: 4-way Split-K over K=2560). Grid: (40, R * min(10, local_e)).
KERNEL(64) void fn_moe_up_1(global const ushort *X, global const int *ACTIVE, global const int *N_ACT,
                            global const uint *GW, global const ushort *GS, global const ushort *GB,
                            global const uint *UW, global const ushort *US, global const ushort *UB,
                            global ushort *MOE_ACT) {
  local float gp[4][16], up[4][16];
  const int pair_idx = (int)get_group_id(1);
  if (pair_idx >= N_ACT[1]) return;
  const int pk = ACTIVE[pair_idx];
  const int row_i = (pk >> 14) & 0x3FFF;
  const int le = pk & 0x3FF;
  global const ushort *xr = X + (long)row_i * 2560;
  const int nb_col = (int)get_group_id(0);
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const long nb = (long)le * 40 + nb_col;
  float gq = 0.0f, uq = 0.0f;
  for (int g = sg * 20; g < (sg + 1) * 20; g++) {
    const short a_lo = (short)xr[g * 32 + lane];
    const short a_hi = (short)xr[g * 32 + 16 + lane];
    const float xs = dpas32_1(a_lo, a_hi, ONES_BF16, ONES_BF16);
    int8 gb0, gb1, ub0, ub1; float gsc, gbi, usc, ubi;
    load_q4_tile(GW, GS, GB, nb, 80, g, lane, &gb0, &gb1, &gsc, &gbi);
    load_q4_tile(UW, US, UB, nb, 80, g, lane, &ub0, &ub1, &usc, &ubi);
    const float gd = dpas32_1(a_lo, a_hi, gb0, gb1);
    const float ud = dpas32_1(a_lo, a_hi, ub0, ub1);
    gq = fma(xs, gbi, fma(gd, gsc, gq));
    uq = fma(xs, ubi, fma(ud, usc, uq));
  }
  gp[sg][lane] = gq;
  up[sg][lane] = uq;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg == 0) {
    const float gr = (((gp[0][lane]  + gp[1][lane]) + gp[2][lane] ) + gp[3][lane]);
    const float ur = (((up[0][lane] + up[1][lane]) + up[2][lane]) + up[3][lane]);
    MOE_ACT[(long)pair_idx * 640 + nb_col * 16 + lane] = to_bf(bsilu(rbf(gr)) * rbf(ur));
  }
}

// Multi-row Expert-Grouped Routed Gate+Up on Xe2 DPAS: WG = 256 (4 M-sgs x 4 N-sgs = [32 pairs x 64 cols]).
KERNEL(256) void fn_moe_up_dpas(global const ushort *X, global const int *ACTIVE, global const int *EXP_GRP,
                                global const int *N_ACT,
                                global const uint *GW, global const ushort *GS, global const ushort *GB,
                                global const uint *UW, global const ushort *US, global const ushort *UB,
                                global ushort *MOE_ACT) {
  local int rows_s[32];
  local ushort xl[32][160];
  const int grp_idx = (int)get_group_id(1);
  if (grp_idx >= N_ACT[0]) return;
  const uint grp = (uint)EXP_GRP[grp_idx];
  const int le = (int)((grp >> 22) & 0x1FFu);
  const int m = (int)((grp >> 16) & 0x3Fu);
  const int p0 = (int)(grp & 0xFFFFu);
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id(), lid = get_local_id(0);
  if (lid < m) rows_s[lid] = (ACTIVE[p0 + lid] >> 14) & 0x3FFF;
  barrier(CLK_LOCAL_MEM_FENCE);

  const int sg_m = sg >> 2, sg_n = sg & 3;
  const int nb_col = (int)get_group_id(0) * 4 + sg_n;
  const bool nb_ok = (nb_col < 40);
  const long safe_nb = nb_ok ? ((long)le * 40 + nb_col) : 0;
  const int m_pad = min(32, (m + 7) & ~7);
  float8 ga = 0.0f, ua = 0.0f;

  for (int q = 0; q < 4; q++) {
    float8 g0 = 0.0f, u0 = 0.0f;
    for (int g = q * 20; g < (q + 1) * 20; g += 5) {
      for (int e = lid; e < m_pad * 40; e += 256) {
        const int rr = e / 40, cc = (e % 40) * 4;
        const ushort4 v = (rr < m) ? vload4(0, X + (long)rows_s[rr] * 2560 + g * 32 + cc) : (ushort4)(0);
        xl[rr][cc + 0] = v.s0; xl[rr][cc + 1] = v.s1; xl[rr][cc + 2] = v.s2; xl[rr][cc + 3] = v.s3;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (sg_m * 8 < m) {
        for (int sub = 0; sub < 5; sub++) {
          short8 a0, a1;
          const int kb = sub * 32;
          for (int r = 0; r < 8; r++) {
            a0[r] = (short)xl[sg_m * 8 + r][kb + lane];
            a1[r] = (short)xl[sg_m * 8 + r][kb + 16 + lane];
          }
          const float8 xs = dpas32_8(a0, a1, ONES_BF16, ONES_BF16);
          int8 b0 = 0, b1 = 0;
          float sc = 0.0f, bi = 0.0f;
          if (nb_ok) load_q4_tile(GW, GS, GB, safe_nb, 80, g + sub, lane, &b0, &b1, &sc, &bi);
          const float8 gd = dpas32_8(a0, a1, b0, b1);
          for (int r = 0; r < 8; r++) g0[r] = fma(xs[r], bi, fma(gd[r], sc, g0[r]));
          if (nb_ok) load_q4_tile(UW, US, UB, safe_nb, 80, g + sub, lane, &b0, &b1, &sc, &bi);
          const float8 ud = dpas32_8(a0, a1, b0, b1);
          for (int r = 0; r < 8; r++) u0[r] = fma(xs[r], bi, fma(ud[r], sc, u0[r]));
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    ga = (q == 0) ? g0 : (ga + g0);
    ua = (q == 0) ? u0 : (ua + u0);
  }

  if (nb_ok) {
    const int col = nb_col * 16 + lane;
    for (int r = 0; r < 8; r++) {
      const int j0 = sg_m * 8 + r;
      if (j0 < m) MOE_ACT[(long)(p0 + j0) * 640 + col] = to_bf(bsilu(rbf(ga[r])) * rbf(ua[r]));
    }
  }
}

// Per-pair Routed Down for R = 1..16 (K=640 -> 20 groups, N=2560 = 160 tiles). Grid: (160, R * min(10, local_e)).
KERNEL(64) void fn_moe_down_1(global const ushort *MOE_ACT, global const int *ACTIVE, global const int *N_ACT,
                              global const uint *DW, global const ushort *DS, global const ushort *DB,
                              global float *MOE_DOWN) {
  local float kp[4][16];
  const int pair_idx = (int)get_group_id(1);
  if (pair_idx >= N_ACT[1]) return;
  const int le = ACTIVE[pair_idx] & 0x3FF;
  const int nb_col = (int)get_group_id(0);
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const long nb = (long)le * 160 + nb_col;
  global const ushort *xa = MOE_ACT + (long)pair_idx * 640;
  float qacc = 0.0f;
  for (int g = sg * 5; g < (sg + 1) * 5; g++) {
    const short a_lo = (short)xa[g * 32 + lane];
    const short a_hi = (short)xa[g * 32 + 16 + lane];
    const float xs = dpas32_1(a_lo, a_hi, ONES_BF16, ONES_BF16);
    int8 b0, b1; float sc, bi;
    load_q4_tile(DW, DS, DB, nb, 20, g, lane, &b0, &b1, &sc, &bi);
    const float d = dpas32_1(a_lo, a_hi, b0, b1);
    qacc = fma(xs, bi, fma(d, sc, qacc));
  }
  kp[sg][lane] = qacc;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg == 0) {
    MOE_DOWN[(long)pair_idx * 2560 + nb_col * 16 + lane] = (((kp[0][lane]  + kp[1][lane]) + kp[2][lane] ) + kp[3][lane]);
  }
}

// Multi-row Expert-Grouped Routed Down on Xe2 DPAS: WG = 128 ([32 active pairs x 64 cols], K=640).
KERNEL(128) void fn_moe_down_dpas(global const ushort *MOE_ACT, global const int *EXP_GRP, global const int *N_ACT,
                                  global const uint *DW, global const ushort *DS, global const ushort *DB,
                                  global float *MOE_DOWN) {
  local ushort xl[32][160];
  const int grp_idx = (int)get_group_id(1);
  if (grp_idx >= N_ACT[0]) return;
  const uint grp = (uint)EXP_GRP[grp_idx];
  const int le = (int)((grp >> 22) & 0x1FFu);
  const int m = (int)((grp >> 16) & 0x3Fu);
  const int p0 = (int)(grp & 0xFFFFu);
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id(), lid = get_local_id(0);
  const int sg_m = sg >> 2, sg_n = sg & 3;
  const int nb_col = (int)get_group_id(0) * 4 + sg_n;
  const long nb = (long)le * 160 + nb_col;
  const int m_pad = min(32, (m + 7) & ~7);
  float8 acc0 = 0.0f, acc1 = 0.0f;

  for (int q = 0; q < 4; q++) {
    for (int e = lid; e < m_pad * 40; e += 128) {
      const int rr = e / 40, cc = (e % 40) * 4;
      const ushort4 v = (rr < m) ? vload4(0, MOE_ACT + (long)(p0 + rr) * 640 + q * 160 + cc) : (ushort4)(0);
      xl[rr][cc + 0] = v.s0; xl[rr][cc + 1] = v.s1; xl[rr][cc + 2] = v.s2; xl[rr][cc + 3] = v.s3;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float8 q0 = 0.0f, q1 = 0.0f;
    if (sg_m * 16 < m) {
      for (int lg = 0; lg < 5; lg++) {
        const int g = q * 5 + lg;
        int8 b0, b1; float sc, bi;
        load_q4_tile(DW, DS, DB, nb, 20, g, lane, &b0, &b1, &sc, &bi);
        short8 a0_0, a0_1, a1_0, a1_1;
        for (int r = 0; r < 8; r++) {
          a0_0[r] = (short)xl[sg_m * 16 + r][lg * 32 + lane];
          a0_1[r] = (short)xl[sg_m * 16 + r][lg * 32 + 16 + lane];
          a1_0[r] = (short)xl[sg_m * 16 + 8 + r][lg * 32 + lane];
          a1_1[r] = (short)xl[sg_m * 16 + 8 + r][lg * 32 + 16 + lane];
        }
        const float8 xs0 = dpas32_8(a0_0, a0_1, ONES_BF16, ONES_BF16);
        const float8 d0 = dpas32_8(a0_0, a0_1, b0, b1);
        for (int r = 0; r < 8; r++) q0[r] = fma(xs0[r], bi, fma(d0[r], sc, q0[r]));
        if (sg_m * 16 + 8 < m) {
          const float8 xs1 = dpas32_8(a1_0, a1_1, ONES_BF16, ONES_BF16);
          const float8 d1 = dpas32_8(a1_0, a1_1, b0, b1);
          for (int r = 0; r < 8; r++) q1[r] = fma(xs1[r], bi, fma(d1[r], sc, q1[r]));
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    acc0 = (q == 0) ? q0 : (acc0 + q0);
    acc1 = (q == 0) ? q1 : (acc1 + q1);
  }

  const int col = nb_col * 16 + lane;
  for (int r = 0; r < 8; r++) {
    const int j0 = sg_m * 16 + r;
    if (j0 < m) MOE_DOWN[(long)(p0 + j0) * 2560 + col] = acc0[r];
    const int j1 = j0 + 8;
    if (j1 < m) MOE_DOWN[(long)(p0 + j1) * 2560 + col] = acc1[r];
  }
}

// 1..16-row Shared Expert SwiGLU Up on Xe2 DPAS (grid: (sh_w / 16, R), 128 B SLM).
KERNEL(64) void fn_moe_sh_up_1(global const ushort *X,
                               global const uint *GW, global const ushort *GS, global const ushort *GB,
                               global const uint *UW, global const ushort *US, global const ushort *UB,
                               global ushort *SH_ACT, int sh_w) {
  local float gp[4][16], up[4][16];
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int nb = (int)get_group_id(0), row_i = (int)get_group_id(1);
  if (nb >= sh_w / 16) return;
  global const ushort *xr = X + (long)row_i * 2560;
  float gq = 0.0f, uq = 0.0f;
  for (int g = sg * 20; g < (sg + 1) * 20; g++) {
    const short a_lo = (short)xr[g * 32 + lane];
    const short a_hi = (short)xr[g * 32 + 16 + lane];
    const float xs = dpas32_1(a_lo, a_hi, ONES_BF16, ONES_BF16);
    int8 gb0, gb1, ub0, ub1; float gsc, gbi, usc, ubi;
    load_q4_tile(GW, GS, GB, nb, 80, g, lane, &gb0, &gb1, &gsc, &gbi);
    load_q4_tile(UW, US, UB, nb, 80, g, lane, &ub0, &ub1, &usc, &ubi);
    const float gd = dpas32_1(a_lo, a_hi, gb0, gb1);
    const float ud = dpas32_1(a_lo, a_hi, ub0, ub1);
    gq = fma(xs, gbi, fma(gd, gsc, gq));
    uq = fma(xs, ubi, fma(ud, usc, uq));
  }
  gp[sg][lane] = gq;
  up[sg][lane] = uq;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg == 0) {
    const float gr = (((gp[0][lane] + gp[1][lane]) + gp[2][lane]) + gp[3][lane]);
    const float ur = (((up[0][lane] + up[1][lane]) + up[2][lane]) + up[3][lane]);
    SH_ACT[(long)row_i * sh_w + nb * 16 + lane] = to_bf(bsilu(rbf(gr)) * rbf(ur));
  }
}

KERNEL(128) void fn_moe_sh_up_dpas(global const ushort *X,
                                   global const uint *GW, global const ushort *GS, global const ushort *GB,
                                   global const uint *UW, global const ushort *US, global const ushort *UB,
                                   global ushort *SH_ACT, int sh_w, int R) {
  local ushort xl[32][64];
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id(), lid = get_local_id(0);
  const int sg_m = sg >> 1, sg_n = sg & 1;
  const int m0 = (int)get_group_id(1) * 32;
  const int nb = (int)get_group_id(0) * 2 + sg_n;
  const int col = nb * 16 + lane;
  const int nb_max = sh_w / 16;
  const bool nb_ok = (nb < nb_max);
  const long safe_nb = nb_ok ? (long)nb : 0;
  float8 ga = 0.0f, ua = 0.0f;
  for (int q = 0; q < 4; q++) {
    float8 g0 = 0.0f, u0 = 0.0f;
    for (int g = q * 20; g < (q + 1) * 20; g += 2) {
      for (int e = lid; e < 32 * 16; e += 128) {
        const int rr = e >> 4, cc = (e & 15) * 4;
        const ushort4 v = (m0 + rr < R) ? vload4(0, X + (long)(m0 + rr) * 2560 + g * 32 + cc) : (ushort4)(0);
        xl[rr][cc + 0] = v.s0; xl[rr][cc + 1] = v.s1; xl[rr][cc + 2] = v.s2; xl[rr][cc + 3] = v.s3;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int sub = 0; sub < 2; sub++) {
        short8 a0, a1;
        const int kb = sub * 32;
        for (int r = 0; r < 8; r++) {
          a0[r] = (short)xl[sg_m * 8 + r][kb + lane];
          a1[r] = (short)xl[sg_m * 8 + r][kb + 16 + lane];
        }
        const float8 xs = dpas32_8(a0, a1, ONES_BF16, ONES_BF16);
        int8 b0 = 0, b1 = 0;
        float sc = 0.0f, bi = 0.0f;
        if (nb_ok) load_q4_tile(GW, GS, GB, safe_nb, 80, g + sub, lane, &b0, &b1, &sc, &bi);
        const float8 gd = dpas32_8(a0, a1, b0, b1);
        for (int r = 0; r < 8; r++) g0[r] = fma(xs[r], bi, fma(gd[r], sc, g0[r]));
        if (nb_ok) load_q4_tile(UW, US, UB, safe_nb, 80, g + sub, lane, &b0, &b1, &sc, &bi);
        const float8 ud = dpas32_8(a0, a1, b0, b1);
        for (int r = 0; r < 8; r++) u0[r] = fma(xs[r], bi, fma(ud[r], sc, u0[r]));
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    ga = (q == 0) ? g0 : (ga + g0);
    ua = (q == 0) ? u0 : (ua + u0);
  }
  if (col < sh_w) {
    for (int r = 0; r < 8; r++) {
      const int row0 = m0 + sg_m * 8 + r;
      if (row0 < R) SH_ACT[(long)row0 * sh_w + col] = to_bf(bsilu(rbf(ga[r])) * rbf(ua[r]));
    }
  }
}

// Pure-bandwidth MoE Combine in exact ascending slot order (0..9 + shared slot 10): Grid (10, R), WG = 256.
KERNEL(256) void fn_moe_combine(global const float *MOE_DOWN, global const float *SH_OUT,
                                global const int *ACTIVE, global const int *ROW_LIST, global const float *WTS,
                                global float *OUT, int has_sh, int R) {
  const int col = (int)get_group_id(0) * 256 + (int)get_local_id(0);
  const int r = (int)get_group_id(1);
  if (col >= 2560 || r >= R) return;
  float total = 0.0f;
  for (int c = 0; c < 10; c++) {
    const int pair_idx = ROW_LIST[(long)r * 11 + c];
    if (pair_idx < 0) break;
    const int slot = (ACTIVE[pair_idx] >> 10) & 0xF;
    total = fma(MOE_DOWN[(long)pair_idx * 2560 + col], WTS[(long)r * 11 + slot], total);
  }
  if (has_sh) {
    total = fma(SH_OUT[(long)r * 2560 + col], WTS[(long)r * 11 + 10], total);
  }
  OUT[(long)r * 2560 + col] = total;
}
