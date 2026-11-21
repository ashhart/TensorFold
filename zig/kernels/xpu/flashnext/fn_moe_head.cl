// Qwen3.8 Flash Next MoE router, expert-grouped routed/shared MLP, and logits kernels.

// Router logits [R, 513] f32 = X [R, 2560] bf16 . W [513, 2560] bf16. Grid: ((513 + 3) / 4, R), WG = 64 (4 subgroups).
KERNEL(64) void fn_router_logits(global const ushort *X, global const ushort *W, global float *OUT, int R) {
  const int e = get_group_id(0) * 4 + get_sub_group_id();
  const int r = get_group_id(1);
  if (e >= 513 || r >= R) return;
  const int lane = get_sub_group_local_id();
  global const ushort *xr = X + (long)r * 2560;
  global const ushort *wr = W + (long)e * 2560;
  float acc = 0.0f;
  for (int i = lane * 8; i < 2560; i += 16 * 8) {
    const ushort8 xv = vload8(0, xr + i);
    const ushort8 wv = vload8(0, wr + i);
    for (int j = 0; j < 8; j++) acc = fma(bf(xv[j]), bf(wv[j]), acc);
  }
  acc = sub_group_reduce_add(acc);
  if (lane == 0) OUT[(long)r * 513 + e] = acc;
}

// Exact top-10 of 512 routed experts (ties: lowest id) + slot 10 shared-expert sigmoid gate. Grid: (R), WG = 128 (8
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
    // Subgroup best (tie -> lower id)
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

// Compact active (row, slot, local_expert) triples on this rank, grouped by local_expert (up to 16 rows/group) while
KERNEL(128) void fn_moe_compact(global const int *PICK, global int *ACTIVE, global int *EXP_GRP, global int *N_ACT,
                                global int *ROW_LIST, int R, int first_e, int local_e) {
  local int pick_s[64 * 10];
  local int slot_pair[64 * 10];
  local int e_cnt[512];
  local int e_off[512];
  local int g_off[512];
  const int tid = get_local_id(0);
  if (R == 1) {
    if (tid == 0) {
      int rc = 0;
      for (int s = 0; s < 10; s++) {
        const int le = PICK[s] - first_e;
        if (le >= 0 && le < local_e) {
          ACTIVE[rc] = (s << 10) | le;
          EXP_GRP[rc] = (le << 20) | (1 << 12) | rc;
          ROW_LIST[rc] = rc;
          rc++;
        }
      }
      ROW_LIST[rc] = -1;
      N_ACT[0] = rc;
    }
    return;
  }
  const int total_slots = R * 10;
  for (int idx = tid; idx < total_slots; idx += 128) {
    const int r = idx / 10, s = idx % 10;
    pick_s[idx] = PICK[(long)r * 11 + s] - first_e;
  }
  for (int le = tid; le < local_e; le += 128) e_cnt[le] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int le = tid; le < local_e; le += 128) {
    int c = 0;
    for (int idx = 0; idx < total_slots; idx++) {
      if (pick_s[idx] == le) c++;
    }
    e_cnt[le] = c;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (tid == 0) {
    int p_acc = 0, g_acc = 0;
    for (int le = 0; le < local_e; le++) {
      const int c = e_cnt[le];
      e_off[le] = p_acc;
      g_off[le] = g_acc;
      p_acc += c;
      g_acc += (c + 15) >> 4;
    }
    N_ACT[0] = g_acc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int le = tid; le < local_e; le += 128) {
    const int c = e_cnt[le];
    if (c == 0) continue;
    int p_at = e_off[le];
    for (int idx = 0; idx < total_slots; idx++) {
      if (pick_s[idx] == le) {
        const int r = idx / 10, s = idx % 10;
        const int pair_idx = p_at++;
        ACTIVE[pair_idx] = (r << 16) | (s << 10) | le;
        slot_pair[idx] = pair_idx;
      }
    }
    const int p0 = e_off[le], g0 = g_off[le];
    for (int g = 0; g * 16 < c; g++) {
      const int m = min(16, c - g * 16);
      EXP_GRP[g0 + g] = (le << 20) | (m << 12) | (p0 + g * 16);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int r = tid; r < R; r += 128) {
    int rc = 0;
    for (int s = 0; s < 10; s++) {
      const int le = pick_s[r * 10 + s];
      if (le >= 0 && le < local_e) {
        ROW_LIST[r * 11 + rc++] = slot_pair[r * 10 + s];
      }
    }
    ROW_LIST[r * 11 + rc] = -1;
  }
}

// Routed expert SwiGLU up projection grouped by local expert (up to 16 rows share 1 weight load!): Grid: (640 / 16,
KERNEL(256) void fn_moe_routed_up(global const ushort *X, global const int *ACTIVE, global const int *EXP_GRP,
                                  global const int *N_ACT,
                                  global const uint *GW, global const ushort *GS, global const ushort *GB,
                                  global const uint *UW, global const ushort *US, global const ushort *UB,
                                  global ushort *MOE_ACT) {
  local int rows_s[16];
  const int grp_idx = get_group_id(1);
  if (grp_idx >= N_ACT[0]) return;
  const int grp = EXP_GRP[grp_idx];
  const int le = (grp >> 20) & 0x3FF;
  const int m = (grp >> 12) & 0xFF;
  const int p0 = grp & 0xFFF;
  const int lid = get_local_id(0);
  if (lid < m) rows_s[lid] = ACTIVE[p0 + lid] >> 16;
  barrier(CLK_LOCAL_MEM_FENCE);

  const int col = get_group_id(0) * 16 + get_sub_group_id();
  const int lane = get_sub_group_local_id();
  const long erow = (long)le * 640 + col;
  const int words = 2560 / 8, groups = 2560 / 32;
  global const uint *gw = GW + erow * words, *uw = UW + erow * words;
  global const ushort *gs = GS + erow * groups, *gb = GB + erow * groups;
  global const ushort *us = US + erow * groups, *ub = UB + erow * groups;
  float g_acc[16], u_acc[16];
  for (int j = 0; j < m; j++) {
    g_acc[j] = 0.0f;
    u_acc[j] = 0.0f;
  }
  for (int wi = lane; wi < words; wi += 16) {
    const int g = wi >> 2;
    const uint pg = gw[wi], pu = uw[wi];
    const float gs_v = bf(gs[g]), gb_v = bf(gb[g]);
    const float us_v = bf(us[g]), ub_v = bf(ub[g]);
    const float qg0 = (float)(pg & 15u), qg1 = (float)((pg >> 4) & 15u), qg2 = (float)((pg >> 8) & 15u), qg3 = (float)((pg >> 12) & 15u);
    const float qg4 = (float)((pg >> 16) & 15u), qg5 = (float)((pg >> 20) & 15u), qg6 = (float)((pg >> 24) & 15u), qg7 = (float)((pg >> 28) & 15u);
    const float qu0 = (float)(pu & 15u), qu1 = (float)((pu >> 4) & 15u), qu2 = (float)((pu >> 8) & 15u), qu3 = (float)((pu >> 12) & 15u);
    const float qu4 = (float)((pu >> 16) & 15u), qu5 = (float)((pu >> 20) & 15u), qu6 = (float)((pu >> 24) & 15u), qu7 = (float)((pu >> 28) & 15u);
    for (int j = 0; j < m; j++) {
      const ushort8 xv = vload8(0, X + (long)rows_s[j] * 2560 + wi * 8);
      const float x0 = bf(xv.s0), x1 = bf(xv.s1), x2 = bf(xv.s2), x3 = bf(xv.s3);
      const float x4 = bf(xv.s4), x5 = bf(xv.s5), x6 = bf(xv.s6), x7 = bf(xv.s7);
      const float sx = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
      const float dg = x0 * qg0 + x1 * qg1 + x2 * qg2 + x3 * qg3 + x4 * qg4 + x5 * qg5 + x6 * qg6 + x7 * qg7;
      const float du = x0 * qu0 + x1 * qu1 + x2 * qu2 + x3 * qu3 + x4 * qu4 + x5 * qu5 + x6 * qu6 + x7 * qu7;
      g_acc[j] += gs_v * dg + gb_v * sx;
      u_acc[j] += us_v * du + ub_v * sx;
    }
  }
  for (int j = 0; j < m; j++) {
    const float gr = sub_group_reduce_add(g_acc[j]);
    const float ur = sub_group_reduce_add(u_acc[j]);
    if (lane == 0) {
      MOE_ACT[(long)(p0 + j) * 640 + col] = to_bf(bsilu(rbf(gr)) * rbf(ur));
    }
  }
}

// Shared expert SwiGLU up projection over this rank's `sh_w` columns (640 / TP): Grid: ((sh_w + 15) / 16, (R + 15) /
KERNEL(256) void fn_moe_shared_up(global const ushort *X, global const uint *GW, global const ushort *GS,
                                  global const ushort *GB, global const uint *UW, global const ushort *US,
                                  global const ushort *UB, global ushort *SH_ACT, int sh_w, int R) {
  const int col = get_group_id(0) * 16 + get_sub_group_id();
  const int r0 = get_group_id(1) * 16;
  if (col >= sh_w || r0 >= R) return;
  const int m = min(16, R - r0);
  float g_acc[16], u_acc[16];
  q4g32_dot16(GW, GS, GB, X + (long)r0 * 2560, 2560, m, col, 2560, g_acc);
  q4g32_dot16(UW, US, UB, X + (long)r0 * 2560, 2560, m, col, 2560, u_acc);
  if (get_sub_group_local_id() == 0) {
    for (int r = 0; r < m; r++) {
      SH_ACT[(long)(r0 + r) * sh_w + col] = to_bf(bsilu(rbf(g_acc[r])) * rbf(u_acc[r]));
    }
  }
}

// Combined local MoE down projection (routed active pairs + sharded shared expert) -> OUT [R, 2560] fp32! Grid: (2560
KERNEL(256) void fn_moe_down_combine(global const ushort *MOE_ACT, global const ushort *SH_ACT,
                                     global const int *ACTIVE, global const int *ROW_LIST, global const float *WTS,
                                     global const uint *DW, global const ushort *DS, global const ushort *DB,
                                     global const uint *SDW, global const ushort *SDS, global const ushort *SDB,
                                     global float *OUT, int sh_w, int R) {
  const int col = get_group_id(0) * 16 + get_sub_group_id();
  const int r = get_group_id(1);
  if (col >= 2560 || r >= R) return;
  const int lane = get_sub_group_local_id();
  float total = 0.0f;

  // 1. Active local routed experts for row r in ascending slot order
  for (int c = 0; c < 10; c++) {
    const int pair_idx = ROW_LIST[r * 11 + c];
    if (pair_idx < 0) break;
    const int packed = ACTIVE[pair_idx];
    const int slot = (packed >> 10) & 0x3F;
    const int le = packed & 0x3FF;
    const float wt = WTS[(long)r * 11 + slot];
    const long drow = (long)le * 2560 + col;
    // K = 640 -> 80 words = 5 words per lane!
    global const uint *wr = DW + drow * 80;
    global const ushort *sr = DS + drow * 20, *br = DB + drow * 20;
    global const ushort *xa = MOE_ACT + (long)pair_idx * 640;
    float acc = 0.0f;
    for (int wi = lane; wi < 80; wi += 16) {
      const uint pk = wr[wi];
      const int g = wi >> 2;
      const ushort8 xv = vload8(0, xa + wi * 8);
      const float x0 = bf(xv.s0), x1 = bf(xv.s1), x2 = bf(xv.s2), x3 = bf(xv.s3);
      const float x4 = bf(xv.s4), x5 = bf(xv.s5), x6 = bf(xv.s6), x7 = bf(xv.s7);
      const float dot = x0 * (float)(pk & 15u) + x1 * (float)((pk >> 4) & 15u) + x2 * (float)((pk >> 8) & 15u) +
                        x3 * (float)((pk >> 12) & 15u) + x4 * (float)((pk >> 16) & 15u) + x5 * (float)((pk >> 20) & 15u) +
                        x6 * (float)((pk >> 24) & 15u) + x7 * (float)((pk >> 28) & 15u);
      const float sx = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
      acc += bf(sr[g]) * dot + bf(br[g]) * sx;
    }
    acc = sub_group_reduce_add(acc);
    total = fma(acc, wt, total);
  }

  // 2. Shared expert slice (sh_w = 640 / TP -> sh_w/8 words, sh_w/32 groups)
  if (sh_w > 0) {
    const int swords = sh_w / 8, sgroups = sh_w / 32;
    global const uint *swr = SDW + (long)col * swords;
    global const ushort *ssr = SDS + (long)col * sgroups, *sbr = SDB + (long)col * sgroups;
    global const ushort *sxa = SH_ACT + (long)r * sh_w;
    float sacc = 0.0f;
    for (int wi = lane; wi < swords; wi += 16) {
      const uint pk = swr[wi];
      const int g = wi >> 2;
      const ushort8 xv = vload8(0, sxa + wi * 8);
      const float x0 = bf(xv.s0), x1 = bf(xv.s1), x2 = bf(xv.s2), x3 = bf(xv.s3);
      const float x4 = bf(xv.s4), x5 = bf(xv.s5), x6 = bf(xv.s6), x7 = bf(xv.s7);
      const float dot = x0 * (float)(pk & 15u) + x1 * (float)((pk >> 4) & 15u) + x2 * (float)((pk >> 8) & 15u) +
                        x3 * (float)((pk >> 12) & 15u) + x4 * (float)((pk >> 16) & 15u) + x5 * (float)((pk >> 20) & 15u) +
                        x6 * (float)((pk >> 24) & 15u) + x7 * (float)((pk >> 28) & 15u);
      const float sx = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
      sacc += bf(ssr[g]) * dot + bf(sbr[g]) * sx;
    }
    sacc = sub_group_reduce_add(sacc);
    total = fma(sacc, WTS[(long)r * 11 + 10], total);
  }

  if (lane == 0) OUT[(long)r * 2560 + col] = total;
}

// 8. Logits Argmax, Log-Sum-Exp (MTP confidence), and Top-64 Candidates

// Two-stage argmax + log-sum-exp over BF16 logits [R, V]: Stage 1: grid (256, R), WG = 256 -> PV[r*256 + g] = max,
KERNEL(256) void fn_logits_part(global const ushort *L, int V, global float *PV, global int *PI, global float *PS) {
  local float bv[256], bs[256];
  local int bi[256];
  const int lid = get_local_id(0), gid = get_group_id(0), row = get_group_id(1);
  const int per = (V + 255) / 256, lo = gid * per, hi = min(V, lo + per);
  float best = -INFINITY;
  int idx = 0x7fffffff;
  for (int i = lo + lid; i < hi; i += 256) {
    const float v = bf(L[(long)row * V + i]);
    if (v > best) {
      best = v;
      idx = i;
    }
  }
  bv[lid] = best;
  bi[lid] = idx;
  barrier(CLK_LOCAL_MEM_FENCE);
  float gmax = bv[0];
  int gidx = bi[0];
  for (int i = 1; i < 256; i++) {
    if (bv[i] > gmax || (bv[i] == gmax && bi[i] < gidx)) {
      gmax = bv[i];
      gidx = bi[i];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float s = 0.0f;
  if (gmax > -INFINITY) {
    for (int i = lo + lid; i < hi; i += 256) s += exp(bf(L[(long)row * V + i]) - gmax);
  }
  bs[lid] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    float tot = 0.0f;
    for (int i = 0; i < 256; i++) tot += bs[i];
    PV[row * 256 + gid] = gmax;
    PI[row * 256 + gid] = gidx;
    PS[row * 256 + gid] = tot;
  }
}

// Stage 2: grid (R), WG = 256 -> OUT_TOK[r] = argmax, OUT_BEST[r*4 + 0..2] = (max, lse, id_bits).
KERNEL(256) void fn_logits_best(global const float *PV, global const int *PI, global const float *PS,
                                global int *OUT_TOK, global float *OUT_BEST) {
  local float bv[256], bs[256];
  local int bi[256];
  const int lid = get_local_id(0), row = get_group_id(0);
  bv[lid] = PV[row * 256 + lid];
  bi[lid] = PI[row * 256 + lid];
  bs[lid] = PS[row * 256 + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    float best = bv[0];
    int idx = bi[0];
    for (int i = 1; i < 256; i++) {
      if (bv[i] > best || (bv[i] == best && bi[i] < idx)) {
        best = bv[i];
        idx = bi[i];
      }
    }
    float sum = 0.0f;
    if (best > -INFINITY) {
      for (int i = 0; i < 256; i++) {
        if (bv[i] > -INFINITY) sum += bs[i] * exp(bv[i] - best);
      }
    }
    OUT_TOK[row] = idx;
    OUT_BEST[row * 4 + 0] = best;
    OUT_BEST[row * 4 + 1] = best + log(sum);
    OUT_BEST[row * 4 + 2] = as_float(idx);
    OUT_BEST[row * 4 + 3] = 0.0f;
  }
}

// Top-64 candidates of one BF16 logits row `L + row * V`: Stage 1 keeps top-64 per 256-slice, Stage 2 merges. Grid:
KERNEL(64) void fn_topk64_part(global const ushort *L, int V, int row, global float *TV, global int *TI) {
  local float sv[64];
  local int si[64];
  const int tid = get_local_id(0), gid = get_group_id(0);
  const int per = (V + 255) / 256, lo = gid * per, hi = min(V, lo + per);
  float my_v[32];
  int my_i[32];
  int cnt = 0;
  for (int i = lo + tid; i < hi; i += 64) {
    my_v[cnt] = bf(L[(long)row * V + i]);
    my_i[cnt] = i;
    cnt++;
  }
  for (int k = 0; k < 64; k++) {
    float bv = -INFINITY;
    int bi = 0x7fffffff, bj = -1;
    for (int j = 0; j < cnt; j++) {
      if (my_v[j] > bv || (my_v[j] == bv && my_i[j] < bi)) {
        bv = my_v[j];
        bi = my_i[j];
        bj = j;
      }
    }
    sv[tid] = bv;
    si[tid] = bi;
    barrier(CLK_LOCAL_MEM_FENCE);
    float wv = sv[0];
    int wi = si[0];
    for (int u = 1; u < 64; u++) {
      if (sv[u] > wv || (sv[u] == wv && si[u] < wi)) {
        wv = sv[u];
        wi = si[u];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (bj >= 0 && my_i[bj] == wi) my_v[bj] = -INFINITY;
    if (tid == 0) {
      TV[gid * 64 + k] = wv;
      TI[gid * 64 + k] = wi;
    }
  }
}

// Convert BF16 row [V] to FP32 row [V] (for full-vocab host sampling when top_k == 0 or top_k > 64).
KERNEL(256) void fn_bf16_to_f32(global const ushort *IN, global float *OUT, int N) {
  const int i = get_global_id(0);
  if (i < N) OUT[i] = bf(IN[i]);
}
