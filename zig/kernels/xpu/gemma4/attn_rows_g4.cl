// Prompt rows: dp_tile's scores and P over each tile of a block's rows in order, folded into (m, l, o) as g4_attn_merge
KERNEL(256) void g4_attn_dp_rows(global const ushort *Q, global const ushort *KC, global const ushort *VC, int p0, int T, int window, int ring, int krows,
                                 global half *H) {
  local ushort Pl[16 * KEYS];
  local float MX[NSG * 16], SM[NSG * 16];
  const int kvh = get_group_id(0), t0 = get_group_id(1) * 4, lane = get_sub_group_local_id(), sg = get_sub_group_id();
  const int min_first = first_tile(p0 + t0, window), max_last = (p0 + min(t0 + 3, T - 1)) / KEYS;
  const int token = t0 + lane / 4, hq = lane % 4, qpos = p0 + token, first = window ? max(0, qpos - window + 1) : 0;
  const bool qok = token < T;
  const int first_l = first_tile(qpos, window), last_l = qpos / KEYS;
  const int maxpos = p0 + min(t0 + 3, T - 1);
  global const uint *qrow = (global const uint *)(Q + ((long)(qok ? token : 0) * 8 + kvh * 4 + hq) * HD);
  float mrun = -INFINITY, lrun = 0.0f;
  float8 orun[2][DTS];
  for (int h = 0; h < 2; h++)
    for (int j = 0; j < DTS; j++) orun[h][j] = 0.0f;
  for (int tile = min_first; tile <= max_last; tile++) {
    const int ts = tile * KEYS, kb = ts + sg * 8;
    float8 s = 0.0f;
    for (int ds = 0; ds < HD / 16; ds++) {
      ushort8 ka;
      intel_sub_group_2d_block_read_16b_8r16x1c((global void *)(KC + kvh * HD), HD * 2, krows, 2 * HD * 2, (int2)(ds * 16, SLOT(kb)), (private ushort *)&ka);
      const int8 qb = qok ? as_int8(vload8(0, qrow + ds * 8)) : (int8)(0);
      s = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(ka), qb, s);
    }
    // scores and the per-slot max, as dp_tile
    float mx = -INFINITY;
    for (int r = 0; r < 8; r++) {
      const int kk = kb + r;
      const float x = (qok && kk >= first && kk <= qpos) ? s[r] : -INFINITY;
      s[r] = x;
      mx = fmax(mx, x);
    }
    MX[sg * 16 + lane] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    float m = -INFINITY;
    for (int i = 0; i < NSG; i++) m = fmax(m, MX[i * 16 + lane]);
    float sum = 0.0f;
    for (int r = 0; r < 8; r++) {
      const float p = s[r] == -INFINITY ? 0.0f : exp(s[r] - m);
      Pl[lane * KEYS + sg * 8 + r] = f2bf(p);
      sum += p;
    }
    SM[sg * 16 + lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    // l, summed in the same order by every subgroup: no round trip through local memory
    float l = 0.0f;
    for (int i = 0; i < NSG; i++) l += SM[i * 16 + lane];
    int st = 0;
    float a = 1.0f, b = 0.0f;
    if (qok && tile >= first_l && tile <= last_l) {
      if (tile == first_l) {
        st = 1;
        mrun = m;
        lrun = l;
      } else {
        st = 2;
        const float mm = fmax(mrun, m);
        a = exp(mrun - mm);
        b = exp(m - mm);
        lrun = lrun * a + l * b;
        mrun = mm;
      }
    }
    // o = P V a column block of V at a time, the fold factors of slot 8 h + i by broadcast from its lane
#pragma unroll
    for (int j = 0; j < DTS; j++) {
      float8 ot[2] = {(float8)(0.0f), (float8)(0.0f)};
      for (int c = 0; c < KEYS / 16; c++) {
        if (ts + c * 16 > maxpos) break;
        short8 a0, a1;
        for (int mm = 0; mm < 8; mm++) {
          a0[mm] = (short)Pl[mm * KEYS + c * 16 + lane];
          a1[mm] = (short)Pl[(8 + mm) * KEYS + c * 16 + lane];
        }
        uint8 vw;
        intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)(VC + kvh * HD), HD * 2, krows, 2 * HD * 2, (int2)((sg * DTS + j) * 16, SLOT(ts + c * 16)), (private uint *)&vw);
        ot[0] = intel_sub_group_bf16_bf16_matrix_mad_k16(a0, as_int8(vw), ot[0]);
        ot[1] = intel_sub_group_bf16_bf16_matrix_mad_k16(a1, as_int8(vw), ot[1]);
      }
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const int sv = sub_group_broadcast(st, 8 * h + i);
          const float fa = sub_group_broadcast(a, 8 * h + i), fb = sub_group_broadcast(b, 8 * h + i);
          if (sv == 1) orun[h][j][i] = ot[h][i];
          else if (sv == 2) orun[h][j][i] = orun[h][j][i] * fa + ot[h][i] * fb;
        }
    }
  }
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int slot = 8 * h + i, tok = t0 + slot / 4, head = kvh * 4 + slot % 4;
      const float lf = sub_group_broadcast(lrun, slot);
      if (tok >= T) continue;
#pragma unroll
      for (int j = 0; j < DTS; j++) H[(long)tok * 8 * HD + head * HD + (sg * DTS + j) * 16 + lane] = convert_half_rte(rbf(orun[h][j][i] / lf));
    }
}
