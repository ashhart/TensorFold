// Qwen3.8-27B decode attention with XPU_ATTN_PREF: attn_pref, attn_tail and attn_merge2. Included by decode.cl.
#if Q27_GQA == 6
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
// The committed prefix on bf16 DPAS: a subgroup per (tile of 8 queries, K/V head, chunk of CH keys), P in bf16.
KERNEL(16) void attn_pref(global const ushort *Q, global const ushort *KC, global const ushort *VC, global float *PMP,
                          global float *PLP, global float *POP, int T, int NCH, int CH, float scale,
                          global const ulong *MT, global const int *POS) {
  // Query f = tile * 8 + r is row f / 6, head f % 6; with MT, one row a tile over its own cache and prefix.
  local short qa[16 * 16 * 8];                            // [dim block][lane][query of the tile]
  const int tile = get_group_id(0), hk = get_group_id(1), c = get_group_id(2), lane = get_sub_group_local_id();
  if (MT) {
    if ((c + 1) * CH > pref_base(POS[tile], CH)) return;
    KC = (global const ushort *)MT[tile * 8];
    VC = (global const ushort *)MT[tile * 8 + 1];
  }
  for (int i = lane; i < 16 * 16 * 8; i += 16) {
    const int tb = i >> 7, l = (i >> 3) & 15, r = i & 7, f = tile * 8 + r;
    const int row = MT ? tile : f / Q27_GQA, a = MT ? r : f % Q27_GQA;
    qa[i] = row < T && a < Q27_GQA ? (short)Q[((long)row * Q27_AQ + hk * Q27_GQA + a) * 256 + tb * 16 + l] : (short)0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float m[8], ll[8];
  float8 acc[16];
  for (int i = 0; i < 8; i++) {
    m[i] = -INFINITY;
    ll[i] = 0.0f;
  }
  for (int j = 0; j < 16; j++) acc[j] = (float8)(0.0f);
  const long k0 = (long)c * CH;
  for (int u = 0; u < CH / 16; u++) {
    const global uint *krow = (const global uint *)(KC + ((k0 + u * 16 + lane) * Q27_AKV + hk) * 256);
    float8 s = (float8)(0.0f);
    __attribute__((opencl_unroll_hint))
    for (int tb = 0; tb < 16; tb++)
      s = intel_sub_group_bf16_bf16_matrix_mad_k16(vload8(tb * 16 + lane, qa), as_int8(vload8(0, krow + tb * 8)), s);
    short8 own;
    float8 al;
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 8; i++) {
      const float x = s[i] * scale;
      const float mn = fmax(m[i], sub_group_reduce_max(x));
      const float p = bf2f(f2bf(exp(x - mn)));
      al[i] = exp(m[i] - mn);
      ll[i] = ll[i] * al[i] + sub_group_reduce_add(p);
      own[i] = (short)f2bf(p);
      m[i] = mn;
    }
    __attribute__((opencl_unroll_hint))
    for (int hb = 0; hb < 4; hb++) {
      ushort4 vv[16];
      __attribute__((opencl_unroll_hint))
      for (int k = 0; k < 16; k++)
        vv[k] = intel_sub_group_block_read_us4(VC + ((k0 + u * 16 + k) * Q27_AKV + hk) * 256 + hb * 64);
      __attribute__((opencl_unroll_hint))
      for (int e = 0; e < 4; e++) {
        int8 b;
        __attribute__((opencl_unroll_hint))
        for (int d = 0; d < 8; d++) b[d] = (int)vv[2 * d][e] | ((int)vv[2 * d + 1][e] << 16);
        acc[hb * 4 + e] = intel_sub_group_bf16_bf16_matrix_mad_k16(own, b, acc[hb * 4 + e] * al);
      }
    }
  }
  for (int i = 0; i < 8; i++) {
    const int f = tile * 8 + i, row = MT ? tile : f / Q27_GQA, a = MT ? i : f % Q27_GQA;
    if (row >= T || a >= Q27_GQA) break;
    const long base = ((long)row * Q27_AQ + hk * Q27_GQA + a) * NCH + c;
    if (lane == 0) {
      PMP[base] = m[i];
      PLP[base] = ll[i];
    }
    for (int j = 0; j < 16; j++) POP[base * 256 + j * 16 + lane] = acc[j][i];
  }
}

// A row's slices past its prefix: attn_decode's arithmetic from pref_base on.
KERNEL(16) void attn_tail(global const ushort *Q, global const ushort *KC, global const ushort *VC, global float *PM,
                          global float *PL, global float *PO, int p0, int NS, float scale, global const int *TREE,
                          global const ulong *MT, global const int *POS, int cap, int PRE) {
  attn_slices(Q, KC, VC, PM, PL, PO, p0, NS, scale, TREE, MT, POS, cap, PRE);
}

// Merge of a row's prefix chunks then its slices, in order: O bf16 [T, Q27_AQ, 256]. One subgroup per (head, row).
KERNEL(16) void attn_merge2(global const float *PMP, global const float *PLP, global const float *POP,
                            global const float *PM, global const float *PL, global const float *PO, global ushort *O,
                            int p0, int NCH, int CH, int NS, global const int *TREE, global const int *POS) {
  const int hh = get_group_id(0), t = get_group_id(1), lane = get_sub_group_local_id();
  const int last = POS ? POS[t] : p0 + DEPTH(t), pb = pref_base(last, CH), nc = pb / CH, used = (last - pb) / SPLIT + 1;
  const long bp = ((long)t * Q27_AQ + hh) * NCH, base = ((long)t * Q27_AQ + hh) * NS;
  float m = -INFINITY;
  for (int s = 0; s < nc; s++) m = fmax(m, PMP[bp + s]);
  for (int s = 0; s < used; s++) m = fmax(m, PM[base + s]);
  float l = 0.0f, o[16];
  for (int i = 0; i < 16; i++) o[i] = 0.0f;
  for (int s = 0; s < nc; s++) {
    const float w = exp(PMP[bp + s] - m);
    l += PLP[bp + s] * w;
    for (int i = 0; i < 16; i++) o[i] += POP[(bp + s) * 256 + i * 16 + lane] * w;
  }
  for (int s = 0; s < used; s++) {
    const float w = exp(PM[base + s] - m);
    l += PL[base + s] * w;
    for (int i = 0; i < 16; i++) o[i] += PO[(base + s) * 256 + i * 16 + lane] * w;
  }
  for (int i = 0; i < 16; i++) O[((long)t * Q27_AQ + hh) * 256 + i * 16 + lane] = f2bf(o[i] / l);
}
#endif
