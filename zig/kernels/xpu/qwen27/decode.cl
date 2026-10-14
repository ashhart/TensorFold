// Qwen3.8-27B decode kernels on Intel Xe2: serial GDN (a few tokens) and attention over the cache.
#pragma OPENCL FP_CONTRACT OFF

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) {
  return intel_convert_bfloat16_as_ushort(f); // hardware conversion (round to nearest even)
}

// Tree table of a verified block (<= 16 rows): TREE[t*17] = depth | SAVE | LOAD flags, then the ancestor row per depth.
#define DEPTH(t) (TREE[(t) * 17] & 255)
#define SAVE(t) (TREE[(t) * 17] & 256)
#define LOAD(t) (TREE[(t) * 17] & 512)
#define ANC(t, d) TREE[(t) * 17 + 1 + (d)]

// Serial GDN over T tokens: one subgroup per (8 v rows, V head), the lane holds 8 k columns; ROWS, TREE, MT: modes.
KERNEL(16) void gdn_step(global const ushort *Q, global const ushort *K, global const ushort *V, global const float *G,
                         global const float *BETA, global const float *S, global float *SO, global ushort *Y, int T,
                         global const int *ROWS, global const int *TREE, global float *SL, global const ulong *MT) {
  // MT (if non-null): several streams, grid (16, 48, rows); a window's first row starts from its stream's state.
  const int vb = get_group_id(0), h = get_group_id(1), lane = get_sub_group_local_id(), kh = h / 3;
  int t0 = 0;
  if (MT) {
    t0 = get_group_id(2);
    if ((int)MT[t0 * 8 + 6] != t0) return;            // not a window's first row
    S = (global const float *)MT[t0 * 8 + 4];
    SO = (global float *)MT[t0 * 8 + 5];
    T = t0 + (int)MT[t0 * 8 + 7];
  }
  float s[8][8];
  for (int r = 0; r < 8; r++)
    for (int i = 0; i < 8; i++) s[r][i] = S[((long)h * 128 + vb * 8 + r) * 128 + i * 16 + lane];
  for (int t = t0; t < T; t++) {
    const int row = ROWS ? ROWS[t] : t;
    if (TREE && DEPTH(t) > 0 && LOAD(t)) {              // the parent's state, from its slot
      const int par = ANC(t, DEPTH(t) - 1);
      for (int r = 0; r < 8; r++)
        for (int i = 0; i < 8; i++) s[r][i] = SL[(((long)par * 48 + h) * 128 + vb * 8 + r) * 128 + i * 16 + lane];
    } else if (TREE && DEPTH(t) == 0)
      for (int r = 0; r < 8; r++)
        for (int i = 0; i < 8; i++) s[r][i] = S[((long)h * 128 + vb * 8 + r) * 128 + i * 16 + lane];
    float q[8], k[8];
    for (int i = 0; i < 8; i++) {
      q[i] = bf2f(Q[((long)row * 16 + kh) * 128 + i * 16 + lane]);
      k[i] = bf2f(K[((long)row * 16 + kh) * 128 + i * 16 + lane]);
    }
    const float g = G[(long)row * 48 + h], beta = BETA[(long)row * 48 + h];
    for (int r = 0; r < 8; r++) {
      float mem = 0.0f;
      for (int i = 0; i < 8; i++) {
        s[r][i] *= g;
        mem += s[r][i] * k[i];
      }
      mem = sub_group_reduce_add(mem);
      const float u = (bf2f(V[((long)row * 48 + h) * 128 + vb * 8 + r]) - mem) * beta;
      float y = 0.0f;
      for (int i = 0; i < 8; i++) {
        s[r][i] += k[i] * u;
        y += s[r][i] * q[i];
      }
      y = sub_group_reduce_add(y);
      if (lane == 0) Y[((long)t * 48 + h) * 128 + vb * 8 + r] = f2bf(y);
    }
    if (TREE && SAVE(t))
      for (int r = 0; r < 8; r++)
        for (int i = 0; i < 8; i++) SL[(((long)t * 48 + h) * 128 + vb * 8 + r) * 128 + i * 16 + lane] = s[r][i];
  }
  if (!TREE)
    for (int r = 0; r < 8; r++)
      for (int i = 0; i < 8; i++) SO[((long)h * 128 + vb * 8 + r) * 128 + i * 16 + lane] = s[r][i];
}

// Decode attention by key slices: one subgroup per (slice, K/V head, row) handles 6 query heads; partials PM, PL, PO.
#define SPLIT 32
KERNEL(16) void attn_decode(global const ushort *Q, global const ushort *KC, global const ushort *VC, global float *PM,
                            global float *PL, global float *PO, int p0, int NS, float scale, global const int *TREE,
                            global const ulong *MT, global const int *POS, int cap) {
  // MT (if non-null): several streams, one row each; with TREE, tree windows; cap > 0: int8 cache (256 bytes per row).
  const int sp = get_group_id(0), hk = get_group_id(1), t = get_group_id(2), lane = get_sub_group_local_id();
  int base = p0, wf = 0;
  if (MT) {
    KC = (global const ushort *)MT[t * 8];
    VC = (global const ushort *)MT[t * 8 + 1];
    wf = (int)MT[t * 8 + 6];
    base = TREE ? POS[wf] : 0x7fffffff;
  }
  const int last = MT ? POS[t] : p0 + DEPTH(t), j0 = sp * SPLIT, j1 = min(j0 + SPLIT, last + 1);
  float q[6][16], o[6][16], m[6], l[6];
  for (int a = 0; a < 6; a++) {
    for (int i = 0; i < 16; i++) {
      q[a][i] = bf2f(Q[((long)t * 24 + hk * 6 + a) * 256 + i * 16 + lane]);
      o[a][i] = 0.0f;
    }
    m[a] = -INFINITY;
    l[a] = 0.0f;
  }
  for (int j = j0; j < j1; j++) {
    const long kr = j < base ? j : base + ANC(t, j - base) - wf; // cache row of key j
    float kv[16];
    if (cap) {
      const global char *k8 = (const global char *)KC + (kr * 4 + hk) * 256;
      const float ks = ((const global float *)((const global char *)KC + (long)cap * 1024))[kr * 4 + hk];
      for (int i = 0; i < 16; i++) kv[i] = (float)k8[i * 16 + lane] * ks;
    } else
      for (int i = 0; i < 16; i++) kv[i] = bf2f(KC[(kr * 4 + hk) * 256 + i * 16 + lane]);
    float sc[6];
    for (int a = 0; a < 6; a++) {
      float d = 0.0f;
      for (int i = 0; i < 16; i++) d += q[a][i] * kv[i];
      sc[a] = sub_group_reduce_add(d) * scale;
    }
    if (cap) {
      const global char *v8 = (const global char *)VC + (kr * 4 + hk) * 256;
      const float vs = ((const global float *)((const global char *)VC + (long)cap * 1024))[kr * 4 + hk];
      for (int i = 0; i < 16; i++) kv[i] = (float)v8[i * 16 + lane] * vs;
    } else
      for (int i = 0; i < 16; i++) kv[i] = bf2f(VC[(kr * 4 + hk) * 256 + i * 16 + lane]);
    for (int a = 0; a < 6; a++) {
      const float mn = fmax(m[a], sc[a]);
      const float al = exp(m[a] - mn), p = exp(sc[a] - mn);
      l[a] = l[a] * al + p;
      for (int i = 0; i < 16; i++) o[a][i] = o[a][i] * al + p * kv[i];
      m[a] = mn;
    }
  }
  for (int a = 0; a < 6; a++) {
    const long base = ((long)t * 24 + hk * 6 + a) * NS + sp;
    if (lane == 0) {
      PM[base] = m[a];
      PL[base] = l[a];
    }
    for (int i = 0; i < 16; i++) PO[base * 256 + i * 16 + lane] = o[a][i];
  }
}

// Merge of the slices, in order: O bf16 [T, 24, 256]. One subgroup per (head, row).
KERNEL(16) void attn_merge(global const float *PM, global const float *PL, global const float *PO, global ushort *O,
                           int p0, int NS, global const int *TREE, global const int *POS) {
  const int hh = get_group_id(0), t = get_group_id(1), lane = get_sub_group_local_id();
  const int used = (POS ? POS[t] : p0 + DEPTH(t)) / SPLIT + 1; // non-empty slices of row t
  const long base = ((long)t * 24 + hh) * NS;
  float m = -INFINITY;
  for (int s = 0; s < used; s++) m = fmax(m, PM[base + s]);
  float l = 0.0f, o[16];
  for (int i = 0; i < 16; i++) o[i] = 0.0f;
  for (int s = 0; s < used; s++) {
    const float w = exp(PM[base + s] - m);
    l += PL[base + s] * w;
    for (int i = 0; i < 16; i++) o[i] += PO[(base + s) * 256 + i * 16 + lane] * w;
  }
  for (int i = 0; i < 16; i++) O[((long)t * 24 + hh) * 256 + i * 16 + lane] = f2bf(o[i] / l);
}

// Commit of a kept path: its K, V rows move to positions p0 + depth in depth order, never overwriting an unread row.
KERNEL(16) void kv_move(global ushort *KC, global ushort *VC, int p0, global const int *PATH, int n, int cap) {
  const int hk = get_group_id(0), lane = get_sub_group_local_id();
  for (int d = 0; d < n; d++) {
    const long src = ((long)(p0 + PATH[d]) * 4 + hk) * 256, dst = ((long)(p0 + d) * 4 + hk) * 256;
    if (src == dst) continue;
    if (cap) {                                         // int8: 256 bytes (16 per lane) and the scale
      global uint *k = (global uint *)KC, *v = (global uint *)VC;
      for (int i = 0; i < 4; i++) {
        k[dst / 4 + i * 16 + lane] = k[src / 4 + i * 16 + lane];
        v[dst / 4 + i * 16 + lane] = v[src / 4 + i * 16 + lane];
      }
      if (lane == 0) {
        global float *ks = (global float *)((global char *)KC + (long)cap * 1024),
                     *vs = (global float *)((global char *)VC + (long)cap * 1024);
        ks[dst / 256] = ks[src / 256];
        vs[dst / 256] = vs[src / 256];
      }
      continue;
    }
    for (int i = 0; i < 16; i++) {
      KC[dst + i * 16 + lane] = KC[src + i * 16 + lane];
      VC[dst + i * 16 + lane] = VC[src + i * 16 + lane];
    }
  }
}

// EXL3 probe (mul1 codebook, docs/recipes/exl3.md): X * W_q over K-tile slices, W_q decoded on the fly from TR.
KERNEL(16) void exl3_gemv(global const ushort *TR, global const half *X, global float *Y, int K, int N, int T, int bits,
                          int kslices) {
  // the tile is read once per subgroup and distributed by shuffle, like the activations of its 16 K rows
  const int nt = get_group_id(0), ks = get_group_id(1), lane = get_sub_group_local_id();
  const int per = K / 16 / kslices, kt0 = ks * per, R = 256 * bits, nw = 8 * bits;
  const float c1 = (float)as_half((ushort)0x1EEE), c0 = (float)as_half((ushort)0xC931);
  float acc0[8], acc1[8];
  for (int t = 0; t < 8; t++) acc0[t] = acc1[t] = 0.0f;
  for (int kt = kt0; kt < kt0 + per; kt++) {
    global const uint *tw = (global const uint *)(TR + ((long)kt * (N / 16) + nt) * 16 * bits);
    const uint r0 = tw[lane], r1 = lane + 16 < nw ? tw[lane + 16] : 0u;
    float xr[8];
    for (int t = 0; t < 8; t++) xr[t] = t < T ? (float)X[(long)t * K + kt * 16 + lane] : 0.0f;
    __attribute__((opencl_unroll_hint))
    for (int v = 0; v < 16; v++) {
      const int p = lane * 16 + v;
      int a = (p + 1) * bits - 16;
      a += a < 0 ? R : 0;
      const int wi = a >> 5, wj = wi + 1 == nw ? 0 : wi + 1, sh = a & 31;
      const uint w0a = sub_group_shuffle(r0, wi & 15), w0b = sub_group_shuffle(r1, wi & 15);
      const uint w1a = sub_group_shuffle(r0, wj & 15), w1b = sub_group_shuffle(r1, wj & 15);
      const uint w0 = wi < 16 ? w0a : w0b, w1 = wj < 16 ? w1a : w1b;
      const uint hi = sh ? (w0 << sh) | (w1 >> (32 - sh)) : w0; // the 32 bits starting at bit a
      const uint s = hi >> 16;
      const uint xx = s * 0x83DCD12Du;
      const uint bs = (xx & 0x00FF00FFu) + ((xx >> 8) & 0x00FF00FFu);
      const float w = (float)(1024u + (bs & 0xFFFFu) + (bs >> 16)) * c1 + c0;
      const int l = p >> 3, j = p & 7, row = 2 * (l & 3) + (j & 1) + 8 * ((j >> 1) & 1);
      for (int t = 0; t < 8; t++) {
        if (t >= T) break;
        const float xv = sub_group_shuffle(xr[t], row);
        if (j >> 2) acc1[t] += xv * w; else acc0[t] += xv * w;
      }
    }
  }
  for (int t = 0; t < T && t < 8; t++) {
    const float a0 = acc0[t] + sub_group_shuffle_xor(acc0[t], 1), a1 = acc1[t] + sub_group_shuffle_xor(acc1[t], 1);
    if ((lane & 1) == 0) {
      const long base = ((long)ks * T + t) * N + nt * 16 + lane / 2;
      Y[base] = a0;
      Y[base + 8] = a1;
    }
  }
}

// Prompt attention (FlashAttention-2 on bf16 DPAS), causal, 256-wide heads; fixed key order, so a row's bits ignore T.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(128, 1, 1)))
kernel void attn_prompt(global const ushort *Q, global const ushort *KC, global const ushort *VC, global ushort *O,
                        int p0, int T, float scale) {
  local ushort P[6 * 8 * 128];                         // A tiles: (tile of 8 queries, 16 keys), r × 16 + lane
  local float MX[3][2][16], SM[3][2][16], A[48];
  const int t0 = get_group_id(0) * 8, hk = get_group_id(1), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int nkeys = p0 + min(t0 + 7, T - 1) + 1;
  const bool qk = sg < 6;
  const int g = sg / 2, kh = sg % 2;
  const int tq = lane % 8, hq = 2 * g + lane / 8;      // the lane's query: token t0 + tq, head 6 hk + hq
  const int qpos = p0 + t0 + tq;
  const bool qok = t0 + tq < T;
  global const uint *qrow = (global const uint *)(Q + ((long)(t0 + tq) * 24 + hk * 6 + hq) * 256);
  const float sl = scale * M_LOG2E_F;
  float8 o[6][2];
  for (int i = 0; i < 6; i++) o[i][0] = o[i][1] = 0.0f;
  float m = -INFINITY, l = 0.0f;
  for (int k0 = 0; k0 < nkeys; k0 += 128) {
    if (qk) {
      const int kb = k0 + kh * 64;                   // the 64 keys of this subgroup
      float8 s[8];
      for (int b = 0; b < 8; b++) s[b] = 0.0f;
      for (int ds = 0; ds < 16; ds++) {
        const int8 qb = qok ? as_int8(vload8(0, qrow + ds * 8)) : (int8)(0);
        __attribute__((opencl_unroll_hint))
        for (int h2 = 0; h2 < 2; h2++) {
          ushort ka[32];                               // 32 keys × 16 dimensions: 4 A tiles of 8 keys
          intel_sub_group_2d_block_read_16b_32r16x1c((global void *)(KC + hk * 256), 256 * 2, nkeys, 4 * 256 * 2,
                                                     (int2)(ds * 16, kb + h2 * 32), ka);
          __attribute__((opencl_unroll_hint))
          for (int b = 0; b < 4; b++) {
            short8 a;
            for (int r = 0; r < 8; r++) a[r] = (short)ka[b * 8 + r];
            s[h2 * 4 + b] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, qb, s[h2 * 4 + b]);
          }
        }
      }
      float mx = -INFINITY;
      __attribute__((opencl_unroll_hint))
      for (int b = 0; b < 8; b++)
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) {
          const float x = kb + b * 8 + r > qpos ? -INFINITY : s[b][r] * sl;
          s[b][r] = x;
          mx = fmax(mx, x);
        }
      MX[g][kh][lane] = mx;
      barrier(CLK_LOCAL_MEM_FENCE);
      const float mn = fmax(m, fmax(mx, MX[g][kh ^ 1][lane]));
      const float alpha = mn == -INFINITY ? 1.0f : native_exp2(m - mn);
      float sum = 0.0f;
      local ushort *pt = P + (hq * 8) * 128 + tq * 16;
      __attribute__((opencl_unroll_hint))
      for (int b = 0; b < 8; b++)
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) {
          const float p = mn == -INFINITY ? 0.0f : native_exp2(s[b][r] - mn);
          sum += p;
          const int key = kh * 64 + b * 8 + r;      // within the block
          pt[(key / 16) * 128 + key % 16] = f2bf(p);
        }
      SM[g][kh][lane] = sum;
      if (kh == 0) A[hq * 8 + tq] = alpha;
      m = mn;
      barrier(CLK_LOCAL_MEM_FENCE);
      l = l * alpha + SM[g][0][lane] + SM[g][1][lane];
    } else {
      barrier(CLK_LOCAL_MEM_FENCE);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 6; i++)
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 8; r++) {
        const float al = A[i * 8 + r];
        o[i][0][r] *= al;
        o[i][1][r] *= al;
      }
    for (int c = 0; c < 8; c++) {
      if (k0 + c * 16 >= nkeys) break;
      short8 pa[6];
      __attribute__((opencl_unroll_hint))
      for (int i = 0; i < 6; i++) pa[i] = as_short8(intel_sub_group_block_read_us8(P + (i * 8 + c) * 128));
      __attribute__((opencl_unroll_hint))
      for (int dt = 0; dt < 2; dt++) {
        uint8 vw;
        intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)(VC + hk * 256), 256 * 2, nkeys,
                                                             4 * 256 * 2, (int2)((sg * 2 + dt) * 16, k0 + c * 16),
                                                             (private uint *)&vw);
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < 6; i++) o[i][dt] = intel_sub_group_bf16_bf16_matrix_mad_k16(pa[i], as_int8(vw), o[i][dt]);
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (qk && kh == 0) A[hq * 8 + tq] = 1.0f / l;
  barrier(CLK_LOCAL_MEM_FENCE);
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 6; i++)
    __attribute__((opencl_unroll_hint))
    for (int r = 0; r < 8; r++) {
      if (t0 + r >= T) break;
      const float inv = A[i * 8 + r];
      global ushort *dst = O + ((long)(t0 + r) * 24 + hk * 6 + i) * 256 + sg * 32 + lane;
      dst[0] = f2bf(o[i][0][r] * inv);
      dst[16] = f2bf(o[i][1][r] * inv);
    }
}

// Prompt attention over the int8 cache: same partitioning as attn_prompt, int8 DPAS for S and fp16 for O += P*V.
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(128, 1, 1)))
kernel void attn_prompt8(global const ushort *Q, global const char *KC, global const char *VC, global ushort *O,
                         int p0, int T, float scale, int cap) {
  local ushort P[6 * 8 * 128];
  local uint QI[3 * 8 * 8 * 16];                       // (group, 32 dimensions, word) × lane: Q in int8, 4 per word
  local float MX[3][2][16], SM[3][2][16], A[48], QS[48];
  const int t0 = get_group_id(0) * 8, hk = get_group_id(1), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int nkeys = p0 + min(t0 + 7, T - 1) + 1;
  const bool qk = sg < 6;
  const int g = sg / 2, kh = sg % 2;
  const int tq = lane % 8, hq = 2 * g + lane / 8;
  const int qpos = p0 + t0 + tq;
  const bool qok = t0 + tq < T;
  global const float *KS = (global const float *)(KC + (long)cap * 1024), *VS = (global const float *)(VC + (long)cap * 1024);
  if (qk && kh == 0) {                                 // the lane's query, Q in int8
    global const ushort *qrow = Q + ((long)(t0 + tq) * 24 + hk * 6 + hq) * 256;
    float mq = 0.0f;
    if (qok)
      for (int e = 0; e < 16; e++) {
        const ushort16 x = vload16(e, qrow);
        for (int b = 0; b < 16; b++) mq = fmax(mq, fabs(bf2f(x[b])));
      }
    const float qs = mq > 0.0f ? mq / 127.0f : 1.0f, iq = 1.0f / qs;
    for (int e = 0; e < 16; e++) {                     // 16 dimensions: 4 words
      const ushort16 x = qok ? vload16(e, qrow) : (ushort16)(0);
      for (int j = 0; j < 4; j++) {
        uint u = 0;
        for (int b = 0; b < 4; b++) u |= ((uint)(uchar)(char)rint(bf2f(x[j * 4 + b]) * iq)) << (8 * b);
        const int w = e * 4 + j;
        QI[((g * 8 + w / 8) * 8 + w % 8) * 16 + lane] = u;
      }
    }
    QS[hq * 8 + tq] = qs;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float qsl = qk ? QS[hq * 8 + tq] * scale * M_LOG2E_F : 0.0f;
  float8 o[6][2];
  for (int i = 0; i < 6; i++) o[i][0] = o[i][1] = 0.0f;
  float m = -INFINITY, l = 0.0f;
  for (int k0 = 0; k0 < nkeys; k0 += 128) {
    if (qk) {
      const int kb = k0 + kh * 64;
      int8 si[8];
      for (int b = 0; b < 8; b++) si[b] = 0;
      for (int ds = 0; ds < 8; ds++) {                 // steps of 32 dimensions
        const int8 qb = as_int8(intel_sub_group_block_read8(QI + (g * 8 + ds) * 128));
        __attribute__((opencl_unroll_hint))
        for (int h2 = 0; h2 < 2; h2++) {
          ushort ka[32];                               // 32 keys × 32 bytes: 4 A tiles of 8 keys
          intel_sub_group_2d_block_read_16b_32r16x1c((global void *)(KC + hk * 256), 256, nkeys, 4 * 256,
                                                     (int2)(ds * 16, kb + h2 * 32), ka);
          __attribute__((opencl_unroll_hint))
          for (int b = 0; b < 4; b++) {
            short8 a;
            for (int r = 0; r < 8; r++) a[r] = (short)ka[b * 8 + r];
            si[h2 * 4 + b] = intel_sub_group_i8_i8_matrix_mad_k32(a, qb, si[h2 * 4 + b]);
          }
        }
      }
      float ksc[4];                                    // scales of the 64 keys, 4 per lane
      for (int c = 0; c < 4; c++) {
        const int key = kb + c * 16 + lane;
        ksc[c] = key < nkeys ? KS[(long)key * 4 + hk] : 0.0f;
      }
      float8 s[8];
      float mx = -INFINITY;
      __attribute__((opencl_unroll_hint))
      for (int b = 0; b < 8; b++)
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) {
          const int kk = b * 8 + r;
          const float ks = sub_group_broadcast(ksc[kk / 16], kk % 16);
          const float x = kb + kk > qpos ? -INFINITY : (float)si[b][r] * ks * qsl;
          s[b][r] = x;
          mx = fmax(mx, x);
        }
      MX[g][kh][lane] = mx;
      barrier(CLK_LOCAL_MEM_FENCE);
      const float mn = fmax(m, fmax(mx, MX[g][kh ^ 1][lane]));
      const float alpha = mn == -INFINITY ? 1.0f : native_exp2(m - mn);
      float vsc[4];
      for (int c = 0; c < 4; c++) {
        const int key = kb + c * 16 + lane;
        vsc[c] = key < nkeys ? VS[(long)key * 4 + hk] : 0.0f;
      }
      float sum = 0.0f;
      local ushort *pt = P + (hq * 8) * 128 + tq * 16;
      __attribute__((opencl_unroll_hint))
      for (int b = 0; b < 8; b++)
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) {
          const int kk = b * 8 + r;
          const float p = mn == -INFINITY ? 0.0f : native_exp2(s[b][r] - mn);
          sum += p;
          const int key = kh * 64 + kk;
          pt[(key / 16) * 128 + key % 16] = as_ushort(convert_half(p * sub_group_broadcast(vsc[kk / 16], kk % 16)));
        }
      SM[g][kh][lane] = sum;
      if (kh == 0) A[hq * 8 + tq] = alpha;
      m = mn;
      barrier(CLK_LOCAL_MEM_FENCE);
      l = l * alpha + SM[g][0][lane] + SM[g][1][lane];
    } else {
      barrier(CLK_LOCAL_MEM_FENCE);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 6; i++)
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 8; r++) {
        const float al = A[i * 8 + r];
        o[i][0][r] *= al;
        o[i][1][r] *= al;
      }
    for (int c2 = 0; c2 < 4; c2++) {                   // 32 keys at a time
      if (k0 + c2 * 32 >= nkeys) break;
      __attribute__((opencl_unroll_hint))
      for (int dt = 0; dt < 2; dt++) {
        uint w8[8];                                    // lane: dimension 16 (2 sg + dt) + lane; word j: keys 4j … 4j + 3
        intel_sub_group_2d_block_read_transform_8b_32r16x1c((global void *)(VC + hk * 256), 256, nkeys, 4 * 256,
                                                            (int2)((sg * 2 + dt) * 16, k0 + c2 * 32), w8);
        __attribute__((opencl_unroll_hint))
        for (int hc = 0; hc < 2; hc++) {               // 16 keys: 8 fp16 pairs (keys 2j, 2j + 1), int8 → fp16 exact
          uint8 vw;
          __attribute__((opencl_unroll_hint))
          for (int j = 0; j < 4; j++) {
            const half4 h = convert_half4(as_char4(w8[hc * 4 + j]));
            vw[2 * j] = as_uint(h.s01);
            vw[2 * j + 1] = as_uint(h.s23);
          }
          __attribute__((opencl_unroll_hint))
          for (int i = 0; i < 6; i++) {
            const short8 pa = as_short8(intel_sub_group_block_read_us8(P + (i * 8 + c2 * 2 + hc) * 128));
            o[i][dt] = intel_sub_group_f16_f16_matrix_mad_k16(pa, as_int8(vw), o[i][dt]);
          }
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (qk && kh == 0) A[hq * 8 + tq] = 1.0f / l;
  barrier(CLK_LOCAL_MEM_FENCE);
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 6; i++)
    __attribute__((opencl_unroll_hint))
    for (int r = 0; r < 8; r++) {
      if (t0 + r >= T) break;
      const float inv = A[i * 8 + r];
      global ushort *dst = O + ((long)(t0 + r) * 24 + hk * 6 + i) * 256 + sg * 32 + lane;
      dst[0] = f2bf(o[i][0][r] * inv);
      dst[16] = f2bf(o[i][1][r] * inv);
    }
}
