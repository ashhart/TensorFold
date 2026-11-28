// Qwen3.8 Flash Next Layer 1 PLE, Gated DeltaNet, and QSA sparse attention kernels.

// Grid: (R), WG = 512 (32 subgroups).
KERNEL(512) void fn_ple_gate(global const ushort *KEYS, global const ushort *VALS, global const ushort *H,
                             global const float *NK, global const float *NQ, global ushort *GATED,
                             global float *PSS, float eps) {
  local float red[32];
  const int r = get_group_id(0), lid = get_local_id(0);
  const int D = 2560, S = 4;
  const float fd = (float)D;
  for (int s = 0; s < S; s++) {
    const long row = (long)r * (S * D) + (long)s * D;
    float kss = 0.0f, qss = 0.0f;
    for (int d = lid; d < D; d += 512) {
      const float kv = bf(KEYS[row + d]), hv = bf(H[row + d]);
      kss = fma(kv, kv, kss);
      qss = fma(hv, hv, qss);
    }
    kss = wg_sum(kss, red, 32);
    qss = wg_sum(qss, red, 32);
    const float krinv = rsqrt(kss / fd + eps);
    const float qrinv = rsqrt(qss / fd + eps);
    float dot = 0.0f;
    for (int d = lid; d < D; d += 512) {
      const float kn = rbf((bf(KEYS[row + d]) * krinv) * NK[s * D + d]);
      const float qn = rbf((bf(H[row + d]) * qrinv) * NQ[s * D + d]);
      dot += rbf(kn * qn);
    }
    dot = wg_sum(dot, red, 32);
    float gate = rbf(dot / sqrt(fd));
    const float mag = sqrt(fmax(fabs(gate), 1e-6f));
    gate = rbf(gate > 0.0f ? mag : (gate < 0.0f ? -mag : 0.0f));
    const float sg = bsig(gate);
    float gss = 0.0f;
    for (int d = lid; d < D; d += 512) {
      const float gv = rbf(sg * bf(VALS[(long)r * D + d]));
      GATED[row + d] = to_bf(gv);
      gss = fma(gv, gv, gss);
    }
    gss = wg_sum(gss, red, 32);
    if (lid == 0) PSS[(long)r * S + s] = gss;
  }
}

// Grid: (R, 20), WG = 512 (20 * 512 = 10240 = WIDE). TAPS = 4, DIL = 3, nt = 9.
KERNEL(512) void fn_ple_conv(global const ushort *GATED, global const float *PSS, global const float *NC,
                             global const ushort *TAIL, global const ushort *CW, global ushort *H,
                             global ushort *NROW, float eps) {
  const int r = get_group_id(0), e = get_group_id(1) * 512 + get_local_id(0);
  const int D = 2560, S = 4, w = 10240, TAPS = 4, DIL = 3, nt = 9;
  if (e >= w) return;
  const int s = e / D;
  const float fd = (float)D;
  float acc = 0.0f;
  for (int t = 0; t < TAPS; t++) {
    const int at = r + t * DIL;
    float xv;
    if (at < nt) {
      xv = bf(TAIL[(long)at * w + e]);
    } else {
      const int rr = at - nt;
      const float rinv = 1.0f / sqrt(PSS[rr * S + s] / fd + eps);
      xv = rbf((bf(GATED[(long)rr * w + e]) * rinv) * NC[e]);
    }
    acc = fma(bf(CW[(long)e * TAPS + t]), xv, acc);
  }
  const float act = bsilu(rbf(acc));
  const float g = bf(GATED[(long)r * w + e]);
  const float ple = rbf(g + act);
  const long at = (long)r * w + e;
  H[at] = to_bf(bf(H[at]) + ple);
  const float rinv = 1.0f / sqrt(PSS[r * S + s] / fd + eps);
  NROW[at] = to_bf((g * rinv) * NC[e]);
}

// Shift history window [taps, C] bf16 past `n` rows of stride `stride` (taps <= 16). Grid: ((C + 255) / 256).
KERNEL(256) void fn_shift_window(global ushort *WIN, global const ushort *ROWS, long stride, int n, int taps, int C) {
  const int ch = get_global_id(0);
  if (ch >= C) return;
  ushort next[16];
  for (int j = 0; j < taps; j++) {
    const int src = j + n;
    next[j] = src < taps ? WIN[(long)src * C + ch] : ROWS[(long)(src - taps) * stride + ch];
  }
  for (int j = 0; j < taps; j++) WIN[(long)j * C + ch] = next[j];
}

// 5. Gated DeltaNet (Parameterized by rank's NK, NV: full 16/48, TP=2 8/24, TP=4 4/12)

static inline float gdn_conv_act(global const ushort *P, global const ushort *cs, global const ushort *cw,
                                 int r, int c, int C, int PW) {
  float acc = 0.0f;
  for (int tap = 0; tap < 4; tap++) {
    const int at = r + tap;
    const float x = at < 3 ? bf(cs[(long)at * C + c]) : bf(P[(long)(at - 3) * PW + c]);
    acc += bf(cw[(long)c * 4 + tap]) * x;
  }
  return rbf(acc * tf_sigmoid(acc));
}

// Grid: (R, NK + NV), WG = 256.
KERNEL(256) void fn_gdn_front(global const ushort *P, global const ushort *cs, global const ushort *cw,
                              global const float *a_log, global const float *dt_bias,
                              global float *qn, global float *kn, global float *v, global float *gb,
                              int NK, int NV) {
  local float x[2][128];
  local float red[8];
  const int r = get_group_id(0), u = get_group_id(1), t = get_local_id(0);
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int C = 2 * NK * 128 + NV * 128;
  const int PW = C + NV * 128 + 2 * NV;
  if (u < NK) {
    const int hk = u;
    x[t / 128][t % 128] = gdn_conv_act(P, cs, cw, r, t < 128 ? hk * 128 + t : NK * 128 + hk * 128 + (t - 128), C, PW);
    barrier(CLK_LOCAL_MEM_FENCE);
    // Subgroup 0 norms q (128 floats, 8 per lane), subgroup 1 norms k (128 floats, 8 per lane).
    if (sg < 2) {
      float v8[8], ss = 0.0f;
      for (int i = 0; i < 8; i++) {
        v8[i] = x[sg][lane * 8 + i];
        ss += v8[i] * v8[i];
      }
      ss = sub_group_reduce_add(ss);
      float inv = rsqrt(ss + 1e-6f);
      if (sg == 0) inv *= rsqrt(128.0f);
      global float *dst = (sg == 0) ? (qn + ((long)r * NK + hk) * 128) : (kn + ((long)r * NK + hk) * 128);
      for (int i = 0; i < 8; i++) dst[lane * 8 + i] = v8[i] * inv;
    }
    (void)red;
  } else {
    const int hv = u - NK;
    if (t < 128) {
      v[((long)r * NV + hv) * 128 + t] = gdn_conv_act(P, cs, cw, r, 2 * NK * 128 + hv * 128 + t, C, PW);
    } else if (t == 128) {
      const float b = bf(P[(long)r * PW + C + NV * 128 + hv]);
      const float a = bf(P[(long)r * PW + C + NV * 128 + NV + hv]);
      gb[((long)r * NV + hv) * 2 + 0] = exp(-exp(a_log[hv]) * softplus_f(a + dt_bias[hv]));
      gb[((long)r * NV + hv) * 2 + 1] = rbf(tf_sigmoid(b));
    }
  }
}

// Recurrence over `rows` tokens: 1 subgroup (16 lanes) per (8 V-rows, V-head), 8 K-cols/lane. Grid: (16, NV).
KERNEL(16) void fn_gdn_rows(global const float *qn, global const float *kn, global const float *v,
                            global const float *gb, global const float *state_in, global float *state_out,
                            int rows, global float *ys_out, int NK, int NV) {
  const int vb = get_group_id(0), hv = get_group_id(1), lane = get_sub_group_local_id();
  const int hk = hv / (NV / NK), v0 = vb * 8;
  const long sbase = (long)hv * (128 * 128);
  float s[8][8];
  for (int j = 0; j < 8; j++)
    for (int i = 0; i < 8; i++)
      s[j][i] = state_in[sbase + (long)(v0 + j) * 128 + i * 16 + lane];

  for (int r = 0; r < rows; r++) {
    float q8[8], k8[8];
    for (int i = 0; i < 8; i++) {
      q8[i] = qn[((long)r * NK + hk) * 128 + i * 16 + lane];
      k8[i] = kn[((long)r * NK + hk) * 128 + i * 16 + lane];
    }
    const float g = gb[((long)r * NV + hv) * 2 + 0];
    const float beta = gb[((long)r * NV + hv) * 2 + 1];
    for (int j = 0; j < 8; j++) {
      float kv = 0.0f;
      for (int i = 0; i < 8; i++) {
        s[j][i] *= g;
        kv += s[j][i] * k8[i];
      }
      kv = sub_group_reduce_add(kv);
      const float delta = (v[((long)r * NV + hv) * 128 + v0 + j] - kv) * beta;
      float o = 0.0f;
      for (int i = 0; i < 8; i++) {
        s[j][i] += k8[i] * delta;
        o += s[j][i] * q8[i];
      }
      o = rbf(sub_group_reduce_add(o));
      if (lane == 0) ys_out[((long)r * NV + hv) * 128 + v0 + j] = o;
    }
  }
  for (int j = 0; j < 8; j++)
    for (int i = 0; i < 8; i++)
      state_out[sbase + (long)(v0 + j) * 128 + i * 16 + lane] = s[j][i];
}

// GDN output RMSNorm + sigmoid(z) gate (FlashNext output_gate_type = "sigmoid"!): Grid: (R, NV), WG = 128 (8
KERNEL(128) void fn_gdn_norm(global const float *ys_in, global const ushort *P, global const ushort *norm_w,
                             float eps, global ushort *out, int NK, int NV) {
  local float red[8];
  const int r = get_group_id(0), hv = get_group_id(1), t = get_local_id(0);
  const int C = 2 * NK * 128 + NV * 128;
  const int PW = C + NV * 128 + 2 * NV;
  const float y = ys_in[((long)r * NV + hv) * 128 + t];
  const float tot = wg_sum(y * y, red, 8);
  const float rinv = rsqrt(tot / 128.0f + eps);
  const float yn = rbf(rbf(y * rinv) * bf(norm_w[t]));
  const float z = bf(P[(long)r * PW + C + hv * 128 + t]);
  const float o = rbf(yn * tf_sigmoid(z));
  out[((long)r * NV + hv) * 128 + t] = to_bf(o);
}

// 6. Sparse Block-Indexed Attention (QSA: 512 blocks of 4 keys + tail)

static inline int partner_dim(int d, int half_d) {
  return d < half_d ? d + half_d : (d < 2 * half_d ? d - half_d : d);
}

static inline float apply_rope(float xn, float xpn, int d, int half_d, int pos, global const float *inv) {
  if (d >= 2 * half_d) return xn;
  const int i = d < half_d ? d : d - half_d;
  const float ang = (float)pos * inv[i];
  const float c = cos(ang), s = sin(ang);
  return d < half_d ? (xn * c - xpn * s) : (xpn * s + xn * c);
}

// Grid: (R, NQ + NKV + 5), WG = 256. (NI = 4, IHD = 128, HD = 256, HALF = 32; head NQ+NKV+4 writes IKC).
KERNEL(256) void fn_attn_prep(global const ushort *P, int pos0, global const float *QW, global const float *KW,
                              global const float *IW, global const float *INV, global ushort *Q,
                              global ushort *KC, global ushort *VC, global ushort *IQ, global ushort *IKC,
                              float eps, int PW, int NQ, int NKV) {
  local float red[16];
  const int r = get_group_id(0), head = get_group_id(1), d = get_local_id(0);
  const int NI = 4, IHD = 128, HD = 256, HALF = 32;
  const int pos = pos0 + r;
  const long row = (long)r * PW;
  if (head >= NQ + NKV + NI) {
    if (d < IHD) IKC[(long)pos * IHD + d] = P[row + NQ * 2 * HD + 2 * NKV * HD + NI * IHD + d];
    return;
  }
  const bool index = head >= NQ + NKV;
  const int width = index ? IHD : HD;
  const bool live = d < width;
  const long src = row + (head < NQ ? head * 2 * HD
                                    : (head < NQ + NKV ? NQ * 2 * HD + (head - NQ) * HD
                                                       : NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD));
  global const float *w = head < NQ ? QW : (head < NQ + NKV ? KW : IW);
  const float x = live ? bf(P[src + d]) : 0.0f;
  const float tot = wg_sum(x * x, red, 16);
  const float rinv = rsqrt(tot / (float)width + eps);
  if (!live) return;
  const int p = partner_dim(d, HALF);
  const float xn = rbf((x * rinv) * w[d]);
  const float xpn = rbf((bf(P[src + p]) * rinv) * w[p]);
  const ushort out = to_bf(apply_rope(xn, xpn, d, HALF, pos, INV));
  if (head < NQ) {
    Q[((long)r * NQ + head) * HD + d] = out;
  } else if (!index) {
    const long at = ((long)pos * NKV + (head - NQ)) * HD;
    KC[at + d] = out;
    VC[at + d] = P[row + NQ * 2 * HD + NKV * HD + (head - NQ) * HD + d];
  } else {
    IQ[((long)r * NI + (head - NQ - NKV)) * IHD + d] = out;
  }
}

// Pool completed blocks of 4 indexer keys: grid ((pos0 + R)/4 - pos0/4 + 1), WG = 128.
KERNEL(128) void fn_qsa_pool(global const ushort *IKC, global ushort *POOLED, int pos0, global const float *W,
                             global const float *INV, float eps, int R) {
  local float red[8];
  local float xs[128];
  const int DI = 128, HALF = 32, RATIO = 4;
  const int b = pos0 / RATIO + get_group_id(0), d = get_local_id(0);
  if (RATIO * b + RATIO > pos0 + R) return;
  const long first = (long)RATIO * b;
  float acc = bf(IKC[first * DI + d]);
  for (int k = 1; k < RATIO; k++) acc += bf(IKC[(first + k) * DI + d]);
  const float x = rbf(acc * 0.25f);
  xs[d] = x;
  const float tot = wg_sum(x * x, red, 8);
  const float rinv = rsqrt(tot / (float)DI + eps);
  barrier(CLK_LOCAL_MEM_FENCE);
  const int p = partner_dim(d, HALF);
  const float xn = rbf((x * rinv) * W[d]);
  const float xpn = rbf((xs[p] * rinv) * W[p]);
  POOLED[(long)b * DI + d] = to_bf(apply_rope(xn, xpn, d, HALF, RATIO * b, INV));
}

// Score blocks past TOP=512: grid ((R + 15) / 16, (complete_max + 63) / 64), WG=256 (64 blocks x 16 queries).
KERNEL(256) void fn_qsa_scores(global const ushort *IQ, global const ushort *POOLED, int pos0, global float *SC,
                               int NB, int R) {
  local ushort qs[16][512];
  const int r0 = (int)get_group_id(0) * 16, j = (int)get_group_id(1), lid = get_local_id(0);
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  if (r0 >= R) return;
  const int m_rows = min(16, R - r0);
  const int complete_max = (pos0 + r0 + m_rows) / 4;
  if (complete_max <= 512 || j * 64 >= complete_max) return;
  for (int e = lid; e < m_rows * 512; e += 256) qs[e >> 9][e & 511] = IQ[(long)r0 * 512 + e];
  barrier(CLK_LOCAL_MEM_FENCE);
  const float scale = sqrt(128.0f);
  for (int bs = 0; bs < 4; bs++) {
    const int b = j * 64 + bs * 16 + sg;
    if (b >= complete_max) break;
    ushort8 kb = vload8(0, POOLED + (long)b * 128 + lane * 8);
    float k8[8];
    for (int i = 0; i < 8; i++) k8[i] = bf(kb[i]);
    for (int rr = 0; rr < m_rows; rr++) {
      const int r = r0 + rr;
      const int complete_r = (pos0 + r + 1) / 4;
      if (complete_r <= 512 || b >= complete_r) continue;
      float total = 0.0f;
      for (int h = 0; h < 4; h++) {
        float part = 0.0f;
        for (int i = 0; i < 8; i++) part = fma(k8[i], bf(qs[rr][h * 128 + lane * 8 + i]), part);
        total += fmax(sub_group_reduce_add(part), 0.0f);
      }
      if (lane == 0) SC[(long)r * NB + b] = total / scale;
    }
  }
}

static inline uint qsa_order_key(float v) {
  const uint bits = as_uint(v);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Select top-512 blocks of 4 keys + tail: grid (R), WG = 256.
KERNEL(256) void fn_qsa_select(global const float *SC, int pos0, global int *IDS, global int *NKR, global int *SPR,
                               int NB) {
  local int hist[256];
  local uint cut_s;
  local int need_s;
  local int scan_gt[256], scan_eq[256], high_s;
  const int r = get_group_id(0), t = get_local_id(0);
  const int IDW = 2052, TOP = 512, RATIO = 4;
  const int end = pos0 + r + 1;
  const int complete = end / RATIO;
  if (complete <= TOP) {
    if (t == 0) {
      NKR[r] = end;
      SPR[r] = 0;
    }
    return;
  }
  global const float *row = SC + (long)r * NB;
  global int *out = IDS + (long)r * IDW;
  uint cut = 0;
  int need = TOP;
  for (int p = 0; p < 4; p++) {
    const int shift = 24 - 8 * p;
    hist[t] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int b = t; b < complete; b += 256) {
      const uint k = qsa_order_key(row[b]);
      if (p == 0 || (k >> (shift + 8)) == (cut >> (shift + 8))) {
        atomic_inc(&hist[(k >> shift) & 0xFFu]);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
      int s = 0, best = 0, rest = need;
      bool found = false;
      for (int d = 255; d >= 1; d--) {
        if (!found && s + hist[d] >= need) {
          best = d;
          rest = need - s;
          found = true;
          break;
        }
        s += hist[d];
      }
      if (!found) {
        best = 0;
        rest = need - s;
      }
      cut_s = cut | ((uint)best << shift);
      need_s = rest;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    cut = cut_s;
    need = need_s;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // Contiguous slice per thread to preserve block order for ties!
  const int per = (complete + 255) / 256;
  const int lo = t * per, hi = min(complete, lo + per);
  int gt = 0, eq = 0;
  for (int b = lo; b < hi; b++) {
    const uint k = qsa_order_key(row[b]);
    if (k > cut) gt++;
    else if (k == cut) eq++;
  }
  scan_gt[t] = gt;
  scan_eq[t] = eq;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t == 0) {
    int sg_acc = 0, sq_acc = 0;
    for (int i = 0; i < 256; i++) {
      const int g0 = scan_gt[i], e0 = scan_eq[i];
      scan_gt[i] = sg_acc;
      scan_eq[i] = sq_acc;
      sg_acc += g0;
      sq_acc += e0;
    }
    high_s = sg_acc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int high = high_s, rem_need = TOP - high;
  int at_gt = scan_gt[t], at_eq = scan_eq[t];
  for (int b = lo; b < hi; b++) {
    const uint k = qsa_order_key(row[b]);
    if (k > cut) {
      for (int j = 0; j < RATIO; j++) out[at_gt * RATIO + j] = b * RATIO + j;
      at_gt++;
    } else if (k == cut && at_eq < rem_need) {
      for (int j = 0; j < RATIO; j++) out[(high + at_eq) * RATIO + j] = b * RATIO + j;
      at_eq++;
    }
  }
  if (t < RATIO) {
    const int tail = RATIO * complete + t;
    if (tail < end) out[TOP * RATIO + t] = tail;
  }
  if (t == 0) {
    NKR[r] = TOP * RATIO + (end - RATIO * complete);
    SPR[r] = 1;
  }
}

// Sparse/dense attention over key slices of SPLIT=32 keys (at most 65 slices since NKR <= 2051!). One subgroup (16
KERNEL(16) void fn_attn_sparse(global const ushort *Q, global const ushort *KC, global const ushort *VC,
                               global const int *IDS, global const int *NKR, global const int *SPR,
                               global float *PM, global float *PL, global float *PO,
                               int NS, int NQ, int NKV) {
  const int sp = get_group_id(0), qh = get_group_id(1), r = get_group_id(2), lane = get_sub_group_local_id();
  const int GQA = NQ / NKV, hk = qh / GQA;
  const int nkeys = NKR[r], sparse = SPR[r];
  const int j0 = sp * 32, j1 = min(j0 + 32, nkeys);
  const float scale = 1.0f / 16.0f; // 1 / sqrt(256)
  float q[16], o[16];
  for (int i = 0; i < 16; i++) {
    q[i] = bf(Q[((long)r * NQ + qh) * 256 + i * 16 + lane]);
    o[i] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  for (int j = j0; j < j1; j++) {
    const long key = sparse ? (long)IDS[(long)r * 2052 + j] : (long)j;
    const long kbase = (key * NKV + hk) * 256;
    float d = 0.0f;
    for (int i = 0; i < 16; i++) d += q[i] * bf(KC[kbase + i * 16 + lane]);
    const float sc = sub_group_reduce_add(d) * scale;
    const float mn = fmax(m, sc);
    const float al = (m == -INFINITY) ? 0.0f : exp(m - mn);
    const float p = exp(sc - mn);
    l = l * al + p;
    for (int i = 0; i < 16; i++) o[i] = o[i] * al + p * bf(VC[kbase + i * 16 + lane]);
    m = mn;
  }
  const long base = ((long)r * NQ + qh) * NS + sp;
  if (lane == 0) {
    PM[base] = m;
    PL[base] = l;
  }
  for (int i = 0; i < 16; i++) PO[base * 256 + i * 16 + lane] = o[i];
}

// Merge key slices in ascending order: Grid: (NQ, R), WG = 16.
KERNEL(16) void fn_attn_merge(global const float *PM, global const float *PL, global const float *PO,
                              global const int *NKR, global ushort *AO, int NS, int NQ) {
  const int qh = get_group_id(0), r = get_group_id(1), lane = get_sub_group_local_id();
  const int used = (NKR[r] + 31) / 32;
  const long base = ((long)r * NQ + qh) * NS;
  float m = -INFINITY;
  for (int s = 0; s < used; s++) m = fmax(m, PM[base + s]);
  float l = 0.0f, o[16];
  for (int i = 0; i < 16; i++) o[i] = 0.0f;
  for (int s = 0; s < used; s++) {
    const float pm = PM[base + s];
    if (pm == -INFINITY) continue;
    const float w = exp(pm - m);
    l += PL[base + s] * w;
    for (int i = 0; i < 16; i++) o[i] += PO[(base + s) * 256 + i * 16 + lane] * w;
  }
  const float inv = l > 0.0f ? (1.0f / l) : 0.0f;
  for (int i = 0; i < 16; i++) AO[((long)r * NQ + qh) * 256 + i * 16 + lane] = to_bf(rbf(o[i] * inv));
}

// Attention sigmoid gate: OUT[r, qh*256 + d] = bf16(AO / (1 + e^-gate)). Grid: (R, NQ), WG = 256.
KERNEL(256) void fn_attn_gate(global const ushort *AO, global const ushort *P, global ushort *OUT, int PW, int NQ) {
  const int r = get_group_id(0), h = get_group_id(1), d = get_local_id(0);
  const float o = bf(AO[((long)r * NQ + h) * 256 + d]);
  const float g = bf(P[(long)r * PW + h * 512 + 256 + d]);
  OUT[(long)r * (NQ * 256) + h * 256 + d] = to_bf(o * tf_sigmoid(g));
}

// 7. MoE Router + TP-Sharded Routed Experts & Shared Expert
