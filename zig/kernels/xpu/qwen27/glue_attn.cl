// Qwen3.8-27B glue, part 2 (included by glue.cl): Hadamard and int8 K/V helpers, the attention kernels.

// Attention, before the kernel: q, k RMS per head with weight, RoPE on the first 64 dims; K, V written to the cache.

// Hadamard rotation of a head of 256 (normalized by 1/16) over the subgroup; its own inverse, so (qH)·(kH) = q·k.
static void fwht256(float v[16]) {
  const int lane = get_sub_group_local_id();
  for (int m = 1; m < 16; m <<= 1)
    for (int i = 0; i < 16; i++) {
      const float o = sub_group_shuffle_xor(v[i], m);
      v[i] = (lane & m) ? o - v[i] : v[i] + o;
    }
  for (int m = 1; m < 16; m <<= 1)
    for (int i = 0; i < 16; i++)
      if (!(i & m)) {
        const float a = v[i], b = v[i | m];
        v[i] = a + b;
        v[i | m] = a - b;
      }
  for (int i = 0; i < 16; i++) v[i] *= 1.0f / 16.0f;
}

// K/V cache study: a head quantized to symmetric integers of `levels` steps (127: int8, 7: int4), then back to float.
static void quant8_head(float v[16], float levels) {
  float m = 0.0f;
  for (int i = 0; i < 16; i++) m = fmax(m, fabs(v[i]));
  m = sub_group_reduce_max(m);
  const float s = m > 0.0f ? m / levels : 1.0f;
  for (int i = 0; i < 16; i++) v[i] = rint(v[i] / s) * s;
}

// Attention output brought back from the rotated basis (KV8 & 1): O bf16 [T, 24, 256], grid (T, 24).
KERNEL(16) void rot_heads(global ushort *O) {
  const int t = get_group_id(0), hh = get_group_id(1), lane = get_sub_group_local_id();
  global ushort *o = O + ((long)t * 24 + hh) * 256;
  float v[16];
  for (int i = 0; i < 16; i++) v[i] = bf2f(o[i * 16 + lane]);
  fwht256(v);
  for (int i = 0; i < 16; i++) o[i * 16 + lane] = f2bf(v[i]);
}

// One K or V head in the int8 cache (cap positions): 256 int8 at row `row`, the scale after the cap · 4 rows.
static void store_kv8(global ushort *C, long row, int h, const float v[16], int cap) {
  const int lane = get_sub_group_local_id();
  float m = 0.0f;
  for (int i = 0; i < 16; i++) m = fmax(m, fabs(v[i]));
  m = sub_group_reduce_max(m);
  const float s = m > 0.0f ? m / 127.0f : 1.0f;
  global char *c = (global char *)C + (row * 4 + h) * 256;
  for (int i = 0; i < 16; i++) c[i * 16 + lane] = (char)rint(v[i] / s);
  if (lane == 0) ((global float *)((global char *)C + (long)cap * 1024))[row * 4 + h] = s;
}

KERNEL(256) void attn_prep(global const half *P, global const ushort *QN, global const ushort *KN,
                           global const float *INVF, global ushort *Q, global ushort *KC, global ushort *VC, int p0,
                           float eps, global const int *TREE, global const ulong *MT, global const int *POS, int kv8,
                           int cap) {
  // cap > 0: int8 K/V cache (store_kv8)

  // kv8 (K/V cache study): & 1 rotates q, k, v by Hadamard; & 2 quantizes k, v per head (int4 with & 4)

  // TREE (if non-null): row t's position is p0 + its depth, K, V stay at p0 + t; with MT, POS[t] is its position

  // MT (if non-null): several streams, one row each: caches MT[8t], MT[8t + 1], position POS[t], where K, V go.
  const int t = get_group_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const float pos = (float)(MT ? POS[t] : p0 + (TREE ? TREE[t * 17] & 255 : t));
  const int wf = MT ? (int)MT[t * 8 + 6] : 0;
  const long krow = MT ? (TREE ? POS[wf] + t - wf : POS[t]) : p0 + t; // cache row where K and V go
  if (MT) {
    KC = (global ushort *)MT[t * 8];
    VC = (global ushort *)MT[t * 8 + 1];
  }
  for (int hh = sg; hh < 32; hh += 16) {               // 0-23 : q, 24-27 : k, 28-31 : v
    if (hh >= 28) {
      float vv[16];
      for (int i = 0; i < 16; i++) vv[i] = (float)P[(long)t * 14336 + 13312 + (hh - 28) * 256 + i * 16 + lane];
      if (kv8 & 1) fwht256(vv);
      if (kv8 & 2) quant8_head(vv, (kv8 & 4) ? 7.0f : 127.0f);
      if (cap)
        store_kv8(VC, krow, hh - 28, vv, cap);
      else
        for (int i = 0; i < 16; i++) VC[(krow * 4 + hh - 28) * 256 + i * 16 + lane] = f2bf(vv[i]);
      continue;
    }
    const bool isq = hh < 24;
    const long src = (long)t * 14336 + (isq ? hh * 512 : 12288 + (hh - 24) * 256);
    float v[16], ss = 0.0f;
    for (int i = 0; i < 16; i++) {
      v[i] = (float)P[src + i * 16 + lane];
      ss += v[i] * v[i];
    }
    const float inv = rsqrt(sub_group_reduce_add(ss) / 256.0f + eps);
    for (int i = 0; i < 16; i++) v[i] = rbf(v[i] * inv * bf2f((isq ? QN : KN)[i * 16 + lane]));
    // RoPE: dimensions 0-31 (x1) and 32-63 (x2), i.e. v[0], v[1] (x1) and v[2], v[3] (x2) for this lane.
    for (int i = 0; i < 2; i++) {
      const float ang = pos * INVF[i * 16 + lane], c = cos(ang), s = sin(ang);
      const float x1 = v[i], x2 = v[i + 2];
      v[i] = x1 * c - x2 * s;
      v[i + 2] = x2 * c + x1 * s;
    }
    if (kv8 & 1) fwht256(v);
    if ((kv8 & 2) && !isq) quant8_head(v, (kv8 & 4) ? 7.0f : 127.0f);
    if (!isq && cap) {
      store_kv8(KC, krow, hh - 24, v, cap);
      continue;
    }
    global ushort *dst = isq ? Q + ((long)t * 24 + hh) * 256 : KC + (krow * 4 + hh - 24) * 256;
    for (int i = 0; i < 16; i++) dst[i * 16 + lane] = f2bf(v[i]);
  }
}

// Attention output: fp16(bf16(o·σ(gate))) [T, 6 144] and its row sums, or (A8) its W4A8 quantization.
KERNEL(256) void attn_gate(global const ushort *O, global const half *P, global half *H, global half *AHI,
                           global half *ALO, int gs, int rot, global char *A8, global float *SA, int MS) {
  local half hl[6144];
  const int t = get_group_id(0), lane = get_sub_group_local_id();
  for (int b = get_sub_group_id(); b < 48; b += get_num_sub_groups()) {   // blocks of 128, block reads
    const ushort8 ov = intel_sub_group_block_read_us8(O + (long)t * 6144 + b * 128);
    const half8 gv = as_half8(intel_sub_group_block_read_us8((global const ushort *)(P + (long)t * 14336 + (b / 2) * 512 + 256 + (b % 2) * 128)));
    half8 o;
    for (int i = 0; i < 8; i++) {
      o[i] = convert_half_rte(rbf(bf2f(ov[i]) * sigmoid_f((float)gv[i])));
      hl[b * 128 + i * 16 + lane] = o[i];
    }
    if (!rot) intel_sub_group_block_write_us8((global ushort *)(H + (long)t * 6144 + b * 128), as_ushort8(o));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  out_rows(hl, 6144, H, t, rot, gs, A8, SA, MS, AHI, ALO);
}

// SwiGLU: act = fp16(bf16(SiLU(g)·u)) and its row sums, a group of 256 per 1 024 columns; ilv: gate|up by 16.
KERNEL(256) void swiglu(global const half *GU, global half *H, global half *AHI, global half *ALO, int gs, int ilv,
                        int rot) {
  local half hl[1024];
  const int t = get_group_id(0), blk = get_group_id(1), lid = get_local_id(0);
  {                                                   // 4 consecutive columns per work-item, vector reads and write
    const int e = lid * 4, c = blk * 1024 + e;
    const long gc = ilv ? (c / 16) * 32 + c % 16 : c, uc = ilv ? gc + 16 : 17408 + c;
    const half4 gv = vload4(0, GU + (long)t * 34816 + gc), uv = vload4(0, GU + (long)t * 34816 + uc);
    half4 o;
    for (int i = 0; i < 4; i++) o[i] = convert_half_rte(rbf(silu((float)gv[i]) * (float)uv[i]));
    if (!rot) vstore4(o, 0, H + (long)t * 17408 + c);
    vstore4(o, 0, hl + e);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (rot) {                                          // the 8 blocks of 128 of these 1 024 columns
    if (get_sub_group_id() < 8) {
      const int o = get_sub_group_id() * 128 + get_sub_group_local_id() * 8;
      vstore8(rot128_reg(vload8(0, hl + o)), 0, hl + o);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    vstore4(vload4(0, hl + lid * 4), 0, H + (long)t * 17408 + blk * 1024 + lid * 4);
  }
  const int lane = get_sub_group_local_id(), per = gs / 16, ng = 1024 / gs;
  for (int g = get_sub_group_id(); g < ng; g += 16) {
    float s = 0.0f;
    for (int k = 0; k < per; k++) s += (float)hl[g * gs + lane * per + k];
    s = sub_group_reduce_add(s) * (1.0f / gs);
    if (lane == 0) {
      const half hi = convert_half_rte(s);
      AHI[(long)t * (17408 / gs) + blk * ng + g] = hi;
      ALO[(long)t * (17408 / gs) + blk * ng + g] = convert_half_rte(s - (float)hi);
    }
  }
}

// Row sums of the activation written by the SwiGLU epilogue (H fp16 [T, 17 408]): grid T, groups of 256.
KERNEL(128) void rowsums_ff(global half *H, global half *AHI, global half *ALO, int gs, int rot) {
  // the sums of rowsums read directly from global memory (same additions, same order: same bits)
  const int t = get_group_id(0), lane = get_sub_group_local_id(), per = gs / 16;
  global half *h = H + (long)t * 17408;
  if (rot) {                                          // rotated weights: rotate each block in place, then sum
    for (int b = get_sub_group_id(); b < 17408 / 128; b += get_num_sub_groups()) {
      float v[8];
      for (int i = 0; i < 8; i++) v[i] = (float)h[b * 128 + i * 16 + lane];
      for (int m = 1; m < 16; m <<= 1)
        for (int i = 0; i < 8; i++) {
          const float o = sub_group_shuffle_xor(v[i], m);
          v[i] = (lane & m) ? o - v[i] : v[i] + o;
        }
      for (int m = 1; m < 8; m <<= 1)
        for (int i = 0; i < 8; i++)
          if (!(i & m)) {
            const float x = v[i], y = v[i | m];
            v[i] = x + y;
            v[i | m] = x - y;
          }
      float s = 0.0f;
      for (int i = 0; i < 8; i++) {
        const half o = convert_half_rte(v[i] * 0.08838834764831845f);
        h[b * 128 + i * 16 + lane] = o;
        s += (float)o;
      }
      s = sub_group_reduce_add(s) * (1.0f / 128);
      if (lane == 0) {
        const half hi = convert_half_rte(s);
        AHI[(long)t * 136 + b] = hi;
        ALO[(long)t * 136 + b] = convert_half_rte(s - (float)hi);
      }
    }
    return;
  }
  for (int g = get_sub_group_id(); g < 17408 / gs; g += get_num_sub_groups()) {
    float s = 0.0f;
    if (per == 8) {
      const half8 x = vload8(0, h + g * gs + lane * 8);
      for (int k = 0; k < 8; k++) s += (float)x[k];
    } else {
      const half4 x = vload4(0, h + g * gs + lane * 4);
      for (int k = 0; k < 4; k++) s += (float)x[k];
    }
    s = sub_group_reduce_add(s) * (1.0f / gs);
    if (lane == 0) {
      const half hi = convert_half_rte(s);
      AHI[(long)t * (17408 / gs) + g] = hi;
      ALO[(long)t * (17408 / gs) + g] = convert_half_rte(s - (float)hi);
    }
  }
}
