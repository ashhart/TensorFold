// Qwen3.8-27B glue, part 3 (included by glue.cl): activation quantization (W4A8, W4A4) and weight repacking.

// W4A8 study: each row of X quantized to int8 per groups of qg, back to fp16 in place, row sums redone; rot: Hadamard.
KERNEL(256) void fakequant_rows(global half *X, global half *AHI, global half *ALO, int K, int qg, int gs, int rot,
                                int levels) {
  local half hl[17408];
  const int t = get_group_id(0), lane = get_sub_group_local_id();
  for (int e = get_local_id(0) * 4; e < K; e += 1024) vstore4(vload4(0, X + (long)t * K + e), 0, hl + e);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int b = get_sub_group_id(); rot && b < K / 128; b += get_num_sub_groups()) {
    float v[8];                                      // element 16 i + lane of the block
    for (int i = 0; i < 8; i++) v[i] = (float)hl[b * 128 + i * 16 + lane];
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
    for (int i = 0; i < 8; i++) hl[b * 128 + i * 16 + lane] = convert_half_rte(v[i] * 0.08838834764831845f);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int g = get_sub_group_id(); g < K / qg; g += get_num_sub_groups()) {
    float m = 0.0f;
    for (int k = lane; k < qg; k += 16) m = fmax(m, fabs((float)hl[g * qg + k]));
    m = sub_group_reduce_max(m);
    const float s = m > 0.0f ? m / (float)levels : 1.0f;
    for (int k = lane; k < qg; k += 16) {
      const float q = clamp(rint((float)hl[g * qg + k] / s), -(float)levels - (levels == 7 ? 1.0f : 0.0f), (float)levels);
      hl[g * qg + k] = convert_half_rte(q * s);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int e = get_local_id(0) * 4; e < K; e += 1024) vstore4(vload4(0, hl + e), 0, X + (long)t * K + e);
  rowsums(hl, K, AHI, ALO, t, gs);
}

// Prompt W4A8 activation quantization: row t to int8 per group of 128 in A8; AHI, ALO: hi/lo sum of dequantized.
static void q8_block(half8 xv, int K, int g, global char *A8, global float *SA, int MS, global half *AHI,
                     global half *ALO, int t) {
  const int lane = get_sub_group_local_id();
  float m = 0.0f;
  for (int j = 0; j < 8; j++) m = fmax(m, fabs((float)xv[j]));
  m = sub_group_reduce_max(m);
  const float s = m > 0.0f ? m / 127.0f : 1.0f, r = 1.0f / s; // one division per block, not per value
  char4 q0, q1;                                       // double word (w & m), then ((w >> 4) & m)
  int qs = 0;
  for (int j = 0; j < 8; j++) {
    const int v = (int)clamp(rint((float)xv[j] * r), -127.0f, 127.0f);
    const int b = (j % 2) * 2 + j / 4;                // byte within its double word
    if ((j % 4) < 2) q0[b] = (char)v; else q1[b] = (char)v;
    qs += v;
  }
  global char *blk = A8 + (long)t * K + g * 128 + (lane / 2) * 16;
  vstore4(q0, 0, blk + (lane % 2) * 4);
  vstore4(q1, 0, blk + 8 + (lane % 2) * 4);
  qs = sub_group_reduce_add(qs);
  if (lane == 0) {
    SA[(long)g * MS + t] = s;
    const float a = s * (float)qs * (1.0f / 128.0f);
    const half hi = convert_half_rte(a);
    AHI[(long)t * (K / 128) + g] = hi;
    ALO[(long)t * (K / 128) + g] = convert_half_rte(a - (float)hi);
  }
}
static void quant_row(local const half *hl, int K, global char *A8, global float *SA, int MS, global half *AHI,
                      global half *ALO, int t) {
  for (int g = get_sub_group_id(); g < K / 128; g += get_num_sub_groups())
    q8_block(vload8(0, hl + g * 128 + get_sub_group_local_id() * 8), K, g, A8, SA, MS, AHI, ALO, t);
}

// Output of a kernel that stored row t in local memory: rotate 128-blocks if rot, then W4A8 (A8) or row sums per gs.
static void out_rows(local const half *hl, int K, global half *H, int t, int rot, int gs, global char *A8,
                     global float *SA, int MS, global half *AHI, global half *ALO) {
  if (!rot && !A8) {
    rowsums(hl, K, AHI, ALO, t, gs);
    return;
  }
  const int lane = get_sub_group_local_id();
  for (int g = get_sub_group_id(); g < K / 128; g += get_num_sub_groups()) {
    half8 xv = vload8(0, hl + g * 128 + lane * 8);
    if (rot) {
      xv = rot128_reg(xv);
      vstore8(xv, 0, H + (long)t * K + g * 128 + lane * 8);
    }
    if (A8) {
      q8_block(xv, K, g, A8, SA, MS, AHI, ALO, t);
      continue;
    }
    float s = 0.0f;
    for (int j = 0; j < 8; j++) s += (float)xv[j];
    s = sub_group_reduce_add(s) * (1.0f / 128);
    if (lane == 0) {
      const half hi = convert_half_rte(s);
      AHI[(long)t * (K / 128) + g] = hi;
      ALO[(long)t * (K / 128) + g] = convert_half_rte(s - (float)hi);
    }
  }
}

// W4A4 (rotated weights): one group of 128 of a row in int4, the lane's word in the repacked weights' nibble order.
static void q4_block(half8 xv, int K, int g, global uint *A4, global float *SA, int MS, global half *AHI,
                     global half *ALO, int t) {
  const int lane = get_sub_group_local_id();
  float m = 0.0f;
  for (int j = 0; j < 8; j++) m = fmax(m, fabs((float)xv[j]));
  m = sub_group_reduce_max(m);
  const float s = m > 0.0f ? m / 7.0f : 1.0f, r = 1.0f / s;
  uint w = 0;
  int qs = 0;
  for (int i = 0; i < 8; i++) {
    const int j = i < 4 ? 2 * i : 2 * (i - 4) + 1;
    const int v = (int)clamp(rint((float)xv[j] * r), -8.0f, 7.0f);
    w |= ((uint)v & 0xFu) << (4 * i);
    qs += v;
  }
  A4[((long)t * K / 2 + g * 64) / 4 + lane] = w;
  qs = sub_group_reduce_add(qs);
  if (lane == 0) {
    SA[(long)g * MS + t] = s;
    const float a = s * (float)qs * (1.0f / 128.0f);
    const half hi = convert_half_rte(a);
    AHI[(long)t * (K / 128) + g] = hi;
    ALO[(long)t * (K / 128) + g] = convert_half_rte(a - (float)hi);
  }
}
// Quantization alone, row X fp16 [T, K]: one group of 128 per row, in registers (rot: block rotated first).
KERNEL(128) void quant_a4(global const half *X, global uint *A4, global float *SA, global half *AHI, global half *ALO,
                          int K, int MS, int rot) {
  const int t = get_group_id(0);
  for (int g = get_sub_group_id(); g < K / 128; g += get_num_sub_groups()) {
    half8 xv = vload8(0, X + (long)t * K + g * 128 + get_sub_group_local_id() * 8);
    if (rot) xv = rot128_reg(xv);
    q4_block(xv, K, g, A4, SA, MS, AHI, ALO, t);
  }
}

KERNEL(128) void quant_a8(global const half *X, global char *A8, global float *SA, global half *AHI, global half *ALO,
                          int K, int MS, int rot) {
  const int t = get_group_id(0);
  for (int g = get_sub_group_id(); g < K / 128; g += get_num_sub_groups()) {
    half8 xv = vload8(0, X + (long)t * K + g * 128 + get_sub_group_local_id() * 8);
    if (rot) xv = rot128_reg(xv);
    q8_block(xv, K, g, A8, SA, MS, AHI, ALO, t);
  }
}

// Arg max of each row of fp16 logits [T, V] (first index on ties), in two kernels: partial maxima, then their maximum.
KERNEL(256) void argmax_part(global const half *L, int V, global float *PV, global int *PI) {
  local float bv[256];
  local int bi[256];
  const int lid = get_local_id(0), gid = get_group_id(0), row = get_group_id(1), per = (V + 255) / 256;
  float best = -INFINITY;
  int idx = 0x7fffffff;
  for (int i = gid * per + lid; i < min(V, (gid + 1) * per); i += 256) {
    const float v = (float)L[(long)row * V + i];
    if (v > best) { best = v; idx = i; }
  }
  bv[lid] = best;
  bi[lid] = idx;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    for (int i = 1; i < 256; i++)
      if (bv[i] > best || (bv[i] == best && bi[i] < idx)) { best = bv[i]; idx = bi[i]; }
    PV[row * 256 + gid] = best;
    PI[row * 256 + gid] = idx;
  }
}
// Structured output: argmax_part over logits rows ROWS[j], among the tokens grammar mask j allows; same ties.
KERNEL(256) void argmax_part_masked(global const half *L, int V, global const uint *MASK, int W, global const int *ROWS,
                                    global float *PV, global int *PI) {
  local float bv[256];
  local int bi[256];
  const int lid = get_local_id(0), gid = get_group_id(0), j = get_group_id(1), per = (V + 255) / 256;
  const long row = ROWS[j];
  global const uint *m = MASK + (long)j * W;
  float best = -INFINITY;
  int idx = 0x7fffffff;
  for (int i = gid * per + lid; i < min(V, (gid + 1) * per); i += 256) {
    if (i / 32 >= W || !((m[i / 32] >> (i % 32)) & 1u)) continue;
    const float v = (float)L[row * V + i];
    if (v > best || idx == 0x7fffffff) { best = v; idx = i; }
  }
  bv[lid] = best;
  bi[lid] = idx;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    for (int i = 1; i < 256; i++)
      if (bi[i] != 0x7fffffff && (idx == 0x7fffffff || bv[i] > best || (bv[i] == best && bi[i] < idx))) { best = bv[i]; idx = bi[i]; }
    PV[j * 256 + gid] = best;
    PI[j * 256 + gid] = idx;
  }
}
KERNEL(256) void argmax(global const float *PV, global const int *PI, global int *OUT) {
  local float bv[256];
  local int bi[256];
  const int lid = get_local_id(0), row = get_group_id(0);
  bv[lid] = PV[row * 256 + lid];
  bi[lid] = PI[row * 256 + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    float best = bv[0];
    int idx = bi[0];
    for (int i = 1; i < 256; i++)
      if (bv[i] > best || (bv[i] == best && bi[i] < idx)) { best = bv[i]; idx = bi[i]; }
    OUT[row] = idx;
  }
}

// Taps for the drafter: TAPS[t][slot·D + e] = bf16(x + r) (the full hidden state after the layer), [T, 5 D].
KERNEL(256) void tap(global const float *X, global const half *R, global ushort *TAPS, int D, int slot) {
  const int t = get_group_id(1), e = get_global_id(0);
  if (e < D) TAPS[(long)t * 5 * D + slot * D + e] = f2bf(rbf(X[(long)t * D + e] + (float)R[(long)t * D + e]));
}

// Convolution state after a kept path of KEEP rows: COUT = last 3 rows of concat(CIN, qkv of the path), qkv from P.
KERNEL(256) void conv_commit(global const half *CIN, global const half *P, global half *COUT, int keep,
                             global const int *PATH) {
  const int c = get_global_id(0);
  for (int r = 0; r < 3; r++) {
    const int i = keep + r;                           // row of concat(state, qkv), 3 + keep rows
    COUT[r * NCONV + c] = i < 3 ? CIN[i * NCONV + c] : P[(long)(PATH ? PATH[i - 3] : i - 3) * 16640 + c];
  }
}

// Load-time repack of an MLX projection into the GEMM layout: interleaved nibbles, fp16 scales, biases × 64.
kernel void repack(global const uint *MW, global const ushort *MS, global const ushort *MB, global uint *W,
                   global half *S, global half *BG, int N, int K, int NT, int n0, int ILV) {
  const int n = get_global_id(0), w = get_global_id(1);
  if (n >= N || w >= K / 8) return;
  const long col = ILV ? (n / 16) * 32 + (ILV - 1) * 16 + n % 16 : n0 + n;
  const uint m = MW[(long)n * (K / 8) + w];
  uint o = 0;
  for (int p = 0; p < 4; p++) {
    o |= ((m >> (8 * p)) & 0xFu) << (4 * p);
    o |= ((m >> (8 * p + 4)) & 0xFu) << (16 + 4 * p);
  }
  W[(long)w * NT + col] = o;
  if (w % (GS / 8) == 0) {
    const int g = w / (GS / 8);
    S[(long)g * NT + col] = convert_half_rte(bf2f(MS[(long)n * (K / GS) + g]));
    BG[(long)g * NT + col] = convert_half_rte(bf2f(MB[(long)n * (K / GS) + g]) * GS);
  }
}

// Load-time repack of a symmetric 4-bit GPTQ projection into the GEMM layout: same words, S = s, bias × G = −8·s·G.
kernel void repack_gptq(global const uint *QW, global const half *SC, global uint *W, global half *S, global half *BG,
                        int N, int K, int NT, int n0, int G, int ILV) {
  const int n = get_global_id(0), w = get_global_id(1);
  if (n >= N || w >= K / 8) return;
  const long col = ILV ? (n / 16) * 32 + (ILV - 1) * 16 + n % 16 : n0 + n;
  const uint m = QW[(long)w * N + n];
  uint o = 0;
  for (int p = 0; p < 4; p++) {
    o |= ((m >> (8 * p)) & 0xFu) << (4 * p);
    o |= ((m >> (8 * p + 4)) & 0xFu) << (16 + 4 * p);
  }
  W[(long)w * NT + col] = o;
  if (w % (G / 8) == 0) {
    const int g = w / (G / 8);
    const half s = SC[(long)g * N + n];
    S[(long)g * NT + col] = s;
    BG[(long)g * NT + col] = convert_half_rte((float)s * (-8.0f * G));
  }
}

// Affine 4-bit quantization (groups of 64) of an fp16 weight [N, K] by MLX's algorithm, into the layout `repack` reads.
kernel void quantize_mlx(global const half *Wt, int N, int K, global uint *MW, global ushort *MS, global ushort *MB) {
  const int n = get_global_id(0), g = get_global_id(1);
  if (n >= N || g >= K / 64) return;
  global const half *row = Wt + (long)n * K + g * 64;
  float lo = INFINITY, hi = -INFINITY;
  for (int i = 0; i < 64; i++) {
    const float v = (float)row[i];
    lo = fmin(lo, v);
    hi = fmax(hi, v);
  }
  const bool neg = fabs(lo) > fabs(hi);
  float scale = fmax((hi - lo) / 15.0f, 1e-7f);
  scale = neg ? scale : -scale;
  const float edge = neg ? lo : hi;
  const float q0 = rint(edge / scale);
  if (q0 != 0.0f) scale = edge / q0;
  const float bias = q0 == 0.0f ? 0.0f : edge;
  const float s = rbf(scale), b = rbf(bias);
  MS[(long)n * (K / 64) + g] = f2bf(s);
  MB[(long)n * (K / 64) + g] = f2bf(b);
  for (int w = 0; w < 8; w++) {
    uint word = 0;
    for (int j = 0; j < 8; j++) {
      const int q = clamp((int)rint(((float)row[w * 8 + j] - b) / s), 0, 15);
      word |= (uint)q << (4 * j);
    }
    MW[(long)n * (K / 8) + g * 8 + w] = word;
  }
}
