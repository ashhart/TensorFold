// The Qwen3.5 vision tower on Xe2 (transformers' Qwen3_5VisionModel in bf16), as kernels/cuda/qwen_vision.cu runs it:
// the GEMMs sum in fp32 on DPAS, the epilogues add the bias to the fp32 sums and round once.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define KWG(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
  uint u = as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (ushort)((u >> 16) | 0x40u); // NaN stays NaN
  u += 0x7fffu + ((u >> 16) & 1u);
  return (ushort)(u >> 16);
}
inline float rnd(float f) { return bf(to_bf(f)); }

// C [M, N] fp32 = A [M, K] bf16 x W [N, K]^T bf16 (a Linear's weight as stored). K is a multiple of 16, rows are
// 16-byte multiples. A group of 8 sub-groups holds a 128 x 64 tile: 4 sub-groups down M by 2 across N, 32 x 32 each.
// The 2-D block reads return zeros past M and N, and the writes stop there.
KWG(128) void qv_gemm(global const ushort *A, global const ushort *W, global float *C, int M, int N, int K) {
  const int sg = get_sub_group_id();
  const int m0 = get_group_id(0) * 128 + (sg % 4) * 32;
  const int n0 = get_group_id(1) * 64 + (sg / 4) * 32;
  float8 acc[4][2];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++) acc[i][j] = 0.0f;
  for (int k = 0; k < K; k += 16) {
    short8 a[4];
    uint8 b[2];
    for (int i = 0; i < 4; i++)
      intel_sub_group_2d_block_read_16b_8r16x1c((global void *)A, K * 2, M, K * 2, (int2)(k, m0 + 8 * i), (private ushort *)&a[i]);
    for (int j = 0; j < 2; j++) // lane l: W row n0 + 16 j + l, k .. k + 15 in pairs
      intel_sub_group_2d_block_read_transpose_32b_16r8x1c((global void *)W, K * 2, N, K * 2, (int2)(k / 2, n0 + 16 * j), (private uint *)&b[j]);
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 2; j++) acc[i][j] = intel_sub_group_bf16_bf16_matrix_mad_k16(a[i], as_int8(b[j]), acc[i][j]);
  }
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
      intel_sub_group_2d_block_write_32b_8r16x1c((global void *)C, N * 4, M, N * 4, (int2)(n0 + 16 * j, m0 + 8 * i), (private uint *)&acc[i][j]);
}

// A group a row: the sum of its 256 items' values.
inline float row_sum(float v, local float *part) {
  v = sub_group_reduce_add(v);
  if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (int i = 0; i < 16; i++) t += part[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

// The patch embedding plus the resampled position grid: x = bf16(acc + b) + bf16(sum_t pos[idx_t] w_t). A group a row.
KWG(256) void qv_embed(global const float *acc, global const ushort *bias, global const ushort *pos, global const int *idx,
                       global const float *wts, global ushort *out, int D) {
  const long row = get_group_id(0);
  for (int i = get_local_id(0); i < D; i += 256) {
    const float h = rnd(acc[row * D + i] + bf(bias[i]));
    float p = 0.0f;
    for (int t = 0; t < 4; t++) p += bf(pos[(long)idx[row * 4 + t] * D + i]) * wts[row * 4 + t];
    out[row * D + i] = to_bf(h + rnd(p));
  }
}

// LayerNorm of bf16 rows in fp32 (the mean, then the biased variance about it), bf16 out. A group a row.
KWG(256) void qv_layernorm(global const ushort *x, global const ushort *w, global const ushort *b, global ushort *y, int D,
                           float eps) {
  local float part[16];
  const long row = get_group_id(0);
  global const ushort *xr = x + row * D;
  float s = 0.0f;
  for (int i = get_local_id(0); i < D; i += 256) s += bf(xr[i]);
  const float mean = row_sum(s, part) / D;
  float v = 0.0f;
  for (int i = get_local_id(0); i < D; i += 256) {
    const float d = bf(xr[i]) - mean;
    v += d * d;
  }
  const float r = rsqrt(row_sum(v, part) / D + eps);
  for (int i = get_local_id(0); i < D; i += 256) y[row * D + i] = to_bf((bf(xr[i]) - mean) * r * bf(w[i]) + bf(b[i]));
}

// qkv = bf16(acc + b) split into q, k, v [heads][rows][hd]; q and k get the axial 2-D rotary in fp32 (h, then w).
// Groups (rows, heads), 128 items.
KWG(128) void qv_qkv_rope(global const float *acc, global const ushort *bias, global const int *hw, global const float *inv,
                          global ushort *q, global ushort *k, global ushort *v, int rows, int heads, int hd) {
  const long row = get_group_id(0);
  const int head = get_group_id(1);
  const int D = heads * hd, hh = hd / 2, quarter = hd / 4;
  const float ph = (float)hw[row * 2], pw = (float)hw[row * 2 + 1];
  global const float *a = acc + row * 3 * D;
  const long o = ((long)head * rows + row) * hd;
  for (int j = get_local_id(0); j < hd; j += 128) {
    const int c = head * hd + j;
    // rotate_half pairs j with j +- hd/2; the angle is h inv[jj] for the first quarter's frequencies, w inv after
    const int pc = head * hd + (j < hh ? j + hh : j - hh);
    const int jj = j % hh;
    const float ang = jj < quarter ? ph * inv[jj] : pw * inv[jj - quarter];
    const float cs = cos(ang), sn = sin(ang);
    const float sign = j < hh ? -1.0f : 1.0f;
    const float qx = rnd(a[c] + bf(bias[c])), qp = rnd(a[pc] + bf(bias[pc]));
    const float kx = rnd(a[D + c] + bf(bias[D + c])), kp = rnd(a[D + pc] + bf(bias[D + pc]));
    q[o + j] = to_bf(qx * cs + sign * qp * sn);
    k[o + j] = to_bf(kx * cs + sign * kp * sn);
    v[o + j] = to_bf(a[2 * D + c] + bf(bias[2 * D + c]));
  }
}

// Attention within one image's rows [s0, s0 + n) at a head width of HD (72): an item a query row, an online softmax in
// fp32 over key tiles of 64 rows in local memory. Groups (query tiles of 64, heads), 64 items.
#define HD 72
KWG(64) void qv_attention(global const ushort *q, global const ushort *k, global const ushort *v, global ushort *out,
                          int rows, int heads, int s0, int n, float scale) {
  local float ks[64][HD];
  local float vs[64][HD];
  const int head = get_group_id(1);
  const int qi = get_group_id(0) * 64 + get_local_id(0);
  const bool live = qi < n;
  const long base = (long)head * rows + s0;
  float qr[HD], acc[HD];
  float m = -INFINITY, l = 0.0f;
  for (int d = 0; d < HD; d++) {
    qr[d] = live ? bf(q[(base + qi) * HD + d]) * scale : 0.0f;
    acc[d] = 0.0f;
  }
  for (int t0 = 0; t0 < n; t0 += 64) {
    const int count = min(64, n - t0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = get_local_id(0); i < count * HD; i += 64) {
      const int r = i / HD, d = i % HD;
      ks[r][d] = bf(k[(base + t0 + r) * HD + d]);
      vs[r][d] = bf(v[(base + t0 + r) * HD + d]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int r = 0; r < count; r++) {
      float s = 0.0f;
      for (int d = 0; d < HD; d++) s += qr[d] * ks[r][d];
      const float mn = fmax(m, s);
      const float corr = exp(m - mn);
      const float p = exp(s - mn);
      l = l * corr + p;
      for (int d = 0; d < HD; d++) acc[d] = acc[d] * corr + p * vs[r][d];
      m = mn;
    }
  }
  if (!live) return;
  const float il = 1.0f / l;
  global ushort *op = out + (long)(s0 + qi) * heads * HD + head * HD;
  for (int d = 0; d < HD; d++) op[d] = to_bf(acc[d] * il);
}

// y = gelu(bf16(acc + b)), the tanh form (gelu_pytorch_tanh) or the erf form (nn.GELU), in fp32, bf16 out.
KWG(256) void qv_bias_gelu(global const float *acc, global const ushort *bias, global ushort *y, int D, int tanh_form) {
  const long row = get_group_id(0);
  for (int i = get_local_id(0); i < D; i += 256) {
    const float x = rnd(acc[row * D + i] + bf(bias[i]));
    const float g = tanh_form ? 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)))
                              : 0.5f * x * (1.0f + erf(x * 0.7071067811865476f));
    y[row * D + i] = to_bf(g);
  }
}

// x += bf16(acc + b), the residual add in bf16 (both operands bf16, the sum rounded).
KWG(256) void qv_bias_residual(global const float *acc, global const ushort *bias, global ushort *x, int D) {
  const long row = get_group_id(0);
  for (int i = get_local_id(0); i < D; i += 256) {
    const float y = rnd(acc[row * D + i] + bf(bias[i]));
    x[row * D + i] = to_bf(bf(x[row * D + i]) + y);
  }
}

// y = bf16(acc + b) as fp32: the merger's last Linear, rows the language model's residual stream takes as they are.
KWG(256) void qv_bias_round(global const float *acc, global const ushort *bias, global float *y, int D) {
  const long row = get_group_id(0);
  for (int i = get_local_id(0); i < D; i += 256) y[row * D + i] = rnd(acc[row * D + i] + bf(bias[i]));
}
