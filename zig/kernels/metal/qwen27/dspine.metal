// DSpine's code bridge and exact-sender greedy table (qwen27/dspine), rounding where Tencent's dspine_mlx rounds.
#include <metal_stdlib>
using namespace metal;

// Sum of v over a threadgroup of up to 1024 threads; every thread gets it.
static float group_sum(float v, threadgroup float* red, uint t, uint lane, uint sg, uint threads) {
  v = simd_sum(v);
  if (sg == 0) red[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) red[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total = simd_sum(red[lane]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

static float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

struct Rows { uint rows, width, block; float eps, scale; };

// Y = bf16(A + B), the layer's residual adds.
kernel void tf_dspine_add(device const bfloat* A [[buffer(0)]], device const bfloat* B [[buffer(1)]], constant Rows& p [[buffer(2)]],
    device bfloat* Y [[buffer(3)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.width || i.y >= p.rows) return;
  const size_t at = size_t(i.y) * p.width + i.x;
  Y[at] = bfloat(float(A[at]) + float(B[at]));
}

// Per row (one threadgroup): RMS = sqrt(max(mean RAW^2, tiny)); XN = bf16(RAW / (RMS + eps)).
kernel void tf_dspine_read(device const bfloat* RAW [[buffer(0)]], constant Rows& p [[buffer(1)]], device bfloat* XN [[buffer(2)]],
    device float* RMS [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float red[32];
  const size_t base = size_t(row) * p.width;
  float ss = 0.0f;
  for (uint i = t; i < p.width; i += threads) { const float x = float(RAW[base + i]); ss = fma(x, x, ss); }
  const float rms = sqrt(max(group_sum(ss, red, t, lane, sg, threads) / float(p.width), 1.17549435e-38f));
  if (t == 0) RMS[row] = rms;
  for (uint i = t; i < p.width; i += threads) XN[base + i] = bfloat(float(RAW[base + i]) / (rms + p.eps));
}

// Codes `width` (C) wide: a block's anchor row bf16(scale * CB[ids]), the rest bf16(scale * P / (|P| + eps)).
kernel void tf_dspine_code(device const bfloat* P [[buffer(0)]], device const bfloat* CB [[buffer(1)]], device const uint* IDS [[buffer(2)]],
    constant Rows& p [[buffer(3)]], device bfloat* OUT [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float red[32];
  const size_t base = size_t(row) * p.width;
  if (row % p.block == 0) {
    const size_t src = size_t(IDS[row]) * p.width;
    for (uint i = t; i < p.width; i += threads) OUT[base + i] = bfloat(p.scale * float(CB[src + i]));
    return;
  }
  float ss = 0.0f;
  for (uint i = t; i < p.width; i += threads) { const float x = float(P[base + i]); ss = fma(x, x, ss); }
  const float n = sqrt(group_sum(ss, red, t, lane, sg, threads)) + p.eps;
  for (uint i = t; i < p.width; i += threads) OUT[base + i] = bfloat(p.scale * float(P[base + i]) / n);
}

// Row t's message: 0 at an anchor, else bf16(sigmoid(Q[t] + S[t-1] + GB) * V[t-1]); PACKED rows are [Q|S|V].
kernel void tf_dspine_message(device const bfloat* PACKED [[buffer(0)]], device const bfloat* GB [[buffer(1)]], constant Rows& p [[buffer(2)]],
    device bfloat* OUT [[buffer(3)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.width || i.y >= p.rows) return;
  const uint r = p.width, t = i.y;
  float m = 0.0f;
  if (t % p.block != 0) {
    const size_t q = size_t(t) * 3 * r, prev = size_t(t - 1) * 3 * r;
    m = sigmoid(float(PACKED[q + i.x]) + float(PACKED[prev + r + i.x]) + float(GB[i.x])) * float(PACKED[prev + 2 * r + i.x]);
  }
  OUT[size_t(t) * r + i.x] = bfloat(m);
}

// RES = RAW at an anchor, else bf16(RAW + RMS * DELTA); XN = bf16(bf16(RES * rsqrt(mean RES^2 + eps)) * W).
kernel void tf_dspine_write(device const bfloat* RAW [[buffer(0)]], device const bfloat* DELTA [[buffer(1)]], device const float* RMS [[buffer(2)]],
    device const bfloat* W [[buffer(3)]], constant Rows& p [[buffer(4)]], device bfloat* RES [[buffer(5)]], device bfloat* XN [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float red[32];
  const bool anchor = row % p.block == 0;
  const float rms = RMS[row];
  const size_t base = size_t(row) * p.width;
  float ss = 0.0f;
  for (uint i = t; i < p.width; i += threads) {
    const float raw = float(RAW[base + i]);
    const float out = anchor ? raw : float(bfloat(raw + rms * float(DELTA[base + i])));
    RES[base + i] = bfloat(out);
    ss = fma(out, out, ss);
  }
  const float inv = precise::rsqrt(group_sum(ss, red, t, lane, sg, threads) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += threads) XN[base + i] = bfloat(float(bfloat(float(RES[base + i]) * inv)) * float(W[i]));
}

// Each head row's K senders' bf16(scale * CB[sid]): the anchor for row 0, else the row before's top-k (bf16_topk Rows).
struct Senders { uint rows, k, code, row_words, anchor; float scale; };
kernel void tf_dspine_senders(device const uint* TOPK [[buffer(0)]], device const bfloat* CB [[buffer(1)]], constant Senders& p [[buffer(2)]],
    device bfloat* OUT [[buffer(3)]], device uint* SIDS [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint2 tpg [[threads_per_threadgroup]]) {
  const uint row = g.y, j = g.x;
  const uint sid = row == 0 ? p.anchor : TOPK[size_t(row - 1) * p.row_words + 2 + 2 * j];
  if (t == 0) SIDS[row * p.k + j] = sid;
  const size_t src = size_t(sid) * p.code, dst = size_t(row * p.k + j) * p.code;
  for (uint i = t; i < p.code; i += tpg.x) OUT[dst + i] = bfloat(p.scale * float(CB[src + i]));
}

// BASE[i][j] = sum_h RAW[first + i][h] * NORM[h] * bf16(dequant HEAD[c_j][h]) in f32, HEAD in MLX's affine-4 layout.
struct Base { uint rows, k, hidden, row_words, raw_first; };
kernel void tf_dspine_base(device const bfloat* RAW [[buffer(0)]], device const bfloat* NORM [[buffer(1)]], device const uint* HW [[buffer(2)]],
    device const bfloat* HS [[buffer(3)]], device const bfloat* HB [[buffer(4)]], device const uint* TOPK [[buffer(5)]], constant Base& p [[buffer(6)]],
    device float* OUT [[buffer(7)]], uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint2 tpg [[threads_per_threadgroup]]) {
  threadgroup float red[32];
  const uint i = g.y, j = g.x;
  const uint id = TOPK[size_t(i) * p.row_words + 2 + 2 * j];
  const size_t raw = size_t(p.raw_first + i) * p.hidden, words = size_t(id) * (p.hidden / 8), groups = size_t(id) * (p.hidden / 64);
  float d = 0.0f;
  for (uint w = t; w < p.hidden / 8; w += tpg.x) {
    const uint bits = HW[words + w];
    const float scale = float(HS[groups + w / 8]), bias = float(HB[groups + w / 8]);
    for (uint x = 0; x < 8; x++) {
      const uint h = 8 * w + x;
      const float head = float(bfloat(scale * float((bits >> (4 * x)) & 15u) + bias));
      d = fma(float(RAW[raw + h]) * float(NORM[h]), head, d);
    }
  }
  d = group_sum(d, red, t, lane, sg, tpg.x);
  if (t == 0) OUT[i * p.k + j] = d;
}

// Per (row i, sender s): m = sigmoid(Q_i + G_s + b) * V_s (f32); OUT[i][s][j] = BASE[i][j] + RMS_i * m . TG[c_j].
struct Table { uint rows, k, rank, row_words, rms_first; };
kernel void tf_dspine_table(device const bfloat* PACKED [[buffer(0)]], device const bfloat* TSV [[buffer(1)]], device const bfloat* GB [[buffer(2)]],
    device const bfloat* TG [[buffer(3)]], device const uint* TOPK [[buffer(4)]], device const float* BASE [[buffer(5)]],
    device const float* RMS [[buffer(6)]], constant Table& p [[buffer(7)]], device float* OUT [[buffer(8)]],
    uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint2 tpg [[threads_per_threadgroup]]) {
  threadgroup float m[1024];
  const uint i = g.y, s = g.x, r = p.rank;
  const size_t q = size_t(p.rms_first + i) * 3 * r, sv = size_t(i * p.k + s) * 2 * r;
  for (uint x = t; x < r; x += tpg.x) m[x] = sigmoid(float(PACKED[q + x]) + float(TSV[sv + x]) + float(GB[x])) * float(TSV[sv + r + x]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float rms = RMS[p.rms_first + i];
  device const uint* row = TOPK + size_t(i) * p.row_words + 2;
  for (uint j = sg; j < p.k; j += tpg.x / 32) {
    const uint id = row[2 * j];
    float d = 0.0f;
    for (uint x = lane; x < r; x += 32) d = fma(m[x], float(TG[size_t(id) * r + x]), d);
    d = simd_sum(d);
    if (lane == 0) OUT[(size_t(i) * p.k + s) * p.k + j] = BASE[i * p.k + j] + rms * d;
  }
}
