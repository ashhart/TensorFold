// GLM-5.3-Flash decode glue: MLX 0.32.3's arithmetic where the Python family calls MLX ops, our own elsewhere.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

struct GlmRows {
  int rows, width, x_stride, y_stride;
  float eps;
};

// bf16 -> fp32, as x.astype(mx.float32) (the router's input).
kernel void glm_cast_f32(const device bfloat* x [[buffer(0)]], device float* y [[buffer(1)]],
                         constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] = float(x[i]);
}

// MLX's rms_single_row on strided rows: thread t squares 4t..4t+3, simd_sum, a 32-slot simd_sum, precise rsqrt.
kernel void glm_rms(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                    device bfloat* y [[buffer(2)]], constant GlmRows& a [[buffer(3)]],
                    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sg_sums[32];
  threadgroup float inv_rms[1];
  const uint width = uint(a.width), lo = 4 * t;
  const device bfloat* src = x + size_t(row) * a.x_stride + lo;
  device bfloat* dst = y + size_t(row) * a.y_stride + lo;
  const device bfloat* gain = w + lo;
  const bool whole = lo + 4 <= width;
  float v[4];
  float sq = 0.0f;
  if (whole) {
    for (int j = 0; j < 4; ++j) {
      v[j] = src[j];
      sq += v[j] * v[j];
    }
  } else {
    for (int j = 0; j < 4; ++j) {
      v[j] = lo + j < width ? float(src[j]) : 0.0f;
      sq += v[j] * v[j];
    }
  }
  sq = simd_sum(sq);
  if (sg == 0) sg_sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sg_sums[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float all = simd_sum(sg_sums[lane]);
    if (lane == 0) inv_rms[0] = metal::precise::rsqrt(all / float(width) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (whole) {
    for (int j = 0; j < 4; ++j) dst[j] = gain[j] * static_cast<bfloat>(v[j] * inv_rms[0]);
  } else {
    for (int j = 0; j < 4; ++j)
      if (lo + j < width) dst[j] = gain[j] * static_cast<bfloat>(v[j] * inv_rms[0]);
  }
}

// MLX's layer_norm_single_row (8 values a thread, 32 threads a row): the indexer's key norm with weight and bias.
inline float glm_tg_sum(float x, threadgroup float* xs, uint lane, uint sg) {
  x = simd_sum(x);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) xs[sg] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  x = xs[lane];
  return simd_sum(x);
}

kernel void glm_layer_norm(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                           const device bfloat* b [[buffer(2)]], device bfloat* y [[buffer(3)]],
                           constant GlmRows& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 8;
  float thread_x[N_READS] = {0};
  threadgroup float local_buffer[32];
  if (sg == 0) local_buffer[lane] = 0;
  const int axis = a.width;
  const device bfloat* xr = x + size_t(row) * a.x_stride + lid * N_READS;
  device bfloat* out = y + size_t(row) * a.y_stride + lid * N_READS;
  const device bfloat* wr = w + lid * N_READS;
  const device bfloat* br = b + lid * N_READS;
  const bool safe = int(lid) * N_READS + N_READS <= axis;
  const int n = axis - int(lid) * N_READS;
  if (safe) {
    for (int i = 0; i < N_READS; i++) thread_x[i] = xr[i];
  } else {
    for (int i = 0; i < n; i++) thread_x[i] = xr[i];
  }
  float mean = 0;
  for (int i = 0; i < N_READS; i++) mean += thread_x[i];
  mean = glm_tg_sum(mean, local_buffer, lane, sg);
  mean /= axis;
  float normalizer = 0;
  if (!safe) {
    for (int i = max(n, 0); i < N_READS; i++) thread_x[i] = mean;
  }
  for (int i = 0; i < N_READS; i++) {
    thread_x[i] -= mean;
    normalizer += thread_x[i] * thread_x[i];
  }
  normalizer = glm_tg_sum(normalizer, local_buffer, lane, sg);
  normalizer = metal::precise::rsqrt(normalizer / axis + a.eps);
  if (safe) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] *= normalizer;
      out[i] = wr[i] * static_cast<bfloat>(thread_x[i]) + br[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      thread_x[i] *= normalizer;
      out[i] = wr[i] * static_cast<bfloat>(thread_x[i]) + br[i];
    }
  }
}

// The absorb q_nope [b, 256] @ W_k[h] [256, 512] (h = b % HEADS, kv_b's key rows) in affine_qvm's 4-bit sum order.
kernel void glm_absorb(const device uint32_t* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]],
                       const device bfloat* biases [[buffer(2)]], const device bfloat* x [[buffer(3)]],
                       device bfloat* y [[buffer(4)]], constant uint& q_stride [[buffer(5)]],
                       uint3 tid [[threadgroup_position_in_grid]], uint simd_gid [[simdgroup_index_in_threadgroup]],
                       uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int HEADS = 64, K = 256, N = 512, ROWS_PER_HEAD = 512, NW = N / 8, NG = N / 64;
  const int b = int(tid.z), h = b % HEADS;
  const int out_col = 32 * (int(tid.y) * 2 + int(simd_gid));
  const device uint32_t* ws = w + size_t(h * ROWS_PER_HEAD) * NW + out_col / 8 + simd_lid * NW;
  const device bfloat* sc = scales + size_t(h * ROWS_PER_HEAD) * NG + out_col / 64 + simd_lid * NG;
  const device bfloat* bi = biases + size_t(h * ROWS_PER_HEAD) * NG + out_col / 64 + simd_lid * NG;
  const device bfloat* xp = x + size_t(b / HEADS) * q_stride + size_t(h) * K + simd_lid;
  device bfloat* yp = y + size_t(b) * N + out_col;
  float result[32] = {0};
  for (int i = 0; i < K; i += 32) {
    const float xl = *xp;
    const float scale = *sc;
    const float bias = *bi;
    uint32_t wl[4];
    for (int j = 0; j < 4; j++) wl[j] = ws[j];
    const thread uint8_t* wb = (const thread uint8_t*)wl;
    const float s[2] = {scale, scale / 16.0f};
    for (int j = 0; j < 16; j++) {
      result[2 * j] += xl * (s[0] * (wb[j] & 0x0f) + bias);
      result[2 * j + 1] += xl * (s[1] * (wb[j] & 0xf0) + bias);
    }
    xp += 32;
    sc += 32 * NG;
    bi += 32 * NG;
    ws += 32 * NW;
  }
  for (int k = 0; k < 32; k++) result[k] = simd_sum(result[k]);
  if (simd_lid == 0)
    for (int k = 0; k < 32; k++) yp[k] = static_cast<bfloat>(result[k]);
}

// The unabsorb latent [b, 512] -> values [b, 256] (kv_b's value rows) in qmv_fast's 4-bit sum order, batched by head.
inline float glm_load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
inline float glm_qdot16(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}

kernel void glm_unabsorb(const device uint32_t* W [[buffer(0)]], const device bfloat* S [[buffer(1)]],
                         const device bfloat* B [[buffer(2)]], const device bfloat* X [[buffer(3)]],
                         device bfloat* OUT [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  constexpr int HEADS = 64, K = 512, N = 256, RPS = 4, KB = K / 2, KG = K / 64;
  const int b = int(tg.z), h = b % HEADS;
  const int row0 = h * 512 + 256 + int(tg.y) * RPS;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 8;
  const device bfloat* sc = S + size_t(row0) * KG + lane / 4;
  const device bfloat* bi = B + size_t(row0) * KG + lane / 4;
  const device bfloat* x = X + size_t(b) * K + lane * 16;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    const float sum = glm_load16(x, xt);
    for (int j = 0; j < RPS; j++) acc[j] += glm_qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 256; sc += 8; bi += 8; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[size_t(b) * N + int(tg.y) * RPS + j] = bfloat(v);
  }
}

// y = bf16(x * s) for a power-of-two s (exact): the scaled queries of MLX's unfused attention, the indexer weights.
kernel void glm_scale(const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]],
                      constant GlmRows& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (int(p.x) >= a.width || int(p.y) >= a.rows) return;
  y[size_t(p.y) * a.y_stride + p.x] = bfloat(float(x[size_t(p.y) * a.x_stride + p.x]) * a.eps);
}

// Pooled indexer keys of blocks [first, first + count): caches.pool_blocks' fp32 ops, each rounded alone.
#pragma clang fp contract(off)
kernel void glm_pool(const device bfloat* ik [[buffer(0)]], const device bfloat* ig [[buffer(1)]],
                     const device bfloat* ape [[buffer(2)]], device bfloat* pool [[buffer(3)]],
                     constant uint2& span [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
  constexpr int DI = 128, KP = 4;
  if (p.y >= span.y) return;
  const uint blk = span.x + p.y, d = p.x;
  float logit[KP], k[KP];
  for (int j = 0; j < KP; j++) {
    const size_t at = size_t(blk * KP + j) * DI + d;
    logit[j] = float(ig[at]) + float(ape[j * DI + d]);
    k[j] = float(ik[at]);
  }
  float top = logit[0];
  for (int j = 1; j < KP; j++) top = metal::max(top, logit[j]);
  float e[KP];
  for (int j = 0; j < KP; j++) e[j] = metal::precise::exp(logit[j] - top);
  float total = e[0];
  for (int j = 1; j < KP; j++) total = total + e[j];
  float out = (e[0] / total) * k[0];
  for (int j = 1; j < KP; j++) out = out + (e[j] / total) * k[j];
  pool[size_t(blk) * DI + d] = bfloat(out);
}

// The streams' fp32 mean (summed in stream order, * 1/4) as bf16: model.final_norm before its RMSNorm.
kernel void glm_stream_mean(const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]],
                            constant uint2& dims [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  const uint D = dims.x;
  if (p.x >= D || p.y >= dims.y) return;
  const device bfloat* r = x + size_t(p.y) * 4 * D + p.x;
  float raw = float(r[0]);
  for (int s = 1; s < 4; s++) raw = raw + float(r[s * D]);
  y[size_t(p.y) * D + p.x] = bfloat(raw * 0.25f);
}

// out = bf16(a + b): the MTP block's residual adds.
kernel void glm_add(const device bfloat* a [[buffer(0)]], const device bfloat* b [[buffer(1)]],
                    device bfloat* out [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < n) out[i] = bfloat(float(a[i]) + float(b[i]));
}
#pragma clang fp contract(on)

// MLX's bfloat math overloads (bf16_math.h) and fused.py's sigmoid_fast: nn.silu's compiled x * sigmoid(x).
#ifndef METAL_FUNC
#define METAL_FUNC inline __attribute__((__always_inline__))
#endif
namespace metal {
METAL_FUNC bfloat16_t abs(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); }
METAL_FUNC bfloat16_t exp(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); }
}
template <typename U>
inline U sigmoid_fast(U x) {
  U e = static_cast<U>(metal::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

// The dense MLP's SwiGLU over [gate | up] rows, clipped at `limit`: the MoE kernel's (and the eager ops') bits.
kernel void glm_swiglu(const device bfloat* gu [[buffer(0)]], device bfloat* act [[buffer(1)]],
                       constant GlmRows& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (int(p.x) >= a.width || int(p.y) >= a.rows) return;
  const device bfloat* row = gu + size_t(p.y) * a.x_stride;
  const float lim = float(bfloat(a.eps));
  const bfloat gt = bfloat(metal::min(float(row[p.x]), lim));
  const bfloat up = bfloat(metal::min(metal::max(float(row[a.width + p.x]), -lim), lim));
  const bfloat sl = gt * sigmoid_fast(gt);
  act[size_t(p.y) * a.y_stride + p.x] = sl * up;
}

// Each row's argmax over bf16 logits (the larger value, the lower index on a tie), 1024 threads a row.
kernel void glm_argmax(const device bfloat* logits [[buffer(0)]], device uint* out [[buffer(1)]],
                       constant uint& vocab [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                       uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
  const device bfloat* x = logits + size_t(row) * vocab;
  float best = -INFINITY;
  uint at = 0xffffffffu;
  for (uint i = t; i < vocab; i += 1024) {
    const float v = float(x[i]);
    if (v > best) { best = v; at = i; }
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o);
    const uint oa = simd_shuffle_xor(at, o);
    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
  }
  threadgroup float vb[32];
  threadgroup uint va[32];
  if (lane == 0) { vb[sg] = best; va[sg] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  best = vb[lane];
  at = va[lane];
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o);
    const uint oa = simd_shuffle_xor(at, o);
    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
  }
  if (lane == 0) out[row] = at;
}

// Row r's (position p0 + r) fp32 block scores over its whole blocks: sum over heads in order of w_h relu(q_h . pool_b).
struct GlmScoreArgs {
  uint p0, q_stride, w_stride, s_stride;
};

kernel void glm_index_scores(const device bfloat* iq [[buffer(0)]], const device bfloat* iw [[buffer(1)]],
                             const device bfloat* pool [[buffer(2)]], device float* scores [[buffer(3)]],
                             constant GlmScoreArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int HI = 32, DI = 128, SGS = 8;
  const uint row = tg.y;
  const uint blk = tg.x * SGS + sg;
  const uint blocks = (a.p0 + row + 1) / 4;
  if (tg.x * SGS >= blocks) return;
  threadgroup float qs[HI * DI];
  threadgroup float ws[HI];
  const device bfloat* q = iq + size_t(row) * a.q_stride;
  for (uint i = sg * 32 + lane; i < HI * DI; i += SGS * 32) qs[i] = float(q[i]);
  if (sg == 0) ws[lane] = float(iw[size_t(row) * a.w_stride + lane]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (blk >= blocks) return;
  float k[4];
  for (int j = 0; j < 4; j++) k[j] = float(pool[size_t(blk) * DI + lane * 4 + j]);
  float total = 0.0f;
  for (int h = 0; h < HI; h++) {
    float dot = 0.0f;
    for (int j = 0; j < 4; j++) dot = fma(qs[h * DI + lane * 4 + j], k[j], dot);
    dot = simd_sum(dot);
    total = fma(ws[h], metal::max(dot, 0.0f), total);
  }
  if (lane == 0) scores[size_t(row) * a.s_stride + blk] = total;
}

// Row r's key list: its TOP best blocks (ties: lower block) as keys in block order, its tail, -1 to `width` (a radix select).
struct GlmSelectArgs {
  uint p0, top, width, s_stride, i_stride;
};

// Exclusive prefix sum over a 1024-thread threadgroup (32 simdgroups), `part` 32 slots of scratch.
inline uint glm_scan(uint v, threadgroup uint* part, uint t) {
  const uint lane = t % 32, sg = t / 32;
  const uint inside = simd_prefix_exclusive_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 31) part[sg] = inside + v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint base = 0;
  for (uint i = 0; i < sg; i++) base += part[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return base + inside;
}

inline uint glm_order_key(float v) {
  const uint u = as_type<uint>(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

kernel void glm_index_select(const device float* scores [[buffer(0)]], device int* indices [[buffer(1)]],
                             constant GlmSelectArgs& g [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                             uint t [[thread_position_in_threadgroup]]) {
  constexpr uint NT = 1024;
  struct { uint blocks, top, position, width, s_stride, i_stride; } a = {
      (g.p0 + row + 1) / 4, g.top, g.p0 + row, g.width, g.s_stride, g.i_stride};
  const device float* s = scores + size_t(row) * a.s_stride;
  device int* out = indices + size_t(row) * a.i_stride;
  threadgroup atomic_uint hist[256];
  threadgroup uint shared_prefix[1], shared_need[1];
  threadgroup uint counts[32];
  uint prefix = 0, need = a.top;
  if (a.top < a.blocks) {
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (uint i = t; i < 256; i += NT) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
      for (uint b = t; b < a.blocks; b += NT) {
        const uint key = glm_order_key(s[b]);
        if ((key & hi_mask) == (prefix & hi_mask))
          atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (t == 0) {
        uint acc = 0, digit = 0;
        for (int d = 255; d >= 0; d--) {
          const uint c = atomic_load_explicit(&hist[d], memory_order_relaxed);
          if (acc + c >= need) { digit = uint(d); break; }
          acc += c;
        }
        shared_prefix[0] = prefix | (digit << shift);
        shared_need[0] = need - acc;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      prefix = shared_prefix[0];
      need = shared_need[0];
    }
  }
  // blocks above the threshold key, plus the first `need` equal to it in block order: each thread a run of blocks
  const bool all_blocks = a.top >= a.blocks;
  const uint per = (a.blocks + NT - 1) / NT;
  const uint lo = min(t * per, a.blocks), hi = min(lo + per, a.blocks);
  uint above = 0, equal = 0;
  if (!all_blocks)
    for (uint b = lo; b < hi; b++) {
      const uint key = glm_order_key(s[b]);
      above += key > prefix ? 1u : 0u;
      equal += key == prefix ? 1u : 0u;
    }
  const uint equal_before = glm_scan(equal, counts, t);
  uint take = 0;
  if (all_blocks) take = hi - lo;
  else take = above + (equal_before >= need ? 0u : min(equal, need - equal_before));
  const uint before = glm_scan(take, counts, t);
  uint eq_seen = equal_before;
  uint at = before;
  for (uint b = lo; b < hi; b++) {
    bool chosen = all_blocks;
    if (!all_blocks) {
      const uint key = glm_order_key(s[b]);
      if (key > prefix) chosen = true;
      else if (key == prefix) { chosen = eq_seen < need; eq_seen++; }
    }
    if (chosen) {
      for (uint j = 0; j < 4; j++) out[4 * at + j] = int(4 * b + j);
      at++;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  const uint chosen_blocks = min(a.top, a.blocks);
  const uint tail = (a.position + 1) % 4;
  for (uint i = 4 * chosen_blocks + t; i < a.width; i += NT) {
    const uint j = i - 4 * chosen_blocks;
    out[i] = j < tail ? int(a.position + 1 - tail + j) : -1;
  }
}

// y = exp(x) with MLX's Exp (precise): the KDA decay rates A = exp(A_log), once at load.
kernel void glm_exp_f32(const device float* x [[buffer(0)]], device float* y [[buffer(1)]],
                        constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] = metal::precise::exp(x[i]);
}

// The embedding rows copied into the four hyper-connection streams: [rows, D] -> [rows, 4, D].
kernel void glm_streams(const device bfloat* h [[buffer(0)]], device bfloat* x [[buffer(1)]],
                        constant uint2& dims [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  const uint D = dims.x;
  if (p.x >= D || p.y >= dims.y) return;
  const bfloat v = h[size_t(p.y) * D + p.x];
  for (int s = 0; s < 4; s++) x[(size_t(p.y) * 4 + s) * D + p.x] = v;
}

// dst[i] = src[i] for n u32 values (a pick becoming the next window's token or the MTP head's input).
kernel void glm_copy_u32(const device uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                         constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}

// The prompt's routing: each row's top experts and weights in the decode route's arithmetic, one simdgroup a row.
template <typename U>
inline U glm_sigmoid_precise(U x) {
  U e = static_cast<U>(metal::precise::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

kernel void glm_route_rows(const device float* LOGITS [[buffer(0)]], const device float* BIAS [[buffer(1)]],
                           constant float& SCALE [[buffer(2)]], device uint* PICK [[buffer(3)]],
                           device float* WTS [[buffer(4)]], constant uint& rows [[buffer(5)]],
                           uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NE = 288, TOPK = 8, PER = (NE + 31) / 32;
  const int r = int(gid / 32);
  if (r >= int(rows)) return;
  float c[PER], sc[PER];
  for (int j = 0; j < PER; j++) {
    const int id = j * 32 + int(lane);
    if (id < NE) {
      sc[j] = glm_sigmoid_precise(LOGITS[r * NE + id]);
      c[j] = sc[j] + BIAS[id];
    } else {
      sc[j] = 0.0f; c[j] = -INFINITY;
    }
  }
  float w[TOPK];
  for (int k = 0; k < TOPK; k++) {
    float best = -INFINITY, bsc = 0.0f;
    int bid = NE;
    for (int j = 0; j < PER; j++) {
      const int id = j * 32 + int(lane);
      if (id < NE && (c[j] > best || (c[j] == best && id < bid))) { best = c[j]; bid = id; bsc = sc[j]; }
    }
    for (int off = 16; off > 0; off /= 2) {
      const float ob = simd_shuffle_xor(best, off);
      const int oi = simd_shuffle_xor(bid, off);
      const float os = simd_shuffle_xor(bsc, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; bsc = os; }
    }
    w[k] = bsc;
    if (int(lane) == bid % 32) c[bid / 32] = -INFINITY;
    if (lane == 0) PICK[r * TOPK + k] = uint(bid);
  }
  if (lane == 0) {
    float total = w[0];
    for (int k = 1; k < TOPK; k++) total = total + w[k];
    for (int k = 0; k < TOPK; k++) WTS[r * TOPK + k] = (w[k] / total) * SCALE;
  }
}

// The routed experts' activation for sorted pairs: the MoE kernel's SwiGLU on separate gate and up rows.
kernel void glm_act2(const device bfloat* G [[buffer(0)]], const device bfloat* U [[buffer(1)]],
                     device bfloat* ACT [[buffer(2)]], constant float& limit [[buffer(3)]],
                     constant uint& count [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  const float lim = float(bfloat(limit));
  const bfloat gt = bfloat(metal::min(float(G[i]), lim));
  const bfloat up = bfloat(metal::min(metal::max(float(U[i]), -lim), lim));
  const bfloat sl = gt * sigmoid_fast(gt);
  ACT[i] = sl * up;
}

// Key lists of rows that read every key (positions p0 + r with at most `dense` keys): 0 .. position, then -1.
kernel void glm_dense_indices(device int* indices [[buffer(0)]], constant uint4& a [[buffer(1)]],
                              uint2 p [[thread_position_in_grid]]) {
  const uint width = a.x, p0 = a.y, rows = a.z;
  if (p.x >= width || p.y >= rows) return;
  const uint position = p0 + p.y;
  indices[size_t(p.y) * width + p.x] = p.x <= position ? int(p.x) : -1;
}
