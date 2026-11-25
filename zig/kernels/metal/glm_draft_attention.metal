// GLM's DFlash2 drafter attention: a block's rows over the rolling K/V ring and their own keys, a key a lane.
#include <metal_stdlib>
using namespace metal;

// TF_QH, TF_KVH and TF_D come from the host.
#define GLM_G (TF_QH / TF_KVH)
#define GLM_CHUNK 128
#define GLM_BLOCK 32
#define GLM_PART (TF_D + 4)
// a lane's slice of a head's output: TF_D / 32 consecutive dims (128: 4, 64: 2)
#if TF_D == 128
typedef float4 glm_vec;
typedef bfloat4 glm_bvec;
#else
typedef float2 glm_vec;
typedef bfloat2 glm_bvec;
#endif

// Rows: streams' blocks; stream s's keys: its context at ring slots (at + key) % window, then its own KO/VO rows.
struct glm_draft_args { uint rows, block, streams, mask_stride, scale_bits, window, chunks, pad; };
struct glm_draft_stream { uint keys, context, at, pad; };
#define GLM_STREAMS 8

inline const device bfloat* glm_key(const device bfloat* ring, const device bfloat* own, uint key, uint kh, uint s,
                                    constant glm_draft_stream& st, constant glm_draft_args& a) {
  const size_t row = key < st.context ? (size_t)((st.at + key) % a.window) : (size_t)(s * a.block + key - st.context);
  return (key < st.context ? ring : own) + (row * TF_KVH + kh) * TF_D;
}

// Stream s's ring among the eight bound (one buffer argument each).
#define GLM_RING(R, s) ((s) == 0 ? R##0 : (s) == 1 ? R##1 : (s) == 2 ? R##2 : (s) == 3 ? R##3 : (s) == 4 ? R##4 : (s) == 5 ? R##5 : (s) == 6 ? R##6 : R##7)
#define GLM_RINGS(R, at) const device bfloat* R##0 [[buffer(at)]], const device bfloat* R##1 [[buffer(at + 1)]], \
    const device bfloat* R##2 [[buffer(at + 2)]], const device bfloat* R##3 [[buffer(at + 3)]], \
    const device bfloat* R##4 [[buffer(at + 4)]], const device bfloat* R##5 [[buffer(at + 5)]], \
    const device bfloat* R##6 [[buffer(at + 6)]], const device bfloat* R##7 [[buffer(at + 7)]]

// The NaN-keeping maximum: a NaN score or maximum stays NaN, so no chunk reads as empty.
inline float glm_max(float x, float y) {
  return isnan(x) || isnan(y) ? NAN : max(x, y);
}

// A simdgroup: a row's heads of a KV head over a chunk of its stream's keys; partials [row][kv head][chunk][head].
[[kernel, max_total_threads_per_threadgroup(32)]] void glm_draft_attend(
    const device bfloat* Q [[buffer(0)]], const device bfloat* KO [[buffer(1)]], const device bfloat* VO [[buffer(2)]],
    const device uchar* mask [[buffer(3)]], constant glm_draft_args& a [[buffer(4)]],
    constant glm_draft_stream* streams [[buffer(5)]], device float* part [[buffer(6)]], GLM_RINGS(K, 8), GLM_RINGS(V, 16),
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const uint kh = tg.x, row = tg.y, chunk = tg.z;
  if (row >= a.rows) return;
  const uint s = row / a.block;
  constant glm_draft_stream& st = streams[s];
  if (chunk * GLM_CHUNK >= st.keys) return;
  const device bfloat* K = GLM_RING(K, s);
  const device bfloat* V = GLM_RING(V, s);
  const float scale = as_type<float>(a.scale_bits);
  const device bfloat4* q = (const device bfloat4*)(Q + ((size_t)row * TF_QH + kh * GLM_G) * TF_D);
  glm_vec out[GLM_G];
  float maximum[GLM_G], total[GLM_G];
  for (uint g = 0; g < GLM_G; ++g) {
    maximum[g] = -INFINITY;
    total[g] = 0.0f;
    out[g] = glm_vec(0.0f);
  }
  const uint first = chunk * GLM_CHUNK, last = min(st.keys, first + GLM_CHUNK);
  for (uint base = first; base < last; base += GLM_BLOCK) {
    const uint key = base + lane;
    const bool live = key < last && mask[(size_t)row * a.mask_stride + key] != 0;
    float dot[GLM_G];
    for (uint g = 0; g < GLM_G; ++g) dot[g] = 0.0f;
    if (key < last) {
      const device bfloat4* k = (const device bfloat4*)glm_key(K, KO, key, kh, s, st, a);
      for (uint d = 0; d < TF_D / 4; ++d) {
        const float4 kv = float4(k[d]);
        for (uint g = 0; g < GLM_G; ++g) {
          const float4 qv = float4(q[g * (TF_D / 4) + d]);
          dot[g] = fma(qv.w, kv.w, fma(qv.z, kv.z, fma(qv.y, kv.y, fma(qv.x, kv.x, dot[g]))));
        }
      }
    }
    float p[GLM_G];
    for (uint g = 0; g < GLM_G; ++g) {
      const float score = live ? dot[g] * scale : -INFINITY;
      // a NaN score keeps the maximum NaN, so the merge cannot skip its chunk as empty
      const bool nan = simd_any(isnan(score)) || isnan(maximum[g]);
      const float next = nan ? NAN : max(maximum[g], simd_max(score));
      const float old_weight = maximum[g] == -INFINITY ? 0.0f : precise::exp(maximum[g] - next);
      p[g] = live ? precise::exp(score - next) : 0.0f;
      total[g] = fma(total[g], old_weight, simd_sum(p[g]));
      out[g] *= old_weight;
      maximum[g] = next;
    }
    // a masked key's value is selected away, never multiplied, so whatever it holds never enters the sums
    const uint count = min(uint(GLM_BLOCK), last - base);
    // lane j's key address offset (elements past its buffer's start) and buffer, passed to every lane
    const uint at = key < last ? uint(glm_key(V, VO, key, kh, s, st, a) - (key < st.context ? V : VO)) : 0u;
    for (uint j = 0; j < count; ++j) {
      const bool seen = simd_shuffle(uint(live), ushort(j)) != 0;
      const device bfloat* row_j = (base + j < st.context ? V : VO) + simd_shuffle(at, ushort(j));
      const glm_vec v = glm_vec(((const device glm_bvec*)row_j)[lane]);
      const glm_vec x = seen ? v : glm_vec(0.0f);
      for (uint g = 0; g < GLM_G; ++g) out[g] = fma(glm_vec(simd_shuffle(p[g], ushort(j))), x, out[g]);
    }
  }
  for (uint g = 0; g < GLM_G; ++g) {
    device float* o = part + ((((size_t)row * TF_KVH + kh) * a.chunks + chunk) * GLM_G + g) * GLM_PART;
    ((device glm_vec*)o)[lane] = out[g];
    if (lane == 0) { o[TF_D] = maximum[g]; o[TF_D + 1] = total[g]; }
  }
}

// One query head and row: its chunks' partials rescaled to the largest maximum and added in chunk order.
[[kernel, max_total_threads_per_threadgroup(32)]] void glm_draft_attend_merge(
    const device float* part [[buffer(0)]], constant glm_draft_args& a [[buffer(1)]], device bfloat* O [[buffer(2)]],
    constant glm_draft_stream* streams [[buffer(3)]], uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  const uint head = tg.x, row = tg.y;
  if (head >= TF_QH || row >= a.rows) return;
  const uint chunks = (streams[row / a.block].keys + GLM_CHUNK - 1) / GLM_CHUNK;
  const uint kh = head / GLM_G, g = head % GLM_G;
  const device float* first = part + (((size_t)row * TF_KVH + kh) * a.chunks * GLM_G + g) * GLM_PART;
  const size_t step = (size_t)GLM_G * GLM_PART;
  float maximum = -INFINITY;
  for (uint c = 0; c < chunks; ++c) {
    const float m = first[c * step + TF_D];
    maximum = glm_max(maximum, m);
  }
  glm_vec out = glm_vec(0.0f);
  float total = 0.0f;
  for (uint c = 0; c < chunks; ++c) {
    const device float* p = first + c * step;
    if (p[TF_D] == -INFINITY) continue;
    const float w = precise::exp(p[TF_D] - maximum);
    total = fma(p[TF_D + 1], w, total);
    out = fma(((const device glm_vec*)p)[lane], glm_vec(w), out);
  }
  const size_t qi = ((size_t)row * TF_QH + head) * TF_D;
  ((device glm_bvec*)(O + qi))[lane] = glm_bvec(total == 0.0f ? glm_vec(0.0f) : out / total);
}
