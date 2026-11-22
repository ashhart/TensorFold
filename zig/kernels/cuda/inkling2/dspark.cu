// Inkling 2 Mini CUDA DSpark kernels: block-8 proposal drafter.
// Ported from zig/kernels/metal/inkling2/dspark.metal (11 kernels) plus a new
// top-1/Gumbel pick kernel for the recurrent steps. Shapes: H=2304, Q=3072,
// KV=1024, heads 24/8 x dim 128, rank=256, rnn=768, inter=9216, vocab=200058.
// Metal threadgroup/block maps: threadgroup_position_in_grid -> blockIdx.x,
// thread_position_in_threadgroup -> threadIdx.x, dispatchGroups -> 1D grid,
// dispatchThreads -> 1D grid. bfloat(x) is __float2bfloat16_rn.
// Built for compute_121a/sm_121a only (kept arch-gated with the sm121a file set).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace ink2d {

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 tobf(float x) { return __float2bfloat16_rn(x); }

constexpr unsigned DSP_H = 2304;
constexpr unsigned DSP_Q = 3072;
constexpr unsigned DSP_KV = 1024;
constexpr unsigned DSP_HEAD = 128;
constexpr unsigned DSP_RANK = 256;
constexpr unsigned DSP_RNN = 768;
constexpr unsigned DSP_VOCAB = 200058;
constexpr unsigned DSP_SPLITS = 16;
constexpr float DSP_EPS = 1e-6f;
constexpr float DSP_SCALE = 0x1.6a09e6p-4f;

__device__ __forceinline__ float rope_angle(unsigned d, unsigned position) {
    const unsigned pair = d % 64;
    const float frequency = powf(1e6f, -float(pair) / 64.0f);
    return float(position) * frequency;
}

// splitmix64 of (seed, position, id): matches lanes/sampling.zig keyBits.
__device__ __forceinline__ uint64_t mix64(uint64_t v) {
    uint64_t x = v;
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
__device__ __forceinline__ float uniform24(uint64_t seed, unsigned position, unsigned id) {
    uint64_t base = mix64(mix64(seed + 0x9E3779B97F4A7C15ULL) ^ (uint64_t(position) * 0xD1B54A32D192ED03ULL));
    uint64_t x = mix64(base ^ id);
    uint32_t top = (uint32_t)(x >> 40);
    return (float(top) + 0.5f) * (1.0f / 16777216.0f);
}

} // namespace ink2d

// Row 0 copies the target's normalized anchor; rows 1..7 copy the mask embedding.
extern "C" __global__ void tf_ink2d_init_rows(const __nv_bfloat16* anchor, const __nv_bfloat16* mask,
                                              __nv_bfloat16* rows) {
    const unsigned row = blockIdx.x, t = threadIdx.x;
    for (unsigned d = t; d < ink2d::DSP_H; d += 256) {
        const __nv_bfloat16* src = row == 0 ? anchor : mask;
        rows[row * ink2d::DSP_H + d] = src[d];
    }
}

// Taps [row][tap][H] -> c [fusion][row][H] (fuse softmax + RMS + dflare norm); newest_f is fusion 3's last row.
extern "C" __global__ void tf_ink2d_fuse_taps(const __nv_bfloat16* taps, const __nv_bfloat16* fuse_weight,
                                              const __nv_bfloat16* norm_weight, __nv_bfloat16* c,
                                              __nv_bfloat16* newest_f, unsigned rows) {
    const unsigned group = blockIdx.x, t = threadIdx.x;
    const unsigned fusion = group / rows, row = group % rows;
    __shared__ float alpha[4];
    __shared__ float scale;
    if (t == 0) {
        float scores[4];
        float peak = -INFINITY;
        for (unsigned j = 0; j < 4; ++j) { scores[j] = ink2d::bf(fuse_weight[fusion * 4 + j]); peak = max(peak, scores[j]); }
        float sum = 0.0f;
        for (unsigned j = 0; j < 4; ++j) { scores[j] = expf(scores[j] - peak); sum += scores[j]; }
        for (unsigned j = 0; j < 4; ++j) alpha[j] = ink2d::bf(__float2bfloat16_rn(scores[j] / sum));
        float square = 0.0f;
        for (unsigned d = 0; d < ink2d::DSP_H; ++d) {
            float mixed = 0.0f;
            for (unsigned j = 0; j < 4; ++j) mixed += alpha[j] * ink2d::bf(taps[(row * 4 + j) * ink2d::DSP_H + d]);
            square += mixed * mixed;
        }
        scale = rsqrtf(square / float(ink2d::DSP_H) + ink2d::DSP_EPS);
    }
    __syncthreads();
    for (unsigned d = t; d < ink2d::DSP_H; d += 256) {
        float mixed = 0.0f;
        for (unsigned j = 0; j < 4; ++j) mixed += alpha[j] * ink2d::bf(taps[(row * 4 + j) * ink2d::DSP_H + d]);
        const __nv_bfloat16 out = ink2d::tobf(mixed * scale * ink2d::bf(norm_weight[fusion * ink2d::DSP_H + d]));
        c[(fusion * rows + row) * ink2d::DSP_H + d] = out;
        if (fusion == 3 && row + 1 == rows) newest_f[d] = out;
    }
}

// Raw projections -> per-head RMSNorm rotate-half Q/K + unchanged V (8 block rows); the per-head
// scale is a 128-thread parallel reduction (warp shuffles + 4-warp shared tree) instead of a serial loop.
extern "C" __global__ void tf_ink2d_block_qkv(const __nv_bfloat16* q_raw, const __nv_bfloat16* k_raw,
                                               const __nv_bfloat16* v_raw, const __nv_bfloat16* q_weight,
                                               const __nv_bfloat16* k_weight, __nv_bfloat16* q, __nv_bfloat16* k,
                                               __nv_bfloat16* v, unsigned base_position) {
    const unsigned group = blockIdx.x, d = threadIdx.x;
    const unsigned row = group / 40, work = group % 40;
    const unsigned lane = d & 31, warp = d >> 5;
    __shared__ float wsum[4];
    __shared__ float scale;
    if (work < 32) {
        float sum = 0.0f;
        if (work < 24) {
            for (unsigned i = d; i < ink2d::DSP_HEAD; i += 128) { const float x = ink2d::bf(q_raw[row * ink2d::DSP_Q + work * ink2d::DSP_HEAD + i]); sum += x * x; }
        } else {
            const unsigned h = work - 24;
            for (unsigned i = d; i < ink2d::DSP_HEAD; i += 128) { const float x = ink2d::bf(k_raw[row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + i]); sum += x * x; }
        }
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (d == 0) scale = rsqrtf((wsum[0] + wsum[1] + wsum[2] + wsum[3]) / float(ink2d::DSP_HEAD) + ink2d::DSP_EPS);
    }
    __syncthreads();
    if (work >= 32) {
        const unsigned h = work - 32;
        v[row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + d] = v_raw[row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + d];
        return;
    }
    const bool query = work < 24;
    const unsigned h = query ? work : work - 24;
    const unsigned at = row * (query ? ink2d::DSP_Q : ink2d::DSP_KV) + h * ink2d::DSP_HEAD + d;
    const float x = ink2d::bf(ink2d::tobf(ink2d::bf(query ? q_raw[at] : k_raw[at]) * scale * ink2d::bf(query ? q_weight[d] : k_weight[d])));
    const unsigned partner = d < 64 ? d + 64 : d - 64;
    const unsigned other_at = row * (query ? ink2d::DSP_Q : ink2d::DSP_KV) + h * ink2d::DSP_HEAD + partner;
    const float other = ink2d::bf(ink2d::tobf(ink2d::bf(query ? q_raw[other_at] : k_raw[other_at]) * scale * ink2d::bf(query ? q_weight[partner] : k_weight[partner])));
    const float angle = ink2d::rope_angle(d, base_position + row);
    const float cs = cosf(angle), sn = sinf(angle);
    const float rotated = d < 64 ? x * cs - other * sn : other * sn + x * cs;
    if (query) q[at] = ink2d::tobf(rotated); else k[at] = ink2d::tobf(rotated);
}

// Device-meta variant of block_qkv: base_position comes from hold_meta[0] so a captured hold graph replays it.
// (The per-head scale loop is shared with tf_ink2d_block_qkv above via identical code; only the position differs.)
extern "C" __global__ void tf_ink2d_block_qkv_dev(const __nv_bfloat16* q_raw, const __nv_bfloat16* k_raw,
                                                     const __nv_bfloat16* v_raw, const __nv_bfloat16* q_weight,
                                                     const __nv_bfloat16* k_weight, __nv_bfloat16* q,
                                                     __nv_bfloat16* k, __nv_bfloat16* v,
                                                     const uint32_t* hold_meta) {
    const unsigned group = blockIdx.x, d = threadIdx.x;
    const unsigned row = group / 40, work = group % 40;
    const unsigned lane = d & 31, warp = d >> 5;
    const unsigned base_position = hold_meta[0];
    __shared__ float wsum[4];
    __shared__ float scale;
    if (work < 32) {
        float sum = 0.0f;
        if (work < 24) {
            for (unsigned i = d; i < ink2d::DSP_HEAD; i += 128) { const float x = ink2d::bf(q_raw[row * ink2d::DSP_Q + work * ink2d::DSP_HEAD + i]); sum += x * x; }
        } else {
            const unsigned h = work - 24;
            for (unsigned i = d; i < ink2d::DSP_HEAD; i += 128) { const float x = ink2d::bf(k_raw[row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + i]); sum += x * x; }
        }
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (d == 0) scale = rsqrtf((wsum[0] + wsum[1] + wsum[2] + wsum[3]) / float(ink2d::DSP_HEAD) + ink2d::DSP_EPS);
    }
    __syncthreads();
    if (work >= 32) {
        const unsigned h = work - 32;
        v[row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + d] = v_raw[row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + d];
        return;
    }
    const bool query = work < 24;
    const unsigned h = query ? work : work - 24;
    const unsigned at = row * (query ? ink2d::DSP_Q : ink2d::DSP_KV) + h * ink2d::DSP_HEAD + d;
    const float x = ink2d::bf(ink2d::tobf(ink2d::bf(query ? q_raw[at] : k_raw[at]) * scale * ink2d::bf(query ? q_weight[d] : k_weight[d])));
    const unsigned partner = d < 64 ? d + 64 : d - 64;
    const unsigned other_at = row * (query ? ink2d::DSP_Q : ink2d::DSP_KV) + h * ink2d::DSP_HEAD + partner;
    const float other = ink2d::bf(ink2d::tobf(ink2d::bf(query ? q_raw[other_at] : k_raw[other_at]) * scale * ink2d::bf(query ? q_weight[partner] : k_weight[partner])));
    const float angle = ink2d::rope_angle(d, base_position + row);
    const float cs = cosf(angle), sn = sinf(angle);
    const float rotated = d < 64 ? x * cs - other * sn : other * sn + x * cs;
    if (query) q[at] = ink2d::tobf(rotated); else k[at] = ink2d::tobf(rotated);
}

// Context K/V at absolute-position modulo ring (committed rows only); the per-head K scale is a
// 128-thread parallel reduction instead of a serial loop.
extern "C" __global__ void tf_ink2d_context_kv(const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
                                               const __nv_bfloat16* weight, __nv_bfloat16* keys,
                                               __nv_bfloat16* values, unsigned base_position, unsigned rows,
                                               unsigned ring) {
    const unsigned group = blockIdx.x, d = threadIdx.x;
    const unsigned row = group / 8, h = group % 8;
    const unsigned lane = d & 31, warp = d >> 5;
    __shared__ float wsum[4];
    __shared__ float scale;
    {
        float sum = 0.0f;
        for (unsigned i = d; i < ink2d::DSP_HEAD; i += 128) { const float x = ink2d::bf(k_raw[row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + i]); sum += x * x; }
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (d == 0) scale = rsqrtf((wsum[0] + wsum[1] + wsum[2] + wsum[3]) / float(ink2d::DSP_HEAD) + ink2d::DSP_EPS);
    }
    __syncthreads();
    const unsigned position = base_position + row;
    const unsigned at = (position % ring) * ink2d::DSP_KV + h * ink2d::DSP_HEAD + d;
    const unsigned src = row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + d;
    const unsigned partner = d < 64 ? d + 64 : d - 64;
    const float x = ink2d::bf(ink2d::tobf(ink2d::bf(k_raw[src]) * scale * ink2d::bf(weight[d])));
    const float other = ink2d::bf(ink2d::tobf(ink2d::bf(k_raw[row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + partner]) * scale * ink2d::bf(weight[partner])));
    const float angle = ink2d::rope_angle(d, position);
    const float cs = cosf(angle), sn = sinf(angle);
    keys[at] = ink2d::tobf(d < 64 ? x * cs - other * sn : other * sn + x * cs);
    values[at] = v_raw[src];
}

// Per-(token, head) symmetric int8 context K/V for global layer 1:
// buffer layout = int8_t codes[ring][8][128] followed by float scales[ring][8].
extern "C" __global__ void tf_ink2d_context_kv_i8(const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
                                                   const __nv_bfloat16* weight, int8_t* keys, int8_t* values,
                                                   unsigned base_position, unsigned rows, unsigned ring) {
    const unsigned group = blockIdx.x, d = threadIdx.x;
    const unsigned row = group / 8, h = group % 8;
    const unsigned lane = d & 31, warp = d >> 5;
    __shared__ float wsum[4];
    __shared__ float scale;
    __shared__ float amax_k_w[4], amax_v_w[4];
    __shared__ float k_inv, v_inv;
    {
        float sum = 0.0f;
        for (unsigned i = d; i < ink2d::DSP_HEAD; i += 128) {
            const float x = ink2d::bf(k_raw[row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + i]);
            sum += x * x;
        }
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (d == 0) scale = rsqrtf((wsum[0] + wsum[1] + wsum[2] + wsum[3]) / float(ink2d::DSP_HEAD) + ink2d::DSP_EPS);
    }
    __syncthreads();
    const unsigned position = base_position + row;
    const size_t slot = (size_t)(position % ring) * 8 + h;
    const size_t at = (size_t)(position % ring) * ink2d::DSP_KV + h * ink2d::DSP_HEAD + d;
    const unsigned src = row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + d;
    const unsigned partner = d < 64 ? d + 64 : d - 64;
    const float x = ink2d::bf(ink2d::tobf(ink2d::bf(k_raw[src]) * scale * ink2d::bf(weight[d])));
    const float other = ink2d::bf(ink2d::tobf(ink2d::bf(k_raw[row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + partner]) * scale * ink2d::bf(weight[partner])));
    const float angle = ink2d::rope_angle(d, position);
    const float cs = cosf(angle), sn = sinf(angle);
    const float kf = ink2d::bf(ink2d::tobf(d < 64 ? x * cs - other * sn : other * sn + x * cs));
    const float vf = ink2d::bf(v_raw[src]);
    float ak = fabsf(kf), av = fabsf(vf);
    for (int o = 16; o > 0; o >>= 1) {
        ak = fmaxf(ak, __shfl_xor_sync(0xffffffffu, ak, o));
        av = fmaxf(av, __shfl_xor_sync(0xffffffffu, av, o));
    }
    if (lane == 0) { amax_k_w[warp] = ak; amax_v_w[warp] = av; }
    __syncthreads();
    if (d == 0) {
        const float amax_k = fmaxf(fmaxf(amax_k_w[0], amax_k_w[1]), fmaxf(amax_k_w[2], amax_k_w[3]));
        const float amax_v = fmaxf(fmaxf(amax_v_w[0], amax_v_w[1]), fmaxf(amax_v_w[2], amax_v_w[3]));
        float* k_scales = reinterpret_cast<float*>(keys + (size_t)ring * ink2d::DSP_KV);
        float* v_scales = reinterpret_cast<float*>(values + (size_t)ring * ink2d::DSP_KV);
        k_scales[slot] = amax_k > 0.0f ? amax_k / 127.0f : 0.0f;
        v_scales[slot] = amax_v > 0.0f ? amax_v / 127.0f : 0.0f;
        k_inv = amax_k > 0.0f ? 127.0f / amax_k : 0.0f;
        v_inv = amax_v > 0.0f ? 127.0f / amax_v : 0.0f;
    }
    __syncthreads();
    int qk = __float2int_rn(kf * k_inv);
    if (qk > 127) qk = 127; else if (qk < -127) qk = -127;
    int qv = __float2int_rn(vf * v_inv);
    if (qv > 127) qv = 127; else if (qv < -127) qv = -127;
    keys[at] = static_cast<int8_t>(qk);
    values[at] = static_cast<int8_t>(qv);
}

__device__ __forceinline__ float ink2d_attn_score(const __nv_bfloat16* q, const __nv_bfloat16* k, unsigned q_at, unsigned k_at) {
    float dot = 0.0f;
    for (unsigned d = 0; d < ink2d::DSP_HEAD; ++d) dot += ink2d::bf(q[q_at + d]) * ink2d::bf(k[k_at + d]);
    return dot * ink2d::DSP_SCALE;
}

// 8 block rows attend over retained committed context + the 8 block rows: 1-pass FlashAttention-style
// online softmax across 128 threads (one dim each). Each thread keeps its own (max, sum, numerator);
// per key the 128-dim dot is cooperatively reduced (warp shuffles + 8-float ping-pong shared, 1 sync),
// so the global `scores` buffer and its traffic are gone. `scores`/`score_stride` stay for ABI.
__device__ __forceinline__ void ink2d_attention_online(const __nv_bfloat16* q, const __nv_bfloat16* block_k,
                                                       const __nv_bfloat16* block_v, const __nv_bfloat16* context_k,
                                                       const __nv_bfloat16* context_v, __nv_bfloat16* out,
                                                       unsigned committed, unsigned ring, unsigned local,
                                                       unsigned from, unsigned row, unsigned h, unsigned d) {
    const unsigned kv = h / 3;
    const unsigned q_at = row * ink2d::DSP_Q + h * ink2d::DSP_HEAD;
    const unsigned first = max(from, local && committed > 1024 ? committed - 1024 : 0u);
    const unsigned retained = committed - first;
    const float qv = ink2d::bf(q[q_at + d]);
    const unsigned warp = d >> 5, lane = d & 31;
    __shared__ float pd[2][4];
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (unsigned j = 0; j < retained; ++j) {
        const unsigned position = first + j;
        const unsigned k_at = (position % ring) * ink2d::DSP_KV + kv * ink2d::DSP_HEAD + d;
        float dot = qv * ink2d::bf(context_k[k_at]);
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if (lane == 0) pd[j & 1][warp] = dot;
        __syncthreads();
        const float s = (pd[j & 1][0] + pd[j & 1][1] + pd[j & 1][2] + pd[j & 1][3]) * ink2d::DSP_SCALE;
        const unsigned v_at = (position % ring) * ink2d::DSP_KV + kv * ink2d::DSP_HEAD + d;
        const float v = ink2d::bf(context_v[v_at]);
        const float m_new = fmaxf(m, s);
        const float a = (l == 0.0f) ? 0.0f : expf(m - m_new), b = expf(s - m_new);
        acc = acc * a + b * v;
        l = l * a + b;
        m = m_new;
    }
    for (unsigned j = 0; j < 8; ++j) {
        const unsigned k_at = j * ink2d::DSP_KV + kv * ink2d::DSP_HEAD + d;
        float dot = qv * ink2d::bf(block_k[k_at]);
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if (lane == 0) pd[(retained + j) & 1][warp] = dot;
        __syncthreads();
        const float s = (pd[(retained + j) & 1][0] + pd[(retained + j) & 1][1] + pd[(retained + j) & 1][2] + pd[(retained + j) & 1][3]) * ink2d::DSP_SCALE;
        const unsigned v_at = j * ink2d::DSP_KV + kv * ink2d::DSP_HEAD + d;
        const float v = ink2d::bf(block_v[v_at]);
        const float m_new = fmaxf(m, s);
        const float a = (l == 0.0f) ? 0.0f : expf(m - m_new), b = expf(s - m_new);
        acc = acc * a + b * v;
        l = l * a + b;
        m = m_new;
    }
    out[row * ink2d::DSP_Q + h * ink2d::DSP_HEAD + d] = ink2d::tobf(acc / l);
}

extern "C" __global__ void tf_ink2d_attention(const __nv_bfloat16* q, const __nv_bfloat16* block_k,
                                              const __nv_bfloat16* block_v, const __nv_bfloat16* context_k,
                                              const __nv_bfloat16* context_v, float* scores, __nv_bfloat16* out,
                                              unsigned committed, unsigned ring, unsigned local, unsigned from,
                                              unsigned score_stride) {
    (void)scores;
    (void)score_stride;
    const unsigned group = blockIdx.x, d = threadIdx.x;
    ink2d_attention_online(q, block_k, block_v, context_k, context_v, out, committed, ring, local, from,
                            group / 24, group % 24, d);
}

// Device-meta variant: committed/from come from hold_meta[1..2] so a captured hold graph replays them.
extern "C" __global__ void tf_ink2d_attention_dev(const __nv_bfloat16* q, const __nv_bfloat16* block_k,
                                                    const __nv_bfloat16* block_v, const __nv_bfloat16* context_k,
                                                    const __nv_bfloat16* context_v, float* scores,
                                                    __nv_bfloat16* out, const uint32_t* hold_meta, unsigned ring,
                                                    unsigned local, unsigned score_stride) {
    (void)scores;
    (void)score_stride;
    const unsigned group = blockIdx.x, d = threadIdx.x;
    ink2d_attention_online(q, block_k, block_v, context_k, context_v, out, hold_meta[1], ring, local,
                            hold_meta[2], group / 24, group % 24, d);
}

__device__ __forceinline__ void ink2d_attention_online_i8(const __nv_bfloat16* q, const __nv_bfloat16* block_k,
                                                          const __nv_bfloat16* block_v, const int8_t* context_k,
                                                          const int8_t* context_v, __nv_bfloat16* out,
                                                          unsigned committed, unsigned ring, unsigned local,
                                                          unsigned from, unsigned row, unsigned h, unsigned d) {
    const unsigned kv = h / 3;
    const unsigned q_at = row * ink2d::DSP_Q + h * ink2d::DSP_HEAD;
    const unsigned first = max(from, local && committed > 1024 ? committed - 1024 : 0u);
    const unsigned retained = committed - first;
    const float qv = ink2d::bf(q[q_at + d]);
    const unsigned warp = d >> 5, lane = d & 31;
    const float* k_scales = reinterpret_cast<const float*>(context_k + (size_t)ring * ink2d::DSP_KV);
    const float* v_scales = reinterpret_cast<const float*>(context_v + (size_t)ring * ink2d::DSP_KV);
    __shared__ float pd[2][4];
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (unsigned j = 0; j < retained; ++j) {
        const unsigned position = first + j;
        const size_t slot = (size_t)(position % ring) * 8 + kv;
        const size_t k_at = (size_t)(position % ring) * ink2d::DSP_KV + kv * ink2d::DSP_HEAD + d;
        const float k_val = static_cast<float>(context_k[k_at]) * k_scales[slot];
        float dot = qv * k_val;
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if (lane == 0) pd[j & 1][warp] = dot;
        __syncthreads();
        const float s = (pd[j & 1][0] + pd[j & 1][1] + pd[j & 1][2] + pd[j & 1][3]) * ink2d::DSP_SCALE;
        const float v = static_cast<float>(context_v[k_at]) * v_scales[slot];
        const float m_new = fmaxf(m, s);
        const float a = (l == 0.0f) ? 0.0f : expf(m - m_new), b = expf(s - m_new);
        acc = acc * a + b * v;
        l = l * a + b;
        m = m_new;
    }
    for (unsigned j = 0; j < 8; ++j) {
        const unsigned k_at = j * ink2d::DSP_KV + kv * ink2d::DSP_HEAD + d;
        float dot = qv * ink2d::bf(block_k[k_at]);
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if (lane == 0) pd[(retained + j) & 1][warp] = dot;
        __syncthreads();
        const float s = (pd[(retained + j) & 1][0] + pd[(retained + j) & 1][1] + pd[(retained + j) & 1][2] + pd[(retained + j) & 1][3]) * ink2d::DSP_SCALE;
        const unsigned v_at = j * ink2d::DSP_KV + kv * ink2d::DSP_HEAD + d;
        const float v = ink2d::bf(block_v[v_at]);
        const float m_new = fmaxf(m, s);
        const float a = (l == 0.0f) ? 0.0f : expf(m - m_new), b = expf(s - m_new);
        acc = acc * a + b * v;
        l = l * a + b;
        m = m_new;
    }
    out[row * ink2d::DSP_Q + h * ink2d::DSP_HEAD + d] = ink2d::tobf(acc / l);
}

extern "C" __global__ void tf_ink2d_attention_i8(const __nv_bfloat16* q, const __nv_bfloat16* block_k,
                                                  const __nv_bfloat16* block_v, const int8_t* context_k,
                                                  const int8_t* context_v, float* scores, __nv_bfloat16* out,
                                                  unsigned committed, unsigned ring, unsigned local, unsigned from,
                                                  unsigned score_stride) {
    (void)scores;
    (void)score_stride;
    const unsigned group = blockIdx.x, d = threadIdx.x;
    ink2d_attention_online_i8(q, block_k, block_v, context_k, context_v, out, committed, ring, local, from,
                              group / 24, group % 24, d);
}

extern "C" __global__ void tf_ink2d_attention_dev_i8(const __nv_bfloat16* q, const __nv_bfloat16* block_k,
                                                      const __nv_bfloat16* block_v, const int8_t* context_k,
                                                      const int8_t* context_v, float* scores, __nv_bfloat16* out,
                                                      const uint32_t* hold_meta, unsigned ring, unsigned local,
                                                      unsigned score_stride) {
    (void)scores;
    (void)score_stride;
    const unsigned group = blockIdx.x, d = threadIdx.x;
    ink2d_attention_online_i8(q, block_k, block_v, context_k, context_v, out, hold_meta[1], ring, local,
                              hold_meta[2], group / 24, group % 24, d);
}

__device__ __forceinline__ void ink2d_attention_splitk_i8_body(
    const __nv_bfloat16* q, const __nv_bfloat16* block_k, const __nv_bfloat16* block_v,
    const int8_t* context_k, const int8_t* context_v, float* part_o, float2* part_ml,
    unsigned committed, unsigned ring, unsigned from, unsigned kv, unsigned sp, unsigned stream_idx) {
    const unsigned t = threadIdx.x;
    const unsigned row = t >> 5;
    const unsigned lane = t & 31;
    const unsigned d0 = lane * 4;
    __shared__ float sk[16][128];
    __shared__ float sv[16][128];

    float qv[3][4];
    for (unsigned g = 0; g < 3; ++g) {
        const unsigned h = kv * 3 + g;
        const unsigned q_at = row * ink2d::DSP_Q + h * ink2d::DSP_HEAD + d0;
        for (unsigned j = 0; j < 4; ++j) qv[g][j] = ink2d::bf(q[q_at + j]);
    }

    float m[3] = { -INFINITY, -INFINITY, -INFINITY };
    float l[3] = { 0.0f, 0.0f, 0.0f };
    float acc[3][4] = {};

    const unsigned retained = committed > from ? committed - from : 0u;
    const unsigned k_per = (retained + ink2d::DSP_SPLITS - 1) / ink2d::DSP_SPLITS;
    const unsigned k_lo = min(sp * k_per, retained);
    const unsigned k_hi = min(k_lo + k_per, retained);

    const float* k_scales = reinterpret_cast<const float*>(context_k + (size_t)ring * ink2d::DSP_KV);
    const float* v_scales = reinterpret_cast<const float*>(context_v + (size_t)ring * ink2d::DSP_KV);

    const unsigned t_key = t >> 4;
    const unsigned t_d = (t & 15) * 8;
    for (unsigned base = k_lo; base < k_hi; base += 16) {
        const unsigned n_tile = min(16u, k_hi - base);
        if (t_key < n_tile) {
            const unsigned ring_pos = (from + base + t_key) % ring;
            const size_t slot = (size_t)ring_pos * 8 + kv;
            const size_t kv_at = (size_t)ring_pos * ink2d::DSP_KV + kv * ink2d::DSP_HEAD + t_d;
            const float ks = k_scales[slot];
            const float vs = v_scales[slot];
            for (unsigned j = 0; j < 8; ++j) {
                sk[t_key][t_d + j] = static_cast<float>(context_k[kv_at + j]) * ks;
                sv[t_key][t_d + j] = static_cast<float>(context_v[kv_at + j]) * vs;
            }
        }
        __syncthreads();
        for (unsigned tk = 0; tk < n_tile; ++tk) {
            const float4 kv4 = *reinterpret_cast<const float4*>(&sk[tk][d0]);
            const float4 vv4 = *reinterpret_cast<const float4*>(&sv[tk][d0]);
            for (unsigned g = 0; g < 3; ++g) {
                float dot = qv[g][0] * kv4.x + qv[g][1] * kv4.y + qv[g][2] * kv4.z + qv[g][3] * kv4.w;
                for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
                const float s = dot * ink2d::DSP_SCALE;
                const float m_new = fmaxf(m[g], s);
                const float a = (l[g] == 0.0f) ? 0.0f : expf(m[g] - m_new);
                const float b = expf(s - m_new);
                acc[g][0] = acc[g][0] * a + b * vv4.x;
                acc[g][1] = acc[g][1] * a + b * vv4.y;
                acc[g][2] = acc[g][2] * a + b * vv4.z;
                acc[g][3] = acc[g][3] * a + b * vv4.w;
                l[g] = l[g] * a + b;
                m[g] = m_new;
            }
        }
        __syncthreads();
    }

    if (sp == 0) {
        if (t_key < 8) {
            const size_t kv_at = (size_t)t_key * ink2d::DSP_KV + kv * ink2d::DSP_HEAD + t_d;
            for (unsigned j = 0; j < 8; ++j) {
                sk[t_key][t_d + j] = ink2d::bf(block_k[kv_at + j]);
                sv[t_key][t_d + j] = ink2d::bf(block_v[kv_at + j]);
            }
        }
        __syncthreads();
        for (unsigned tk = 0; tk < 8; ++tk) {
            const float4 kv4 = *reinterpret_cast<const float4*>(&sk[tk][d0]);
            const float4 vv4 = *reinterpret_cast<const float4*>(&sv[tk][d0]);
            for (unsigned g = 0; g < 3; ++g) {
                float dot = qv[g][0] * kv4.x + qv[g][1] * kv4.y + qv[g][2] * kv4.z + qv[g][3] * kv4.w;
                for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
                const float s = dot * ink2d::DSP_SCALE;
                const float m_new = fmaxf(m[g], s);
                const float a = (l[g] == 0.0f) ? 0.0f : expf(m[g] - m_new);
                const float b = expf(s - m_new);
                acc[g][0] = acc[g][0] * a + b * vv4.x;
                acc[g][1] = acc[g][1] * a + b * vv4.y;
                acc[g][2] = acc[g][2] * a + b * vv4.z;
                acc[g][3] = acc[g][3] * a + b * vv4.w;
                l[g] = l[g] * a + b;
                m[g] = m_new;
            }
        }
    }

    for (unsigned g = 0; g < 3; ++g) {
        const size_t out_slot = ((size_t)(stream_idx * 8 + row) * 24 + (kv * 3 + g)) * ink2d::DSP_SPLITS + sp;
        if (lane == 0) part_ml[out_slot] = make_float2(m[g], l[g]);
        *reinterpret_cast<float4*>(part_o + out_slot * ink2d::DSP_HEAD + d0) =
            make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
    }
}

extern "C" __global__ void tf_ink2d_attention_splitk_dev_i8(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ block_k,
    const __nv_bfloat16* __restrict__ block_v, const int8_t* __restrict__ context_k,
    const int8_t* __restrict__ context_v, float* __restrict__ part_o, float2* __restrict__ part_ml,
    const uint32_t* __restrict__ hold_meta, unsigned ring) {
    const unsigned kv = blockIdx.x / ink2d::DSP_SPLITS;
    const unsigned sp = blockIdx.x % ink2d::DSP_SPLITS;
    ink2d_attention_splitk_i8_body(q, block_k, block_v, context_k, context_v, part_o, part_ml,
                                   hold_meta[1], ring, hold_meta[2], kv, sp, 0);
}

extern "C" __global__ void tf_ink2d_attention_splitk_merge(
    const float* __restrict__ part_o, const float2* __restrict__ part_ml, __nv_bfloat16* __restrict__ out) {
    const unsigned d = threadIdx.x;
    const size_t slot_base = (size_t)blockIdx.x * ink2d::DSP_SPLITS;
    float M = -INFINITY, L = 0.0f, O = 0.0f;
    for (unsigned sp = 0; sp < ink2d::DSP_SPLITS; ++sp) {
        const float2 ml = part_ml[slot_base + sp];
        if (ml.y <= 0.0f) continue;
        const float o = part_o[(slot_base + sp) * ink2d::DSP_HEAD + d];
        if (L == 0.0f) {
            M = ml.x;
            L = ml.y;
            O = o;
        } else {
            const float m_new = fmaxf(M, ml.x);
            const float a = expf(M - m_new);
            const float b = expf(ml.x - m_new);
            O = O * a + o * b;
            L = L * a + ml.y * b;
            M = m_new;
        }
    }
    out[(size_t)blockIdx.x * ink2d::DSP_HEAD + d] = ink2d::tobf(L > 0.0f ? (O / L) : 0.0f);
}

// RMSNorm consuming the fp32 residual sum; stores the rounded residual. The row sum of squares is a
// 256-thread parallel reduction (warp shuffles + 8-warp shared tree) instead of a serial loop.
extern "C" __global__ void tf_ink2d_residual_norm(const __nv_bfloat16* x, const __nv_bfloat16* residual,
                                                  const __nv_bfloat16* weight, __nv_bfloat16* new_residual,
                                                  __nv_bfloat16* normalized, unsigned rows, unsigned init) {
    const unsigned row = blockIdx.x, t = threadIdx.x;
    const unsigned lane = t & 31, warp = t >> 5;
    __shared__ float wsum[8];
    __shared__ float scale;
    {
        float sum = 0.0f;
        for (unsigned d = t; d < ink2d::DSP_H; d += 256) {
            const unsigned at = row * ink2d::DSP_H + d;
            const float v = init ? ink2d::bf(x[at]) : ink2d::bf(x[at]) + ink2d::bf(residual[at]);
            sum += v * v;
        }
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (t == 0) scale = rsqrtf((wsum[0] + wsum[1] + wsum[2] + wsum[3] + wsum[4] + wsum[5] + wsum[6] + wsum[7]) / float(ink2d::DSP_H) + ink2d::DSP_EPS);
    }
    __syncthreads();
    for (unsigned n = 0; n < 9; ++n) {
        const unsigned d = n * 256 + t;
        if (d >= ink2d::DSP_H) break;
        const unsigned at = row * ink2d::DSP_H + d;
        const float sigma = init ? ink2d::bf(x[at]) : ink2d::bf(x[at]) + ink2d::bf(residual[at]);
        new_residual[at] = ink2d::tobf(sigma);
        normalized[at] = ink2d::tobf(sigma * scale * ink2d::bf(weight[d]));
    }
}

// Plain SwiGLU (no caps on the drafter path): gate*sigmoid(gate)*up.
extern "C" __global__ void tf_ink2d_swiglu(const __nv_bfloat16* g, const __nv_bfloat16* u, __nv_bfloat16* out,
                                            unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float gate = ink2d::bf(g[i]);
    out[i] = ink2d::tobf((gate / (1.0f + expf(-gate))) * ink2d::bf(u[i]));
}

// Kappa rows [rows][768]: y cols 512..2815, newest_f cols 2816..5119, + bias.
extern "C" __global__ void tf_ink2d_kappa(const __nv_bfloat16* y, const __nv_bfloat16* f, const __nv_bfloat16* weight,
                                          const __nv_bfloat16* bias, float* out, unsigned rows) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = i / 768, channel = i % 768;
    if (row >= rows) return;
    float sum = 0.0f;
    for (unsigned d = 0; d < ink2d::DSP_H; ++d) sum += ink2d::bf(y[row * ink2d::DSP_H + d]) * ink2d::bf(weight[channel * 5120 + 512 + d]);
    for (unsigned d = 0; d < ink2d::DSP_H; ++d) sum += ink2d::bf(f[d]) * ink2d::bf(weight[channel * 5120 + 2816 + d]);
    sum += ink2d::bf(bias[channel]);
    out[i] = sum;
}

// One Markov recurrent step: GRU-style blend per lane, tanh lambda.
extern "C" __global__ void tf_ink2d_recurrent(const __nv_bfloat16* state, const float* kappa,
                                              const __nv_bfloat16* rnn_weight, const __nv_bfloat16* markov,
                                              __nv_bfloat16* z, __nv_bfloat16* next_state, __nv_bfloat16* lambda,
                                              unsigned token) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ink2d::DSP_RANK) return;
    float values[3];
    for (unsigned part = 0; part < 3; ++part) {
        const unsigned channel = part * ink2d::DSP_RANK + i;
        float sum = 0.0f;
        for (unsigned d = 0; d < ink2d::DSP_RANK; ++d) sum += ink2d::bf(state[d]) * ink2d::bf(rnn_weight[channel * 5120 + d]);
        for (unsigned d = 0; d < ink2d::DSP_RANK; ++d) sum += ink2d::bf(markov[uint64_t(token) * ink2d::DSP_RANK + d]) * ink2d::bf(rnn_weight[channel * 5120 + 256 + d]);
        sum += kappa[channel];
        const __nv_bfloat16 rounded = ink2d::tobf(sum);
        z[channel] = rounded;
        values[part] = ink2d::bf(rounded);
    }
    const __nv_bfloat16 gamma = ink2d::tobf(1.0f / (1.0f + expf(-values[0])));
    const __nv_bfloat16 candidate = ink2d::tobf(tanhf(values[1]));
    const __nv_bfloat16 keep = ink2d::tobf(1.0f - ink2d::bf(gamma));
    const __nv_bfloat16 left = ink2d::tobf(ink2d::bf(gamma) * ink2d::bf(state[i]));
    const __nv_bfloat16 right = ink2d::tobf(ink2d::bf(keep) * ink2d::bf(candidate));
    next_state[i] = ink2d::tobf(ink2d::bf(left) + ink2d::bf(right));
    lambda[i] = ink2d::tobf(tanhf(values[2]));
}

// Device-side token chain variant of tf_ink2d_recurrent: the token comes from a device pointer
// (the previous step's draw output) instead of by value, and state[0..255] plus the token's markov
// row preload into __shared__ once across the 256 threads. Same ops/order/rounding: bit-identical.
// Lets a captured hold graph replay the whole chain with zero host syncs or D2H copies per step.
extern "C" __global__ void tf_ink2d_recurrent_dev(const __nv_bfloat16* state, const float* kappa,
                                                  const __nv_bfloat16* rnn_weight, const __nv_bfloat16* markov,
                                                  __nv_bfloat16* z, __nv_bfloat16* next_state,
                                                  __nv_bfloat16* lambda, const unsigned* token_ptr) {
    __shared__ float sh_state[ink2d::DSP_RANK];
    __shared__ float sh_mark[ink2d::DSP_RANK];
    const unsigned t = threadIdx.x;
    const unsigned token = token_ptr[0];
    sh_state[t] = ink2d::bf(state[t]);
    sh_mark[t] = ink2d::bf(markov[uint64_t(token) * ink2d::DSP_RANK + t]);
    __syncthreads();
    const unsigned i = blockIdx.x * blockDim.x + t;
    if (i >= ink2d::DSP_RANK) return;
    float values[3];
    for (unsigned part = 0; part < 3; ++part) {
        const unsigned channel = part * ink2d::DSP_RANK + i;
        float sum = 0.0f;
        for (unsigned d = 0; d < ink2d::DSP_RANK; ++d) sum += sh_state[d] * ink2d::bf(rnn_weight[channel * 5120 + d]);
        for (unsigned d = 0; d < ink2d::DSP_RANK; ++d) sum += sh_mark[d] * ink2d::bf(rnn_weight[channel * 5120 + 256 + d]);
        sum += kappa[channel];
        const __nv_bfloat16 rounded = ink2d::tobf(sum);
        z[channel] = rounded;
        values[part] = ink2d::bf(rounded);
    }
    const __nv_bfloat16 gamma = ink2d::tobf(1.0f / (1.0f + expf(-values[0])));
    const __nv_bfloat16 candidate = ink2d::tobf(tanhf(values[1]));
    const __nv_bfloat16 keep = ink2d::tobf(1.0f - ink2d::bf(gamma));
    const __nv_bfloat16 left = ink2d::tobf(ink2d::bf(gamma) * sh_state[i]);
    const __nv_bfloat16 right = ink2d::tobf(ink2d::bf(keep) * ink2d::bf(candidate));
    next_state[i] = ink2d::tobf(ink2d::bf(left) + ink2d::bf(right));
    lambda[i] = ink2d::tobf(tanhf(values[2]));
}


// Raw U*y rows, one BF16 round, no softcap.
extern "C" __global__ void tf_ink2d_base(const __nv_bfloat16* unembed, const __nv_bfloat16* y, __nv_bfloat16* out,
                                         unsigned rows, unsigned vocab) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * vocab) return;
    const unsigned row = i / vocab, id = i % vocab;
    float dot = 0.0f;
    for (unsigned d = 0; d < ink2d::DSP_H; ++d) dot += ink2d::bf(unembed[id * ink2d::DSP_H + d]) * ink2d::bf(y[row * ink2d::DSP_H + d]);
    out[i] = ink2d::tobf(dot);
}

// Folded Phi/beta correction + target-style 30-tanh over the full draft vocab, warp a row:
// 8 warps a block = 8 vocab rows a block, lambda[0..255] in __shared__ once per block, each lane one
// 16-byte uint4 (8 consecutive BF16) for a fully coalesced phi row read. The per-row dot regroups the
// 256 products into 32 lane partials combined by a shfl-xor tree, so logits may differ ~1ulp from the
// old one-thread-a-row kernel; bias/logit/30-tanh stay the exact same formulas.
extern "C" __global__ void tf_ink2d_logits(const __nv_bfloat16* base, const __nv_bfloat16* phi,
                                           const __nv_bfloat16* beta, const __nv_bfloat16* lambda,
                                           __nv_bfloat16* out, unsigned vocab) {
    __shared__ float slam[ink2d::DSP_RANK];
    for (unsigned d = threadIdx.x; d < ink2d::DSP_RANK; d += blockDim.x) slam[d] = ink2d::bf(lambda[d]);
    __syncthreads();
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned v = blockIdx.x * 8 + warp;
    if (v >= vocab) return;
    const uint4 pack = reinterpret_cast<const uint4*>(phi + uint64_t(v) * ink2d::DSP_RANK + lane * 8)[0];
    __align__(16) __nv_bfloat16 e[8];
    *reinterpret_cast<uint4*>(e) = pack;
    float dot = 0.0f;
    for (unsigned j = 0; j < 8; ++j) dot += ink2d::bf(e[j]) * slam[lane * 8 + j];
    for (unsigned o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
    if (lane == 0) {
        const __nv_bfloat16 bias = ink2d::tobf(ink2d::bf(beta[v]) + dot);
        const __nv_bfloat16 logit = ink2d::tobf(ink2d::bf(base[v]) + ink2d::bf(bias));
        out[v] = ink2d::tobf(30.0f * tanhf(ink2d::bf(logit) / 30.0f));
    }
}

// FP8-E4M3 (bias 7, max 448, saturation) helpers for M6b: software round-to-nearest so encode and
// the host LUT decode form a closed loop with zero arch-specific instructions. The 448 scale makes a
// row's peak round-trip exactly (amplitude preserved); all other values keep e4m3 precision.
__device__ __forceinline__ uint8_t ink2d_f32_to_e4m3(float x) {
    const float ax = fabsf(x);
    if (!(ax > 0.0f)) return __float_as_uint(x) >> 24 == 0x80u ? 0x80u : 0x00u;  // +-0 (and any NaN payload -> 0)
    if (!isfinite(ax)) return __signbit(x) ? 0xFEu : 0x7Eu;
    const uint32_t u = __float_as_uint(ax);
    const int e = int(u >> 23) - 127;  // unbiased f32 exponent
    const int e8 = e + 7;
    if (e8 > 15) return __signbit(x) ? 0xFEu : 0x7Eu;  // saturate to +-448
    if (e8 >= 1) {
        const float q = ax * exp2f(float(10 - e8));  // exact: ulp becomes 1.0
        int qi = __float2int_rn(q);                  // RN into [8, 16]
        int ee = e8;
        if (qi >= 16) { qi = 8; ee += 1; }
        if (ee > 15 || (ee == 15 && qi > 14)) return __signbit(x) ? 0xFEu : 0x7Eu;
        return (__signbit(x) ? 0x80u : 0u) | uint8_t((ee << 3) | (qi & 7));
    }
    const float q = ax * 512.0f;  // subnormal ulp 2^-9
    int qi = __float2int_rn(q);   // RN into [0, 8]
    if (qi <= 0) return __signbit(x) ? 0x80u : 0x00u;
    if (qi > 7) return (__signbit(x) ? 0x80u : 0u) | uint8_t((1 << 3) | 0);  // rounds into 2^-6
    return (__signbit(x) ? 0x80u : 0u) | uint8_t(qi);
}

// Folded BF16 phi[vocab, 256] -> FP8-E4M3 phi_fp8 + per-row FP32 phi_scale (amax / 448, zero for 0 rows).
extern "C" __global__ void tf_ink2d_quant_phi_fp8(const __nv_bfloat16* phi, uint8_t* phi_fp8,
                                                     float* phi_scale, unsigned vocab) {
    const unsigned v = blockIdx.x;
    if (v >= vocab) return;
    const unsigned t = threadIdx.x;
    float amax = 0.0f;
    for (unsigned d = t; d < ink2d::DSP_RANK; d += 256) amax = fmaxf(amax, fabsf(ink2d::bf(phi[v * ink2d::DSP_RANK + d])));
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    __shared__ float row_max;
    __shared__ float warp_max[8];
    if (t % 32 == 0) warp_max[t / 32] = amax;
    __syncthreads();
    if (t == 0) {
        float m = warp_max[0];
        for (int w = 1; w < 8; ++w) m = fmaxf(m, warp_max[w]);
        row_max = m;
    }
    __syncthreads();
    const float scale = row_max > 0.0f ? row_max / 448.0f : 0.0f;
    if (t == 0) phi_scale[v] = scale;
    const float inv = scale > 0.0f ? 1.0f / scale : 1.0f;
    for (unsigned d = t; d < ink2d::DSP_RANK; d += 256)
        phi_fp8[v * ink2d::DSP_RANK + d] = ink2d_f32_to_e4m3(ink2d::bf(phi[v * ink2d::DSP_RANK + d]) * inv);
}

// Warp-per-row FP8 logits over the halved phi (51.2 MB): LUT-decoded e4m3 dot, same bias/logit/30-tanh.
extern "C" __global__ void tf_ink2d_logits_fp8(const __nv_bfloat16* base, const uint8_t* phi_fp8,
                                                 const float* phi_scale, const __nv_bfloat16* beta,
                                                 const float* f8lut, const __nv_bfloat16* lambda,
                                                 __nv_bfloat16* out, unsigned vocab) {
    __shared__ float slam[ink2d::DSP_RANK];
    __shared__ float slut[256];
    for (unsigned d = threadIdx.x; d < ink2d::DSP_RANK; d += blockDim.x) slam[d] = ink2d::bf(lambda[d]);
    slut[threadIdx.x] = f8lut[threadIdx.x];
    __syncthreads();
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned v = blockIdx.x * 8 + warp;
    if (v >= vocab) return;
    const uint2 pack = reinterpret_cast<const uint2*>(phi_fp8 + uint64_t(v) * ink2d::DSP_RANK + lane * 8)[0];
    const uint8_t* e = reinterpret_cast<const uint8_t*>(&pack);
    float dot = 0.0f;
    for (unsigned j = 0; j < 8; ++j) dot += slut[e[j]] * slam[lane * 8 + j];
    for (unsigned o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
    if (lane == 0) {
        const float scaled = dot * phi_scale[v];
        const __nv_bfloat16 bias = ink2d::tobf(ink2d::bf(beta[v]) + scaled);
        const __nv_bfloat16 logit = ink2d::tobf(ink2d::bf(base[v]) + ink2d::bf(bias));
        out[v] = ink2d::tobf(30.0f * tanhf(ink2d::bf(logit) / 30.0f));
    }
}
// vocab 200058, rows <= 7: one 256-thread block a row, strided scan + shared reduce.
// Per-row pick: greedy argmax (inv_t == 0) or Gumbel-max with the lanes keyBits rule.
extern "C" __global__ void tf_ink2d_top1_gumbel(const __nv_bfloat16* logits, float inv_t, uint64_t seed,
                                                unsigned position, unsigned vocab, unsigned* out_id) {
    const unsigned row = blockIdx.x;
    const unsigned t = threadIdx.x;
    __shared__ float best_v[256];
    __shared__ unsigned best_i[256];
    float bv = -INFINITY;
    unsigned bi = 0;
    for (unsigned id = t; id < vocab; id += 256) {
        const float l = ink2d::bf(logits[row * vocab + id]);
        float score;
        if (inv_t == 0.0f) {
            score = l;
        } else {
            float u = ink2d::uniform24(seed, position, id);
            u = fminf(fmaxf(u, 1e-7f), 1.0f - 1e-7f);
            score = l * inv_t - logf(-logf(u));
        }
        if (score > bv || (score == bv && id < bi)) { bv = score; bi = id; }
    }
    best_v[t] = bv;
    best_i[t] = bi;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            if (best_v[t + stride] > best_v[t] ||
                (best_v[t + stride] == best_v[t] && best_i[t + stride] < best_i[t])) {
                best_v[t] = best_v[t + stride];
                best_i[t] = best_i[t + stride];
            }
        }
    }
    __syncthreads();
    if (t == 0) out_id[row] = best_i[0];
}

// Device-meta greedy top-1 (graph path only): position reserved in hold_meta[3] for a future sampled
// variant; the greedy pick ignores it, matching tf_ink2d_top1_gumbel with inv_t == 0.
extern "C" __global__ void tf_ink2d_top1_gumbel_dev(const __nv_bfloat16* logits, unsigned* out_id,
                                                     const uint32_t* hold_meta) {
    (void)hold_meta;
    const unsigned t = threadIdx.x;
    __shared__ float best_v[256];
    __shared__ unsigned best_i[256];
    float bv = -INFINITY;
    unsigned bi = 0;
    for (unsigned id = t; id < ink2d::DSP_VOCAB; id += 256) {
        const float l = ink2d::bf(logits[id]);
        if (l > bv || (l == bv && id < bi)) { bv = l; bi = id; }
    }
    best_v[t] = bv;
    best_i[t] = bi;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            if (best_v[t + stride] > best_v[t] ||
                (best_v[t + stride] == best_v[t] && best_i[t + stride] < best_i[t])) {
                best_v[t] = best_v[t + stride];
                best_i[t] = best_i[t + stride];
            }
        }
    }
    __syncthreads();
    if (t == 0) out_id[0] = best_i[0];
}

// One-time GPU binary64 fold of phi[vocab, 256] and beta[vocab] from unembed[vocab, 2304], hidden_out[2304, 256], hidden_bias[2304].
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_fold_phi_beta(
    const __nv_bfloat16* __restrict__ unembed, const __nv_bfloat16* __restrict__ hidden_out,
    const __nv_bfloat16* __restrict__ hidden_bias, __nv_bfloat16* __restrict__ phi,
    __nv_bfloat16* __restrict__ beta, unsigned vocab) {
    const unsigned v = blockIdx.x, r = threadIdx.x;
    if (v >= vocab) return;
    __shared__ float su[ink2d::DSP_H];
    for (unsigned d = r; d < ink2d::DSP_H; d += 256) su[d] = ink2d::bf(unembed[uint64_t(v) * ink2d::DSP_H + d]);
    __syncthreads();
    double acc = 0.0;
    for (unsigned d = 0; d < ink2d::DSP_H; ++d)
        acc += double(su[d]) * double(ink2d::bf(hidden_out[uint64_t(d) * ink2d::DSP_RANK + r]));
    phi[uint64_t(v) * ink2d::DSP_RANK + r] = ink2d::tobf(float(acc));
    double bpart = 0.0;
    for (unsigned d = r; d < ink2d::DSP_H; d += 256)
        bpart += double(su[d]) * double(ink2d::bf(hidden_bias[d]));
    __shared__ double sb[256];
    sb[r] = bpart;
    __syncthreads();
    if (r == 0) {
        double bsum = 0.0;
        for (unsigned i = 0; i < 256; ++i) bsum += sb[i];
        beta[v] = ink2d::tobf(float(bsum));
    }
}

// Batched row init: n_streams * 8 blocks; block s*8 + r copies anchor[s] (r==0) or mask (r>0) into rows[s*8+r].
extern "C" __global__ void tf_ink2d_init_rows_batch(const __nv_bfloat16* anchors, const __nv_bfloat16* mask,
                                                    __nv_bfloat16* rows, unsigned n_streams) {
    const unsigned b = blockIdx.x;
    if (b >= n_streams * 8) return;
    const unsigned s = b >> 3, r = b & 7;
    const __nv_bfloat16* src = (r == 0) ? (anchors + static_cast<size_t>(s) * ink2d::DSP_H) : mask;
    __nv_bfloat16* dst = rows + static_cast<size_t>(b) * ink2d::DSP_H;
    for (unsigned d = threadIdx.x; d < ink2d::DSP_H; d += 256) dst[d] = src[d];
}

// Pack draft hidden states x[s*8 + 1 + step] -> dst[s*depth + step] for one batched unembed.
extern "C" __global__ void tf_ink2d_pack_draft_x(const __nv_bfloat16* x, __nv_bfloat16* dst,
                                                 unsigned n_streams, unsigned depth) {
    const unsigned b = blockIdx.x;
    if (b >= n_streams * depth) return;
    const unsigned s = b / depth, step = b - s * depth;
    const __nv_bfloat16* src = x + static_cast<size_t>(s * 8 + 1 + step) * ink2d::DSP_H;
    __nv_bfloat16* out = dst + static_cast<size_t>(b) * ink2d::DSP_H;
    for (unsigned d = threadIdx.x; d < ink2d::DSP_H; d += 256) out[d] = src[d];
}

// ---- M9.7a: batched absorb + batched hold (S > 1). Bit-exact vs the single-stream kernels: every
// block computes the same (row, work) tile from the same inputs; only the block->stream fan-out and
// the per-stream pointer/ring lookup are new. Row partitions cover [0, total_rows) contiguously.
struct ink2d_lane_desc {
    uint64_t k[3];     // per-layer context K ring bases (device)
    uint64_t v[3];     // per-layer context V ring bases (device)
    uint64_t newest;   // lane newest row (device, kappa reads it)
    uint64_t src;      // absorb only: taps src base (device)
    uint32_t row0;     // absorb only: first fused row in the contiguous batch buffer
    uint32_t n;        // absorb only: row count for this stream
    uint32_t base_pos; // absorb only: absolute position of row0
    uint32_t ring[3];
    uint32_t _pad[2]; // -> 96 bytes
};

// One grid for all streams: block (fusion, orow) fuses stream s = owner(orow) exactly like
// tf_ink2d_fuse_taps fuses its row (same fuse/dflare weights, same formulas, same rounding).
// newest_s gets the stream's last row, mirroring `row + 1 == rows` per stream.
extern "C" __global__ void tf_ink2d_fuse_taps_batch(const ink2d_lane_desc* __restrict__ desc,
                                                      const __nv_bfloat16* __restrict__ fuse_weight,
                                                      const __nv_bfloat16* __restrict__ norm_weight,
                                                      __nv_bfloat16* __restrict__ fused,
                                                      unsigned n_streams, unsigned total_rows) {
    const unsigned b = blockIdx.x, t = threadIdx.x;
    if (b >= total_rows * 4) return;
    const unsigned fusion = b / total_rows, orow = b - fusion * total_rows;
    unsigned s = 0;
    while (s < n_streams && (orow < desc[s].row0 || orow >= desc[s].row0 + desc[s].n)) s += 1;
    if (s >= n_streams) return;
    const __nv_bfloat16* taps = reinterpret_cast<const __nv_bfloat16*>((const void*)(uintptr_t)desc[s].src);
    const unsigned lrow = orow - desc[s].row0, n = desc[s].n;
    __shared__ float alpha[4];
    __shared__ float scale;
    if (t == 0) {
        float scores[4];
        float peak = -INFINITY;
        for (unsigned j = 0; j < 4; ++j) { scores[j] = ink2d::bf(fuse_weight[fusion * 4 + j]); peak = max(peak, scores[j]); }
        float sum = 0.0f;
        for (unsigned j = 0; j < 4; ++j) { scores[j] = expf(scores[j] - peak); sum += scores[j]; }
        for (unsigned j = 0; j < 4; ++j) alpha[j] = ink2d::bf(__float2bfloat16_rn(scores[j] / sum));
        float square = 0.0f;
        for (unsigned d = 0; d < ink2d::DSP_H; ++d) {
            float mixed = 0.0f;
            for (unsigned j = 0; j < 4; ++j) mixed += alpha[j] * ink2d::bf(taps[(lrow * 4 + j) * ink2d::DSP_H + d]);
            square += mixed * mixed;
        }
        scale = rsqrtf(square / float(ink2d::DSP_H) + ink2d::DSP_EPS);
    }
    __syncthreads();
    for (unsigned d = t; d < ink2d::DSP_H; d += 256) {
        float mixed = 0.0f;
        for (unsigned j = 0; j < 4; ++j) mixed += alpha[j] * ink2d::bf(taps[(lrow * 4 + j) * ink2d::DSP_H + d]);
        const __nv_bfloat16 out = ink2d::tobf(mixed * scale * ink2d::bf(norm_weight[fusion * ink2d::DSP_H + d]));
        fused[(fusion * total_rows + orow) * ink2d::DSP_H + d] = out;
        if (fusion == 3 && lrow + 1 == n)
            reinterpret_cast<__nv_bfloat16*>((void*)(uintptr_t)desc[s].newest)[d] = out;
    }
}

// Scatter the contiguous [total_rows][KV] dense outputs into each lane's ring, one (orow, head) block
// exactly like tf_ink2d_context_kv scatters its row (same k_norm weight, same RMS/rope/rounding).
extern "C" __global__ void tf_ink2d_context_kv_batch(const __nv_bfloat16* __restrict__ ck,
                                                       const __nv_bfloat16* __restrict__ cv,
                                                       const __nv_bfloat16* __restrict__ weight,
                                                       const ink2d_lane_desc* __restrict__ desc,
                                                       unsigned layer, unsigned n_streams,
                                                       unsigned total_rows) {
    const unsigned group = blockIdx.x, d = threadIdx.x;
    if (group >= total_rows * 8) return;
    const unsigned orow = group / 8, h = group % 8;
    unsigned s = 0;
    while (s < n_streams && (orow < desc[s].row0 || orow >= desc[s].row0 + desc[s].n)) s += 1;
    if (s >= n_streams) return;
    const unsigned lrow = orow - desc[s].row0;
    const unsigned lane = d & 31, warp = d >> 5;
    __shared__ float wsum[4];
    __shared__ float scale;
    {
        float sum = 0.0f;
        for (unsigned i = d; i < ink2d::DSP_HEAD; i += 128) { const float x = ink2d::bf(ck[orow * ink2d::DSP_KV + h * ink2d::DSP_HEAD + i]); sum += x * x; }
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (d == 0) scale = rsqrtf((wsum[0] + wsum[1] + wsum[2] + wsum[3]) / float(ink2d::DSP_HEAD) + ink2d::DSP_EPS);
    }
    __syncthreads();
    const unsigned position = desc[s].base_pos + lrow;
    const unsigned ring = desc[s].ring[layer];
    __nv_bfloat16* keys = reinterpret_cast<__nv_bfloat16*>((void*)(uintptr_t)desc[s].k[layer]);
    __nv_bfloat16* values = reinterpret_cast<__nv_bfloat16*>((void*)(uintptr_t)desc[s].v[layer]);
    const unsigned at = (position % ring) * ink2d::DSP_KV + h * ink2d::DSP_HEAD + d;
    const unsigned src = orow * ink2d::DSP_KV + h * ink2d::DSP_HEAD + d;
    const unsigned partner = d < 64 ? d + 64 : d - 64;
    const float x = ink2d::bf(ink2d::tobf(ink2d::bf(ck[src]) * scale * ink2d::bf(weight[d])));
    const float other = ink2d::bf(ink2d::tobf(ink2d::bf(ck[orow * ink2d::DSP_KV + h * ink2d::DSP_HEAD + partner]) * scale * ink2d::bf(weight[partner])));
    const float angle = ink2d::rope_angle(d, position);
    const float cs = cosf(angle), sn = sinf(angle);
    keys[at] = ink2d::tobf(d < 64 ? x * cs - other * sn : other * sn + x * cs);
    values[at] = cv[src];
}

extern "C" __global__ void tf_ink2d_context_kv_batch_i8(const __nv_bfloat16* __restrict__ ck,
                                                          const __nv_bfloat16* __restrict__ cv,
                                                          const __nv_bfloat16* __restrict__ weight,
                                                          const ink2d_lane_desc* __restrict__ desc,
                                                          unsigned layer, unsigned n_streams,
                                                          unsigned total_rows) {
    const unsigned group = blockIdx.x, d = threadIdx.x;
    if (group >= total_rows * 8) return;
    const unsigned orow = group / 8, h = group % 8;
    unsigned s = 0;
    while (s < n_streams && (orow < desc[s].row0 || orow >= desc[s].row0 + desc[s].n)) s += 1;
    if (s >= n_streams) return;
    const unsigned lrow = orow - desc[s].row0;
    const unsigned lane = d & 31, warp = d >> 5;
    __shared__ float wsum[4];
    __shared__ float scale;
    __shared__ float amax_k_w[4], amax_v_w[4];
    __shared__ float k_inv, v_inv;
    {
        float sum = 0.0f;
        for (unsigned i = d; i < ink2d::DSP_HEAD; i += 128) {
            const float x = ink2d::bf(ck[orow * ink2d::DSP_KV + h * ink2d::DSP_HEAD + i]);
            sum += x * x;
        }
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (d == 0) scale = rsqrtf((wsum[0] + wsum[1] + wsum[2] + wsum[3]) / float(ink2d::DSP_HEAD) + ink2d::DSP_EPS);
    }
    __syncthreads();
    const unsigned position = desc[s].base_pos + lrow;
    const unsigned ring = desc[s].ring[layer];
    int8_t* keys = reinterpret_cast<int8_t*>((void*)(uintptr_t)desc[s].k[layer]);
    int8_t* values = reinterpret_cast<int8_t*>((void*)(uintptr_t)desc[s].v[layer]);
    const size_t slot = (size_t)(position % ring) * 8 + h;
    const size_t at = (size_t)(position % ring) * ink2d::DSP_KV + h * ink2d::DSP_HEAD + d;
    const unsigned src = orow * ink2d::DSP_KV + h * ink2d::DSP_HEAD + d;
    const unsigned partner = d < 64 ? d + 64 : d - 64;
    const float x = ink2d::bf(ink2d::tobf(ink2d::bf(ck[src]) * scale * ink2d::bf(weight[d])));
    const float other = ink2d::bf(ink2d::tobf(ink2d::bf(ck[orow * ink2d::DSP_KV + h * ink2d::DSP_HEAD + partner]) * scale * ink2d::bf(weight[partner])));
    const float angle = ink2d::rope_angle(d, position);
    const float cs = cosf(angle), sn = sinf(angle);
    const float kf = ink2d::bf(ink2d::tobf(d < 64 ? x * cs - other * sn : other * sn + x * cs));
    const float vf = ink2d::bf(cv[src]);
    float ak = fabsf(kf), av = fabsf(vf);
    for (int o = 16; o > 0; o >>= 1) {
        ak = fmaxf(ak, __shfl_xor_sync(0xffffffffu, ak, o));
        av = fmaxf(av, __shfl_xor_sync(0xffffffffu, av, o));
    }
    if (lane == 0) { amax_k_w[warp] = ak; amax_v_w[warp] = av; }
    __syncthreads();
    if (d == 0) {
        const float amax_k = fmaxf(fmaxf(amax_k_w[0], amax_k_w[1]), fmaxf(amax_k_w[2], amax_k_w[3]));
        const float amax_v = fmaxf(fmaxf(amax_v_w[0], amax_v_w[1]), fmaxf(amax_v_w[2], amax_v_w[3]));
        float* k_scales = reinterpret_cast<float*>(keys + (size_t)ring * ink2d::DSP_KV);
        float* v_scales = reinterpret_cast<float*>(values + (size_t)ring * ink2d::DSP_KV);
        k_scales[slot] = amax_k > 0.0f ? amax_k / 127.0f : 0.0f;
        v_scales[slot] = amax_v > 0.0f ? amax_v / 127.0f : 0.0f;
        k_inv = amax_k > 0.0f ? 127.0f / amax_k : 0.0f;
        v_inv = amax_v > 0.0f ? 127.0f / amax_v : 0.0f;
    }
    __syncthreads();
    int qk = __float2int_rn(kf * k_inv);
    if (qk > 127) qk = 127; else if (qk < -127) qk = -127;
    int qv = __float2int_rn(vf * v_inv);
    if (qv > 127) qv = 127; else if (qv < -127) qv = -127;
    keys[at] = static_cast<int8_t>(qk);
    values[at] = static_cast<int8_t>(qv);
}

// S streams x 320 groups: block (s, group) is stream s's old block `group` (same q_norm/k_norm,
// same anchor from hold_meta, same row/work math and rounding). Strides: 8 rows a stream everywhere.
extern "C" __global__ void tf_ink2d_block_qkv_dev_batch(const __nv_bfloat16* __restrict__ q_raw_b,
                                                          const __nv_bfloat16* __restrict__ k_raw_b,
                                                          const __nv_bfloat16* __restrict__ v_raw_b,
                                                          const __nv_bfloat16* __restrict__ q_weight,
                                                          const __nv_bfloat16* __restrict__ k_weight,
                                                          __nv_bfloat16* __restrict__ q_b,
                                                          __nv_bfloat16* __restrict__ kb_b,
                                                          __nv_bfloat16* __restrict__ vb_b,
                                                          const uint32_t* __restrict__ hold_meta_b) {
    const unsigned b = blockIdx.x;
    const unsigned s = b / 320, group = b - s * 320;
    const unsigned row = group / 40, work = group % 40;
    const unsigned d = threadIdx.x;
    const unsigned lane = d & 31, warp = d >> 5;
    const unsigned base_position = hold_meta_b[s * 4];
    const __nv_bfloat16* q_raw = q_raw_b + (size_t)s * 8 * ink2d::DSP_Q;
    const __nv_bfloat16* k_raw = k_raw_b + (size_t)s * 8 * ink2d::DSP_KV;
    const __nv_bfloat16* v_raw = v_raw_b + (size_t)s * 8 * ink2d::DSP_KV;
    __nv_bfloat16* q = q_b + (size_t)s * 8 * ink2d::DSP_Q;
    __nv_bfloat16* k = kb_b + (size_t)s * 8 * ink2d::DSP_KV;
    __nv_bfloat16* v = vb_b + (size_t)s * 8 * ink2d::DSP_KV;
    __shared__ float wsum[4];
    __shared__ float scale;
    if (work < 32) {
        float sum = 0.0f;
        if (work < 24) {
            for (unsigned i = d; i < ink2d::DSP_HEAD; i += 128) { const float x = ink2d::bf(q_raw[row * ink2d::DSP_Q + work * ink2d::DSP_HEAD + i]); sum += x * x; }
        } else {
            const unsigned h = work - 24;
            for (unsigned i = d; i < ink2d::DSP_HEAD; i += 128) { const float x = ink2d::bf(k_raw[row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + i]); sum += x * x; }
        }
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (d == 0) scale = rsqrtf((wsum[0] + wsum[1] + wsum[2] + wsum[3]) / float(ink2d::DSP_HEAD) + ink2d::DSP_EPS);
    }
    __syncthreads();
    if (work >= 32) {
        const unsigned h = work - 32;
        v[row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + d] = v_raw[row * ink2d::DSP_KV + h * ink2d::DSP_HEAD + d];
        return;
    }
    const bool query = work < 24;
    const unsigned h = query ? work : work - 24;
    const unsigned at = row * (query ? ink2d::DSP_Q : ink2d::DSP_KV) + h * ink2d::DSP_HEAD + d;
    const float x = ink2d::bf(ink2d::tobf(ink2d::bf(query ? q_raw[at] : k_raw[at]) * scale * ink2d::bf(query ? q_weight[d] : k_weight[d])));
    const unsigned partner = d < 64 ? d + 64 : d - 64;
    const unsigned other_at = row * (query ? ink2d::DSP_Q : ink2d::DSP_KV) + h * ink2d::DSP_HEAD + partner;
    const float other = ink2d::bf(ink2d::tobf(ink2d::bf(query ? q_raw[other_at] : k_raw[other_at]) * scale * ink2d::bf(query ? q_weight[partner] : k_weight[partner])));
    const float angle = ink2d::rope_angle(d, base_position + row);
    const float cs = cosf(angle), sn = sinf(angle);
    const float rotated = d < 64 ? x * cs - other * sn : other * sn + x * cs;
    if (query) q[at] = ink2d::tobf(rotated); else k[at] = ink2d::tobf(rotated);
}

// S streams x 192 groups: block (s, group) is stream s's old attention block (same committed/from
// from hold_meta, same ring/lane caches from desc, same online-softmax math). scores stays ABI-only.
extern "C" __global__ void tf_ink2d_attention_dev_batch(const __nv_bfloat16* __restrict__ q_b,
                                                          const __nv_bfloat16* __restrict__ kb_b,
                                                          const __nv_bfloat16* __restrict__ vb_b,
                                                          const ink2d_lane_desc* __restrict__ desc,
                                                          float* scores, __nv_bfloat16* __restrict__ attn_b,
                                                          const uint32_t* __restrict__ hold_meta_b,
                                                          unsigned layer, unsigned local,
                                                          unsigned score_stride) {
    (void)scores;
    (void)score_stride;
    const unsigned b = blockIdx.x;
    const unsigned s = b / 192, group = b - s * 192;
    const unsigned d = threadIdx.x;
    ink2d_attention_online(q_b + (size_t)s * 8 * ink2d::DSP_Q, kb_b + (size_t)s * 8 * ink2d::DSP_KV,
                            vb_b + (size_t)s * 8 * ink2d::DSP_KV,
                            reinterpret_cast<const __nv_bfloat16*>((const void*)(uintptr_t)desc[s].k[layer]),
                            reinterpret_cast<const __nv_bfloat16*>((const void*)(uintptr_t)desc[s].v[layer]),
                            attn_b + (size_t)s * 8 * ink2d::DSP_Q, hold_meta_b[s * 4 + 1],
                            desc[s].ring[layer], local, hold_meta_b[s * 4 + 2], group / 24, group % 24, d);
}

extern "C" __global__ void tf_ink2d_attention_dev_batch_i8(const __nv_bfloat16* __restrict__ q_b,
                                                             const __nv_bfloat16* __restrict__ kb_b,
                                                             const __nv_bfloat16* __restrict__ vb_b,
                                                             const ink2d_lane_desc* __restrict__ desc,
                                                             float* scores, __nv_bfloat16* __restrict__ attn_b,
                                                             const uint32_t* __restrict__ hold_meta_b,
                                                             unsigned layer, unsigned local,
                                                             unsigned score_stride) {
    (void)scores;
    (void)score_stride;
    const unsigned b = blockIdx.x;
    const unsigned s = b / 192, group = b - s * 192;
    const unsigned d = threadIdx.x;
    ink2d_attention_online_i8(q_b + (size_t)s * 8 * ink2d::DSP_Q, kb_b + (size_t)s * 8 * ink2d::DSP_KV,
                              vb_b + (size_t)s * 8 * ink2d::DSP_KV,
                              reinterpret_cast<const int8_t*>((const void*)(uintptr_t)desc[s].k[layer]),
                              reinterpret_cast<const int8_t*>((const void*)(uintptr_t)desc[s].v[layer]),
                              attn_b + (size_t)s * 8 * ink2d::DSP_Q, hold_meta_b[s * 4 + 1],
                              desc[s].ring[layer], local, hold_meta_b[s * 4 + 2], group / 24, group % 24, d);
}

extern "C" __global__ void tf_ink2d_attention_splitk_dev_batch_i8(
    const __nv_bfloat16* __restrict__ q_b, const __nv_bfloat16* __restrict__ kb_b,
    const __nv_bfloat16* __restrict__ vb_b, const ink2d_lane_desc* __restrict__ desc,
    float* __restrict__ part_o, float2* __restrict__ part_ml, const uint32_t* __restrict__ hold_meta_b) {
    const unsigned s = blockIdx.x / (8 * ink2d::DSP_SPLITS);
    const unsigned rem = blockIdx.x % (8 * ink2d::DSP_SPLITS);
    const unsigned kv = rem / ink2d::DSP_SPLITS;
    const unsigned sp = rem % ink2d::DSP_SPLITS;
    ink2d_attention_splitk_i8_body(
        q_b + (size_t)s * 8 * ink2d::DSP_Q,
        kb_b + (size_t)s * 8 * ink2d::DSP_KV,
        vb_b + (size_t)s * 8 * ink2d::DSP_KV,
        reinterpret_cast<const int8_t*>((const void*)(uintptr_t)desc[s].k[1]),
        reinterpret_cast<const int8_t*>((const void*)(uintptr_t)desc[s].v[1]),
        part_o, part_ml, hold_meta_b[s * 4 + 1], desc[s].ring[1], hold_meta_b[s * 4 + 2], kv, sp, s);
}

// Kappa rows for all (stream, step) pairs in one grid: element i is old kernel's element i of the
// (s, step) row (same y rows [s*8+1+step], same lane newest f, same weight/bias math and rounding).
extern "C" __global__ void tf_ink2d_kappa_batch(const __nv_bfloat16* __restrict__ x_b,
                                                  const ink2d_lane_desc* __restrict__ desc,
                                                  const __nv_bfloat16* __restrict__ weight,
                                                  const __nv_bfloat16* __restrict__ bias, float* __restrict__ out,
                                                  unsigned n_streams, unsigned max_d) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_streams * max_d * ink2d::DSP_RNN) return;
    const unsigned im = i / ink2d::DSP_RNN, channel = i - im * ink2d::DSP_RNN;
    const unsigned s = im / max_d, step = im - s * max_d;
    const __nv_bfloat16* y = x_b + (size_t)(s * 8 + 1 + step) * ink2d::DSP_H;
    const __nv_bfloat16* f = reinterpret_cast<const __nv_bfloat16*>((const void*)(uintptr_t)desc[s].newest);
    float sum = 0.0f;
    for (unsigned dd = 0; dd < ink2d::DSP_H; ++dd) sum += ink2d::bf(y[dd]) * ink2d::bf(weight[channel * 5120 + 512 + dd]);
    for (unsigned dd = 0; dd < ink2d::DSP_H; ++dd) sum += ink2d::bf(f[dd]) * ink2d::bf(weight[channel * 5120 + 2816 + dd]);
    sum += ink2d::bf(bias[channel]);
    out[i] = sum;
}

// One block a stream (grid S): block s is stream s's old recurrent_dev step (same state/kappa row,
// same token, same markov row, same preloads, same GRU/lambda math and rounding).
extern "C" __global__ void tf_ink2d_recurrent_dev_batch(const __nv_bfloat16* __restrict__ state_b,
                                                          const float* __restrict__ kappa_b,
                                                          const __nv_bfloat16* __restrict__ rnn_weight,
                                                          const __nv_bfloat16* __restrict__ markov,
                                                          __nv_bfloat16* __restrict__ z_b,
                                                          __nv_bfloat16* __restrict__ next_b,
                                                          __nv_bfloat16* __restrict__ lambda_b,
                                                          const unsigned* __restrict__ tok_b, unsigned max_d,
                                                          unsigned step) {
    const unsigned s = blockIdx.x, t = threadIdx.x;
    __shared__ float sh_state[ink2d::DSP_RANK];
    __shared__ float sh_mark[ink2d::DSP_RANK];
    const unsigned token = tok_b[s * 8 + step];
    sh_state[t] = ink2d::bf(state_b[(size_t)s * ink2d::DSP_RANK + t]);
    sh_mark[t] = ink2d::bf(markov[uint64_t(token) * ink2d::DSP_RANK + t]);
    __syncthreads();
    const unsigned i = t;
    float values[3];
    const float* kappa = kappa_b + (size_t)(s * max_d + step) * ink2d::DSP_RNN;
    __nv_bfloat16* z = z_b + (size_t)s * ink2d::DSP_RNN;
    __nv_bfloat16* next_state = next_b + (size_t)s * ink2d::DSP_RANK;
    __nv_bfloat16* lambda = lambda_b + (size_t)s * ink2d::DSP_RANK;
    for (unsigned part = 0; part < 3; ++part) {
        const unsigned channel = part * ink2d::DSP_RANK + i;
        float sum = 0.0f;
        for (unsigned d = 0; d < ink2d::DSP_RANK; ++d) sum += sh_state[d] * ink2d::bf(rnn_weight[channel * 5120 + d]);
        for (unsigned d = 0; d < ink2d::DSP_RANK; ++d) sum += sh_mark[d] * ink2d::bf(rnn_weight[channel * 5120 + 256 + d]);
        sum += kappa[channel];
        const __nv_bfloat16 rounded = ink2d::tobf(sum);
        z[channel] = rounded;
        values[part] = ink2d::bf(rounded);
    }
    const __nv_bfloat16 gamma = ink2d::tobf(1.0f / (1.0f + expf(-values[0])));
    const __nv_bfloat16 candidate = ink2d::tobf(tanhf(values[1]));
    const __nv_bfloat16 keep = ink2d::tobf(1.0f - ink2d::bf(gamma));
    const __nv_bfloat16 left = ink2d::tobf(ink2d::bf(gamma) * sh_state[i]);
    const __nv_bfloat16 right = ink2d::tobf(ink2d::bf(keep) * ink2d::bf(candidate));
    next_state[i] = ink2d::tobf(ink2d::bf(left) + ink2d::bf(right));
    lambda[i] = ink2d::tobf(tanhf(values[2]));
}

// One block a stream (grid S, greedy only): block s is stream s's old top1_gumbel_dev pick over its
// own vocab row (same argmax/tie-break, same hold_meta-free math).
extern "C" __global__ void tf_ink2d_top1_gumbel_dev_batch(const __nv_bfloat16* __restrict__ logits_b,
                                                             unsigned* __restrict__ out_b, unsigned step,
                                                             unsigned n_streams) {
    (void)n_streams;
    const unsigned s = blockIdx.x;
    const __nv_bfloat16* logits = logits_b + (size_t)s * ink2d::DSP_VOCAB;
    unsigned* out_id = out_b + s * 8 + step + 1;
    const unsigned t = threadIdx.x;
    __shared__ float best_v[256];
    __shared__ unsigned best_i[256];
    float bv = -INFINITY;
    unsigned bi = 0;
    for (unsigned id = t; id < ink2d::DSP_VOCAB; id += 256) {
        const float l = ink2d::bf(logits[id]);
        if (l > bv || (l == bv && id < bi)) { bv = l; bi = id; }
    }
    best_v[t] = bv;
    best_i[t] = bi;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            if (best_v[t + stride] > best_v[t] ||
                (best_v[t + stride] == best_v[t] && best_i[t + stride] < best_i[t])) {
                best_v[t] = best_v[t + stride];
                best_i[t] = best_i[t + stride];
            }
        }
    }
    __syncthreads();
    if (t == 0) out_id[0] = best_i[0];
}

// Batched warp-per-row FP8 logits: reads phi_fp8 ONCE across up to 16 streams.
extern "C" __global__ void tf_ink2d_logits_fp8_batch(const __nv_bfloat16* base, const uint8_t* phi_fp8,
                                                       const float* phi_scale, const __nv_bfloat16* beta,
                                                       const float* f8lut, const __nv_bfloat16* lambda,
                                                       __nv_bfloat16* out, unsigned vocab, unsigned n_streams,
                                                       unsigned depth, unsigned step) {
    __shared__ float slam[16][ink2d::DSP_RANK];
    __shared__ float slut[256];
    slut[threadIdx.x] = f8lut[threadIdx.x];
    for (unsigned s = 0; s < n_streams; ++s)
        slam[s][threadIdx.x] = ink2d::bf(lambda[static_cast<size_t>(s) * ink2d::DSP_RANK + threadIdx.x]);
    __syncthreads();
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned v = blockIdx.x * 8 + warp;
    if (v >= vocab) return;
    const uint2 pack = reinterpret_cast<const uint2*>(phi_fp8 + static_cast<uint64_t>(v) * ink2d::DSP_RANK + lane * 8)[0];
    const uint8_t* e = reinterpret_cast<const uint8_t*>(&pack);
    float w[8];
    for (unsigned j = 0; j < 8; ++j) w[j] = slut[e[j]];
    const float sc = phi_scale[v];
    const float bv = ink2d::bf(beta[v]);
    for (unsigned s = 0; s < n_streams; ++s) {
        float dot = 0.0f;
        for (unsigned j = 0; j < 8; ++j) dot += w[j] * slam[s][lane * 8 + j];
        for (unsigned o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if (lane == 0) {
            const float scaled = dot * sc;
            const __nv_bfloat16 bias = ink2d::tobf(bv + scaled);
            const __nv_bfloat16 logit = ink2d::tobf(ink2d::bf(base[(static_cast<size_t>(s) * depth + step) * vocab + v]) + ink2d::bf(bias));
            out[static_cast<size_t>(s) * vocab + v] = ink2d::tobf(30.0f * tanhf(ink2d::bf(logit) / 30.0f));
        }
    }
}

// M10b: Single-pass top-1 + exact softmax probability + top1-top2 margin (bit-identical top1_id).
extern "C" __global__ void tf_ink2d_top1_prob_dev(const __nv_bfloat16* logits, unsigned* out_id,
                                                  float* out_prob, float* out_margin,
                                                  const uint32_t* hold_meta) {
    (void)hold_meta;
    const unsigned t = threadIdx.x;
    __shared__ float best_v[256];
    __shared__ float best_v2[256];
    __shared__ unsigned best_i[256];
    __shared__ float sm_m[256];
    __shared__ float sm_s[256];
    float bv = -INFINITY, bv2 = -INFINITY;
    unsigned bi = 0;
    float lse_m = -INFINITY, lse_s = 0.0f;
    for (unsigned id = t; id < ink2d::DSP_VOCAB; id += 256) {
        const float l = ink2d::bf(logits[id]);
        if (l > bv || (l == bv && id < bi)) { bv2 = bv; bv = l; bi = id; }
        else if (l > bv2) { bv2 = l; }
        if (l > lse_m) {
            lse_s = (lse_s > 0.0f ? lse_s * expf(lse_m - l) : 0.0f) + 1.0f;
            lse_m = l;
        } else {
            lse_s += expf(l - lse_m);
        }
    }
    best_v[t] = bv;
    best_v2[t] = bv2;
    best_i[t] = bi;
    sm_m[t] = lse_m;
    sm_s[t] = lse_s;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            const unsigned r = t + stride;
            if (best_v[r] > best_v[t] || (best_v[r] == best_v[t] && best_i[r] < best_i[t])) {
                best_v2[t] = fmaxf(best_v[t], best_v2[r]);
                best_v[t] = best_v[r];
                best_i[t] = best_i[r];
            } else {
                best_v2[t] = fmaxf(best_v2[t], best_v[r]);
            }
            const float m = fmaxf(sm_m[t], sm_m[r]);
            sm_s[t] = sm_s[t] * expf(sm_m[t] - m) + sm_s[r] * expf(sm_m[r] - m);
            sm_m[t] = m;
        }
    }
    __syncthreads();
    if (t == 0) {
        out_id[0] = best_i[0];
        out_prob[0] = (sm_s[0] > 0.0f) ? (expf(best_v[0] - sm_m[0]) / sm_s[0]) : 0.0f;
        out_margin[0] = best_v[0] - best_v2[0];
    }
}

extern "C" __global__ void tf_ink2d_top1_prob_dev_batch(const __nv_bfloat16* __restrict__ logits_b,
                                                        unsigned* __restrict__ out_b,
                                                        float* __restrict__ probs_b,
                                                        float* __restrict__ margins_b,
                                                        unsigned step, unsigned n_streams) {
    (void)n_streams;
    const unsigned s = blockIdx.x;
    const __nv_bfloat16* logits = logits_b + static_cast<size_t>(s) * ink2d::DSP_VOCAB;
    const unsigned t = threadIdx.x;
    __shared__ float best_v[256];
    __shared__ float best_v2[256];
    __shared__ unsigned best_i[256];
    __shared__ float sm_m[256];
    __shared__ float sm_s[256];
    float bv = -INFINITY, bv2 = -INFINITY;
    unsigned bi = 0;
    float lse_m = -INFINITY, lse_s = 0.0f;
    for (unsigned id = t; id < ink2d::DSP_VOCAB; id += 256) {
        const float l = ink2d::bf(logits[id]);
        if (l > bv || (l == bv && id < bi)) { bv2 = bv; bv = l; bi = id; }
        else if (l > bv2) { bv2 = l; }
        if (l > lse_m) {
            lse_s = (lse_s > 0.0f ? lse_s * expf(lse_m - l) : 0.0f) + 1.0f;
            lse_m = l;
        } else {
            lse_s += expf(l - lse_m);
        }
    }
    best_v[t] = bv;
    best_v2[t] = bv2;
    best_i[t] = bi;
    sm_m[t] = lse_m;
    sm_s[t] = lse_s;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            const unsigned r = t + stride;
            if (best_v[r] > best_v[t] || (best_v[r] == best_v[t] && best_i[r] < best_i[t])) {
                best_v2[t] = fmaxf(best_v[t], best_v2[r]);
                best_v[t] = best_v[r];
                best_i[t] = best_i[r];
            } else {
                best_v2[t] = fmaxf(best_v2[t], best_v[r]);
            }
            const float m = fmaxf(sm_m[t], sm_m[r]);
            sm_s[t] = sm_s[t] * expf(sm_m[t] - m) + sm_s[r] * expf(sm_m[r] - m);
            sm_m[t] = m;
        }
    }
    __syncthreads();
    if (t == 0) {
        out_b[s * 8 + step + 1] = best_i[0];
        probs_b[s * 8 + step] = (sm_s[0] > 0.0f) ? (expf(best_v[0] - sm_m[0]) / sm_s[0]) : 0.0f;
        margins_b[s * 8 + step] = best_v[0] - best_v2[0];
    }
}

// M10a (dump-only): extract top-16 (token_id, logit) pairs per stream for a draft step.
extern "C" __global__ void tf_ink2d_top16(const __nv_bfloat16* __restrict__ logits_b,
                                          unsigned* __restrict__ top_toks,
                                          float* __restrict__ top_vals,
                                          unsigned step, unsigned max_d) {
    const unsigned s = blockIdx.x;
    const __nv_bfloat16* logits = logits_b + static_cast<size_t>(s) * ink2d::DSP_VOCAB;
    const unsigned t = threadIdx.x;
    __shared__ unsigned picked[16];
    __shared__ float best_v[256];
    __shared__ unsigned best_i[256];
    for (unsigned k = 0; k < 16; ++k) {
        float bv = -INFINITY;
        unsigned bi = 0;
        for (unsigned id = t; id < ink2d::DSP_VOCAB; id += 256) {
            bool skip = false;
            for (unsigned j = 0; j < k; ++j) {
                if (id == picked[j]) { skip = true; break; }
            }
            if (skip) continue;
            const float l = ink2d::bf(logits[id]);
            if (l > bv || (l == bv && id < bi)) { bv = l; bi = id; }
        }
        best_v[t] = bv;
        best_i[t] = bi;
        __syncthreads();
        for (unsigned stride = 128; stride > 0; stride >>= 1) {
            __syncthreads();
            if (t < stride) {
                const unsigned r = t + stride;
                if (best_v[r] > best_v[t] || (best_v[r] == best_v[t] && best_i[r] < best_i[t])) {
                    best_v[t] = best_v[r];
                    best_i[t] = best_i[r];
                }
            }
        }
        __syncthreads();
        if (t == 0) {
            picked[k] = best_i[0];
            const size_t out_idx = (static_cast<size_t>(s) * max_d + step) * 16 + k;
            top_toks[out_idx] = best_i[0];
            top_vals[out_idx] = best_v[0];
        }
        __syncthreads();
    }
}
