// Drama 3 tower kernels after sglang-omni ab095ef9 (decoder_layers.py, audio_decoder.py, sampling.py).
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

using bf16 = __nv_bfloat16;
__device__ inline float f(bf16 x) { return __bfloat162float(x); }
__device__ inline bf16 b(float x) { return __float2bfloat16_rn(x); }
__device__ inline bool semantic(int x, int begin, int end) { return x >= begin && x <= end; }
__device__ inline int code(int x) { return max(0, min(4095, x)); }

__device__ float sum_block(float x, float* scratch) {
    const int t = threadIdx.x;
    scratch[t] = x;
    __syncthreads();
    for (int n = blockDim.x / 2; n; n >>= 1) {
        if (t < n) scratch[t] += scratch[t + n];
        __syncthreads();
    }
    return scratch[0];
}

// Frame rows are [main token, code0..code9]; sums keep the reference's bf16 embedding-bag output and bf16 adds.
extern "C" __global__ void drama_frame_embeddings(
    const int* frames, const bf16* table, bf16* backbone, bf16* shared,
    int hidden, int begin, int end) {
    const int row = blockIdx.x;
    const int* frame = frames + row * 11;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float extra = 0.0f, rest = 0.0f;
        if (semantic(frame[0], begin, end)) {
            extra = f(table[(4096 + code(frame[2])) * hidden + j]);
            for (int c = 2; c < 10; ++c)
                rest += f(table[(c * 4096 + code(frame[1 + c])) * hidden + j]);
        }
        backbone[row * hidden + j] = b(extra);
        shared[row * hidden + j] = b(rest);
    }
}

extern "C" __global__ void drama_mix(
    const bf16* hidden_in, const bf16* embedding, const int* labels,
    const bf16* shared_input, const bf16* enorm, const bf16* hnorm,
    bf16* joined, int width, int code_index, int begin, int end, float eps) {
    __shared__ float sums[2];
    const int row = blockIdx.x, t = threadIdx.x;
    const int* label = labels + row * 11;
    const bool speech = semantic(label[0], begin, end);
    const int id = code(label[1 + code_index]);
    // CuTe's width-2048 RMS path: 8 contiguous elements a lane, FP32 FMA, then an ascending XOR-warp reduction.
    if (t < 32) {
        float e2 = 0, h2 = 0;
        for (int block = 0; block < width; block += 256) {
            for (int v = 0; v < 8; ++v) {
                const int j = block + 8 * t + v;
                float e = speech ? f(embedding[id * width + j]) : 0;
                if (shared_input) e = f(b(e + f(shared_input[row * width + j])));
                const float h = f(hidden_in[row * width + j]);
                e2 = fmaf(e, e, e2);
                h2 = fmaf(h, h, h2);
            }
        }
        for (int offset = 1; offset < 32; offset <<= 1) {
            e2 += __shfl_xor_sync(0xffffffffu, e2, offset);
            h2 += __shfl_xor_sync(0xffffffffu, h2, offset);
        }
        if (t == 0) {
            sums[0] = e2;
            sums[1] = h2;
        }
    }
    __syncthreads();
    const float er = rsqrtf(sums[0] / width + eps);
    const float hr = rsqrtf(sums[1] / width + eps);
    for (int j = t; j < width; j += blockDim.x) {
        float e = speech ? f(embedding[id * width + j]) : 0;
        if (shared_input) e = f(b(e + f(shared_input[row * width + j])));
        joined[row * 2 * width + j] = b(e * er * (1.0f + f(enorm[j])));
        joined[row * 2 * width + width + j] =
            b(f(hidden_in[row * width + j]) * hr * (1.0f + f(hnorm[j])));
    }
}

// Gemma RMSNorm with optional fused residual (stored bf16, normalized from the fp32 sum) in FlashInfer's CuTe order.
extern "C" __global__ void drama_norm(
    const bf16* x, const bf16* residual, const bf16* weight,
    bf16* normalized, bf16* residual_out, int width, float eps) {
    __shared__ float square_sum;
    const int base = blockIdx.x * width, t = threadIdx.x;
    if (t < 32) {
        float square = 0;
        for (int block = 0; block < width; block += 256) {
            for (int v = 0; v < 8; ++v) {
                const int j = block + 8 * t + v;
                const float value = residual ? f(x[base + j]) + f(residual[base + j]) : f(x[base + j]);
                square = fmaf(value, value, square);
            }
        }
        for (int offset = 1; offset < 32; offset <<= 1)
            square += __shfl_xor_sync(0xffffffffu, square, offset);
        if (t == 0) square_sum = square;
    }
    __syncthreads();
    const float r = rsqrtf(square_sum / width + eps);
    for (int j = t; j < width; j += blockDim.x) {
        const float v = residual ? f(x[base + j]) + f(residual[base + j]) : f(x[base + j]);
        if (residual_out) residual_out[base + j] = b(v);
        normalized[base + j] = b(v * r * (1.0f + f(weight[j])));
    }
}

// Q/K Gemma RMSNorm then interleaved RoPE in the reference's arithmetic: CuTe sum order and cos/sin angle rounding.
extern "C" __global__ void drama_rope_cache(
    const bf16* qkv, const bf16* qnorm, const bf16* knorm,
    bf16* queries, bf16* keys, bf16* values,
    int start, int qheads, int kvheads, int dim, int rotary_dim,
    float rope_base, float eps) {
    __shared__ float square_sum;
    const int row = blockIdx.x / (qheads + kvheads);
    const int h = blockIdx.x % (qheads + kvheads), t = threadIdx.x;
    const bool is_q = h < qheads;
    const int kh = h - qheads;
    const int qw = qheads * dim, kw = kvheads * dim;
    const bf16* input = qkv + row * (2 * qw + 2 * kw) + (is_q ? h * dim : 2 * qw + kh * dim);
    const bf16* weight = is_q ? qnorm : knorm;
    if (t < 32) {
        float square = 0;
        for (int block = 0; block < dim; block += 256) {
            for (int v = 0; v < 8; ++v) {
                const float value = f(input[block + 8 * t + v]);
                square = fmaf(value, value, square);
            }
        }
        for (int offset = 1; offset < 32; offset <<= 1)
            square += __shfl_xor_sync(0xffffffffu, square, offset);
        if (t == 0) square_sum = square;
    }
    __syncthreads();
    const float r = rsqrtf(square_sum / dim + eps);
    if (t < dim) {
        float x = f(b(f(input[t]) * r * (1.0f + f(weight[t]))));
        if (t < rotary_dim) {
            const int other = t ^ 1;
            const float y = f(b(f(input[other]) * r * (1.0f + f(weight[other]))));
            const float inverse = __frcp_rn(powf(rope_base, float(t & ~1) / rotary_dim));
            const float angle = __fmul_rn(float(start + row), inverse);
            const float co = cosf(angle), si = sinf(angle);
            x = (t & 1) ? fmaf(x, co, __fmul_rn(y, si)) : fmaf(x, co, -__fmul_rn(y, si));
        }
        if (is_q) queries[row * qw + h * dim + t] = b(x);
        else {
            keys[(start + row) * kw + kh * dim + t] = b(x);
            values[(start + row) * kw + kh * dim + t] = qkv[row * (2 * qw + 2 * kw) + 2 * qw + kw + kh * dim + t];
        }
    }
}

// Eight keys per tile, FP32 online softmax. One CTA per query head and row.
extern "C" __global__ void drama_attention(
    const bf16* query, const bf16* keys, const bf16* values, const bf16* qkv,
    bf16* out, int start, int qheads, int kvheads, int dim) {
    __shared__ float scores[8], probabilities[8], scale, running_max, denominator;
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    const int row = blockIdx.x / qheads, head = blockIdx.x % qheads;
    const int kh = head / (qheads / kvheads), qw = qheads * dim, kw = kvheads * dim;
    const int length = start + row + 1;
    if (t == 0) { running_max = -INFINITY; denominator = 0; }
    __syncthreads();
    float accum = 0;
    for (int first = 0; first < length; first += 8) {
        const int at = first + warp;
        float dot = 0;
        if (at < length)
            for (int j = lane; j < dim; j += 32)
                dot += f(query[row * qw + head * dim + j]) * f(keys[at * kw + kh * dim + j]);
        for (int d = 16; d; d >>= 1) dot += __shfl_down_sync(0xffffffff, dot, d);
        if (lane == 0) scores[warp] = at < length ? dot * rsqrtf(float(dim)) : -INFINITY;
        __syncthreads();
        if (t == 0) {
            float next_max = running_max;
            for (int i = 0; i < 8; ++i) next_max = fmaxf(next_max, scores[i]);
            scale = expf(running_max - next_max);
            float mass = 0;
            for (int i = 0; i < 8; ++i) {
                probabilities[i] = expf(scores[i] - next_max);
                mass += probabilities[i];
            }
            denominator = denominator * scale + mass;
            running_max = next_max;
        }
        __syncthreads();
        if (t < dim) {
            accum *= scale;
            for (int i = 0; i < 8 && first + i < length; ++i)
                accum += probabilities[i] * f(values[(first + i) * kw + kh * dim + t]);
        }
        __syncthreads();
    }
    if (t < dim) {
        const bf16 attended = b(accum / denominator);
        const float gate = f(qkv[row * (2 * qw + 2 * kw) + qw + head * dim + t]);
        const bf16 gated = b(1.0f / (1.0f + expf(-gate)));
        out[row * qw + head * dim + t] = b(f(attended) * f(gated));
    }
}

// drama_attention for the one decode row, its keys scored across a cluster of attention_split CTAs, same arithmetic.
constexpr int attention_split = 4, attention_keys = 8192, attention_ahead = 6;

__device__ inline void copy_async(void* shared, const void* global) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(unsigned(__cvta_generic_to_shared(shared))), "l"(global) : "memory");
}

extern "C" __global__ void __cluster_dims__(attention_split, 1, 1) __launch_bounds__(256) drama_attention_decode(
    const bf16* query, const bf16* keys, const bf16* values, const bf16* qkv,
    bf16* out, int start, int qheads, int kvheads, int dim) {
    __shared__ float scores[attention_keys], maxima[attention_keys / 8], scales[attention_keys / 8], denominator;
    // The value lanes' own BF16 pairs, attention_ahead tiles of copies in flight ahead of their sums.
    __shared__ __nv_bfloat162 ring[attention_ahead][8][32];
    const cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    const int t = threadIdx.x, warp = t / 32, lane = t % 32, warps = blockDim.x / 32;
    const int part = cluster.block_rank(), row = blockIdx.x / attention_split / qheads, head = blockIdx.x / attention_split % qheads;
    const int kh = head / (qheads / kvheads), qw = qheads * dim, kw = kvheads * dim;
    const int length = start + row + 1, tiles = (length + 7) / 8;
    const int share = (tiles + attention_split - 1) / attention_split * 8, last = min(8 * tiles, (part + 1) * share);
    const int lanes = dim / attention_split / 2, j = part * 2 * lanes + 2 * t;
    const auto fetch = [&](int tile) {
        for (int i = 0; i < 8; ++i)
            if (8 * tile + i < length) copy_async(&ring[tile % attention_ahead][i][t], values + (8 * tile + i) * kw + kh * dim + j);
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    // The first values do not wait for the scores.
    if (t < lanes)
        for (int tile = 0; tile < attention_ahead; ++tile) fetch(tile);
    float q[8];
    for (int i = 0; i < 8; ++i) q[i] = f(query[row * qw + head * dim + lane + 32 * i]);
    // A warp scores a tile, its eight keys' loads in flight; keys past the length read the last one and score -inf.
    for (int first = part * share + 8 * warp; first < last; first += 8 * warps) {
        bf16 k[8][8];
        for (int u = 0; u < 8; ++u)
            for (int i = 0; i < 8; ++i) k[u][i] = keys[min(first + u, length - 1) * kw + kh * dim + lane + 32 * i];
        float dot[8] = {};
        for (int i = 0; i < 8; ++i)
            for (int u = 0; u < 8; ++u) dot[u] += q[i] * f(k[u][i]);
        for (int d = 16; d; d >>= 1)
            for (int u = 0; u < 8; ++u) dot[u] += __shfl_down_sync(0xffffffff, dot[u], d);
        if (lane == 0)
            for (int u = 0; u < 8; ++u) scores[first + u] = first + u < length ? dot[u] * rsqrtf(float(dim)) : -INFINITY;
    }
    cluster.sync();
    for (int at = t; at < 8 * tiles; at += blockDim.x)
        if (at / share != part) scores[at] = cluster.map_shared_rank(scores, at / share)[at];
    // Peers read this CTA's scores until all have arrived: the probabilities overwrite them only after the wait.
    auto gathered = cluster.barrier_arrive();
    __syncthreads();
    for (int tile = t; tile < tiles; tile += blockDim.x) {
        float next_max = -INFINITY;
        for (int i = 0; i < 8; ++i) next_max = fmaxf(next_max, scores[8 * tile + i]);
        maxima[tile] = next_max;
    }
    __syncthreads();
    if (warp == 0) {
        float before = -INFINITY;
        for (int first = 0; first < tiles; first += 32) {
            float next_max = first + lane < tiles ? maxima[first + lane] : -INFINITY;
            for (int d = 1; d < 32; d <<= 1) {
                const float other = __shfl_up_sync(0xffffffff, next_max, d);
                if (lane >= d) next_max = fmaxf(next_max, other);
            }
            next_max = fmaxf(before, next_max);
            if (first + lane < tiles) maxima[first + lane] = next_max;
            before = __shfl_sync(0xffffffff, next_max, 31);
        }
    }
    __syncthreads();
    cluster.barrier_wait(std::move(gathered));
    for (int tile = t; tile < tiles; tile += blockDim.x) {
        const float running_max = tile ? maxima[tile - 1] : -INFINITY, next_max = maxima[tile];
        scales[tile] = expf(running_max - next_max);
        for (int i = 0; i < 8; ++i) scores[8 * tile + i] = expf(scores[8 * tile + i] - next_max);
    }
    __syncthreads();
    float accum[2] = {0, 0};
    if (t < lanes) {
        // A tile's operands reach registers while the previous tile's sums run.
        float rescale, p[8];
        __nv_bfloat162 v[8];
        const auto take = [&](int tile) {
            rescale = scales[tile];
            for (int i = 0; i < 8; ++i) { p[i] = scores[8 * tile + i]; v[i] = ring[tile % attention_ahead][i][t]; }
        };
        asm volatile("cp.async.wait_group %0;" :: "n"(attention_ahead - 1) : "memory");
        take(0);
        for (int tile = 0; tile < tiles; ++tile) {
            const float now_rescale = rescale;
            float now_p[8];
            __nv_bfloat162 now_v[8];
            for (int i = 0; i < 8; ++i) { now_p[i] = p[i]; now_v[i] = v[i]; }
            asm volatile("cp.async.wait_group %0;" :: "n"(attention_ahead - 2) : "memory");
            if (tile + 1 < tiles) take(tile + 1);
            accum[0] *= now_rescale;
            accum[1] *= now_rescale;
            for (int i = 0; i < 8; ++i)
                if (8 * tile + i < length) {
                    accum[0] += now_p[i] * f(now_v[i].x);
                    accum[1] += now_p[i] * f(now_v[i].y);
                }
            fetch(tile + attention_ahead);  // into this tile's slots, summed from registers
        }
    } else if (t == blockDim.x - 1) {
        float sum = 0;
        for (int tile = 0; tile < tiles; ++tile) {
            float mass = 0;
            for (int i = 0; i < 8; ++i) mass += scores[8 * tile + i];
            sum = sum * scales[tile] + mass;
        }
        denominator = sum;
    }
    __syncthreads();
    if (t < lanes)
        for (int e = 0; e < 2; ++e) {
            const bf16 attended = b(accum[e] / denominator);
            const float gate = f(qkv[row * (2 * qw + 2 * kw) + qw + head * dim + j + e]);
            const bf16 gated = b(1.0f / (1.0f + expf(-gate)));
            out[row * qw + head * dim + j + e] = b(f(attended) * f(gated));
        }
}

extern "C" __global__ void drama_silu_mul(const bf16* gate_up, bf16* out, int width, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    const int row = i / width, col = i % width;
    const float gate = f(gate_up[row * 2 * width + col]);
    out[i] = b((gate / (1.0f + expf(-gate))) * f(gate_up[row * 2 * width + width + col]));
}

__device__ uint64_t random_word(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// The sampler's order (score descending; equal scores, ±0 alike, by ascending id) as a key ascending in it.
__device__ inline unsigned order_key(float x) {
    const unsigned bits = __float_as_uint(x), magnitude = bits & 0x7fffffffu;
    return bits >> 31 ? 0x80000000u + magnitude : 0x80000000u - magnitude;
}

// Top-p over the whole vocabulary before top-k, ranked by a stable LSD radix sort; 64 KiB dynamic shared memory.
constexpr int sample_rounds = 6;

extern "C" __global__ void __launch_bounds__(1024) drama_sample(
    const void* logits, int is_bf16, int vocab, float temperature, int top_k,
    float top_p, uint64_t seed, uint64_t step, int output_index,
    int begin, int eos, int* history, int* frame, int* error_flag) {
    extern __shared__ unsigned char storage[];
    unsigned short* counts = reinterpret_cast<unsigned short*>(storage);  // [digit][warp]
    float* scores = reinterpret_cast<float*>(counts + 256 * 32);
    unsigned short* ids = reinterpret_cast<unsigned short*>(scores + vocab);
    unsigned short* next = ids + vocab;
    // The counts are no longer needed after the sort; reuse them for reductions.
    float* partial = reinterpret_cast<float*>(counts);
    __shared__ int invalid, selected[2];
    __shared__ unsigned key_and, key_or, warp_offsets[32];
    __shared__ float total;
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    if (t == 0) { invalid = 0; key_and = ~0u; key_or = 0; }
    __syncthreads();
    unsigned all = ~0u, any = 0;
    for (int i = t; i < vocab; i += blockDim.x) {
        const float x = is_bf16 ? f(static_cast<const bf16*>(logits)[i]) : static_cast<const float*>(logits)[i];
        if (!isfinite(x)) atomicExch(&invalid, 1);
        scores[i] = x;
        ids[i] = i;
        all &= order_key(x);
        any |= order_key(x);
    }
    all = __reduce_and_sync(0xffffffff, all);
    any = __reduce_or_sync(0xffffffff, any);
    if (lane == 0) { atomicAnd(&key_and, all); atomicOr(&key_or, any); }
    __syncthreads();
    if (invalid) {
        if (t == 0) {
            *error_flag = 1;
            if (output_index == 0) { frame[0] = eos; frame[1] = 0; }
            else frame[output_index] = 0;
        }
        return;
    }
    // Warp w ranks its run 32 at a time; equal digits keep their order (BF16 keys skip the two zero digits).
    const int run = (vocab + 31) / 32, from = warp * run, end = min(vocab, from + run);
    const unsigned lower = (1u << lane) - 1;
    for (int shift = 0; shift < 32; shift += 8) {
        if (!((key_and ^ key_or) >> shift & 255)) continue;
        reinterpret_cast<uint4*>(counts)[t] = make_uint4(0, 0, 0, 0);
        __syncthreads();
        // Each element's place among its warp's equal digits: the earlier rounds' count, then the lower lanes'.
        int digit[sample_rounds], place[sample_rounds];
        for (int r = 0; r < sample_rounds && from + 32 * r < end; ++r) {
            const int i = from + 32 * r + lane;
            digit[r] = i < end ? order_key(scores[ids[i]]) >> shift & 255 : 0;
            unsigned peers = __ballot_sync(0xffffffff, i < end);
            for (int bit = 0; bit < 8; ++bit) {
                const unsigned set = __ballot_sync(0xffffffff, digit[r] >> bit & 1);
                peers &= digit[r] >> bit & 1 ? set : ~set;
            }
            const int leader = __ffs(peers) - 1;
            unsigned counted = 0;
            if (lane == leader) {
                counted = counts[digit[r] * 32 + warp];
                counts[digit[r] * 32 + warp] = counted + __popc(peers);
            }
            place[r] = __shfl_sync(0xffffffff, counted, leader) + __popc(peers & lower);
            __syncwarp();
        }
        __syncthreads();
        // Exclusive scan of the counts in [digit][warp] order: eight per thread, then across threads.
        const uint4 packed = reinterpret_cast<const uint4*>(counts)[t];
        unsigned c[8] = {packed.x & 0xffff, packed.x >> 16, packed.y & 0xffff, packed.y >> 16,
                         packed.z & 0xffff, packed.z >> 16, packed.w & 0xffff, packed.w >> 16};
        unsigned sum = 0;
        for (int i = 0; i < 8; ++i) { const unsigned n = c[i]; c[i] = sum; sum += n; }
        unsigned before = sum;
        for (int d = 1; d < 32; d <<= 1) {
            const unsigned other = __shfl_up_sync(0xffffffff, before, d);
            if (lane >= d) before += other;
        }
        if (lane == 31) warp_offsets[warp] = before;
        __syncthreads();
        if (warp == 0) {
            const unsigned own = warp_offsets[lane];
            unsigned prefix = own;
            for (int d = 1; d < 32; d <<= 1) {
                const unsigned other = __shfl_up_sync(0xffffffff, prefix, d);
                if (lane >= d) prefix += other;
            }
            warp_offsets[lane] = prefix - own;
        }
        __syncthreads();
        before += warp_offsets[warp] - sum;
        reinterpret_cast<uint4*>(counts)[t] = make_uint4(
            (c[0] + before) | (c[1] + before) << 16, (c[2] + before) | (c[3] + before) << 16,
            (c[4] + before) | (c[5] + before) << 16, (c[6] + before) | (c[7] + before) << 16);
        __syncthreads();
        for (int r = 0; r < sample_rounds && from + 32 * r < end; ++r) {
            const int i = from + 32 * r + lane;
            if (i < end) next[counts[digit[r] * 32 + warp] + place[r]] = ids[i];
        }
        __syncthreads();
        unsigned short* const sorted = next; next = ids; ids = sorted;
    }
    const float top = scores[ids[0]];
    const bool main = output_index == 0;
    for (int draw = 0; draw < (main ? 2 : 1); ++draw) {
        const float inv_t = 1.0f / fmaxf(draw == 1 ? 1.0f : temperature, 1e-5f);
        const float p_limit = draw == 1 ? 0.95f : top_p;
        const int k = min(vocab, draw == 1 ? 30 : top_k);
        float mass = 0;
        for (int i = t; i < vocab; i += blockDim.x) mass += expf((scores[ids[i]] - top) * inv_t);
        const float summed = sum_block(mass, partial);
        if (t == 0) total = summed;
        __syncthreads();
        // Warp 0 takes 32 ranks at a time; lane 0 runs the in-order cumulative sum, cutoff and pick.
        if (warp == 0) {
            float* const weights = partial;
            float* const candidates = partial + 32;
            float cumulative = 0, best = -INFINITY;
            int pick = ids[0];
            const uint64_t stream = main ? uint64_t(draw) : uint64_t(output_index);
            bool cut = false;
            for (int first = 0; first < k && !cut; first += 32) {
                const int rank = first + lane;
                if (rank < k) {
                    const float log_weight = (scores[ids[rank]] - top) * inv_t;
                    weights[lane] = expf(log_weight) / total;
                    const uint64_t word = random_word(seed ^ random_word((step << 20) | (stream << 14) | uint64_t(rank)));
                    const float u = (float(word >> 41) + 0.5f) * (1.0f / 8388608.0f);
                    candidates[lane] = log_weight - logf(-logf(u));
                }
                __syncwarp();
                if (lane == 0)
                    for (int i = 0; i < 32 && first + i < k; ++i) {
                        cumulative += weights[i];
                        if (first + i > 0 && cumulative > p_limit) { cut = true; break; }
                        if (candidates[i] > best) { best = candidates[i]; pick = ids[first + i]; }
                    }
                __syncwarp();
                cut = __shfl_sync(0xffffffff, cut, 0);
            }
            if (lane == 0) selected[draw] = pick;
        }
        __syncthreads();
    }
    if (t == 0) {
        if (main) {
            int first = selected[0] == 0 ? eos : begin + selected[0] - 1;
            bool repeated = false;
            for (int i = 0; i < 16; ++i) repeated |= history[i] == first;
            const int pick = repeated && semantic(first, begin, begin + 4095) ? selected[1] : selected[0];
            frame[0] = pick == 0 ? eos : begin + pick - 1;
            frame[1] = code(frame[0] - begin);
            history[step % 16] = frame[0];
        } else frame[output_index] = selected[0];
    }
}
