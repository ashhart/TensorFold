// Decode-step kernels for a few rows: the MoE router, a wave an expert, and residual tails; sums in the fastest order.

// kind: 1 fp16, 2 bf16 (ops/act.hip's numbering).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "decode/plan.hpp"

namespace {

__device__ __forceinline__ float bf16_up(uint32_t bits) { return __uint_as_float(bits << 16); }

// Four consecutive activations (8 bytes) widened.
template <int KIND>
__device__ __forceinline__ float4 widen4(uint2 v) {
    if constexpr (KIND == 1) {
        const __half2 a = __builtin_bit_cast(__half2, v.x);
        const __half2 b = __builtin_bit_cast(__half2, v.y);
        return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
    } else {
        return make_float4(bf16_up(v.x & 0xffffu), bf16_up(v.x >> 16), bf16_up(v.y & 0xffffu), bf16_up(v.y >> 16));
    }
}

constexpr int kRouterWaves = 4;
constexpr int kRouterSteps = 16;  // float4 loads a lane keeps in flight
constexpr int kRouterRows = 16;

// Four router weights at `at` of row `w`: fp32 rows as stored, or bf16 rows (ROWS16) widened, which is exact.
template <bool ROWS16>
__device__ __forceinline__ float4 router_weights(const void* w, int at) {
    if constexpr (ROWS16) {
        const uint2 v = *reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(w) + at);
        return make_float4(bf16_up(v.x & 0xffffu), bf16_up(v.x >> 16), bf16_up(v.y & 0xffffu), bf16_up(v.y >> 16));
    } else {
        return *reinterpret_cast<const float4*>(static_cast<const float*>(w) + at);
    }
}

// logits[r, e] = x[r] . rows[e], 16 rows a block, a wave an expert; d % 4 == 0; rows 16B aligned (8B ROWS16); x staged.
template <int KIND, bool ROWS16 = false>
__device__ void router_decode(const void* x, const void* rows, float* logits, int r, int d, int e, bool staged = false) {
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x * kRouterWaves + (threadIdx.x >> 5);
    if (expert >= e) return;
    // blockIdx.y picks a block of 16 rows: a row's sums are the same whichever block it is in
    const int first = blockIdx.y * kRouterRows;
    if (!staged) x = static_cast<const uint16_t*>(x) + static_cast<long long>(first) * d;
    logits += static_cast<long long>(first) * e;
    r = r - first < kRouterRows ? r - first : kRouterRows;
    const void* w = ROWS16 ? static_cast<const void*>(static_cast<const uint16_t*>(rows) + static_cast<long long>(expert) * d)
                           : static_cast<const void*>(static_cast<const float*>(rows) + static_cast<long long>(expert) * d);
    float acc[kRouterRows];
#pragma unroll
    for (int k = 0; k < kRouterRows; ++k) acc[k] = 0.f;
    for (int base = 0; base < d; base += kRouterSteps * 128) {
        float4 wv[kRouterSteps];
#pragma unroll
        for (int s = 0; s < kRouterSteps; ++s) {
            const int at = base + (lane + 32 * s) * 4;
            wv[s] = at < d ? router_weights<ROWS16>(w, at) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int k = 0; k < kRouterRows; ++k) {
            if (k >= r) continue;
            const uint16_t* xr = static_cast<const uint16_t*>(x) + static_cast<long long>(k) * d;
#pragma unroll
            for (int s = 0; s < kRouterSteps; ++s) {
                const int at = base + (lane + 32 * s) * 4;
                if (at >= d) continue;
                const float4 xv = widen4<KIND>(*reinterpret_cast<const uint2*>(xr + at));
                acc[k] = fmaf(xv.x, wv[s].x, acc[k]);
                acc[k] = fmaf(xv.y, wv[s].y, acc[k]);
                acc[k] = fmaf(xv.z, wv[s].z, acc[k]);
                acc[k] = fmaf(xv.w, wv[s].w, acc[k]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < kRouterRows; ++k) {
        float v = acc[k];
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) v += __shfl_xor(v, mask, 32);
        if (lane == 0 && k < r) logits[static_cast<long long>(k) * e + expert] = v;
    }
}


__device__ __forceinline__ float load_kind(const void* p, int kind, long long i) {
    return kind == 1 ? __half2float(static_cast<const __half*>(p)[i]) : bf16_up(static_cast<const uint16_t*>(p)[i]);
}

// A float rounded to the activation kind (fp16 or bf16, to nearest even) and widened again.
__device__ __forceinline__ float round_kind(float v, int kind) {
    if (kind == 1) return __half2float(__float2half_rn(v));
    if (v != v) return v;
    const uint32_t u = __float_as_uint(v);
    return __uint_as_float(((u + (((u >> 16) & 1u) + 0x7fffu)) >> 16) << 16);
}

__device__ __forceinline__ void store_kind(void* p, int kind, long long i, float v) {
    if (kind == 1) {
        static_cast<__half*>(p)[i] = __float2half_rn(v);
    } else {
        const uint32_t u = __float_as_uint(v);
        static_cast<uint16_t*>(p)[i] = v != v ? 0x7fc0 : static_cast<uint16_t>((u + (((u >> 16) & 1u) + 0x7fffu)) >> 16);
    }
}

constexpr int kTailThreads = 512;
constexpr int kTailWidth = 8192;

// One row a block, one launch: x = round(x + t), t = y or the slot-ordered weighted sum; normed = rms(x) * weight.
__device__ void tail_body(void* x, const void* y, const float* wts, const float* weight, void* normed, int kind,
                          int slots, int width, float eps) {
    __shared__ float row_v[kTailWidth];
    __shared__ float part[kTailThreads / 32];
    const int row = blockIdx.x;
    const long long at = static_cast<long long>(row) * width;
    float sum = 0.f;
    for (int i = threadIdx.x; i < width; i += kTailThreads) {
        float t;
        if (slots > 0) {
            float acc = 0.f;
            for (int s = 0; s < slots; ++s) {
                acc += static_cast<const float*>(y)[(static_cast<long long>(row) * slots + s) * width + i] *
                       wts[row * slots + s];
            }
            t = round_kind(acc, kind);
        } else {
            t = load_kind(y, kind, at + i);
        }
        const float v = round_kind(load_kind(x, kind, at + i) + t, kind);
        store_kind(x, kind, at + i, v);
        row_v[i] = v;
        sum += v * v;
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sum;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kTailThreads / 32; ++w) total += part[w];
    const float inv = rsqrtf(total / static_cast<float>(width) + eps);
    for (int i = threadIdx.x; i < width; i += kTailThreads) {
        float value = row_v[i] * inv;
        if (weight != nullptr) value *= weight[i];
        store_kind(normed, kind, at + i, value);
    }
}

// Conv launch arguments: x (rows, channels), taps, state, snapshots; q, k (rows, kw), v (rows, vw) out; norm, gate.
struct ConvArgs {
    const void* x;
    int kind;
    const float* weight;
    float* state;
    float* states;
    float* qn;
    float* kn;
    float* v;
    int channels, kernel, rows, kw, vw;
    const float* qw;
    const float* kw_w;
    float eps;
    int norm;
    const void* ga;
    const void* gb;
    const float* a_log;
    const float* dt_bias;
    float* gate;
    float* beta;
    int gcount, heads;
    // A round's plan (plan.count null: the launch's own rows and state): a grid row a slot, rows and state from it.
    PlanArgs plan;
    int layer;
    int slots;  // the launcher's grid rows (the grid says it here)
    // f >= 0 here: the window starts from the last window's snapshot after its row f; null or -1: from `state`
    const int* from;
};

// The conv for a window's rows split into q, k, v: taps in order, then silu; q, k heads rms-normed; gate and beta last.
__device__ void conv_split_body(const ConvArgs& launch) {
    __shared__ float sq[128];
    __shared__ float inv_s;
    ConvArgs c = launch;
    float* samples = nullptr;
    const int conv_blocks = (c.channels + 127) / 128;
    if (launch.plan.count != nullptr) {
        const int slot = blockIdx.y;
        // the gates are one pass over every row of the round, from the first slot's blocks
        if (blockIdx.x >= conv_blocks && slot != 0) return;
        if (blockIdx.x < conv_blocks) {
            const int rows = launch.plan.count[slot];
            if (rows == 0) return;
            const long long row0 = launch.plan.first[slot];
            c.rows = rows;
            c.state = reinterpret_cast<float*>(plan_desc(launch.plan, slot)[plan_first(launch.layer)]);
            c.x = static_cast<const char*>(launch.x) + row0 * c.channels * 2;
            c.qn += row0 * c.kw;
            c.kn += row0 * c.kw;
            c.v += row0 * c.vw;
            // a window of rows keeps its input rows and leaves the window as the round found it, for its keep
            const bool replayed = rows > 1 && slot + 1 < static_cast<int>(gridDim.y);
            c.states = nullptr;
            samples = replayed ? reinterpret_cast<float*>(launch.plan.snaps[2 * launch.layer]) + row0 * c.channels : nullptr;
        }
    }
    if (blockIdx.x >= conv_blocks) {
        const int i = (blockIdx.x - conv_blocks) * 128 + threadIdx.x;
        if (c.ga == nullptr || i >= c.gcount) return;
        const int h = i % c.heads;
        c.beta[i] = 1.f / (1.f + expf(-load_kind(c.gb, c.kind, i)));
        const float xg = load_kind(c.ga, c.kind, i) + c.dt_bias[h];
        const float soft = xg > 20.f ? xg : log1pf(expf(xg));
        c.gate[i] = expf(-expf(c.a_log[h]) * soft);
        return;
    }
    const int channel = blockIdx.x * 128 + threadIdx.x;
    const bool live = channel < c.channels;
    const bool head = c.norm == 128 && static_cast<int>(blockIdx.x) * 128 < 2 * c.kw;
    const float* taps = c.weight + static_cast<long long>(live ? channel : 0) * c.kernel;
    const int kept = c.kernel - 1;
    float history[7];
    const int f = c.from != nullptr && c.states != nullptr ? *c.from : -1;
    const float* start = f >= 0 ? c.states + static_cast<long long>(f) * kept * c.channels : c.state;
    if (live) {
        for (int tap = 0; tap < kept; ++tap) history[tap] = start[static_cast<long long>(tap) * c.channels + channel];
    }
    __atomic_signal_fence(__ATOMIC_SEQ_CST);  // the start is read before any snapshot store (it may be one of them)
    for (int r = 0; r < c.rows; ++r) {
        float y = 0.f;
        float sample = 0.f;
        if (live) {
            float acc = 0.f;
            for (int tap = 0; tap < kept; ++tap) acc += history[tap] * taps[tap];
            sample = load_kind(c.x, c.kind, static_cast<long long>(r) * c.channels + channel);
            acc += sample * taps[kept];
            y = acc * (1.f / (1.f + expf(-acc)));
        }
        if (head) {
            // the norm of the head's 128 values, then the scale by its weight
            __syncthreads();
            sq[threadIdx.x] = y * y;
            __syncthreads();
            if (threadIdx.x < 32) {
                float sum = 0.f;
                for (int j = 0; j < 4; ++j) sum += sq[threadIdx.x + 32 * j];
#pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
                if (threadIdx.x == 0) inv_s = rsqrtf(sum / 128.f + c.eps);
            }
            __syncthreads();
            const bool is_q = channel < c.kw;
            y = y * inv_s * (is_q ? c.qw : c.kw_w)[threadIdx.x];
        }
        if (live) {
            if (channel < c.kw) {
                c.qn[static_cast<long long>(r) * c.kw + channel] = y;
            } else if (channel < 2 * c.kw) {
                c.kn[static_cast<long long>(r) * c.kw + channel - c.kw] = y;
            } else {
                c.v[static_cast<long long>(r) * c.vw + channel - 2 * c.kw] = y;
            }
            for (int tap = 0; tap < kept - 1; ++tap) history[tap] = history[tap + 1];
            if (kept > 0) history[kept - 1] = sample;
            if (samples != nullptr) samples[static_cast<long long>(r) * c.channels + channel] = sample;
            if (c.states != nullptr) {
                for (int tap = 0; tap < kept; ++tap) {
                    c.states[(static_cast<long long>(r) * kept + tap) * c.channels + channel] = history[tap];
                }
            }
        }
    }
    if (live && samples == nullptr) {
        for (int tap = 0; tap < kept; ++tap) c.state[static_cast<long long>(tap) * c.channels + channel] = history[tap];
    }
}

// rms over rows of `width` fp32 values a wave a row as rms_rows; blockIdx.y picks the first or second set of rows.
__device__ void
    rms2_body(const float* x0, const float* w0, float* y0, const float* x1, const float* w1, float* y1, int rows,
                int width, float eps) {
    constexpr int kMost = 1024 / 32;
    const float* x = blockIdx.y == 0 ? x0 : x1;
    const float* weight = blockIdx.y == 0 ? w0 : w1;
    float* y = blockIdx.y == 0 ? y0 : y1;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long long at = static_cast<long long>(row) * width;
    float v[kMost];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < kMost; ++j) {
        const int i = lane + 32 * j;
        v[j] = i < width ? x[at + i] : 0.f;
        if (i < width) sum += v[j] * v[j];
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
    const float inv = rsqrtf(__shfl(sum, 0, 32) / static_cast<float>(width) + eps);
#pragma unroll
    for (int j = 0; j < kMost; ++j) {
        const int i = lane + 32 * j;
        if (i >= width) break;
        float value = v[j] * inv;
        if (weight != nullptr) value *= weight[i];
        y[at + i] = value;
    }
}

// The gated norm out = rms(y) * weight * round(silu(z) or sigmoid(z)) in the kind, a wave a (row, head).
__device__ void
    gnorm_out_body(const float* y, const float* weight, const void* z, void* out, int kind, int rows, int width,
                     float eps, int sigmoid_gate) {
    constexpr int kMost = 1024 / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long long at = static_cast<long long>(row) * width;
    float v[kMost];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < kMost; ++j) {
        const int i = lane + 32 * j;
        v[j] = i < width ? y[at + i] : 0.f;
        if (i < width) sum += v[j] * v[j];
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
    const float inv = rsqrtf(__shfl(sum, 0, 32) / static_cast<float>(width) + eps);
#pragma unroll
    for (int j = 0; j < kMost; ++j) {
        const int i = lane + 32 * j;
        if (i >= width) break;
        float value = v[j] * inv;
        if (weight != nullptr) value *= weight[i];
        const float zv = load_kind(z, kind, at + i);
        const float gate = sigmoid_gate ? 1.f / (1.f + expf(-zv)) : zv / (1.f + expf(-zv));
        store_kind(out, kind, at + i, value * round_kind(gate, kind));
    }
}

}  // namespace

#include "decode/select.hpp"

extern "C" __global__ void __launch_bounds__(32 * kRouterWaves)
    tf_router_select_f16(const void* x, const void* rows, float* logits, int r, int d, int e, int* pick, float* wts,
                         int* items, int* members, int capacity, int top_k, int tile, int staged, unsigned* done) {
    router_select<1>(x, rows, logits, r, d, e, pick, wts, items, members, capacity, top_k, tile, staged, done);
}

extern "C" __global__ void __launch_bounds__(32 * kRouterWaves)
    tf_router_select_bf16(const void* x, const void* rows, float* logits, int r, int d, int e, int* pick, float* wts,
                          int* items, int* members, int capacity, int top_k, int tile, int staged, unsigned* done) {
    router_select<2>(x, rows, logits, r, d, e, pick, wts, items, members, capacity, top_k, tile, staged, done);
}

extern "C" __global__ void __launch_bounds__(32 * kRouterWaves)
    tf_router_decode_f16(const void* x, const float* rows, float* logits, int r, int d, int e) {
    router_decode<1>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(32 * kRouterWaves)
    tf_router_decode_bf16(const void* x, const float* rows, float* logits, int r, int d, int e) {
    router_decode<2>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(kTailThreads)
    tf_tail(void* x, const void* y, const float* wts, const float* weight, void* normed, int kind, int slots, int width,
            float eps) {
    tail_body(x, y, wts, weight, normed, kind, slots, width, eps);
}

extern "C" __global__ void __launch_bounds__(128) tf_conv_split(ConvArgs c) { conv_split_body(c); }

extern "C" __global__ void __launch_bounds__(256)
    tf_rms2(const float* x0, const float* w0, float* y0, const float* x1, const float* w1, float* y1, int rows, int width,
            float eps) {
    rms2_body(x0, w0, y0, x1, w1, y1, rows, width, eps);
}

extern "C" __global__ void __launch_bounds__(256)
    tf_gnorm_out(const float* y, const float* weight, const void* z, void* out, int kind, int rows, int width,
                 float eps, int sigmoid_gate) {
    gnorm_out_body(y, weight, z, out, kind, rows, width, eps, sigmoid_gate);
}

extern "C" __global__ void __launch_bounds__(kSelectThreads)
    tf_select_decode(const float* logits, int* pick, float* wts, int* items, int* members, int capacity, int experts,
                     int top_k) {
    select_body(logits, pick, wts, items, members, capacity, experts, top_k);
}
