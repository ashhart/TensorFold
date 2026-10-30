#pragma once

#include "../ops/common.hpp"

// Flash Next's residual streams: the hyper-connection norm, gates and mix, and the n-gram (PLE) gate and conv.

namespace {

constexpr int kFnThreads = 256;
constexpr int kFnWaves = kFnThreads / 32;
// The widest stream a thread walks in registers: dims / kFnThreads values each.
constexpr int kFnPer = 16;
// The hyper-connection norm's block: a stream's dims over 1024 threads, four values each at most.
constexpr int kNormThreads = 1024;
constexpr int kNormPer = 4;

// A block's sum of a value a thread (`scratch`: a value a wave): each wave by a fixed xor tree, then waves in order.
__device__ float fn_block_sum(float v, float* scratch) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) v += __shfl_xor(v, mask, 32);
    const int wave = threadIdx.x >> 5, waves = (blockDim.x + 31) >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) scratch[wave] = v;
    __syncthreads();
    float total = 0.f;
    for (int w = 0; w < waves; ++w) total += scratch[w];
    return total;
}

// Block (s, r): h_s += round(branch(r, d) * inject[s]) with a branch (rounded to `kind`), then x = h rinv scale.
template <typename Branch>
__device__ __forceinline__ void hc_norm_body(const void* h, bool has_branch, Branch branch, const void* inject,
                                             void* h_out, void* x_out, const float* scale, int kind, int streams,
                                             int dims, float eps) {
    __shared__ float scratch[kNormThreads / 32];
    const int s = blockIdx.x, r = blockIdx.y;
    const long long base = (static_cast<long long>(r) * streams + s) * dims;
    float v[kNormPer];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < kNormPer; ++j) {
        const int d = threadIdx.x + j * kNormThreads;
        v[j] = 0.f;
        if (d >= dims) continue;
        float x = load(h, kind, base + d);
        if (has_branch) {
            const float add = rounded(branch(r, d) * load(inject, kind, static_cast<long long>(r) * streams + s), kind);
            x = rounded(x + add, kind);
        }
        v[j] = x;
        sum += x * x;
    }
    sum = fn_block_sum(sum, scratch);
    const float rinv = rsqrtf(sum / static_cast<float>(dims) + eps);
#pragma unroll
    for (int j = 0; j < kNormPer; ++j) {
        const int d = threadIdx.x + j * kNormThreads;
        if (d >= dims) continue;
        if (h_out != nullptr) store(h_out, kind, base + d, v[j]);
        store(x_out, kind, base + d, (v[j] * rinv) * scale[s * dims + d]);
    }
}

// Block (s, r): h_s += round(branch * inject[s]) (tp: `world` fp32 shares, summed in order), then x = h rinv scale.
extern "C" __global__ void __launch_bounds__(kNormThreads)
tf_fn_hc_norm(const void* h, const void* branch, const void* inject, void* h_out, void* x_out, const float* scale,
              int kind, int streams, int dims, float eps, int world, long long stride) {
    hc_norm_body(
        h, branch != nullptr,
        [&](int r, int d) {
            const long long at = static_cast<long long>(r) * dims + d;
            if (world <= 0) return load(branch, kind, at);
            float b = 0.f;
            for (int w = 0; w < world; ++w) b += static_cast<const float*>(branch)[w * stride + at];
            return rounded(b, kind);
        },
        inject, h_out, x_out, scale, kind, streams, dims, eps);
}

// The same with the MoE's combine as the branch, summed and rounded as moe_combine_kernel: one launch for both.
extern "C" __global__ void __launch_bounds__(kNormThreads)
tf_fn_hc_norm_moe(const void* h, const float* y, const float* wts, const void* inject, void* h_out, void* x_out,
                  const float* scale, int kind, int streams, int dims, float eps, int slots) {
    hc_norm_body(
        h, true,
        [&](int r, int d) {
            float acc = 0.f;
            for (int k = 0; k < slots; ++k) acc += y[(static_cast<long long>(r) * slots + k) * dims + d] * wts[r * slots + k];
            return rounded(acc, kind);
        },
        inject, h_out, x_out, scale, kind, streams, dims, eps);
}

// tf_fn_hc_norm over bf16 streams of 2,560 in a 128-thread block, 16-byte accesses, the old block's sums: same bits.
constexpr int kNormVThreads = 128;
constexpr int kNormVDims = 2560;
extern "C" __global__ void __launch_bounds__(kNormVThreads)
tf_fn_hc_norm_v(const uint16_t* h, const uint16_t* branch, const uint16_t* inject, uint16_t* h_out, uint16_t* x_out,
                const float* scale, int streams, float eps) {
    __shared__ float scratch[kNormThreads / 32];
    const int s = blockIdx.x, r = blockIdx.y, lane = threadIdx.x;
    const long long base = (static_cast<long long>(r) * streams + s) * kNormVDims;
    const float inj = branch != nullptr ? bf16_to_float(inject[static_cast<long long>(r) * streams + s]) : 0.f;
    float v[3][8];
    float part[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) part[e] = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int d0 = 8 * lane + j * kNormThreads;
        if (d0 >= kNormVDims) break;
        const uint4 hv = *reinterpret_cast<const uint4*>(h + base + d0);
        const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
        uint32_t bw[4] = {0, 0, 0, 0};
        if (branch != nullptr) {
            const uint4 bv = *reinterpret_cast<const uint4*>(branch + static_cast<long long>(r) * kNormVDims + d0);
            bw[0] = bv.x, bw[1] = bv.y, bw[2] = bv.z, bw[3] = bv.w;
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float x = bf16_to_float(static_cast<uint16_t>(hw[e >> 1] >> ((e & 1) * 16)));
            if (branch != nullptr) {
                const float b = bf16_to_float(static_cast<uint16_t>(bw[e >> 1] >> ((e & 1) * 16)));
                const float add = rounded(b * inj, 2);
                x = rounded(x + add, 2);
            }
            v[j][e] = x;
            part[e] += x * x;
        }
    }
    // the old wave tree: xor 16 (lane ^ 2), xor 8 (lane ^ 1), then xor 4, 2, 1 inside the lane's 8
#pragma unroll
    for (int e = 0; e < 8; ++e) part[e] = part[e] + __shfl_xor(part[e], 2, 32);
#pragma unroll
    for (int e = 0; e < 8; ++e) part[e] = part[e] + __shfl_xor(part[e], 1, 32);
#pragma unroll
    for (int m = 4; m > 0; m >>= 1) {
        float next[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) next[e] = part[e] + part[e ^ m];
#pragma unroll
        for (int e = 0; e < 8; ++e) part[e] = next[e];
    }
    if ((lane & 3) == 0) scratch[lane >> 2] = part[0];
    __syncthreads();
    float total = 0.f;
    for (int w = 0; w < kNormThreads / 32; ++w) total += scratch[w];
    const float rinv = rsqrtf(total / static_cast<float>(kNormVDims) + eps);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int d0 = 8 * lane + j * kNormThreads;
        if (d0 >= kNormVDims) break;
        uint32_t ho[4], xo[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float a0 = v[j][2 * q], a1 = v[j][2 * q + 1];
            ho[q] = static_cast<uint32_t>(float_to_bf16(a0)) | (static_cast<uint32_t>(float_to_bf16(a1)) << 16);
            const float x0 = (a0 * rinv) * scale[s * kNormVDims + d0 + 2 * q];
            const float x1 = (a1 * rinv) * scale[s * kNormVDims + d0 + 2 * q + 1];
            xo[q] = static_cast<uint32_t>(float_to_bf16(x0)) | (static_cast<uint32_t>(float_to_bf16(x1)) << 16);
        }
        if (h_out != nullptr) *reinterpret_cast<uint4*>(h_out + base + d0) = uint4{ho[0], ho[1], ho[2], ho[3]};
        *reinterpret_cast<uint4*>(x_out + base + d0) = uint4{xo[0], xo[1], xo[2], xo[3]};
    }
}

// tf_fn_hc_norm_v with the MoE's combine as the branch for a prompt's rows: moe_combine_kernel's then its bytes.
constexpr int kNormMoeVMost = 4;
extern "C" __global__ void __launch_bounds__(kNormMoeVMost * kNormVThreads)
tf_fn_hc_norm_moe_v(const uint16_t* h, const float* y, const float* wts, const uint16_t* inject, uint16_t* h_out,
                    uint16_t* x_out, const float* scale, int streams, float eps, int slots) {
    __shared__ float br[kNormVDims];
    __shared__ float scratch[kNormMoeVMost][kNormThreads / 32];
    const int r = blockIdx.y, tid = threadIdx.x;
    for (int d = tid; d < kNormVDims; d += streams * kNormVThreads) {
        float acc = 0.f;
        for (int k = 0; k < slots; ++k) acc += y[(static_cast<long long>(r) * slots + k) * kNormVDims + d] * wts[r * slots + k];
        br[d] = rounded(acc, 2);
    }
    __syncthreads();
    const int s = tid / kNormVThreads, lane = tid % kNormVThreads;
    const long long base = (static_cast<long long>(r) * streams + s) * kNormVDims;
    const float inj = bf16_to_float(inject[static_cast<long long>(r) * streams + s]);
    float v[3][8];
    float part[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) part[e] = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int d0 = 8 * lane + j * kNormThreads;
        if (d0 >= kNormVDims) break;
        const uint4 hv = *reinterpret_cast<const uint4*>(h + base + d0);
        const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float x = bf16_to_float(static_cast<uint16_t>(hw[e >> 1] >> ((e & 1) * 16)));
            const float add = rounded(br[d0 + e] * inj, 2);
            x = rounded(x + add, 2);
            v[j][e] = x;
            part[e] += x * x;
        }
    }
    // tf_fn_hc_norm_v's tree: xor 16 (lane ^ 2), xor 8 (lane ^ 1), then xor 4, 2, 1 inside the lane's 8
#pragma unroll
    for (int e = 0; e < 8; ++e) part[e] = part[e] + __shfl_xor(part[e], 2, 32);
#pragma unroll
    for (int e = 0; e < 8; ++e) part[e] = part[e] + __shfl_xor(part[e], 1, 32);
#pragma unroll
    for (int m = 4; m > 0; m >>= 1) {
        float next[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) next[e] = part[e] + part[e ^ m];
#pragma unroll
        for (int e = 0; e < 8; ++e) part[e] = next[e];
    }
    if ((lane & 3) == 0) scratch[s][lane >> 2] = part[0];
    __syncthreads();
    float total = 0.f;
    for (int w = 0; w < kNormThreads / 32; ++w) total += scratch[s][w];
    const float rinv = rsqrtf(total / static_cast<float>(kNormVDims) + eps);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int d0 = 8 * lane + j * kNormThreads;
        if (d0 >= kNormVDims) break;
        uint32_t ho[4], xo[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float a0 = v[j][2 * q], a1 = v[j][2 * q + 1];
            ho[q] = static_cast<uint32_t>(float_to_bf16(a0)) | (static_cast<uint32_t>(float_to_bf16(a1)) << 16);
            const float x0 = (a0 * rinv) * scale[s * kNormVDims + d0 + 2 * q];
            const float x1 = (a1 * rinv) * scale[s * kNormVDims + d0 + 2 * q + 1];
            xo[q] = static_cast<uint32_t>(float_to_bf16(x0)) | (static_cast<uint32_t>(float_to_bf16(x1)) << 16);
        }
        *reinterpret_cast<uint4*>(h_out + base + d0) = uint4{ho[0], ho[1], ho[2], ho[3]};
        *reinterpret_cast<uint4*>(x_out + base + d0) = uint4{xo[0], xo[1], xo[2], xo[3]};
    }
}

// tf_fn_hc_mix over bf16 activations, 8 neighbouring outputs a thread (16-byte loads of g and x), the same bits.
extern "C" __global__ void tf_fn_hc_mix8(const void* g, int gkind, const uint16_t* x, uint16_t* mixed, int rows,
                                         int streams, int dims) {
    const long long i = gid() * 8;
    if (i >= static_cast<long long>(rows) * dims) return;
    const long long r = i / dims;
    const int d = static_cast<int>(i % dims);
    float total[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) total[e] = 0.f;
    for (int s = 0; s < streams; ++s) {
        const long long at = (r * streams + s) * dims + d;
        float gv[8];
        if (gkind == 2) {
            const uint4 u = *reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(g) + at);
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int e = 0; e < 8; ++e) gv[e] = bf16_to_float(static_cast<uint16_t>(w[e >> 1] >> ((e & 1) * 16)));
        } else {
            const float4 a = *reinterpret_cast<const float4*>(static_cast<const float*>(g) + at);
            const float4 b = *reinterpret_cast<const float4*>(static_cast<const float*>(g) + at + 4);
            gv[0] = a.x, gv[1] = a.y, gv[2] = a.z, gv[3] = a.w, gv[4] = b.x, gv[5] = b.y, gv[6] = b.z, gv[7] = b.w;
        }
        const uint4 xu = *reinterpret_cast<const uint4*>(x + at);
        const uint32_t xw[4] = {xu.x, xu.y, xu.z, xu.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float gate = rounded(sigmoid(rounded(gv[e], 2)), 2);
            total[e] += rounded(gate * bf16_to_float(static_cast<uint16_t>(xw[e >> 1] >> ((e & 1) * 16))), 2);
        }
    }
    uint32_t o[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
        o[q] = static_cast<uint32_t>(float_to_bf16(total[2 * q] / static_cast<float>(streams))) |
               (static_cast<uint32_t>(float_to_bf16(total[2 * q + 1] / static_cast<float>(streams))) << 16);
    *reinterpret_cast<uint4*>(mixed + i) = uint4{o[0], o[1], o[2], o[3]};
}

// From z [rows, low (+ streams with inject)] / streams: silu for the up projection, 2 sigmoid for the inject gates.
extern "C" __global__ void tf_fn_hc_gates(const float* z, void* act, void* inject, int kind, int rows, int low,
                                          int streams) {
    const long long i = gid();
    const int width = low + (inject != nullptr ? streams : 0);
    if (i >= static_cast<long long>(rows) * width) return;
    const int r = static_cast<int>(i / width), c = static_cast<int>(i % width);
    const float v = rounded(rounded(z[i], kind) / static_cast<float>(streams), kind);
    if (c < low) store(act, kind, static_cast<long long>(r) * low + c, silu(v));
    else store(inject, kind, static_cast<long long>(r) * streams + c - low, 2.f * rounded(sigmoid(v), kind));
}

// Row r of x normed over all its `width` values (fp32 sums: each thread's strided values, then the block's sum).
extern "C" __global__ void __launch_bounds__(kFnThreads)
tf_fn_rms_wide(const void* x, const float* scale, void* y, int kind, int width, float eps) {
    __shared__ float scratch[kFnWaves];
    const long long base = static_cast<long long>(blockIdx.x) * width;
    float sum = 0.f;
    for (int i = threadIdx.x; i < width; i += kFnThreads) {
        const float v = load(x, kind, base + i);
        sum += v * v;
    }
    const float rinv = rsqrtf(fn_block_sum(sum, scratch) / static_cast<float>(width) + eps);
    for (int i = threadIdx.x; i < width; i += kFnThreads) {
        store(y, kind, base + i, (load(x, kind, base + i) * rinv) * scale[i]);
    }
}

// mixed[r, d] = mean over the streams (in order) of round(sigmoid(round(g)) * x): g the up projection's fp32 sums.
extern "C" __global__ void tf_fn_hc_mix(const void* g, int gkind, const void* x, void* mixed, int kind, int rows,
                                        int streams, int dims) {
    const long long i = gid();
    if (i >= static_cast<long long>(rows) * dims) return;
    const long long r = i / dims;
    const int d = static_cast<int>(i % dims);
    float total = 0.f;
    for (int s = 0; s < streams; ++s) {
        const long long at = (r * streams + s) * dims + d;
        const float gate = rounded(sigmoid(rounded(load(g, gkind, at), kind)), kind);
        total += rounded(gate * load(x, kind, at), kind);
    }
    store(mixed, kind, i, total / static_cast<float>(streams));
}

// Block (s, r): gated = sigmoid(signed sqrt(k . q / sqrt(dims))) v, k and q normed; normed is the conv input.
extern "C" __global__ void __launch_bounds__(kFnThreads)
tf_fn_ple_gate(const void* kv, const void* h, const float* key_scale, const float* query_scale, const float* conv_scale,
               void* gated, void* normed, int kind, int streams, int dims, float eps) {
    __shared__ float scratch[kFnWaves];
    const int s = blockIdx.x, r = blockIdx.y;
    const long long wide = static_cast<long long>(streams) * dims;
    const long long krow = r * (wide + dims), hrow = r * wide;
    float k[kFnPer], q[kFnPer], v[kFnPer];
    float sk = 0.f, sq = 0.f;
#pragma unroll
    for (int j = 0; j < kFnPer; ++j) {
        const int d = threadIdx.x + j * kFnThreads;
        k[j] = q[j] = v[j] = 0.f;
        if (d >= dims) continue;
        k[j] = load(kv, kind, krow + static_cast<long long>(s) * dims + d);
        q[j] = load(h, kind, hrow + static_cast<long long>(s) * dims + d);
        v[j] = load(kv, kind, krow + wide + d);
        sk += k[j] * k[j];
        sq += q[j] * q[j];
    }
    const float rk = rsqrtf(fn_block_sum(sk, scratch) / static_cast<float>(dims) + eps);
    const float rq = rsqrtf(fn_block_sum(sq, scratch) / static_cast<float>(dims) + eps);
    float dot = 0.f;
#pragma unroll
    for (int j = 0; j < kFnPer; ++j) {
        const int d = threadIdx.x + j * kFnThreads;
        if (d >= dims) continue;
        const float kn = rounded((k[j] * rk) * key_scale[s * dims + d], kind);
        const float qn = rounded((q[j] * rq) * query_scale[s * dims + d], kind);
        dot += rounded(kn * qn, kind);
    }
    dot = fn_block_sum(dot, scratch);
    const float g1 = rounded(rounded(dot, kind) / rounded(sqrtf(static_cast<float>(dims)), kind), kind);
    const float root = rounded(sqrtf(fmaxf(fabsf(g1), 1e-6f)), kind);
    const float g2 = rounded(g1 < 0.f ? -root : (g1 > 0.f ? root : 0.f), kind);
    const float gate = rounded(sigmoid(g2), kind);
    float sc = 0.f;
#pragma unroll
    for (int j = 0; j < kFnPer; ++j) {
        v[j] = rounded(gate * v[j], kind);
        sc += v[j] * v[j];
    }
    const float rc = rsqrtf(fn_block_sum(sc, scratch) / static_cast<float>(dims) + eps);
#pragma unroll
    for (int j = 0; j < kFnPer; ++j) {
        const int d = threadIdx.x + j * kFnThreads;
        if (d >= dims) continue;
        const long long at = hrow + static_cast<long long>(s) * dims + d;
        store(gated, kind, at, v[j]);
        store(normed, kind, at, (v[j] * rc) * conv_scale[s * dims + d]);
    }
}

// Channel c of row r: out = h + (gated + silu(conv)), the dilated causal conv over cin, which holds the tail first.
extern "C" __global__ void tf_fn_ple_conv(const void* cin, const float* weight, const void* gated, const void* h,
                                          void* out, int kind, int rows, int width, int taps, int dilation) {
    const long long i = gid();
    if (i >= static_cast<long long>(rows) * width) return;
    const long long r = i / width;
    const int c = static_cast<int>(i % width);
    float y = 0.f;
    for (int j = 0; j < taps; ++j) y = fmaf(weight[c * taps + j], load(cin, kind, (r + dilation * j) * width + c), y);
    const float yb = rounded(y, kind);
    const float act = rounded(yb * rounded(sigmoid(yb), kind), kind);
    store(out, kind, i, load(h, kind, i) + rounded(load(gated, kind, i) + act, kind));
}

}  // namespace
