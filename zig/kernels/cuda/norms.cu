// RMS norm rows for every CUDA family (float64 references: core/norms_ref.zig), a block a row of 1,024 or 256 threads.

// A row's sums run in one fixed order, so its bits never depend on the row count; kinds 0 fp32, 1 fp16, 2 bf16.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace tf_norms {

constexpr int NT = 1024;

template <int K>
__device__ __forceinline__ float ld(const void* p, int64_t i) {
    if constexpr (K == 0) return static_cast<const float*>(p)[i];
    else if constexpr (K == 1) return __half2float(static_cast<const __half*>(p)[i]);
    else return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
}

template <int K>
__device__ __forceinline__ void st(void* p, int64_t i, float v) {
    if constexpr (K == 0) static_cast<float*>(p)[i] = v;
    else if constexpr (K == 1) static_cast<__half*>(p)[i] = __float2half_rn(v);
    else static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
}

__device__ __forceinline__ float load(const void* p, int kind, int64_t i) {
    if (kind == 0) return static_cast<const float*>(p)[i];
    if (kind == 1) return __half2float(static_cast<const __half*>(p)[i]);
    return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
}

__device__ __forceinline__ void store(void* p, int kind, int64_t i, float v) {
    if (kind == 0) static_cast<float*>(p)[i] = v;
    else if (kind == 1) static_cast<__half*>(p)[i] = __float2half_rn(v);
    else static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
}

// 1 / sqrt(sum / width + eps), IEEE division and square root, of a block's partial sums in one order every row.
__device__ __forceinline__ float inv_rms(float part, int width, float eps) {
    __shared__ float warps[NT / 32];
    __shared__ float inv;
    for (int m = 16; m > 0; m >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xFFFFFFFFu, part, m));
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = part;
    __syncthreads();
    if (threadIdx.x < 32) {
        float s = warps[threadIdx.x];
        for (int m = 16; m > 0; m >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, m));
        if (threadIdx.x == 0) inv = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(s, static_cast<float>(width)), eps)));
    }
    __syncthreads();
    const float r = inv;
    __syncthreads(); // `warps` and `inv` are reused by a second norm in the same block
    return r;
}

}  // namespace tf_norms

using namespace tf_norms;

namespace tf_norms {

constexpr int CACHED = 8; // a thread keeps its values in registers up to CACHED * NT = 8,192 wide

// rms rows; WIDE re-reads the row from memory (wider than CACHED * NT), the same arithmetic in the same order.
template <bool WIDE, int XK, int YK>
__device__ __forceinline__ void rms_row(const void* __restrict__ x, const float* __restrict__ w, void* __restrict__ y,
                                        int width, float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    float part = 0.f;
    if constexpr (WIDE) {
        for (int i = threadIdx.x; i < width; i += NT) {
            const float a = ld<XK>(x, at + i);
            part = __fmaf_rn(a, a, part);
        }
        const float inv = inv_rms(part, width, eps);
        for (int i = threadIdx.x; i < width; i += NT) {
            float a = __fmul_rn(ld<XK>(x, at + i), inv);
            if (w != nullptr) a = __fmul_rn(a, w[i]);
            st<YK>(y, at + i, a);
        }
    } else {
        float v[CACHED], wv[CACHED];
#pragma unroll
        for (int j = 0; j < CACHED; ++j) { // every load issued before any is used
            const int i = threadIdx.x + j * NT;
            v[j] = i < width ? ld<XK>(x, at + i) : 0.f;
            wv[j] = (w != nullptr && i < width) ? w[i] : 1.f;
        }
#pragma unroll
        for (int j = 0; j < CACHED; ++j) part = __fmaf_rn(v[j], v[j], part); // zeros past the row add nothing
        const float inv = inv_rms(part, width, eps);
#pragma unroll
        for (int j = 0; j < CACHED; ++j) {
            const int i = threadIdx.x + j * NT;
            if (i < width)
                st<YK>(y, at + i, w != nullptr ? __fmul_rn(__fmul_rn(v[j], inv), wv[j]) : __fmul_rn(v[j], inv));
        }
    }
}

template <bool WIDE, int XK, int YK>
__device__ __forceinline__ void add_rms_row(const void* __restrict__ x, float* __restrict__ res, const float* __restrict__ w1,
                                            const float* __restrict__ w2, void* __restrict__ y, int width, float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    if constexpr (WIDE) {
        float inv1 = 1.f;
        if (w1 != nullptr) {
            float part = 0.f;
            for (int i = threadIdx.x; i < width; i += NT) {
                const float a = ld<XK>(x, at + i);
                part = __fmaf_rn(a, a, part);
            }
            inv1 = inv_rms(part, width, eps);
        }
        float part = 0.f;
        for (int i = threadIdx.x; i < width; i += NT) {
            float a = ld<XK>(x, at + i);
            if (w1 != nullptr) a = __fmul_rn(__fmul_rn(a, inv1), w1[i]);
            const float r = __fadd_rn(res[at + i], a);
            res[at + i] = r;
            part = __fmaf_rn(r, r, part);
        }
        const float inv2 = inv_rms(part, width, eps);
        for (int i = threadIdx.x; i < width; i += NT) st<YK>(y, at + i, __fmul_rn(__fmul_rn(res[at + i], inv2), w2[i]));
    } else {
        float v[CACHED], rv[CACHED], a1[CACHED], a2[CACHED];
#pragma unroll
        for (int j = 0; j < CACHED; ++j) { // every load issued before any is used
            const int i = threadIdx.x + j * NT;
            const bool in = i < width;
            v[j] = in ? ld<XK>(x, at + i) : 0.f;
            rv[j] = in ? res[at + i] : 0.f;
            a1[j] = (w1 != nullptr && in) ? w1[i] : 0.f;
            a2[j] = in ? w2[i] : 0.f;
        }
        float inv1 = 1.f;
        if (w1 != nullptr) {
            float part = 0.f;
#pragma unroll
            for (int j = 0; j < CACHED; ++j) part = __fmaf_rn(v[j], v[j], part);
            inv1 = inv_rms(part, width, eps);
        }
        float part = 0.f;
#pragma unroll
        for (int j = 0; j < CACHED; ++j) {
            const float a = w1 != nullptr ? __fmul_rn(__fmul_rn(v[j], inv1), a1[j]) : v[j];
            rv[j] = __fadd_rn(rv[j], a);
            part = __fmaf_rn(rv[j], rv[j], part);
        }
        const float inv2 = inv_rms(part, width, eps);
#pragma unroll
        for (int j = 0; j < CACHED; ++j) {
            const int i = threadIdx.x + j * NT;
            if (i < width) {
                res[at + i] = rv[j];
                st<YK>(y, at + i, __fmul_rn(__fmul_rn(rv[j], inv2), a2[j]));
            }
        }
    }
}

}  // namespace tf_norms

// y = x * rsqrt(mean(x^2) + eps) * w (w null: no weight), in fp32, rounded once to y's kind.
extern "C" __global__ void __launch_bounds__(NT) tf_rms_rows(const void* x, int x_kind, const float* w, void* y,
                                                             int y_kind, int width, float eps) {
#define TF_RMS(XK, YK)                                                                                            \
    if (x_kind == XK && y_kind == YK) {                                                                           \
        if (width > CACHED * NT) rms_row<true, XK, YK>(x, w, y, width, eps);                                      \
        else rms_row<false, XK, YK>(x, w, y, width, eps);                                                         \
        return;                                                                                                   \
    }
    TF_RMS(0, 0) TF_RMS(0, 1) TF_RMS(0, 2) TF_RMS(1, 0) TF_RMS(1, 1) TF_RMS(1, 2) TF_RMS(2, 0) TF_RMS(2, 1) TF_RMS(2, 2)
#undef TF_RMS
}

// bf16 rows `x_ld` apart: y = bf16(x * rsqrt(mean(x^2) + eps) * w), then xs = the stored y's sums over each `group`.
extern "C" __global__ void __launch_bounds__(NT) tf_rms_rows_sums(const __nv_bfloat16* __restrict__ x, long long x_ld,
                                                                  const float* __restrict__ w, __nv_bfloat16* __restrict__ y,
                                                                  float* __restrict__ xs, int width, int group, float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * x_ld, out = static_cast<int64_t>(blockIdx.x) * width;
    float part = 0.f;
    for (int i = threadIdx.x; i < width; i += NT) {
        const float a = __bfloat162float(x[at + i]);
        part = __fmaf_rn(a, a, part);
    }
    const float inv = inv_rms(part, width, eps);
    for (int i = threadIdx.x; i < width; i += NT)
        y[out + i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(x[at + i]), inv), w[i]));
    __syncthreads();
    for (int j = threadIdx.x; j < width / group; j += NT) {
        float sum = 0.f;
        for (int k = 0; k < group; ++k) sum = __fadd_rn(sum, __bfloat162float(y[out + j * group + k]));
        xs[static_cast<int64_t>(blockIdx.x) * (width / group) + j] = sum;
    }
}

// res (fp32, in place) += rms(x) * w1 (a sandwich norm; null: res += x), then y = rms(res) * w2 in y's kind.
extern "C" __global__ void __launch_bounds__(NT) tf_add_rms_rows(const void* x, int x_kind, float* res, const float* w1,
                                                                 const float* w2, void* y, int y_kind, int width,
                                                                 float eps) {
#define TF_ADD(XK, YK)                                                                                            \
    if (x_kind == XK && y_kind == YK) {                                                                           \
        if (width > CACHED * NT) add_rms_row<true, XK, YK>(x, res, w1, w2, y, width, eps);                        \
        else add_rms_row<false, XK, YK>(x, res, w1, w2, y, width, eps);                                           \
        return;                                                                                                   \
    }
    TF_ADD(0, 0) TF_ADD(0, 1) TF_ADD(0, 2) TF_ADD(1, 0) TF_ADD(1, 1) TF_ADD(1, 2) TF_ADD(2, 0) TF_ADD(2, 1) TF_ADD(2, 2)
#undef TF_ADD
}

// bf16 rows with 16-byte loads (width % 8, at most 8 * VEC_MOST * VNT, 16 B aligned): a short straight line.
namespace tf_norms {

constexpr int VNT = 256;
constexpr int VEC_MOST = 4; // 8,192 wide

__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

__device__ __forceinline__ uint4 pack8(const float* f) {
    uint4 u;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return u;
}

__device__ __forceinline__ void load8f(const float* p, float* f) {
    const float4 a = reinterpret_cast<const float4*>(p)[0];
    const float4 b = reinterpret_cast<const float4*>(p)[1];
    f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w, f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
}

__device__ __forceinline__ void store8f(float* p, const float* f) {
    reinterpret_cast<float4*>(p)[0] = make_float4(f[0], f[1], f[2], f[3]);
    reinterpret_cast<float4*>(p)[1] = make_float4(f[4], f[5], f[6], f[7]);
}

// 1 / sqrt(sum / width + eps), IEEE division and square root, over a VNT-thread block: warp trees, then the 8 warps'.
template <bool ONCE = false>
__device__ __forceinline__ float inv_rms_v(float part, int width, float eps) {
    __shared__ float warps[VNT / 32];
    for (int m = 16; m > 0; m >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xFFFFFFFFu, part, m));
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = part;
    __syncthreads();
    // every warp folds the warps' sums in the same tree, so each holds the same total: no broadcast
    float s = (threadIdx.x & 31) < VNT / 32 ? warps[threadIdx.x & 31] : 0.f;
    for (int m = 4; m > 0; m >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, m));
    s = __shfl_sync(0xFFFFFFFFu, s, 0);
    const float r = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(s, static_cast<float>(width)), eps)));
    if constexpr (!ONCE) __syncthreads(); // `warps` is reused by a second norm in the same block
    return r;
}

}  // namespace tf_norms

// tf_rms_rows for bf16 x and y (w fp32, null: none), 16-byte loads.
extern "C" __global__ void __launch_bounds__(VNT) tf_rms_rows_bf16(const __nv_bfloat16* __restrict__ x,
                                                                    const float* __restrict__ w,
                                                                    __nv_bfloat16* __restrict__ y, int width, float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    const int chunks = width / 8;
    float v[VEC_MOST][8], wv[VEC_MOST][8];
    float part = 0.f;
#pragma unroll
    for (int j = 0; j < VEC_MOST; ++j) {
        const int c = threadIdx.x + j * VNT;
        if (c < chunks) {
            unpack8(reinterpret_cast<const uint4*>(x + at)[c], v[j]);
            if (w != nullptr) load8f(w + 8 * c, wv[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < VEC_MOST; ++j)
        if (threadIdx.x + j * VNT < chunks)
#pragma unroll
            for (int i = 0; i < 8; ++i) part = __fmaf_rn(v[j][i], v[j][i], part);
    const float inv = inv_rms_v(part, width, eps);
#pragma unroll
    for (int j = 0; j < VEC_MOST; ++j) {
        const int c = threadIdx.x + j * VNT;
        if (c < chunks) {
            float o[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) o[i] = w != nullptr ? __fmul_rn(__fmul_rn(v[j][i], inv), wv[j][i]) : __fmul_rn(v[j][i], inv);
            reinterpret_cast<uint4*>(y + at)[c] = pack8(o);
        }
    }
}

// tf_add_rms_rows for bf16 x and y, 16-byte loads.
extern "C" __global__ void __launch_bounds__(VNT) tf_add_rms_rows_bf16(const __nv_bfloat16* __restrict__ x,
                                                                        float* __restrict__ res,
                                                                        const float* __restrict__ w1,
                                                                        const float* __restrict__ w2,
                                                                        __nv_bfloat16* __restrict__ y, int width,
                                                                        float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    const int chunks = width / 8;
    float v[VEC_MOST][8], rv[VEC_MOST][8], a1[VEC_MOST][8], a2[VEC_MOST][8];
#pragma unroll
    for (int j = 0; j < VEC_MOST; ++j) {
        const int c = threadIdx.x + j * VNT;
        if (c < chunks) {
            unpack8(reinterpret_cast<const uint4*>(x + at)[c], v[j]);
            load8f(res + at + 8 * c, rv[j]);
            if (w1 != nullptr) load8f(w1 + 8 * c, a1[j]);
            load8f(w2 + 8 * c, a2[j]);
        }
    }
    float inv1 = 1.f;
    if (w1 != nullptr) {
        float part = 0.f;
#pragma unroll
        for (int j = 0; j < VEC_MOST; ++j)
            if (threadIdx.x + j * VNT < chunks)
#pragma unroll
                for (int i = 0; i < 8; ++i) part = __fmaf_rn(v[j][i], v[j][i], part);
        inv1 = inv_rms_v(part, width, eps);
    }
    float part = 0.f;
#pragma unroll
    for (int j = 0; j < VEC_MOST; ++j)
        if (threadIdx.x + j * VNT < chunks)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float a = w1 != nullptr ? __fmul_rn(__fmul_rn(v[j][i], inv1), a1[j][i]) : v[j][i];
                rv[j][i] = __fadd_rn(rv[j][i], a);
                part = __fmaf_rn(rv[j][i], rv[j][i], part);
            }
    const float inv2 = inv_rms_v(part, width, eps);
#pragma unroll
    for (int j = 0; j < VEC_MOST; ++j) {
        const int c = threadIdx.x + j * VNT;
        if (c < chunks) {
            float o[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) o[i] = __fmul_rn(__fmul_rn(rv[j][i], inv2), a2[j][i]);
            store8f(res + at + 8 * c, rv[j]);
            reinterpret_cast<uint4*>(y + at)[c] = pack8(o);
        }
    }
}

// Rows that hand on a bf16 residual and fp32 sums over each 64 outputs (the next affine matmul's group sums).
namespace tf_norms {

__device__ __forceinline__ void load8w(const float* p, float* f) { load8f(p, f); }
__device__ __forceinline__ void load8w(const __nv_bfloat16* p, float* f) { unpack8(*reinterpret_cast<const uint4*>(p), f); }

// A thread's 8 stored outputs summed in order, then the group's 8 lanes in a fixed tree (every lane of the warp calls).
__device__ __forceinline__ float group_sum8(const uint4& packed, bool live) {
    float f[8];
    unpack8(packed, f);
    float s = 0.f;
    if (live)
#pragma unroll
        for (int i = 0; i < 8; ++i) s = __fadd_rn(s, f[i]);
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 4));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 2));
    return __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1));
}

// moe_glue's combine at 8 columns d of row `row`: the slots' w[k] * ys[k] fp32 fmas in slot order from 0, then its bf16
// delta (SK: the slot rows' kind, 0 fp32, 2 bf16).
template <int SK>
__device__ __forceinline__ void slot_sum8(const void* __restrict__ ys, const float* __restrict__ wt, int slots, int64_t row,
                                          int width, int d, float* rv) {
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    for (int k = 0; k < slots; ++k) {
        const float wk = wt[row * slots + k];
        const int64_t at = (row * slots + k) * width + d;
        float yv[8];
        if constexpr (SK == 0) load8f(static_cast<const float*>(ys) + at, yv);
        else unpack8(*reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(ys) + at), yv);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = __fmaf_rn(wk, yv[i], acc[i]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) rv[i] = __bfloat162float(__float2bfloat16_rn(acc[i]));
}

// h = bf16(x + r) (r null: h = x), y = bf16(rms(h) * w), xs y's 64-value sums; width % 64, 16-byte loads.
// SK >= 0: r is the expert slots' combine (slot_sum8 over ys, wt), the bytes of combine then add_rms_h in one pass.
template <typename WT, int SK = -1>
__device__ __forceinline__ void add_rms_h(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ r,
                                          const WT* __restrict__ w, __nv_bfloat16* __restrict__ h,
                                          __nv_bfloat16* __restrict__ y, float* __restrict__ xs, int width, float eps,
                                          const void* __restrict__ ys = nullptr, const float* __restrict__ wt = nullptr,
                                          int slots = 0) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    const int chunks = width / 8;
    float v[VEC_MOST][8], wv[VEC_MOST][8];
    uint4 xp[VEC_MOST], rp[VEC_MOST];
#pragma unroll
    for (int j = 0; j < VEC_MOST; ++j) { // every load issued before any is used
        const int c = threadIdx.x + j * VNT;
        if (c < chunks) {
            xp[j] = reinterpret_cast<const uint4*>(x + at)[c];
            if (SK < 0 && r != nullptr) rp[j] = reinterpret_cast<const uint4*>(r + at)[c];
            load8w(w + 8 * c, wv[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < VEC_MOST; ++j) {
        const int c = threadIdx.x + j * VNT;
        if (c < chunks) {
            unpack8(xp[j], v[j]);
            if (SK >= 0 || r != nullptr) {
                float rv[8];
                if constexpr (SK >= 0) slot_sum8<SK>(ys, wt, slots, blockIdx.x, width, 8 * c, rv);
                else unpack8(rp[j], rv);
#pragma unroll
                for (int i = 0; i < 8; ++i) v[j][i] = __fadd_rn(v[j][i], rv[i]);
                const uint4 hp = pack8(v[j]);
                reinterpret_cast<uint4*>(h + at)[c] = hp;
                unpack8(hp, v[j]); // the norm reads h as stored
            }
        }
    }
    float part = 0.f;
#pragma unroll
    for (int j = 0; j < VEC_MOST; ++j)
        if (threadIdx.x + j * VNT < chunks)
#pragma unroll
            for (int i = 0; i < 8; ++i) part = __fmaf_rn(v[j][i], v[j][i], part);
    const float inv = inv_rms_v<true>(part, width, eps);
#pragma unroll
    for (int j = 0; j < VEC_MOST; ++j) {
        const int c = threadIdx.x + j * VNT;
        const bool live = c < chunks;
        float o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __fmul_rn(__fmul_rn(v[j][i], inv), wv[j][i]);
        const uint4 yp = pack8(o);
        if (live) reinterpret_cast<uint4*>(y + at)[c] = yp;
        if (xs != nullptr && j * VNT < chunks) { // whole warps of whole groups: width % 64 == 0
            const float s = group_sum8(yp, live);
            if (live && (c & 7) == 0) xs[static_cast<int64_t>(blockIdx.x) * (width / 64) + c / 8] = s;
        }
    }
}

}  // namespace tf_norms

extern "C" __global__ void __launch_bounds__(VNT) tf_add_rms_rows_bf16h(const __nv_bfloat16* x, const __nv_bfloat16* r,
                                                                         const __nv_bfloat16* w, __nv_bfloat16* h,
                                                                         __nv_bfloat16* y, float* xs, int width, float eps) {
    add_rms_h(x, r, w, h, y, xs, width, eps);
}

extern "C" __global__ void __launch_bounds__(VNT) tf_add_rms_rows_bf16h_f32w(const __nv_bfloat16* x,
                                                                              const __nv_bfloat16* r, const float* w,
                                                                              __nv_bfloat16* h, __nv_bfloat16* y,
                                                                              float* xs, int width, float eps) {
    add_rms_h(x, r, w, h, y, xs, width, eps);
}

// h = bf16(x + combine), combine = moe_glue's tf_moe_combine of the expert slots into bf16 (ys [rows, slots, width] of
// kind ys_kind, 0 fp32 or 2 bf16; wt [rows, slots]), y = bf16(rms(h) * w), xs: add_rms_h's bytes without the delta row.
extern "C" __global__ void __launch_bounds__(VNT) tf_moe_add_rms_rows_bf16h(const __nv_bfloat16* x, const void* ys,
                                                                           int ys_kind, const float* wt, int slots,
                                                                           const __nv_bfloat16* w, __nv_bfloat16* h,
                                                                           __nv_bfloat16* y, float* xs, int width,
                                                                           float eps) {
    if (ys_kind == 0) add_rms_h<__nv_bfloat16, 0>(x, nullptr, w, h, y, xs, width, eps, ys, wt, slots);
    else add_rms_h<__nv_bfloat16, 2>(x, nullptr, w, h, y, xs, width, eps, ys, wt, slots);
}

// Per head of dv = 8 L: out = bf16(silu(z) * rms(y) * w), y and z rows at strides y_ld, z_ld; xs its 64-value sums.
template <int L>
__device__ __forceinline__ void gated_rms(const __nv_bfloat16* __restrict__ y, long long y_ld,
                                          const __nv_bfloat16* __restrict__ z, long long z_ld,
                                          const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
                                          float* __restrict__ xs, int heads, float eps) {
    constexpr int DV = 8 * L;
    const int64_t r = blockIdx.x;
    const int lane = threadIdx.x & 31, sub = lane % L;
    const int hd = (blockIdx.y * blockDim.x + threadIdx.x) / L;
    const bool live = hd < heads;
    float yv[8], zv[8], wv[8];
    if (live) {
        unpack8(*reinterpret_cast<const uint4*>(y + r * y_ld + hd * DV + 8 * sub), yv);
        unpack8(*reinterpret_cast<const uint4*>(z + r * z_ld + hd * DV + 8 * sub), zv);
        unpack8(*reinterpret_cast<const uint4*>(w + 8 * sub), wv);
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) yv[i] = zv[i] = wv[i] = 0.f;
    }
    float part = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) part = __fmaf_rn(yv[i], yv[i], part);
#pragma unroll
    for (int m = L / 2; m > 0; m >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xFFFFFFFFu, part, m));
    const float inv = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(part, static_cast<float>(DV)), eps)));
    float o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float sg = __fdiv_rn(1.f, __fadd_rn(1.f, expf(-zv[i])));
        o[i] = __fmul_rn(__fmul_rn(zv[i], sg), __fmul_rn(__fmul_rn(yv[i], inv), wv[i]));
    }
    const uint4 op = pack8(o);
    const int64_t at = (r * heads + hd) * DV + 8 * sub;
    if (live) *reinterpret_cast<uint4*>(out + at) = op;
    if (xs != nullptr) {
        const float s = group_sum8(op, live);
        if (live && (sub & 7) == 0) xs[at / 64] = s;
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_gated_rms_rows_64(const __nv_bfloat16* y, long long y_ld,
                                                                        const __nv_bfloat16* z, long long z_ld,
                                                                        const __nv_bfloat16* w, __nv_bfloat16* out,
                                                                        float* xs, int heads, float eps) {
    gated_rms<8>(y, y_ld, z, z_ld, w, out, xs, heads, eps);
}

extern "C" __global__ void __launch_bounds__(256) tf_gated_rms_rows_128(const __nv_bfloat16* y, long long y_ld,
                                                                         const __nv_bfloat16* z, long long z_ld,
                                                                         const __nv_bfloat16* w, __nv_bfloat16* out,
                                                                         float* xs, int heads, float eps) {
    gated_rms<16>(y, y_ld, z, z_ld, w, out, xs, heads, eps);
}

extern "C" __global__ void __launch_bounds__(256) tf_gated_rms_rows_256(const __nv_bfloat16* y, long long y_ld,
                                                                         const __nv_bfloat16* z, long long z_ld,
                                                                         const __nv_bfloat16* w, __nv_bfloat16* out,
                                                                         float* xs, int heads, float eps) {
    gated_rms<32>(y, y_ld, z, z_ld, w, out, xs, heads, eps);
}

// The bf16 residual's tails (norms_tail.cuh): a norm on the branch before the add, a bf16 scalar on the sum.
#include "norms_tail.cuh"

// h = bf16(r + bf16(rms(x) wb)) [* sc], y = bf16(rms(h) w), xs y's 64-value sums: each norm rounded once.
extern "C" __global__ void __launch_bounds__(tf_rms_tail::NT) tf_add_rms_rows_bf16h_tail(
    const __nv_bfloat16* x, const __nv_bfloat16* r, const __nv_bfloat16* wb, const __nv_bfloat16* sc,
    const __nv_bfloat16* w, __nv_bfloat16* h, __nv_bfloat16* y, float* xs, int width, float eps) {
    tf_rms_tail::tail<tf_rms_tail::ONCE>(x, r, wb, sc, w, h, y, xs, width, eps);
}

// The same rows with MLX's bf16 steps: each norm bf16(w * bf16(x * inv)), the sum rounded before the scalar.
extern "C" __global__ void __launch_bounds__(tf_rms_tail::NT) tf_add_rms_rows_bf16h_tail_round(
    const __nv_bfloat16* x, const __nv_bfloat16* r, const __nv_bfloat16* wb, const __nv_bfloat16* sc,
    const __nv_bfloat16* w, __nv_bfloat16* h, __nv_bfloat16* y, float* xs, int width, float eps) {
    tf_rms_tail::tail<tf_rms_tail::MLX>(x, r, wb, sc, w, h, y, xs, width, eps);
}

// nGPT rows (norms_l2.cuh): L2 norms with eps on the sum, and the normalized lerp residual (Dory).
#include "norms_l2.cuh"

// y = bf16(l2(s)), s = x, or bf16(x + r) with r (a loop's entry): a block a row, width % 64, at most 8,192.
extern "C" __global__ void __launch_bounds__(tf_l2::NT) tf_l2_rows(const __nv_bfloat16* x, const __nv_bfloat16* r,
                                                                   __nv_bfloat16* y, int width, float eps) {
    tf_l2::l2_rows<tf_l2::SUM>(x, r, y, width, eps);
}

// h = bf16(l2(a + |alpha scale| (b - a))), a = bf16(l2(res)), b = bf16(l2(out)): dory.py's slerp_norm.
extern "C" __global__ void __launch_bounds__(tf_l2::NT) tf_slerp_rows(const __nv_bfloat16* res, const __nv_bfloat16* out,
                                                                      const __nv_bfloat16* alpha, float scale,
                                                                      __nv_bfloat16* h, int width, float eps) {
    tf_l2::slerp_rows<tf_l2::A | tf_l2::B>(res, out, alpha, scale, h, width, eps);
}

// tail's norm on (row, part) blocks: part p of row r at x + r x_ld + p x_part, w + p w_part; y and xs alike.
template <int R>
__device__ __forceinline__ void rms_parts(const __nv_bfloat16* x, long long x_ld, long long x_part, const __nv_bfloat16* w,
                                          long long w_part, __nv_bfloat16* y, long long y_ld, long long y_part, float* xs,
                                          int width, float eps) {
    const int64_t r = blockIdx.x, p = blockIdx.y, yo = r * (y_ld - width) + p * y_part; // tail adds r * width itself
    tf_rms_tail::tail<R>(x + r * (x_ld - width) + p * x_part, nullptr, nullptr, nullptr, w + p * w_part, nullptr, y + yo,
                         xs != nullptr ? xs + yo / 64 : nullptr, width, eps);
}

// y = bf16(rms(x) w) and xs its 64-value sums on strided parts of rows: the tail's norm, rounded once.
extern "C" __global__ void __launch_bounds__(tf_rms_tail::NT) tf_rms_rows_bf16h_parts(
    const __nv_bfloat16* x, long long x_ld, long long x_part, const __nv_bfloat16* w, long long w_part, __nv_bfloat16* y,
    long long y_ld, long long y_part, float* xs, int width, float eps) {
    rms_parts<tf_rms_tail::ONCE>(x, x_ld, x_part, w, w_part, y, y_ld, y_part, xs, width, eps);
}

// The same parts with MLX's bf16 steps: y = bf16(w * bf16(x * inv)).
extern "C" __global__ void __launch_bounds__(tf_rms_tail::NT) tf_rms_rows_bf16h_parts_round(
    const __nv_bfloat16* x, long long x_ld, long long x_part, const __nv_bfloat16* w, long long w_part, __nv_bfloat16* y,
    long long y_ld, long long y_part, float* xs, int width, float eps) {
    rms_parts<tf_rms_tail::MLX>(x, x_ld, x_part, w, w_part, y, y_ld, y_part, xs, width, eps);
}
