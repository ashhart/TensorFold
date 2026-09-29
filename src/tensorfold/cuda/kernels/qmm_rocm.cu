// ROCm decode matmuls on tiled 4-bit words (qmm_groups.py): wmma_kernel (the default) and gemv_kernel, which uses
// v_dot2_f32_bf16 (fastest at one row, compute bound in draft windows). A word's nibbles sit so that (w >> 4j) &
// 0x000F000F holds inputs 2j and 2j + 1 in its two halves; OR 0x4300 makes each the bf16 128 + q exactly. Both sum
// each 64-input group's dot on 128 + q, then acc = fma(xs, b - 128 s, fma(p, s, acc)) in group order, and add K
// slices fixed by the weight's shape in slice order, so a row's bits never depend on how many rows share the call.

#ifndef __HIPCC__
#error "qmm_rocm.cu is the ROCm decode matmul; NVIDIA builds use qmm.cu"
#endif

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

#include <cstdlib>

namespace {

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef short short8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef int int4v __attribute__((ext_vector_type(4)));

constexpr int ROWS = 16;        // rows a launch takes (the host loops over 16-row slices)
constexpr int GPS = 16;         // groups a K slice holds (1,024 inputs)
constexpr int WARPS = 8;        // a block: 8 warps x 16 outputs
constexpr int COLS = 16 * WARPS;

__device__ __forceinline__ float bf16_value(unsigned short v) {
    return __uint_as_float(static_cast<unsigned>(v) << 16);
}

__device__ __forceinline__ unsigned short bf16_round(float f) {
    const unsigned u = __float_as_uint(f);
    return static_cast<unsigned short>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// Inputs 2j and 2j + 1 of a word as the bf16 pair (128 + q, 128 + q).
__device__ __forceinline__ unsigned pair(unsigned w, int j) {
    return ((w >> (4 * j)) & 0x000F000Fu) | 0x43004300u;
}

// One group's reads for a lane: its 16 bytes of words, the output's scale and bias (zeros past the slice or N).
struct Fetch {
    int4 w;
    unsigned short s, b;
};

// Tiled layout (qmm_groups.py): output col of group g at ((col / 16) * kg + g) * 16 + col % 16.
__device__ __forceinline__ size_t tile_at(int g, int kg, int col) {
    return (static_cast<size_t>(col >> 4) * kg + g) * 16 + (col & 15);
}

__device__ __forceinline__ Fetch fetch(const int4* __restrict__ words, const unsigned short* __restrict__ scales,
                                       const unsigned short* __restrict__ biases, int g, bool ok, int kg, int col,
                                       int half) {
    Fetch f{make_int4(0, 0, 0, 0), 0, 0};
    if (ok) {
        const size_t at = tile_at(g, kg, col);
        f.w = words[at * 2 + half];
        f.s = scales[at];
        f.b = biases[at];
    }
    return f;
}

// MR rows (a launch's rows rounded up to 1, 2, 4, 8 or 16; padding rows read zeros and are never written).
template <int MR>
__global__ void __launch_bounds__(32 * WARPS) gemv_kernel(
        const unsigned* __restrict__ x, int ldx2, int m, const float* __restrict__ xs, int kg,
        const int4* __restrict__ words, const unsigned short* __restrict__ scales,
        const unsigned short* __restrict__ biases, int n, float* __restrict__ part, unsigned short* __restrict__ out) {
    __shared__ uint4 xsh[MR][GPS * 8];                          // bf16 pairs of this slice's inputs, 4 pairs a uint4
    __shared__ float xssh[MR][GPS];                             // the rows' group sums
    const int slice = blockIdx.y;
    const int g0 = slice * GPS;
    const int groups = min(GPS, kg - g0);
    for (int i = threadIdx.x; i < MR * GPS * 8; i += blockDim.x) {
        const int r = i / (GPS * 8), c = i % (GPS * 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < m && c < groups * 8) v = reinterpret_cast<const uint4*>(x + static_cast<size_t>(r) * ldx2 + g0 * 32)[c];
        xsh[r][c] = v;
    }
    for (int i = threadIdx.x; i < MR * GPS; i += blockDim.x) {
        const int r = i / GPS, c = i % GPS;
        xssh[r][c] = r < m && c < groups ? xs[static_cast<size_t>(r) * kg + g0 + c] : 0.0f;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int col = blockIdx.x * COLS + warp * 16 + (lane >> 1);
    const int half = lane & 1;                                   // inputs 32 * half .. 32 * half + 31 of a group
    const bool live = col < n;
    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) acc[r] = 0.0f;
    Fetch next[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) next[j] = fetch(words, scales, biases, g0 + j, j < groups && live, kg, col, half);
    for (int j = 0; j < groups; ++j) {
        const Fetch cur = next[0];
        next[0] = next[1];
        next[1] = fetch(words, scales, biases, g0 + j + 2, j + 2 < groups && live, kg, col, half);
        const int4 w = cur.w;
        const unsigned packed[4] = {static_cast<unsigned>(w.x), static_cast<unsigned>(w.y),
                                    static_cast<unsigned>(w.z), static_cast<unsigned>(w.w)};
        unsigned q[16];
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
            for (int b = 0; b < 4; ++b) q[4 * t + b] = pair(packed[t], b);
        const float s = bf16_value(cur.s), bias = fmaf(-128.0f, s, bf16_value(cur.b));
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            const uint4* xr = &xsh[r][j * 8 + 4 * half];
            float p = 0.0f;
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const uint4 pairs = xr[v];
                const unsigned xp[4] = {pairs.x, pairs.y, pairs.z, pairs.w};
#pragma unroll
                for (int k = 0; k < 4; ++k)
                    p = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, q[4 * v + k]),
                                                        __builtin_bit_cast(bf16x2, xp[k]), p, false);
            }
            p = p + __shfl_xor(p, 1);                            // the group's two halves (addition commutes)
            acc[r] = fmaf(xssh[r][j], bias, fmaf(p, s, acc[r]));
        }
    }
    if (!live || half) return;
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        if (r >= m) break;
        if (part != nullptr) part[(static_cast<size_t>(slice) * m + r) * n + col] = acc[r];
        else out[static_cast<size_t>(r) * n + col] = bf16_round(acc[r]);
    }
}

// The same group arithmetic on WMMA (v_wmma_f32_16x16x16_bf16). An item is 128 outputs (8 warps x 16) over one K
// slice, walked in slabs of SLAB groups whose 16 input rows sit in shared memory (8 KB, rows padded off each other's
// banks). Blocks are persistent: each takes items blockIdx.x, + gridDim.x, ..., and its weight reads run a slab ahead
// across items, so a short item does not pay a fresh memory latency. Lane l: B and D column l % 16; half h = l / 16
// holds a group's inputs 32h..32h+31, K step s of four taking inputs 32h + 8s .. 32h + 8s + 7 (word 4h + s) for B
// and for A's row l % 16. D holds rows 8h .. 8h + 7. Padding rows are zeros; the row count changes nothing a row
// computes, and neither does the grid: an item's slice bounds come from the weight's shape.
constexpr int SLAB = 4;         // groups a slab stages (their WMMA chains interleave)

__device__ __forceinline__ Fetch fetch_nt(const int4* __restrict__ words, const unsigned short* __restrict__ scales,
                                          const unsigned short* __restrict__ biases, int g, bool ok, int kg, int col,
                                          int half) {
    Fetch f{make_int4(0, 0, 0, 0), 0, 0};
    if (ok) {
        const size_t at = tile_at(g, kg, col);
        const int4v v = __builtin_nontemporal_load(reinterpret_cast<const int4v*>(words) + at * 2 + half);
        f.w = make_int4(v[0], v[1], v[2], v[3]);                  // read once a step: keep L2 for x and partials
        f.s = scales[at];
        f.b = biases[at];
    }
    return f;
}

// Split K: an item stores its slice's sums; the one that finds its column's other slices done adds them in slice
// order (reduce_kernel's arithmetic) and writes the output. counts[column block] returns to zero for the next call.
// MT 16-row tiles share each weight fetch (a 32-row window reads the weights once); a tile's WMMA sequence is the
// same with one tile or two, so a row's bits do not depend on which pass or tile holds it.
template <int MT>
__global__ void __launch_bounds__(32 * WARPS) wmma_kernel(
        const unsigned* __restrict__ x, int ldx2, int m, const float* __restrict__ xs, int kg, int gps, int slices,
        const int4* __restrict__ words, const unsigned short* __restrict__ scales,
        const unsigned short* __restrict__ biases, int n, float* __restrict__ part, void* __restrict__ out, bool f32,
        int* __restrict__ counts) {
    __shared__ uint4 xsh[MT * 16][SLAB * 8 + 1];                // 8 bf16 a uint4; +1: rows off each other's banks
    __shared__ float xssh[MT * 16][SLAB];
    __shared__ int last;
    const int blocks = (n + COLS - 1) / COLS, items = blocks * slices;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int h = lane >> 4, c = lane & 15;
    int item = blockIdx.x;
    if (item >= items) return;
    // the step after (item, gs) in this block's schedule: the next slab, else the next item's first
    auto after = [&](int it, int gs, int& nit, int& ngs) {
        const int g1 = min(kg, (it / blocks) * gps + gps);
        if (gs + SLAB < g1) {
            nit = it;
            ngs = gs + SLAB;
        } else {
            nit = it + gridDim.x;
            ngs = (nit / blocks) * gps;
        }
    };
    auto column = [&](int it) { return (it % blocks) * COLS + warp * 16 + c; };
    auto slab = [&](int it, int gs, Fetch (&f)[SLAB]) {
        const int col = column(it), g1 = min(kg, (it / blocks) * gps + gps);
        const bool ok = it < items && col < n;
#pragma unroll
        for (int j = 0; j < SLAB; ++j) f[j] = fetch_nt(words, scales, biases, gs + j, ok && gs + j < g1, kg, col, h);
    };
    int gs = (item / blocks) * gps;
    Fetch cur[SLAB], nxt[SLAB];
    slab(item, gs, cur);
    float8 acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = float8{0, 0, 0, 0, 0, 0, 0, 0};
    while (true) {
        const int slice = item / blocks, g1 = min(kg, slice * gps + gps);
        const int groups = min(SLAB, g1 - gs);
        __syncthreads();                                        // the previous slab is consumed
        for (int i = threadIdx.x; i < MT * 16 * SLAB * 8; i += blockDim.x) {
            const int r = i / (SLAB * 8), cc = i % (SLAB * 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (r < m && cc < groups * 8)
                v = reinterpret_cast<const uint4*>(x + static_cast<size_t>(r) * ldx2 + gs * 32)[cc];
            xsh[r][cc] = v;
        }
        if (threadIdx.x < MT * 16 * SLAB) {
            const int r = threadIdx.x / SLAB, cc = threadIdx.x % SLAB;
            xssh[r][cc] = r < m && cc < groups ? xs[static_cast<size_t>(r) * kg + gs + cc] : 0.0f;
        }
        __syncthreads();
        int nit, ngs;
        after(item, gs, nit, ngs);
        slab(nit, ngs, nxt);                                    // the next step's reads, a slab ahead
        float8 p[MT][SLAB];
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int j = 0; j < SLAB; ++j) p[t][j] = float8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int s = 0; s < 4; ++s)
#pragma unroll
            for (int j = 0; j < SLAB; ++j) {                    // independent chains: groups' WMMAs interleave
                const unsigned word = s == 0 ? cur[j].w.x : s == 1 ? cur[j].w.y : s == 2 ? cur[j].w.z : cur[j].w.w;
                const uint4 b = make_uint4(pair(word, 0), pair(word, 1), pair(word, 2), pair(word, 3));
#pragma unroll
                for (int t = 0; t < MT; ++t) {
                    const uint4 a = xsh[16 * t + c][j * 8 + 4 * h + s];
                    p[t][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(short8, a),
                                                                               __builtin_bit_cast(short8, b), p[t][j]);
                }
            }
#pragma unroll
        for (int j = 0; j < SLAB; ++j) {                        // then the groups in order
            if (j < groups) {
                const float sc = bf16_value(cur[j].s), bias = fmaf(-128.0f, sc, bf16_value(cur[j].b));
#pragma unroll
                for (int t = 0; t < MT; ++t)
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        acc[t][i] = fmaf(xssh[16 * t + 8 * h + i][j], bias, fmaf(p[t][j][i], sc, acc[t][i]));
            }
            cur[j] = nxt[j];
        }
        if (nit == item) {
            gs = ngs;
            continue;
        }
        // the item is done: its output, or its slice's sums and perhaps the column's
        const int col = column(item);
        const bool live = col < n;
        if (slices == 1) {
            if (live) {
#pragma unroll
                for (int t = 0; t < MT; ++t)
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int r = 16 * t + 8 * h + i;
                        if (r >= m) break;
                        const size_t at = static_cast<size_t>(r) * n + col;
                        if (f32) static_cast<float*>(out)[at] = acc[t][i];
                        else static_cast<unsigned short*>(out)[at] = bf16_round(acc[t][i]);
                    }
            }
        } else {
            if (live) {
#pragma unroll
                for (int t = 0; t < MT; ++t)
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int r = 16 * t + 8 * h + i;
                        if (r >= m) break;
                        part[(static_cast<size_t>(slice) * m + r) * n + col] = acc[t][i];
                    }
            }
            // release: this item's stores only (a full fence would also wait out the next item's prefetched loads)
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(&counts[item % blocks], 1) == slices - 1;
            __syncthreads();
            if (last) {
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
                if (live) {
#pragma unroll
                    for (int t = 0; t < MT; ++t)
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int r = 16 * t + 8 * h + i;
                            if (r >= m) break;
                            float sum = __hip_atomic_load(&part[static_cast<size_t>(r) * n + col], __ATOMIC_RELAXED,
                                                          __HIP_MEMORY_SCOPE_AGENT);
                            for (int u = 1; u < slices; ++u)
                                sum = sum + __hip_atomic_load(&part[(static_cast<size_t>(u) * m + r) * n + col],
                                                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                            const size_t at = static_cast<size_t>(r) * n + col;
                            if (f32) static_cast<float*>(out)[at] = sum;
                            else static_cast<unsigned short*>(out)[at] = bf16_round(sum);
                        }
                }
                if (threadIdx.x == 0) counts[item % blocks] = 0;
            }
        }
        item = nit;
        if (item >= items) break;
        gs = ngs;
#pragma unroll
        for (int t = 0; t < MT; ++t) acc[t] = float8{0, 0, 0, 0, 0, 0, 0, 0};
    }
}

__global__ void reduce_kernel(const float* __restrict__ part, int slices, size_t total, float* __restrict__ out32,
                              unsigned short* __restrict__ out16) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = part[i];
    for (int s = 1; s < slices; ++s) acc = acc + part[static_cast<size_t>(s) * total + i];
    if (out32 != nullptr) out32[i] = acc;
    else out16[i] = bf16_round(acc);
}

}  // namespace

int gemv_slices(int kg) { return (kg + GPS - 1) / GPS; }

// Persistent WMMA blocks for ``items``: as many as the GPU holds at once, fewer when that evens out their item counts.
// Only scheduling: an item's arithmetic never depends on the grid. ``TF_ROCM_WMMA_GRID`` caps it for tuning.
int persistent_grid(int items, int mt) {
    static const int two = [] {
        int per_cu = 0;
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, reinterpret_cast<const void*>(wmma_kernel<2>),
                                                     32 * WARPS, 0);
        return std::max(1, per_cu) * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    }();
    static const int resident = [] {
        int per_cu = 0;
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, reinterpret_cast<const void*>(wmma_kernel<1>),
                                                     32 * WARPS, 0);
        const char* cap = std::getenv("TF_ROCM_WMMA_GRID");
        const int all = std::max(1, per_cu) * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        return cap != nullptr && std::atoi(cap) > 0 ? std::min(all, std::atoi(cap)) : all;
    }();
    const int held = mt == 2 ? std::min(two, resident) : resident;
    const int rounds = (items + held - 1) / held;
    return (items + rounds - 1) / rounds;
}

// WMMA K slices: the fewest (1, 2, 4, 8 or 16) whose grid reaches ``fill`` blocks, each a whole number of slabs.
// A function of the weight's shape only, never of the row count.
int wmma_slices(int kg, int n, int fill) {
    const int cols = (n + COLS - 1) / COLS;
    int best = 1;
    for (int dks = 1; dks <= 16; dks *= 2) {
        if (kg % (dks * SLAB)) break;
        best = dks;
        if (cols * dks >= fill) break;
    }
    return best;
}

// x (M, K) bf16 rows (even stride), xs (M, K/64) fp32 group sums, tiled words (ceil(N/16), K/64, 16, 8) int32 and
// scales/biases (ceil(N/16), K/64, 16) bf16 -> out (M, N) bf16 or fp32; part (slices * 16 * N) fp32 scratch when
// K is split; counts: zeros, one per 128 outputs (the WMMA kernel's split-K arrivals, left at zero).
void gemv_groups(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& words, const at::Tensor& scales,
                 const at::Tensor& biases, int n, at::Tensor& out, at::Tensor& part, bool wmma, int fill,
                 at::Tensor& counts) {
    const int kg = words.size(1), m = x.size(0);
    const int slices = wmma ? wmma_slices(kg, n, fill) : gemv_slices(kg);
    const int gps = kg / slices;
    auto stream = at::hip::getCurrentHIPStream();
    const bool f32 = out.scalar_type() == at::kFloat;
    const bool direct = slices == 1 && !f32;                   // gemv_kernel: one slice writes bf16 itself
    const int pass = wmma ? 2 * ROWS : ROWS;                   // the WMMA kernel takes 32 rows a pass (two tiles)
    for (int r0 = 0; r0 < m; r0 += pass) {
        const int rows = std::min(pass, m - r0);
        const dim3 grid((n + COLS - 1) / COLS, slices);
        float* p = (wmma ? slices > 1 : !direct) ? part.data_ptr<float>() : nullptr;
        const auto* xr = reinterpret_cast<const unsigned*>(x.data_ptr()) + static_cast<size_t>(r0) * (x.stride(0) / 2);
        const int ld = static_cast<int>(x.stride(0) / 2);
        const float* xsr = xs.data_ptr<float>() + static_cast<size_t>(r0) * kg;
        const auto* w = reinterpret_cast<const int4*>(words.data_ptr());
        const auto* sc = reinterpret_cast<const unsigned short*>(scales.data_ptr());
        const auto* bi = reinterpret_cast<const unsigned short*>(biases.data_ptr());
        unsigned short* o = direct ? reinterpret_cast<unsigned short*>(out.data_ptr()) + static_cast<size_t>(r0) * n
                                   : nullptr;
#define GEMV(MR) hipLaunchKernelGGL(gemv_kernel<MR>, grid, dim3(32 * WARPS), 0, stream, xr, ld, rows, xsr, kg, w, sc, \
                                    bi, n, p, o)
        void* o_w = static_cast<char*>(out.data_ptr()) + static_cast<size_t>(r0) * n * out.element_size();
        if (wmma && rows <= ROWS)
            hipLaunchKernelGGL(wmma_kernel<1>, dim3(persistent_grid(grid.x * slices, 1)), dim3(32 * WARPS), 0, stream,
                               xr, ld, rows, xsr, kg, gps, slices, w, sc, bi, n, p, o_w, f32, counts.data_ptr<int>());
        else if (wmma)
            hipLaunchKernelGGL(wmma_kernel<2>, dim3(persistent_grid(grid.x * slices, 2)), dim3(32 * WARPS), 0, stream,
                               xr, ld, rows, xsr, kg, gps, slices, w, sc, bi, n, p, o_w, f32, counts.data_ptr<int>());
        else if (rows == 1) GEMV(1);
        else if (rows <= 2) GEMV(2);
        else if (rows <= 4) GEMV(4);
        else if (rows <= 8) GEMV(8);
        else GEMV(16);
#undef GEMV
        if (!wmma && !direct) {
            const size_t total = static_cast<size_t>(rows) * n;
            hipLaunchKernelGGL(reduce_kernel, dim3((total + 255) / 256), dim3(256), 0, stream, p, slices, total,
                               f32 ? out.data_ptr<float>() + static_cast<size_t>(r0) * n : nullptr,
                               f32 ? nullptr
                                   : reinterpret_cast<unsigned short*>(out.data_ptr()) + static_cast<size_t>(r0) * n);
        }
    }
}
