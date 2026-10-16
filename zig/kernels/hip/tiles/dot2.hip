// FP16 dot2 schedules: the one-thread reference and the column, tiled, split and wide prefill tiles.

#include "tiles/dot2.hpp"

#include <hip/hip_fp16.h>

namespace tf {
namespace rocm {

__device__ __half code_f16(uint32_t code) {
    return __float2half_rn(code_bf16(code));
}

__global__ void __launch_bounds__(32) affine_dot2_kernel(Affine a) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (row >= a.m || col >= a.n) return;

    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    const uint32_t* wrow = a.words + static_cast<long long>(col) * words;
    const __half* xrow = static_cast<const __half*>(a.x) + static_cast<long long>(row) * a.k;
    float acc = 0.f;
    for (int group = 0; group < groups; ++group) {
        float dot = 0.f;
        float sum = 0.f;
        int base = group * a.group;
        for (int t = 0; t < a.group; t += 2) {
            __half x0 = xrow[base + t];
            __half x1 = xrow[base + t + 1];
            __half q0 = code_f16(affine_code(wrow, base + t, a.bits, words));
            __half q1 = code_f16(affine_code(wrow, base + t + 1, a.bits, words));
            dot = __builtin_amdgcn_fdot2(__halves2half2(x0, x1), __halves2half2(q0, q1), dot, false);
            sum += static_cast<float>(x0);
            sum += static_cast<float>(x1);
        }
        acc = fmaf(dot, a.scale[static_cast<long long>(col) * groups + group], acc);
        acc = fmaf(sum, a.bias[static_cast<long long>(col) * groups + group], acc);
    }
    a.out[static_cast<long long>(row) * a.n + col] = acc;
}

// Column tile: a thread owns a column and walks a staged K chunk in pair order; scale only after the whole group.
constexpr int kFastM = 8;
constexpr int kFastN = 256;
constexpr int kFastK = 64;
constexpr int kFastLd = 68;

// fp16(bf16(code)) for every byte. Filled by code_f16 so the product rounding does not change.
__constant__ __half kByteF16[256];

__global__ void fill_byte_lut(__half* lut) {
    int i = threadIdx.x;
    if (i < 256) lut[i] = code_f16(static_cast<uint32_t>(i));
}

__device__ void store_codes(__half* dst, uint32_t packed) {
    dst[0] = kByteF16[packed & 0xffu];
    dst[1] = kByteF16[(packed >> 8) & 0xffu];
    dst[2] = kByteF16[(packed >> 16) & 0xffu];
    dst[3] = kByteF16[packed >> 24];
}

// Four packed words per load. An 8-bit group starts on a 16-byte boundary.
__device__ void load_byte_tile(__half* tile, const uint32_t* words, int words_per_row, int n, int n_cols, int tid,
                               int chunk, int gbase, int k0) {
    int vecs = chunk >> 4;
    int nvecs = n_cols * vecs;
    for (int e = tid; e < nvecs; e += blockDim.x) {
        int c = e / vecs;
        int v = e - c * vecs;
        int gc = blockIdx.x * n_cols + c;
        int word = ((gbase + k0) >> 2) + (v << 2);
        uint4 packed = make_uint4(0u, 0u, 0u, 0u);
        if (gc < n) {
            const uint32_t* src = words + static_cast<long long>(gc) * words_per_row + word;
            packed = *reinterpret_cast<const uint4*>(src);
        }
        int base = c * kFastLd + (v << 4);
        store_codes(tile + base, packed.x);
        store_codes(tile + base + 4, packed.y);
        store_codes(tile + base + 8, packed.z);
        store_codes(tile + base + 12, packed.w);
    }
}

// One column's chunk of codes in LDS, a 32-code piece a load; 8-bit keeps the byte path.
template <int BITS>
__device__ void load_piece_tile(__half* tile, const uint32_t* words, int words_per_row, int n, int n_cols, int tid,
                                int chunk, int gbase, int k0) {
    const int pieces = chunk / 32;
    for (int e = tid; e < n_cols * pieces; e += blockDim.x) {
        const int c = e / pieces;
        const int p = e - c * pieces;
        const int gc = blockIdx.x * n_cols + c;
        __half* dst = tile + c * kFastLd + p * 32;
        if (gc >= n) {
#pragma unroll
            for (int t = 0; t < 32; ++t) dst[t] = __float2half(0.f);
            continue;
        }
        uint32_t w[BITS];
        load_piece<BITS>(words + static_cast<long long>(gc) * words_per_row + (gbase + k0 + p * 32) / 32 * BITS, w);
#pragma unroll
        for (int t = 0; t < 32; ++t) dst[t] = piece_code<BITS>(w, t);
    }
}

__device__ void load_code_tile(__half* tile, const Affine& a, int words_per_row, int n_cols, int tid, int chunk,
                               int gbase, int k0) {
    switch (a.bits) {
        case 2: load_piece_tile<2>(tile, a.words, words_per_row, a.n, n_cols, tid, chunk, gbase, k0); break;
        case 3: load_piece_tile<3>(tile, a.words, words_per_row, a.n, n_cols, tid, chunk, gbase, k0); break;
        case 4: load_piece_tile<4>(tile, a.words, words_per_row, a.n, n_cols, tid, chunk, gbase, k0); break;
        case 5: load_piece_tile<5>(tile, a.words, words_per_row, a.n, n_cols, tid, chunk, gbase, k0); break;
        case 6: load_piece_tile<6>(tile, a.words, words_per_row, a.n, n_cols, tid, chunk, gbase, k0); break;
        default: load_byte_tile(tile, a.words, words_per_row, a.n, n_cols, tid, chunk, gbase, k0); break;
    }
}

// Prefill tile: each output is still one thread's fdot2 chain in pair order.
constexpr int kDotBM = 16;
constexpr int kDotBN = 16;
constexpr int kDotGroup = 128;

__global__ void affine_dot2_tiled(Affine a) {
    __shared__ __half xs[kDotBM * kDotGroup];
    __shared__ __half ws[kDotBN * kDotGroup];
    int local_row = threadIdx.x / kDotBN;
    int local_col = threadIdx.x % kDotBN;
    int row = blockIdx.y * kDotBM + local_row;
    int col = blockIdx.x * kDotBN + local_col;
    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    bool live = row < a.m && col < a.n;
    float acc = 0.f;
    const __half* x = static_cast<const __half*>(a.x);

    for (int group = 0; group < groups; ++group) {
        int base = group * a.group;
        int span = kDotBM * a.group;
        for (int e = threadIdx.x; e < span; e += blockDim.x) {
            int r = e / a.group;
            int c = e - r * a.group;
            int gr = blockIdx.y * kDotBM + r;
            int gk = base + c;
            __half xv = __float2half(0.f);
            if (gr < a.m && gk < a.k) xv = x[static_cast<long long>(gr) * a.k + gk];
            xs[r * a.group + c] = xv;
        }
        for (int e = threadIdx.x; e < span; e += blockDim.x) {
            int r = e / a.group;
            int c = e - r * a.group;
            int gc = blockIdx.x * kDotBN + r;
            int gk = base + c;
            __half qv = __float2half(0.f);
            if (gc < a.n && gk < a.k) {
                const uint32_t* wrow = a.words + static_cast<long long>(gc) * words;
                qv = code_f16(affine_code(wrow, gk, a.bits, words));
            }
            ws[r * a.group + c] = qv;
        }
        __syncthreads();
        if (live) {
            float dot = 0.f;
            float sum = 0.f;
            const __half* xrow = xs + local_row * a.group;
            const __half* wrow = ws + local_col * a.group;
            for (int t = 0; t < a.group; t += 2) {
                __half x0 = xrow[t];
                __half x1 = xrow[t + 1];
                __half q0 = wrow[t];
                __half q1 = wrow[t + 1];
                dot = __builtin_amdgcn_fdot2(__halves2half2(x0, x1), __halves2half2(q0, q1), dot, false);
                sum += static_cast<float>(x0);
                sum += static_cast<float>(x1);
            }
            int g = group;
            acc = fmaf(dot, a.scale[static_cast<long long>(col) * groups + g], acc);
            acc = fmaf(sum, a.bias[static_cast<long long>(col) * groups + g], acc);
        }
        __syncthreads();
    }
    if (live) a.out[static_cast<long long>(row) * a.n + col] = acc;
}

// affine_dot2_fast's walk over groups [z * per, (z + 1) * per); affine_dot2_fold applies them in group order.
template <int Rows>
__global__ void __launch_bounds__(kFastN, 1) affine_dot2_span(Affine a, float* partial, int per) {
    __shared__ __half xs[Rows * kFastLd];
    __shared__ __half ws[kFastN * kFastLd];
    int tid = threadIdx.x;
    int col = blockIdx.x * kFastN + tid;
    int row0 = blockIdx.y * kFastM;
    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    int chunk = a.group < kFastK ? a.group : kFastK;
    int g0 = blockIdx.z * per;
    int g1 = g0 + per;
    const __half* x = static_cast<const __half*>(a.x);

    for (int group = g0; group < g1; ++group) {
        int gbase = group * a.group;
        float dot[Rows];
        float sumx[Rows];
#pragma unroll
        for (int m = 0; m < Rows; ++m) {
            dot[m] = 0.f;
            sumx[m] = 0.f;
        }
        for (int k0 = 0; k0 < a.group; k0 += chunk) {
            if (a.bits == 8) {
                load_byte_tile(ws, a.words, words, a.n, kFastN, tid, chunk, gbase, k0);
            } else {
                int span = kFastN * chunk;
                for (int e = tid; e < span; e += kFastN) {
                    int c = e / chunk;
                    int kk = e - c * chunk;
                    int gc = blockIdx.x * kFastN + c;
                    int gk = gbase + k0 + kk;
                    __half qv = __float2half(0.f);
                    if (gc < a.n && gk < a.k) {
                        const uint32_t* wrow = a.words + static_cast<long long>(gc) * words;
                        qv = code_f16(affine_code(wrow, gk, a.bits, words));
                    }
                    ws[c * kFastLd + kk] = qv;
                }
            }
            int xspan = Rows * chunk;
            for (int e = tid; e < xspan; e += kFastN) {
                int r = e / chunk;
                int kk = e - r * chunk;
                int gr = row0 + r;
                int gk = gbase + k0 + kk;
                __half xv = __float2half(0.f);
                if (gr < a.m && gk < a.k) xv = x[static_cast<long long>(gr) * a.k + gk];
                xs[r * kFastLd + kk] = xv;
            }
            __syncthreads();
            if (col < a.n) {
                for (int t = 0; t < chunk; t += 2) {
                    __half2 q = __halves2half2(ws[tid * kFastLd + t], ws[tid * kFastLd + t + 1]);
#pragma unroll
                    for (int m = 0; m < Rows; ++m) {
                        __half x0 = xs[m * kFastLd + t];
                        __half x1 = xs[m * kFastLd + t + 1];
                        dot[m] = __builtin_amdgcn_fdot2(__halves2half2(x0, x1), q, dot[m], false);
                        sumx[m] += static_cast<float>(x0);
                        sumx[m] += static_cast<float>(x1);
                    }
                }
            }
            __syncthreads();
        }
        if (col < a.n) {
#pragma unroll
            for (int m = 0; m < Rows; ++m) {
                int row = row0 + m;
                if (row < a.m) {
                    float* slot = partial + ((static_cast<long long>(row) * a.n + col) * groups + group) * 2;
                    slot[0] = dot[m];
                    slot[1] = sumx[m];
                }
            }
        }
    }
}

__global__ void __launch_bounds__(kFastN) affine_dot2_fold(const float* partial, GroupTable scale, GroupTable bias,
                                                          float* out, int m, int n, int groups) {
    int col = blockIdx.x * kFastN + threadIdx.x;
    int row = blockIdx.y;
    if (row >= m || col >= n) return;
    const float* base = partial + (static_cast<long long>(row) * n + col) * groups * 2;
    float acc = 0.f;
    for (int group = 0; group < groups; ++group) {
        acc = fmaf(base[group * 2], scale[static_cast<long long>(col) * groups + group], acc);
        acc = fmaf(base[group * 2 + 1], bias[static_cast<long long>(col) * groups + group], acc);
    }
    out[static_cast<long long>(row) * n + col] = acc;
}

// Prefill tile: 128 rows share one load of 32 columns; a thread owns a column and eight rows.
constexpr int kWideM = 128;
constexpr int kWideN = 32;
constexpr int kWideRows = 8;

__device__ void load_wide_chunk(__half* xs, __half* ws, const Affine& a, int tid, int row_base, int words,
                                 int chunk, int gbase, int k0) {
    int vecs = chunk >> 2;
    int nvecs = kWideM * vecs;
    for (int e = tid; e < nvecs; e += blockDim.x) {
        int r = e / vecs;
        int v = e - r * vecs;
        int kk = v << 2;
        int gr = row_base + r;
        int gk = gbase + k0 + kk;
        uint2 packed = make_uint2(0u, 0u);
        if (gr < a.m && gk + 3 < a.k) {
            const __half* src = static_cast<const __half*>(a.x) + static_cast<long long>(gr) * a.k + gk;
            packed = *reinterpret_cast<const uint2*>(src);
        }
        *reinterpret_cast<uint2*>(xs + r * kFastLd + kk) = packed;
    }
    load_code_tile(ws, a, words, kWideN, tid, chunk, gbase, k0);
}

__device__ void dot_wide_chunk(float dot[kWideRows], float sumx[kWideRows], const __half* xs, const __half* ws,
                               int col_local, int row_group, int chunk) {
    for (int t = 0; t < chunk; t += 2) {
        __half2 q = __halves2half2(ws[col_local * kFastLd + t], ws[col_local * kFastLd + t + 1]);
#pragma unroll
        for (int m = 0; m < kWideRows; ++m) {
            int r = row_group * kWideRows + m;
            __half x0 = xs[r * kFastLd + t];
            __half x1 = xs[r * kFastLd + t + 1];
            dot[m] = __builtin_amdgcn_fdot2(__halves2half2(x0, x1), q, dot[m], false);
            sumx[m] += static_cast<float>(x0);
            sumx[m] += static_cast<float>(x1);
        }
    }
}

__global__ void __launch_bounds__(512, 1) affine_dot2_wide(Affine a) {
    __shared__ __half xs[2][kWideM * kFastLd];
    __shared__ __half ws[2][kWideN * kFastLd];
    int tid = threadIdx.x;
    int col_local = tid % kWideN;
    int row_group = tid / kWideN;
    int col = blockIdx.x * kWideN + col_local;
    int row_base = blockIdx.y * kWideM;
    int my_row = row_base + row_group * kWideRows;
    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    int chunk = a.group < kFastK ? a.group : kFastK;
    int per = a.group / chunk;
    int nchunks = groups * per;
    float acc[kWideRows];
    float dot[kWideRows];
    float sumx[kWideRows];
#pragma unroll
    for (int m = 0; m < kWideRows; ++m) {
        acc[m] = 0.f;
        dot[m] = 0.f;
        sumx[m] = 0.f;
    }
    if (nchunks > 0) load_wide_chunk(xs[0], ws[0], a, tid, row_base, words, chunk, 0, 0);
    __syncthreads();

    int phase = 0;
    for (int ci = 0; ci < nchunks; ++ci) {
        int next = phase ^ 1;
        if (ci + 1 < nchunks) {
            int nci = ci + 1;
            int ng = nci / per;
            int nk0 = (nci - ng * per) * chunk;
            load_wide_chunk(xs[next], ws[next], a, tid, row_base, words, chunk, ng * a.group, nk0);
        }
        if (col < a.n) dot_wide_chunk(dot, sumx, xs[phase], ws[phase], col_local, row_group, chunk);
        if ((ci % per) == per - 1 && col < a.n) {
            int group = ci / per;
            float scale = a.scale[static_cast<long long>(col) * groups + group];
            float biasv = a.bias[static_cast<long long>(col) * groups + group];
#pragma unroll
            for (int m = 0; m < kWideRows; ++m) {
                acc[m] = fmaf(dot[m], scale, acc[m]);
                acc[m] = fmaf(sumx[m], biasv, acc[m]);
                dot[m] = 0.f;
                sumx[m] = 0.f;
            }
        }
        __syncthreads();
        phase = next;
    }
    if (col < a.n) {
#pragma unroll
        for (int m = 0; m < kWideRows; ++m) {
            int row = my_row + m;
            if (row < a.m) a.out[static_cast<long long>(row) * a.n + col] = acc[m];
        }
    }
}

// The split tile's two row counts launches/affine.zig resolves.
template __global__ void affine_dot2_span<1>(Affine, float*, int);
template __global__ void affine_dot2_span<kFastM>(Affine, float*, int);

}  // namespace rocm
}  // namespace tf
