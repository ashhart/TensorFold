// GLM-5.3-Flash's EXL3 routed experts (4-bit "mcg" trellis, ExLlamaV3's format: MIT, Copyright (c) 2025 Turboderp)
// on CUDA: the grouped trellis GEMV/GEMM and the Hadamard rotations around it. From TensorFold's python-0.6
// src/tensorfold/families/glm5_next/cuda/exl3.cu, without torch, over the routed pairs alone (the shared expert runs
// on the dense linears), with items of 64 pairs for prompt chunks (each expert's tiles read once a chunk).
//
// A 16x16 tile is 32 little-endian 32-bit words; lane L of a warp decodes the tile's values 8L..8L+7 from words L-1
// and L, and those eight values are exactly the B fragments of two mma.m16n8k16 (columns 0-7 and 8-15 of the tile).
// Every output depends only on its own pair: pairs share an mma tile but mma keeps rows independent, the K range of
// every warp and split is fixed by the shape, and warps and splits are summed in a fixed order, never with atomics.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Two values of the "mcg" codebook from two 16-bit states, as a half2 (first state in .x).
__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}

// This lane's eight values of a 4-bit tile (word = tile[lane]) as the B fragments of its two n8 halves.
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg2((w >> 4) & 0xffffu, w & 0xffffu);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

// Grid (item, n block, mat * SK + split) -> Z[mat][split][pair][n]; an item is an expert and up to 16 MT of its pairs
// (the plan's members); warps sum fixed k ranges, added in warp order.
template <int MT, int NT, int W>
__device__ __forceinline__ void grouped(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK) {
    const int item = blockIdx.x;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    const int split = blockIdx.z % SK;
    const int mat = blockIdx.z / SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;
    constexpr int ROWS = 16 * MT;

    __shared__ int rows_sh[ROWS];
    for (int i = threadIdx.x; i < ROWS; i += W * 32) rows_sh[i] = i < cnt ? members[first + i] : -1;
    __syncthreads();
    int r0[MT], r1[MT];
    const half* x0[MT];
    const half* x1[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        r0[m] = rows_sh[m * 16 + g];
        r1[m] = rows_sh[m * 16 + g + 8];
        x0[m] = X + (size_t)(r0[m] < 0 ? 0 : r0[m]) * K + 2 * t;
        x1[m] = X + (size_t)(r1[m] < 0 ? 0 : r1[m]) * K + 2 * t;
    }

    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = blockIdx.y * NT;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[MT][NT][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;

    for (int kt = kt0; kt < kt0 + per_warp; ++kt) {
        uint32_t words[NT];
#pragma unroll
        for (int i = 0; i < NT; ++i) words[i] = __ldg(tile + i * 32);
        const int k = kt * 16;
        uint32_t a[MT][4];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            a[m][0] = load_pair(x0[m] + k, r0[m] >= 0);
            a[m][1] = load_pair(x1[m] + k, r1[m] >= 0);
            a[m][2] = load_pair(x0[m] + k + 8, r0[m] >= 0);
            a[m][3] = load_pair(x1[m] + k + 8, r1[m] >= 0);
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                mma16816(acc[m][i][0], a[m], b0);
                mma16816(acc[m][i][1], a[m], b1);
            }
        }
        tile += (size_t)NTILES * 32;
    }

    // warps' partial sums through shared memory, added in warp order
    __shared__ float red[W][ROWS][NT * 16];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][m * 16 + g][col] = acc[m][i][h][0];
                red[warp][m * 16 + g][col + 1] = acc[m][i][h][1];
                red[warp][m * 16 + g + 8][col] = acc[m][i][h][2];
                red[warp][m * 16 + g + 8][col + 1] = acc[m][i][h][3];
            }
    __syncthreads();
    for (int idx = threadIdx.x; idx < ROWS * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
    }
}

// Fast Walsh-Hadamard transform of 128 values held 4 per lane (lane L: values 4L..4L+3), natural order, fixed
// butterfly order: strides 1, 2 in registers, 4..64 across lanes.
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }


// -- prompt chunks: the staged kernel -------------------------------------------------------------------------------
// An item is an expert and up to PM of its pairs (the plan's members, tile PM). Block (item, column block of
// NW * NTW n-tiles, matrix): the four warps split the columns, not K, so a warp's sums need no cross-warp reduction
// and every decoded tile feeds PM / 8 mma; the member rows' k-slices and the tiles' words are staged in shared
// memory by cp.async, STAGES deep, and the A fragments come from the staged rows by ldmatrix. The same sums as the
// grouped kernel for every output, in a fixed order (one warp owns each output; K runs in order).
//
// decode_tile's values with fewer integer instructions: the byte-aligned 16-bit states taken with one PRMT each
// and w >> 4 shared by two states (MiaAI-Lab, GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold patch 0020, Apache-2.0).
// The same states, so the same bits.
__device__ __forceinline__ void decode_tile_p(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    const uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    const uint32_t s = __funnelshift_r(w, p, 20);
    const uint32_t t = w >> 4;
    b0[0] = mcg2(__byte_perm(s, 0u, 0x4421), (s >> 4) & 0xffffu);          // (s >> 8) & 0xffff, (s >> 4) & 0xffff
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2(__byte_perm(t, 0u, 0x4421), __byte_perm(w, 0u, 0x4421));   // (w >> 12) & 0xffff, (w >> 8) & 0xffff
    b1[1] = mcg2(__byte_perm(t, 0u, 0x4410), w & 0xffffu);                  // (w >> 4) & 0xffff, w & 0xffff
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    const int n = valid ? 16 : 0;   // src-size 0: the 16 bytes come as zeros (a padding row)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&a)[4], const half* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}

// xshift: the X row a member pair reads is pair >> xshift (3: one rotated row a token, when every expert of the layer
// shares one input rotation for gate and up; 0: a row a pair).
// SK: K splits -> Z[mat][split][pair][n] (the grouped kernel's layout: glm_exl3_gateup adds the splits in order). A
// small chunk, whose items hold few pairs, gets SK times the blocks at a fraction of the K each.
// Grid: x = (column block, matrix of ``mats``, split), y = item: one item's blocks are adjacent in launch order, so
// its staged rows come from L2 after the first block reads them.
template <int PM, int NW, int NTW, int KS, int STAGES>
__device__ __forceinline__ void prompt_grouped(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int xshift, int SK, int mats) {
    constexpr int T = NW * 32, MT = PM / 16, NTB = NW * NTW, XS = 16 * KS + 8;   // XS: a staged row's halves (padded: ldmatrix conflict-free)
    constexpr int XCH = 2 * KS;                                                 // 16-byte chunks a staged row takes
    static_assert(PM * XCH % T == 0 && NTB * KS * 8 % T == 0, "the stage's chunks spread evenly over the threads");
    __shared__ __align__(16) half xs[STAGES][PM][XS];
    __shared__ __align__(16) uint32_t ws[STAGES][NTB][KS][32];
    __shared__ int rows_sh[PM];
    const int item = blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    const int cb = blockIdx.x / (mats * SK), ms = blockIdx.x % (mats * SK);
    const int mat = ms / SK, split = ms % SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* Tm = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;
    const int per_split = KT / SK, kt_base = split * per_split;   // KS divides per_split (K = 4096 or 1024, SK <= 4)
    const int nt0 = cb * NTB;
    for (int i = threadIdx.x; i < PM; i += T) rows_sh[i] = i < cnt ? members[first + i] : -1;
    __syncthreads();
    const int mt_n = (cnt + 15) >> 4;   // m-tiles with a member (uniform over the block)
    const uint32_t* tbase = Tm + ((size_t)e * KT * NTILES + nt0) * 32;

    auto issue = [&](int stage, int kt) {   // the stage's rows (k-tiles kt .. kt + KS - 1) and words
#pragma unroll
        for (int j = 0; j < PM * XCH / T; ++j) {
            const int c = threadIdx.x + j * T;
            const int row = c / XCH, part = c % XCH;
            if (row >= mt_n * 16) continue;   // an m-tile without a member: never read
            const int r = rows_sh[row];
            const half* src = X + (size_t)(r < 0 ? 0 : (r >> xshift)) * K + kt * 16 + part * 8;
            cp_async16(&xs[stage][row][part * 8], src, r >= 0);
        }
#pragma unroll
        for (int j = 0; j < NTB * KS * 8 / T; ++j) {
            const int c = threadIdx.x + j * T;
            const int tile = c / (KS * 8), kk = (c / 8) % KS, part = c % 8;
            const uint32_t* src = tbase + ((size_t)(kt + kk) * NTILES + tile) * 32 + part * 4;
            cp_async16(&ws[stage][tile][kk][part * 4], src, true);
        }
        cp_async_commit();
    };

    float acc[MT][NTW][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < NTW; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;

    const int nstages = per_split / KS;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nstages) issue(s, kt_base + s * KS);
        else cp_async_commit();
    }
    for (int st = 0; st < nstages; ++st) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int stage = st % STAGES;
        {   // the stage after the ones in flight: its buffer was consumed one iteration ago
            const int nxt = st + STAGES - 1;
            if (nxt < nstages) issue(nxt % STAGES, kt_base + nxt * KS);
            else cp_async_commit();
        }
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            uint32_t b[NTW][2][2];
#pragma unroll
            for (int i = 0; i < NTW; ++i) decode_tile_p(ws[stage][warp * NTW + i][kk][lane], lane, b[i][0], b[i][1]);
            for (int m = 0; m < mt_n; ++m) {
                uint32_t a[4];
                ldmatrix_x4(a, &xs[stage][m * 16 + (lane & 15)][kk * 16 + (lane >> 4) * 8]);
#pragma unroll
                for (int i = 0; i < NTW; ++i) {
                    mma16816(acc[m][i][0], a, b[i][0]);
                    mma16816(acc[m][i][1], a, b[i][1]);
                }
            }
        }
    }
    cp_async_wait<0>();
    // this warp's columns, rows g and g + 8 of each m-tile, straight to Z[mat][split][pair][n]
    float* Zs = Z + (size_t)(mat * SK + split) * P * N;
    for (int m = 0; m < mt_n; ++m) {
        const int r0 = rows_sh[m * 16 + g], r1 = rows_sh[m * 16 + g + 8];
#pragma unroll
        for (int i = 0; i < NTW; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = (nt0 + warp * NTW + i) * 16 + h * 8 + 2 * t;
                if (r0 >= 0) *reinterpret_cast<float2*>(Zs + (size_t)r0 * N + col) = make_float2(acc[m][i][h][0], acc[m][i][h][1]);
                if (r1 >= 0) *reinterpret_cast<float2*>(Zs + (size_t)r1 * N + col) = make_float2(acc[m][i][h][2], acc[m][i][h][3]);
            }
    }
}

}  // namespace

// prompt chunks: items of 128 pairs, 4 warps x 2 n-tiles (128 columns a block), 2 k-tiles a stage, 3 stages
extern "C" __global__ void __launch_bounds__(128) glm_exl3_prompt128(const half* __restrict__ X0, const half* __restrict__ X1,
        const uint32_t* __restrict__ T0, const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int xshift, int SK, int mats) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    prompt_grouped<128, 4, 2, 2, 3>(X0, X1, T0, T1, items, counts, members, Z, K, N, P, xshift, SK, mats);
}

// the same with 8 warps (256 columns a block): the item's rows staged half as often
extern "C" __global__ void __launch_bounds__(256) glm_exl3_prompt128w(const half* __restrict__ X0, const half* __restrict__ X1,
        const uint32_t* __restrict__ T0, const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int xshift, int SK, int mats) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    prompt_grouped<128, 8, 2, 2, 3>(X0, X1, T0, T1, items, counts, members, Z, K, N, P, xshift, SK, mats);
}

// Program (row, 128-block of K): Xh[row][block] = fp16((x[row] * suh) @ H), one rotation a token for the gate and up
// projections of a layer whose experts all share one suh (then every pair of the row reads Xh[row]: the same bits
// glm_exl3_rot_in gives each pair). One warp a program.
extern "C" __global__ void glm_exl3_rot_rows(const __nv_bfloat16* __restrict__ x, int x_stride, const half* __restrict__ suh,
        half* __restrict__ out, int K) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int row = blockIdx.x, blk = blockIdx.y;
    const int lane = threadIdx.x;
    const half* sh = suh + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(sh[j]);
    fwht128(v, lane);
    half* o = out + (size_t)row * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

namespace {
}  // namespace

// decode windows: items of 16 pairs, 8 n-tiles a block, 4 warps
extern "C" __global__ void __launch_bounds__(128) glm_exl3_grouped(const half* __restrict__ X0, const half* __restrict__ X1,
        const uint32_t* __restrict__ T0, const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    grouped<1, 8, 4>(X0, X1, T0, T1, items, counts, members, Z, K, N, P, SK);
}

// prompt chunks: items of 64 pairs, 2 n-tiles a block, 4 warps
extern "C" __global__ void __launch_bounds__(128) glm_exl3_grouped64(const half* __restrict__ X0, const half* __restrict__ X1,
        const uint32_t* __restrict__ T0, const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    grouped<4, 2, 4>(X0, X1, T0, T1, items, counts, members, Z, K, N, P, SK);
}

// Program (pair, 128-block of K, matrix): Xh[mat][pair][block] = fp16((x[row] * suh[mat][e]) @ H) for the gate and up
// projections of every routed pair (pair = row * 8 + slot). One warp a program.
extern "C" __global__ void glm_exl3_rot_in(const __nv_bfloat16* __restrict__ x, int x_stride, const int* __restrict__ pick,
        const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0, half* __restrict__ out1, int K) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int row = p >> 3;
    const int lane = threadIdx.x;
    const int e = pick[p];
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (pair, 128-block of the rank's expert width): gate and up summed over the splits in order, rotated, scaled
// by svh; GLM's limited SwiGLU with the grouped kernels' bf16 roundings; then the down projection's input rotation:
// Xd[pair][block] = fp16((act * suh_d[e]) @ H).
extern "C" __global__ void glm_exl3_gateup(const float* __restrict__ Z, const int* __restrict__ pick, const half* __restrict__ svh_g,
        const half* __restrict__ svh_u, const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK, float limit) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    const int p = blockIdx.x, blk = blockIdx.y;
    const int lane = threadIdx.x;
    const int e = pick[p];
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
        float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit), limit);
        float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (pair, 128-block of the model width): the down projection summed over the splits in order, rotated and
// scaled by svh -> y[row][slot][:] (fp32, rows of ``ys`` slots: the shared expert's last). This rank's share.
extern "C" __global__ void glm_exl3_down(const float* __restrict__ Z, const int* __restrict__ pick, const half* __restrict__ svh_d,
        float* __restrict__ y, int P, int D, int SK, int ys) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    const int p = blockIdx.x, blk = blockIdx.y;
    const int lane = threadIdx.x;
    const int e = pick[p];
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    fwht128(v, lane);
    float* o = y + ((size_t)(p >> 3) * ys + (p & 7)) * D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
}

// -- prompt chunks, the transposed design: Y^T = W^T X^T -------------------------------------------------------------
// After MiaAI-Lab's prompt kernels (GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold, patches 0004 and 0009, Apache-2.0),
// written for this engine. A decoded trellis tile is, reordered, an m16n8k16 A fragment of W^T (lane (g, t):
// a0 = b0[0], a1 = b1[0], a2 = b0[1], a3 = b1[1]), so each warp decodes its own tiles straight into registers and
// every decoded fragment feeds the whole item's member rows (PM = 64 pairs) as B fragments from shared memory. A
// block covers 16 weight tiles: gate and up's 8 each of one 128-column Hadamard block (the member rows staged once
// for both), or the down projection's 256 columns; 8 warps hold 2 tiles each. Each output is one fp32 chain of mma
// over ascending k tiles (no split K, no partials in memory), so a row's bits depend on its own values only. The
// epilogues (glm_exl3_gateup's and glm_exl3_down's formulas) run on the sums in shared memory: no round trip.
namespace {

constexpr int TPM = 64, TNW = 8, TMT = 16 / TNW, TNB = TPM / 8, TT = TNW * 32, TPH = TPM / 2;
constexpr int TBK = 32, TKT = TBK / 16, TSTAGES = 3, TXS = TBK + 8;   // a stage: 32 k values a row; +16 B: ldmatrix rows on distinct banks
constexpr int TWORDS = 16 * TKT * 32;                                  // a stage's trellis words (16 tiles x 2 k tiles x 32)
constexpr int GU_ES = 128 + 4, DN_ES = 256 + 4;                        // the epilogue sums' row strides (floats)
template <int XR>
constexpr int tstage_halves() { return XR * TXS + 2 * TWORDS; }
constexpr int tsmem_bytes() {
    const int gu = TSTAGES * tstage_halves<2 * TPM>() * 2;   // gate and up rows apart (no shared rotation)
    const int ep = 2 * TPH * GU_ES * 4, dn = TPH * DN_ES * 4;
    return gu > ep ? (gu > dn ? gu : dn) : (ep > dn ? ep : dn);
}

// Which (matrix, weight tile) warp-tile q (0..15) of column block cb is.
template <int MATS>
__device__ __forceinline__ void ttile(int q, int cb, int& mat, int& nt) {
    mat = MATS == 2 ? q >> 3 : 0;
    nt = MATS == 2 ? cb * 8 + (q & 7) : cb * 16 + q;
}

// The main loop: acc[i][nb] = this warp's tile i (16 output columns) x members nb * 8 .. + 7, over K. XR: the rows a
// stage holds (TPM when both matrices read the same rows: X0 == X1, xshift 3; 2 * TPM otherwise).
template <int MATS, int XR>
__device__ __forceinline__ void tmain(const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
        const uint32_t* __restrict__ T1, const int* rows_sh, int cnt, int e, int K, int NTILES, int cb, int xshift,
        half* stage, float (&acc)[TMT][TNB][4]) {
    constexpr int SH = tstage_halves<XR>();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int mat, nt_unused;
    ttile<MATS>(warp * TMT, cb, mat, nt_unused);
    const int KT = K >> 4, chunks = K / TBK;
    const int nblocks = (cnt + 15) >> 4;   // 16-member blocks in use
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int j = 0; j < TNB; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
    auto load = [&](int chunk) {
        half* dst = stage + (size_t)(chunk % TSTAGES) * SH;
        const int k0 = chunk * TBK;
        for (int idx = threadIdx.x; idx < XR * (TBK / 8); idx += TT) {
            const int m = idx / (TPM * (TBK / 8)), rem = idx % (TPM * (TBK / 8)), i = rem / (TBK / 8), j = rem % (TBK / 8);
            if (i >= nblocks * 16) continue;   // a 16-member block without a member: never read
            const int r = rows_sh[i];
            const half* src = (m ? X1 : X0) + (size_t)(r < 0 ? 0 : (r >> xshift)) * K + k0 + j * 8;
            cp_async16(dst + ((size_t)m * TPM + i) * TXS + j * 8, src, r >= 0);
        }
        uint32_t* wdst = reinterpret_cast<uint32_t*>(dst + (size_t)XR * TXS);
        for (int idx = threadIdx.x; idx < 16 * TKT * 8; idx += TT) {   // 16-byte pieces of the 128-byte tile words
            const int q = idx / (TKT * 8), kk = (idx / 8) % TKT, part = idx % 8;
            int wm, nt;
            ttile<MATS>(q, cb, wm, nt);
            const uint32_t* src = (wm ? T1 : T0) + (((size_t)e * KT + chunk * TKT + kk) * NTILES + nt) * 32 + part * 4;
            cp_async16(wdst + ((size_t)q * TKT + kk) * 32 + part * 4, src, true);
        }
    };
#pragma unroll
    for (int c = 0; c < TSTAGES - 1; ++c) {
        if (c < chunks) load(c);
        cp_async_commit();
    }
    for (int c = 0; c < chunks; ++c) {
        cp_async_wait<TSTAGES - 2>();
        __syncthreads();
        if (c + TSTAGES - 1 < chunks) load(c + TSTAGES - 1);
        cp_async_commit();
        const half* base = stage + (size_t)(c % TSTAGES) * SH;
        const half* xs = base + (size_t)(XR == TPM ? 0 : mat) * TPM * TXS;
        const uint32_t* ws = reinterpret_cast<const uint32_t*>(base + (size_t)XR * TXS);
#pragma unroll
        for (int kk = 0; kk < TKT; ++kk) {
            uint32_t a[TMT][4];
#pragma unroll
            for (int i = 0; i < TMT; ++i) {
                uint32_t b0[2], b1[2];
                decode_tile_p(ws[((size_t)(warp * TMT + i) * TKT + kk) * 32 + lane], lane, b0, b1);
                a[i][0] = b0[0]; a[i][1] = b1[0]; a[i][2] = b0[1]; a[i][3] = b1[1];
            }
#pragma unroll
            for (int jb = 0; jb < TPM / 16; ++jb) {
                if (jb < nblocks) {
                    uint32_t r[4];
                    const int row = jb * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                    ldmatrix_x4(r, xs + (size_t)row * TXS + kk * 16 + ((lane >> 3) & 1) * 8);
                    const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
#pragma unroll
                    for (int i = 0; i < TMT; ++i) {
                        mma16816(acc[i][2 * jb], a[i], bl);
                        mma16816(acc[i][2 * jb + 1], a[i], bh);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();   // the stages become the epilogue's sums
}

// acc (this warp's tiles, members TPH * hf .. + TPH - 1) -> E[member - TPH * hf][col] fp32, row stride ES; col0: the
// first tile's first column within the block's E columns.
__device__ __forceinline__ void tspill(float* E, int ES, int col0, const float (&acc)[TMT][TNB][4], int hf) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int q = 0; q < TPM / 16; ++q) {
            const int nb = hf * (TPM / 16) + q;
            const int tok = q * 8 + 2 * t, col = col0 + i * 16 + g;
            E[(size_t)tok * ES + col] = acc[i][nb][0];
            E[(size_t)(tok + 1) * ES + col] = acc[i][nb][1];
            E[(size_t)tok * ES + col + 8] = acc[i][nb][2];
            E[(size_t)(tok + 1) * ES + col + 8] = acc[i][nb][3];
        }
}

template <int XR>
__device__ __forceinline__ void tgateup(const half* __restrict__ Xg, const half* __restrict__ Xu, const uint32_t* __restrict__ Tg,
        const uint32_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
        const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d, half* __restrict__ xd,
        int K, int N, int xshift, float limit, unsigned char* smem_raw, int* rows_sh) {
    const int cb = blockIdx.x, item = blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][TNB][4];
    tmain<2, XR>(Xg, Xu, Tg, Tu, rows_sh, cnt, e, K, N >> 4, cb, xshift, reinterpret_cast<half*>(smem_raw), acc);
    float* Eg = reinterpret_cast<float*>(smem_raw);
    int mat, nt;
    ttile<2>(warp * TMT, cb, mat, nt);
    const int col0 = (nt - cb * 8) * 16;
    const int n = cb * 128 + 4 * lane;
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) __syncthreads();   // the first half's rows are read
        tspill(Eg + (size_t)mat * TPH * GU_ES, GU_ES, col0, acc, hf);
        __syncthreads();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            const int p = rows_sh[hf * TPH + i];
            float gv[4], uv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j];
                uv[j] = Eg[(size_t)(TPH + i) * GU_ES + 4 * lane + j];
            }
            fwht128(gv, lane);
            fwht128(uv, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit), limit);
                float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
            }
            fwht128(v, lane);
            half* o = xd + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
        }
    }
}

}  // namespace

// Prompt chunks (items of 64 pairs): block (128-column block of the expert width, item), 8 warps: Xd[pair] = the
// gate/up epilogue of the item's pairs. xshift 3: one rotated row a token read by gate and up alike (the rows staged
// once); 0: a row a pair in Xg and Xu.
extern "C" __global__ void __launch_bounds__(256, 2) glm_exl3_prompt_gu(const half* __restrict__ Xg, const half* __restrict__ Xu,
        const uint32_t* __restrict__ Tg, const uint32_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
        half* __restrict__ xd, int K, int N, int xshift, float limit) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    __shared__ __align__(16) unsigned char smem_raw[tsmem_bytes()];   // one buffer for either instantiation
    __shared__ int rows_sh[TPM];
    if (xshift) tgateup<TPM>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, xshift, limit, smem_raw, rows_sh);
    else tgateup<2 * TPM>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, 0, limit, smem_raw, rows_sh);
}

// Block (256-column block of the model width, item): y[row][slot][:] = the down epilogue of the item's pairs (this
// rank's share).
extern "C" __global__ void __launch_bounds__(256, 2) glm_exl3_prompt_dn(const half* __restrict__ Xd, const uint32_t* __restrict__ Td,
        const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
        float* __restrict__ y, int K, int D, int ys) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    __shared__ __align__(16) unsigned char smem_raw[tsmem_bytes()];
    __shared__ int rows_sh[TPM];
    const int cb = blockIdx.x, item = blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][TNB][4];
    tmain<1, TPM>(Xd, Xd, Td, Td, rows_sh, cnt, e, K, D >> 4, cb, 0, reinterpret_cast<half*>(smem_raw), acc);
    float* Ed = reinterpret_cast<float*>(smem_raw);
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) __syncthreads();
        tspill(Ed, DN_ES, warp * TMT * 16, acc, hf);
        __syncthreads();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            const int p = rows_sh[hf * TPH + i];
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const int n = cb * 256 + blk * 128 + 4 * lane;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = Ed[(size_t)i * DN_ES + blk * 128 + 4 * lane + j];
                fwht128(v, lane);
                float* o = y + ((size_t)(p >> 3) * ys + (p & 7)) * D + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            }
        }
    }
}
