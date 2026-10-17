// GLM-5.3-Flash's EXL3 routed experts (4-bit mcg trellis, ExLlamaV3's format: MIT, (c) 2025 Turboderp): decode, prompt.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

// Lets a dependent launch start on its weights early (sm_90 and newer; older targets run in plain stream order).
__device__ __forceinline__ void launch_dependents() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

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

// Grid (item, n block, mat * SK + split) -> Z[mat][split][pair][n]: an expert and <= 16 MT of its pairs; warp order.
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

// Fast Walsh-Hadamard transform of 128 values, 4 a lane, natural order: strides 1, 2 in registers, 4..64 across lanes.
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

// The SwiGLU limits as torch.clamp takes them: a NaN stays NaN (fminf and fmaxf would drop it for the bound).
__device__ __forceinline__ float clamp_hi(float x, float hi) { return x > hi ? hi : x; }
__device__ __forceinline__ float clamp_both(float x, float l) { return x < -l ? -l : (x > l ? l : x); }


// -- helpers the transposed prompt kernels share (glm_exl3_transposed.cuh) --

// decode_tile via PRMT (MiaAI-Lab GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold patch 0020, Apache-2.0): same bits.
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

}  // namespace

// Program (row, K block): Xh[row][block] = fp16((x[row] * suh) @ H) when a layer's experts share suh; rot_in's bits.
extern "C" __global__ void glm_exl3_rot_rows(const __nv_bfloat16* __restrict__ x, int x_stride, const half* __restrict__ suh,
        half* __restrict__ out, int K) {
    launch_dependents();
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
    launch_dependents();  // a dependent matmul may start on its weights
    grouped<1, 8, 4>(X0, X1, T0, T1, items, counts, members, Z, K, N, P, SK);
}

// Program (pair, K block, matrix): Xh[mat][pair][block] = fp16((x[row] * suh[mat][e]) @ H), pair = row * 8 + slot.
extern "C" __global__ void glm_exl3_rot_in(const __nv_bfloat16* __restrict__ x, int x_stride, const int* __restrict__ pick,
        const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0, half* __restrict__ out1, int K) {
    launch_dependents();  // a dependent matmul may start on its weights
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

// Program (pair, width block): gate, up over splits in order, rotated, svh, limited SwiGLU, Xd = fp16(act suh_d H).
extern "C" __global__ void glm_exl3_gateup(const float* __restrict__ Z, const int* __restrict__ pick, const half* __restrict__ svh_g,
        const half* __restrict__ svh_u, const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK, float limit) {
    launch_dependents();  // a dependent matmul may start on its weights
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
        float gg = clamp_hi(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
        float uu = clamp_both(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), limit);
        float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (pair, model-width block): down over splits in order, rotated, svh -> y[row][slot] fp32; this rank's share.
extern "C" __global__ void glm_exl3_down(const float* __restrict__ Z, const int* __restrict__ pick, const half* __restrict__ svh_d,
        float* __restrict__ y, int P, int D, int SK, int ys) {
    launch_dependents();  // a dependent matmul may start on its weights
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

// -- prompt chunks, the transposed design: Y^T = W^T X^T, one fp32 mma chain an output over ascending k tiles -------

// After MiaAI-Lab's prompt kernels (GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold patches 0004, 0009, Apache-2.0).
#include "glm_exl3_transposed.cuh"

// Prompt items of 64 pairs: block (128 expert columns, item), 8 warps: Xd = gate/up epilogue; xshift 3 or 0 as above.
extern "C" __global__ void __launch_bounds__(256, 2) glm_exl3_prompt_gu(const half* __restrict__ Xg, const half* __restrict__ Xu,
        const uint32_t* __restrict__ Tg, const uint32_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
        half* __restrict__ xd, int K, int N, int xshift, float limit) {
    launch_dependents();
    __shared__ __align__(16) unsigned char smem_raw[tsmem_bytes()];   // one buffer for either instantiation
    __shared__ int rows_sh[TPM];
    if (xshift) tgateup<TPM>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, xshift, limit, smem_raw, rows_sh);
    else tgateup<2 * TPM>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, 0, limit, smem_raw, rows_sh);
}

// Block (256 model-width columns, item): y[row][slot][:] = the down epilogue of the item's pairs (this rank's share).
extern "C" __global__ void __launch_bounds__(256, 2) glm_exl3_prompt_dn(const half* __restrict__ Xd, const uint32_t* __restrict__ Td,
        const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
        float* __restrict__ y, int K, int D, int ys) {
    launch_dependents();
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

// Program (row): the routed picks of a row of `slots` (the shared expert last) as a row of `slots - 1`, for the plan.
extern "C" __global__ void glm_exl3_picks(const int* __restrict__ pick, int* __restrict__ out, int rows, int slots) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * (slots - 1)) return;
    out[i] = pick[(i / (slots - 1)) * slots + i % (slots - 1)];
}
