// Grouped EXL3 expert GEMM for prompt chunks: each program decodes a trellis tile once and multiplies it against up to
// MTP member tiles of 16 rows (the decode kernel re-decodes the tile for every 16 rows). Same Z layout as
// tensorfold/cuda/exl3/experts_grouped.cuh with one K split, so its epilogues apply unchanged.
#include <torch/extension.h>

#include "experts_grouped.cuh"

namespace tf_exl3x {

template <int CB, int K2, int NT, int MTP>
__device__ __forceinline__ void warp_tiles_multi(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt,
                                                 int nt0, const half* const (&x0)[MTP], const half* const (&x1)[MTP],
                                                 const bool (&ok0)[MTP], const bool (&ok1)[MTP], int lane,
                                                 float (&acc)[MTP][NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;
    uint32_t cur[NT][LW], nxt[NT][LW];
#pragma unroll
    for (int i = 0; i < NT; ++i) load_words<K2>(cur[i], tp + i * TW, lane);
    for (int it = 0; it < nkt; ++it) {
        if (it + 1 < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(nxt[i], tp + (size_t)(it + 1) * kstride + i * TW, lane);
        uint32_t b0[NT][2], b1[NT][2];
#pragma unroll
        for (int i = 0; i < NT; ++i) decode_tile<CB, K2>(cur[i], map, lane, b0[i], b1[i]);
        const int k = (kt0 + it) * 16;
#pragma unroll
        for (int m = 0; m < MTP; ++m) {
            uint32_t a[4] = {load_pair(x0[m] + k, ok0[m]), load_pair(x1[m] + k, ok1[m]),
                             load_pair(x0[m] + k + 8, ok0[m]), load_pair(x1[m] + k + 8, ok1[m])};
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                mma16816(acc[m][i][0], a, b0[i]);
                mma16816(acc[m][i][1], a, b1[i]);
            }
        }
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int l = 0; l < LW; ++l) cur[i][l] = nxt[i][l];
    }
}

// Program (expert u, n block of 16 * NT, member group of 16 * MTP rows, matrix); K split over the W warps only.
template <int CB, int NT, int W, int MTP>
__global__ void __launch_bounds__(W * 32) grouped_prompt_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int maxm, int slots) {
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int MG = (maxm + 16 * MTP - 1) / (16 * MTP);
    const int mgroup = blockIdx.z % MG;
    const int mat = blockIdx.z / MG;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16 * MTP];
    for (int i = threadIdx.x; i < 16 * MTP; i += W * 32) {
        const int m = mgroup * 16 * MTP + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                            // members come first, so this group is empty
    const half* x0[MTP];
    const half* x1[MTP];
    bool ok0[MTP], ok1[MTP];
#pragma unroll
    for (int m = 0; m < MTP; ++m) {
        const int r0 = rows_sh[m * 16 + g], r1 = rows_sh[m * 16 + g + 8];
        x0[m] = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
        x1[m] = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;
        ok0[m] = r0 >= 0;
        ok1[m] = r1 >= 0;
    }
    const int per_warp = KT / W;
    const int kt0 = warp * per_warp;
    const int nt0 = blockIdx.y * NT;
    float acc[MTP][NT][2][4];
#pragma unroll
    for (int m = 0; m < MTP; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;
    switch (k2) {
        case 4: warp_tiles_multi<CB, 4, NT, MTP>(T, NTILES, kt0, per_warp, nt0, x0, x1, ok0, ok1, lane, acc); break;
        case 5: warp_tiles_multi<CB, 5, NT, MTP>(T, NTILES, kt0, per_warp, nt0, x0, x1, ok0, ok1, lane, acc); break;
        case 6: warp_tiles_multi<CB, 6, NT, MTP>(T, NTILES, kt0, per_warp, nt0, x0, x1, ok0, ok1, lane, acc); break;
        case 8: warp_tiles_multi<CB, 8, NT, MTP>(T, NTILES, kt0, per_warp, nt0, x0, x1, ok0, ok1, lane, acc); break;
        default: __trap();
    }
    // warps' partial sums through shared memory, added in warp order, one member tile at a time
    __shared__ float red[W][16][NT * 16];
#pragma unroll
    for (int m = 0; m < MTP; ++m) {
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][g][col] = acc[m][i][h][0];
                red[warp][g][col + 1] = acc[m][i][h][1];
                red[warp][g + 8][col] = acc[m][i][h][2];
                red[warp][g + 8][col + 1] = acc[m][i][h][3];
            }
        __syncthreads();
        for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[m * 16 + row];
            if (r < 0) continue;
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][row][col];
            Z[((size_t)mat * P + r) * N + nt0 * 16 + col] = s;
        }
        __syncthreads();
    }
}

}  // namespace tf_exl3x

template <int NT, int W, int MTP>
static void launch(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                   const at::Tensor& K2_0, const at::Tensor& K2_1, const at::Tensor& uids, const at::Tensor& ucount,
                   const at::Tensor& members, at::Tensor& Z, int64_t mats, int64_t K, int64_t N, int64_t P,
                   int64_t slots) {
    TORCH_CHECK((K / 16) % W == 0 && N % (16 * NT) == 0, "prompt expert kernel: shape does not tile");
    const int maxm = (int)members.size(1);
    const int MG = (maxm + 16 * MTP - 1) / (16 * MTP);
    dim3 grid((unsigned)uids.numel(), (unsigned)(N / (16 * NT)), (unsigned)(mats * MG));
    tf_exl3x::grouped_prompt_kernel<2, NT, W, MTP><<<grid, W * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X0.data_ptr()), reinterpret_cast<const half*>(X1.data_ptr()),
        TP0.data_ptr<int64_t>(), TP1.data_ptr<int64_t>(), K2_0.data_ptr<int>(), K2_1.data_ptr<int>(),
        uids.data_ptr<int>(), ucount.data_ptr<int>(), members.data_ptr<int>(), Z.data_ptr<float>(), (int)K, (int)N,
        (int)P, maxm, (int)slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// config: 0 NT8/MTP1, 1 NT8/MTP2, 2 NT4/MTP2, 3 NT4/MTP4, 4 NT2/MTP4, 5 NT8/MTP4
void grouped_prompt(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                    const at::Tensor& K2_0, const at::Tensor& K2_1, const at::Tensor& uids, const at::Tensor& ucount,
                    const at::Tensor& members, at::Tensor& Z, int64_t mats, int64_t K, int64_t N, int64_t P,
                    int64_t slots, int64_t cb, int64_t config) {
    TORCH_CHECK(cb == 2, "prompt expert kernel: mul1 codebook only");
#define TF_CFG(ID, NT_, MTP_) \
    case ID: launch<NT_, 4, MTP_>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, mats, K, N, P, slots); break;
    switch (config) {
        TF_CFG(0, 8, 1)
        TF_CFG(1, 8, 2)
        TF_CFG(2, 4, 2)
        TF_CFG(3, 4, 4)
        TF_CFG(4, 2, 4)
        TF_CFG(5, 8, 4)
        default: TORCH_CHECK(false, "unknown prompt expert config");
    }
#undef TF_CFG
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("grouped_prompt", &grouped_prompt); }
