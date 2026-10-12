// Copy of src/tensorfold/families/deepseek_v41/cuda/csa2/topk_cuda.cu (blob 839ee8a85f0e): sass_check.sh
#define DSV41_ATTN_NO_TORCH



// Indexer decode top-k (TF_DSV41_ATTN_CUDA=1): top K of unique keys (score image, then lower position), exact.

// Radix select of the K-th 32-bit image over a cluster of <= 8 CTAs a row (DSMEM histograms), then scan-order pick.

// Image: s == 0 -> 0, NaN -> 0x7FC00000, negative -> bits ^ 0x7FFFFFFF, then ^ 0x80000000 (0 marks missing).

#include <cstdint>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace dsv41_topk {
namespace cg = cooperative_groups;

constexpr int NT = 512, NW = NT / 32, MAXCL = 8, MAXEPT = 31;
constexpr unsigned FULL = 0xffffffffu;

struct Job {
    const float* s; long long ss;    // scores [R, n] (unit stride)
    const int* cand; long long cs;   // mode 3: ascending candidate blocks [R, cols] (-1 last)
    int* out; long long os;          // [R, k]
    int* cnt;                        // [R] visible counts or null
    int nk, k, mode, ept;            // entries, K, 0 dense / 2 blocks / 3 candidate positions, entries a thread
};

struct Args {
    Job job[2];
    const void* pos; int pos64;      // int32 [1] (POS + r) or int64 [R]
    int ratio, bs;
};

__device__ __forceinline__ uint32_t image(float s) {
    int b = __float_as_int(s);
    if (s == 0.f) b = 0;
    if (s != s) b = 0x7FC00000;
    if (b < 0) b ^= 0x7FFFFFFF;
    return static_cast<uint32_t>(b) ^ 0x80000000u;
}

// exclusive scan of one int a thread over the CTA (512 threads); returns the prefix, *total the sum
__device__ int block_scan(int v, int* ws, int* total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(FULL, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) ws[w] = x;
    __syncthreads();
    if (w == 0) {
        int z = lane < NW ? ws[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(FULL, z, o);
            if (lane >= o) z += y;
        }
        if (lane < NW) ws[NW + lane] = z;
    }
    __syncthreads();
    const int pre = (w ? ws[NW + w - 1] : 0) + x - v;
    *total = ws[2 * NW - 1];
    __syncthreads();
    return pre;
}

__global__ void __launch_bounds__(NT, 1) topk_kernel(Args a) {
    cg::cluster_group cl = cg::this_cluster();
    const Job J = blockIdx.z == 0 ? a.job[0] : a.job[1];        // (no dynamic indexing: no local memory)
    const int CL = static_cast<int>(cl.num_blocks()), rank = static_cast<int>(cl.block_rank());
    const int r = blockIdx.y, tid = threadIdx.x, lane = tid & 31;
    extern __shared__ uint32_t img[];                            // SL = 512 x ept images (0: missing)
    __shared__ int hist[2][256];
    __shared__ int tot[256];
    __shared__ int ws[2 * NW];
    __shared__ int ex[2];                                        // (above, equal) of this CTA, read by the cluster
    __shared__ int dec[4];
    const int SL = NT * J.ept, lo = rank * SL, n = max(0, min(SL, J.nk - lo));
    const int q = a.pos64 ? static_cast<int>(static_cast<const long long*>(a.pos)[r])
                          : static_cast<const int*>(a.pos)[0] + r;
    const int nvis = (q + 1) / a.ratio;
    if (rank == 0 && tid == 0 && J.cnt) J.cnt[r] = 0;            // before the first cluster barrier (release)
    const float* srow = J.s + static_cast<long long>(r) * J.ss;
    const int* crow = J.cand ? J.cand + static_cast<long long>(r) * J.cs : nullptr;
    for (int i = tid; i < SL; i += NT) {                         // images, coalesced
        const int e = lo + i;
        uint32_t u = 0;
        if (i < n) {
            if (J.mode == 2) {                                   // block e: max over its visible members, newest +inf
                float m = 0.f;                                   // tl.max (maxnumf) over the members, masked -inf
                for (int j = 0; j < a.bs; ++j) {
                    const int p = e * a.bs + j;
                    const float v = p < nvis ? srow[p] : -INFINITY;
                    m = j ? fmaxf(m, v) : v;
                }
                if (e == (nvis - 1) / a.bs) m = INFINITY;
                if (e * a.bs < nvis) u = image(m);
            } else if (J.mode == 3) {
                if (crow[e / a.bs] >= 0) u = image(srow[e]);
            } else {
                u = image(srow[e]);
            }
        }
        img[i] = u;
    }
    __syncthreads();
    uint32_t prefix = 0, pmask = 0;
    int above = 0, quota_mode = 0;                               // 0: every match selected; 1: equal images capped
    for (int p = 0; p < 4; ++p) {
        int* h = hist[p & 1];
        if (tid < 256) h[tid] = 0;
        __syncthreads();
        const int shift = 24 - 8 * p;
        for (int i0 = 0; i0 < SL; i0 += NT) {
            const uint32_t v = img[i0 + tid];
            const bool m = v != 0 && (v & pmask) == prefix;
            const unsigned act = __ballot_sync(FULL, m);
            if (m) {
                const int d = (v >> shift) & 255;
                const unsigned peers = __match_any_sync(act, d);
                if (lane == __ffs(peers) - 1) atomicAdd(&h[d], __popc(peers));
            }
        }
        cl.sync();
        if (tid < 256) {
            int v[MAXCL];
#pragma unroll
            for (int x = 0; x < MAXCL; ++x) v[x] = x < CL ? cl.map_shared_rank(h, x)[tid] : 0;
            int sum = 0;
#pragma unroll
            for (int x = 0; x < MAXCL; ++x) sum += v[x];
            tot[tid] = sum;
        }
        __syncthreads();
        if (tid < 32) {                                          // suffix sums over 256 digits, 8 a lane; the digit
            int loc[8], run = 0;
            for (int x = 7; x >= 0; --x) { run += tot[lane * 8 + x]; loc[x] = run; }
            int suf = run;                                       // inclusive suffix over lanes >= this one
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_down_sync(FULL, suf, o);
                if (lane + o < 32) suf += y;
            }
            const int after = suf - run;                         // lanes above this one
            int best = -1, bge = 0;
            for (int x = 7; x >= 0; --x)
                if (best < 0 && above + after + loc[x] >= J.k) { best = lane * 8 + x; bge = after + loc[x]; }
            const unsigned has = __ballot_sync(FULL, best >= 0);
            const int src = has ? 31 - __clz(has) : 0;
            const int dg = __shfl_sync(FULL, best, src), ge = __shfl_sync(FULL, bge, src);
            const int all = __shfl_sync(FULL, suf, 0);
            if (lane == 0) { dec[0] = dg; dec[1] = ge; dec[2] = all; }
        }
        __syncthreads();
        const int dg = dec[0], ge = dec[1], all = dec[2];
        if (p == 0 && all <= J.k) break;                         // every valid entry (prefix / mask stay 0)
        const int hd = tot[dg];
        above += ge - hd;
        prefix |= static_cast<uint32_t>(dg) << shift;
        pmask |= 255u << shift;
        if (above + hd == J.k) break;                            // the prefix's entries complete the top K
        if (p == 3) quota_mode = 1;                              // ties at the K-th image: the first ones win
    }
    // classify: 2 above (selected), 1 equal (selected up to the quota, in scan order), 0 out
    const int e0 = tid * J.ept;
    int gt = 0, eq = 0;
    for (int x = 0; x < J.ept; ++x) {
        const uint32_t v = img[e0 + x];
        if (v == 0) continue;
        const uint32_t mv = v & pmask;
        gt += mv > prefix;
        eq += mv == prefix;
    }
    int tgt, teq;
    const int pre_eq = block_scan(eq, ws, &teq);
    block_scan(gt, ws, &tgt);
    if (tid == 0) { ex[0] = tgt; ex[1] = teq; }
    cl.sync();
    int base = 0, eq_before = 0, total = 0, quota = teq;
    {
        const int need = J.k - above;                            // equal images to take (quota mode)
        for (int x = 0; x < CL; ++x) {
            const int* rx = cl.map_shared_rank(ex, x);
            const int xg = rx[0], xe = rx[1];
            const int xq = quota_mode ? max(0, min(xe, need - eq_before)) : xe;
            if (x < rank) base += xg + xq;
            if (x == rank) quota = xq;
            eq_before += xe;
            total += xg + xq;
        }
    }
    cl.sync();                                                   // no DSMEM access after this: CTAs may exit
    int sel = 0;
    {
        int er = pre_eq;
        for (int x = 0; x < J.ept; ++x) {
            const uint32_t v = img[e0 + x];
            if (v == 0) continue;
            const uint32_t mv = v & pmask;
            if (mv > prefix) ++sel;
            else if (mv == prefix) { sel += er < quota; ++er; }
        }
    }
    int tsel;
    const int off = base + block_scan(sel, ws, &tsel);
    int* orow = J.out + static_cast<long long>(r) * J.os;
    int vis = 0;
    {
        int er = pre_eq, o = off;
        for (int x = 0; x < J.ept; ++x) {
            const uint32_t v = img[e0 + x];
            if (v == 0) continue;
            const uint32_t mv = v & pmask;
            bool s = mv > prefix;
            if (!s && mv == prefix) { s = er < quota; ++er; }
            if (!s) continue;
            const int e = lo + e0 + x;
            const int pp = J.mode == 3 ? crow[e / a.bs] * a.bs + e % a.bs : e;
            orow[o++] = pp;
            vis += pp < nvis;
        }
    }
    if (J.cnt) {
        int tv;
        block_scan(vis, ws, &tv);
        if (tid == 0 && tv) atomicAdd(J.cnt + r, tv);
    }
    if (rank == 0)
        for (int i = total + tid; i < J.k; i += NT) orow[i] = -1;
}

// Shared memory a CTA for ``ept`` entries a thread (the images; the rest is static).
inline size_t smem(int ept) { return static_cast<size_t>(NT) * ept * 4; }

inline cudaError_t launch(const Args& a, int jobs, int R, int CL, int ept, cudaStream_t stream) {
    if (R < 1 || jobs < 1 || jobs > 2 || CL < 1 || CL > MAXCL || ept < 1 || ept > MAXEPT)
        return cudaErrorInvalidValue;
    static bool init = false;
    if (!init) {
        const cudaError_t e = cudaFuncSetAttribute(topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   static_cast<int>(smem(MAXEPT)));
        if (e != cudaSuccess) return e;
        init = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CL, R, jobs);
    cfg.blockDim = dim3(NT);
    cfg.dynamicSmemBytes = smem(ept);
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = CL;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, topk_kernel, a);
}

}  // namespace dsv41_topk

#ifndef DSV41_ATTN_NO_TORCH
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {
dsv41_topk::Job job_of(const at::Tensor& s, const at::Tensor& cand, at::Tensor& out, at::Tensor& cnt, int64_t nk,
                       int64_t k, int64_t mode, int64_t ept) {
    dsv41_topk::Job j{};
    j.s = static_cast<const float*>(s.data_ptr()); j.ss = s.stride(0);
    j.cand = cand.numel() ? static_cast<const int*>(cand.data_ptr()) : nullptr;
    j.cs = cand.numel() ? cand.stride(0) : 0;
    j.out = static_cast<int*>(out.data_ptr()); j.os = out.stride(0);
    j.cnt = cnt.numel() ? static_cast<int*>(cnt.data_ptr()) : nullptr;
    j.nk = static_cast<int>(nk); j.k = static_cast<int>(k); j.mode = static_cast<int>(mode);
    j.ept = static_cast<int>(ept);
    return j;
}
}

void dsv41_topk_run_cuda(const at::Tensor& s0, const at::Tensor& c0, at::Tensor& o0, at::Tensor& n0, int64_t nk0,
                         int64_t k0, int64_t m0, int64_t e0, const at::Tensor& s1, const at::Tensor& c1,
                         at::Tensor& o1, at::Tensor& n1, int64_t nk1, int64_t k1, int64_t m1, int64_t e1,
                         int64_t jobs, const at::Tensor& pos, int64_t ratio, int64_t bs, int64_t R, int64_t CL) {
    dsv41_topk::Args a{};
    a.job[0] = job_of(s0, c0, o0, n0, nk0, k0, m0, e0);
    a.job[1] = jobs > 1 ? job_of(s1, c1, o1, n1, nk1, k1, m1, e1) : a.job[0];
    a.pos = pos.data_ptr(); a.pos64 = pos.scalar_type() == at::kLong ? 1 : 0;
    a.ratio = static_cast<int>(ratio); a.bs = static_cast<int>(bs);
    const int ept = static_cast<int>(std::max(e0, jobs > 1 ? e1 : e0));
    C10_CUDA_CHECK(dsv41_topk::launch(a, static_cast<int>(jobs), static_cast<int>(R), static_cast<int>(CL), ept,
                                      at::cuda::getCurrentCUDAStream()));
}
#endif
