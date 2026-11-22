// One-shot RoCE all-gather after b12x (github.com/local-inference-lab/b12x, Apache-2.0): docs/roce-gather.md.

#include <stdint.h>
#include "glm_post_math.cuh"

namespace {

constexpr int C_SEQ = 0, C_NBYTES = 1, C_ERROR = 3, C_HCA = 4, C_PEER = 5;  // ctrl words (C_NBYTES + slot)
constexpr int FLAG_STRIDE = 128;
// counters (device memory, u32): epoch, poison, then stage and tail arrivals for each grid size class
constexpr int K_EPOCH = 0, K_POISON = 1, K_CLASSES = 4;

__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint4 ld_relaxed_sys4(const uint4* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_relaxed_sys(uint32_t* p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release_gpu(uint32_t* p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t arrive(uint32_t* p) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(p) : "memory");
    return old;
}
__device__ __forceinline__ void fence_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }

}  // namespace

// in: this rank's shard (16-byte packs); out: every rank's in rank order; recv and flags: by peer number.
enum class Post { none, residual, hc };

template <Post post>
__device__ __forceinline__ void gather_body(
    const uint4* __restrict__ in, uint4* __restrict__ out, int packs, int rank, char* recv, char* flags, char* send,
    uint32_t* ctrl, long long slot_bytes, uint32_t* counters, int klass, uint32_t spin_limit, int n_hca, int world,
    int64_t d, const __nv_bfloat16* x, __nv_bfloat16* xout, const float* post_w, const float* comb) {
    const int tid = threadIdx.x;
    const int peers = world - 1;
    uint32_t* stage = counters + 2 + klass;
    uint32_t* tail = counters + 2 + K_CLASSES + klass;
    if (ld_relaxed_gpu(counters + K_POISON) != 0u) return;
    const uint32_t seq = ld_relaxed_gpu(counters + K_EPOCH) + 1u;
    const uint32_t slot = seq & 1u;
    const int at = blockIdx.x * blockDim.x + tid, stride = gridDim.x * blockDim.x;
    uint4* mine = reinterpret_cast<uint4*>(send + (long long)slot * slot_bytes);
    for (int i = at; i < packs; i += stride) mine[i] = in[i];
    __syncthreads();
    if (tid == 0) {
        fence_sys();
        if ((arrive(stage) + 1u) % gridDim.x == 0u) {
            st_relaxed_sys(ctrl + C_NBYTES + slot, (uint32_t)packs * 16u);
            fence_sys();
            st_relaxed_sys(ctrl + C_SEQ, seq);
        }
    }
    if (tid < peers * n_hca) {
        const int p = tid / n_hca, h = tid - p * n_hca;
        const uint32_t* flag = reinterpret_cast<const uint32_t*>(flags + (((long long)slot * peers + p) * n_hca + h) * FLAG_STRIDE);
        uint32_t polls = 0;
        while (ld_acquire_sys(flag) != seq) {
            if (++polls >= spin_limit) {
                st_relaxed_sys(ctrl + C_HCA, (uint32_t)h);
                st_relaxed_sys(ctrl + C_PEER, (uint32_t)(p < rank ? p : p + 1));
                st_relaxed_sys(ctrl + C_ERROR, seq);
                st_release_gpu(counters + K_POISON, seq);
                break;
            }
        }
    }
    __syncthreads();
    if (ld_relaxed_gpu(counters + K_POISON) == 0u) {
        for (int i = at; i < packs; i += stride) {
            if constexpr (post == Post::none) {
                (out + (long long)rank * packs)[i] = in[i];
                for (int p = 0; p < peers; ++p) {
                    const int r = p < rank ? p : p + 1;
                    const uint4* theirs = reinterpret_cast<const uint4*>(recv + ((long long)slot * peers + p) * slot_bytes);
                    (out + (long long)r * packs)[i] = ld_relaxed_sys4(theirs + i);
                }
            } else {
                const uint4 theirs = ld_relaxed_sys4(reinterpret_cast<const uint4*>(recv + (long long)slot * slot_bytes) + i);
                const uint4 rank0 = rank == 0 ? in[i] : theirs, rank1 = rank == 0 ? theirs : in[i];
                if constexpr (post == Post::residual) {
                    glm_post::residual4(rank0, rank1, x + 4ll * i, xout + 4ll * i);
                } else {
                    const int64_t col = 4ll * i, r = col / d, c = col - r * d;
                    for (int s = 0; s < 4; ++s)
                    glm_post::hc4(rank0, rank1, x + (r * 4) * d + c, xout + (r * 4 + s) * d + c,
                                  d, s, post_w + r * 4, comb + r * 16);
                }
            }
        }
    }
    fence_gpu();
    __syncthreads();
    if (tid == 0 && (arrive(tail) + 1u) % gridDim.x == 0u) {
        fence_gpu();
        if (ld_relaxed_sys(ctrl + C_ERROR) == 0u) st_release_gpu(counters + K_EPOCH, seq);
    }
}

#define ROCE_ARGS const uint4* __restrict__ in, uint4* __restrict__ out, int packs, int rank, char* recv, char* flags, char* send, uint32_t* ctrl, long long slot_bytes, uint32_t* counters, int klass, uint32_t spin_limit, int n_hca, int world
#define ROCE_CALL in, out, packs, rank, recv, flags, send, ctrl, slot_bytes, counters, klass, spin_limit, n_hca, world

extern "C" __global__ void __launch_bounds__(512) roce_gather(ROCE_ARGS) {
    gather_body<Post::none>(ROCE_CALL, 0, nullptr, nullptr, nullptr, nullptr);
}
extern "C" __global__ void __launch_bounds__(512) roce_gather_residual_add(
    ROCE_ARGS, int64_t d, const __nv_bfloat16* x, __nv_bfloat16* xout) {
    gather_body<Post::residual>(ROCE_CALL, d, x, xout, nullptr, nullptr);
}
extern "C" __global__ void __launch_bounds__(512) roce_gather_hc_post(
    ROCE_ARGS, int64_t d, const __nv_bfloat16* x, __nv_bfloat16* xout, const float* post, const float* comb) {
    gather_body<Post::hc>(ROCE_CALL, d, x, xout, post, comb);
}
