// GPU side of the small two-rank all-gather over RoCE (rdma_proxy.c has the host side and the region layout).
//
// One launch of GRID blocks:
//   1. every block copies its share of the local shard into the pinned send slot (seq & 1), then fences system-wide;
//   2. the last block to arrive (a free-running counter, compared modulo GRID) stores the slot's byte count and rings
//      ctrl.seq for the proxy thread, which RDMA-writes the slot and then a flag to the peer;
//   3. each block waits for the peer's flag of this slot to read seq (system-scope acquire loads; a bounded wait
//      records the sequence in ctrl[4] and stops further launches);
//   4. every block copies both shards to the output in rank order (its own from the input, the peer's from recv);
//   5. the last block to leave advances the device epoch, so a CUDA graph replays the exchange with the next seq.
#include <ATen/ATen.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>

namespace {

constexpr int GRID = 8;

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_relaxed_sys(uint32_t* p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint4 ld_sys_v4(const uint4* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

__global__ void __launch_bounds__(256) gather_kernel(const uint4* __restrict__ in, uint4* __restrict__ out, int packs,
                                                     uint32_t nbytes, char* region, long long flag_off,
                                                     long long send_off, long long recv_off, long long slot_bytes,
                                                     uint32_t* state, uint32_t spin, int rank) {
    // state: [0] epoch (last completed seq), [1] arrivals, [2] departures, [3] stopped (a wait timed out)
    __shared__ int ok;
    const int tid = threadIdx.x;
    const uint32_t seq = ld_relaxed_gpu(state) + 1u, slot = seq & 1u;
    uint32_t* ctrl = reinterpret_cast<uint32_t*>(region);
    if (ld_relaxed_gpu(state + 3) != 0u) return;
    const int index = blockIdx.x * blockDim.x + tid, stride = gridDim.x * blockDim.x;

    uint4* send = reinterpret_cast<uint4*>(region + send_off + slot * slot_bytes);           // 1. stage
    for (int i = index; i < packs; i += stride) send[i] = in[i];
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {                                                                           // 2. doorbell
        const uint32_t prior = atomicAdd(state + 1, 1u);
        if ((prior + 1u) % gridDim.x == 0u) {
            __threadfence_system();
            st_relaxed_sys(ctrl + 1 + slot, nbytes);
            st_release_sys(ctrl, seq);
        }
        const uint32_t* flag = reinterpret_cast<const uint32_t*>(region + flag_off + slot * 64);   // 3. wait
        uint32_t polls = 0;
        ok = 1;
        while (ld_acquire_sys(flag) != seq) {
            if (++polls >= spin) {
                st_relaxed_sys(ctrl + 4, seq);
                atomicExch(state + 3, seq);
                ok = 0;
                break;
            }
        }
    }
    __syncthreads();
    if (ok) {                                                                                 // 4. copy out
        const uint4* peer = reinterpret_cast<const uint4*>(region + recv_off + slot * slot_bytes);
        uint4* mine_out = out + (long long)rank * packs;
        uint4* peer_out = out + (long long)(1 - rank) * packs;
        for (int i = index; i < packs; i += stride) {
            mine_out[i] = in[i];
            peer_out[i] = ld_sys_v4(peer + i);
        }
    }
    __threadfence();                                                                          // 5. epoch
    __syncthreads();
    if (tid == 0) {
        const uint32_t prior = atomicAdd(state + 2, 1u);
        if ((prior + 1u) % gridDim.x == 0u && ld_relaxed_gpu(state + 3) == 0u) atomicExch(state, seq);
    }
}

}  // namespace

void rdma_gather(const at::Tensor& in, at::Tensor& out, int64_t region, int64_t flag_off, int64_t send_off,
                 int64_t recv_off, int64_t slot_bytes, at::Tensor& state, int64_t spin, int64_t rank) {
    TORCH_CHECK(in.is_cuda() && out.is_cuda() && state.is_cuda(), "rdma gather: CUDA tensors");
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous(), "rdma gather: contiguous tensors");
    const int64_t nbytes = in.numel() * in.element_size();
    TORCH_CHECK(nbytes % 16 == 0 && nbytes <= slot_bytes, "rdma gather: 16-byte multiple within a slot");
    TORCH_CHECK(out.numel() * out.element_size() == 2 * nbytes, "rdma gather: out holds both shards");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(in.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
                "rdma gather: 16-byte aligned tensors");
    const at::cuda::CUDAGuard guard(in.device());
    gather_kernel<<<GRID, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4*>(in.data_ptr()), reinterpret_cast<uint4*>(out.data_ptr()), (int)(nbytes / 16),
        (uint32_t)nbytes, reinterpret_cast<char*>(region), flag_off, send_off, recv_off, slot_bytes,
        reinterpret_cast<uint32_t*>(state.data_ptr()), (uint32_t)spin, (int)rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

