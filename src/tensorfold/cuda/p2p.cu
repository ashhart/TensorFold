// Two ranks' row-parallel sum over a peer mapping: rank 0's partial plus rank 1's in fp32, rounded once.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int BLOCKS = 32, THREADS = 512;
constexpr long long SPIN_LIMIT_NS = 30LL * 1000 * 1000 * 1000;    // a peer gone for 30 s: trap, never hang silently

__device__ __forceinline__ void release(unsigned long long* p, unsigned long long v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ unsigned long long acquire(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ unsigned long long now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// signals: [2 phases][BLOCKS][2 ranks] u64 in each rank's memory; a rank writes its arrival into the peer's copy
__device__ __forceinline__ void meet(unsigned long long* mine, unsigned long long* peer, int phase, int rank,
                                     unsigned long long round) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        release(peer + (phase * BLOCKS + blockIdx.x) * 2 + rank, round);
        const unsigned long long* wait = mine + (phase * BLOCKS + blockIdx.x) * 2 + (1 - rank);
        const unsigned long long t0 = now();
        while (acquire(wait) < round) {
            if (now() - t0 > SPIN_LIMIT_NS) __trap();
        }
    }
    __syncthreads();
}

template <typename T>
__device__ __forceinline__ float load_peer(const T* p) {
    if constexpr (sizeof(T) == 4) return __ldcv(reinterpret_cast<const float*>(p));
    else {
        unsigned short bits = __ldcv(reinterpret_cast<const unsigned short*>(p));
        return __bfloat162float(__ushort_as_bfloat16(bits));
    }
}

template <typename T>
__device__ __forceinline__ float to_float(T v) {
    if constexpr (sizeof(T) == 4) return v;
    else return __bfloat162float(v);
}

template <typename T>
__global__ void __launch_bounds__(THREADS) p2p_sum_kernel(const T* __restrict__ local, T* stage, const T* peer_stage,
                                                          unsigned long long* sig, unsigned long long* peer_sig,
                                                          __nv_bfloat16* __restrict__ out, long long n, int rank,
                                                          unsigned long long round,
                                                          const unsigned long long* __restrict__ base) {
    if (base != nullptr) round += *base;              // a captured call: offset past the base set before replay
    const long long step = static_cast<long long>(BLOCKS) * THREADS;
    for (long long i = static_cast<long long>(blockIdx.x) * THREADS + threadIdx.x; i < n; i += step) stage[i] = local[i];
    meet(sig, peer_sig, 0, rank, round);              // this block's slice is staged on both ranks
    for (long long i = static_cast<long long>(blockIdx.x) * THREADS + threadIdx.x; i < n; i += step) {
        const float mine = to_float(local[i]), other = load_peer(peer_stage + i);
        out[i] = __float2bfloat16_rn(rank == 0 ? mine + other : other + mine);
    }
    meet(sig, peer_sig, 1, rank, round);              // and read on both: the next round may overwrite it
}

}  // namespace

int64_t alloc(int64_t bytes) {
    void* p = nullptr;
    C10_CUDA_CHECK(cudaMalloc(&p, bytes));
    C10_CUDA_CHECK(cudaMemset(p, 0, bytes));
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    return reinterpret_cast<int64_t>(p);
}

at::Tensor handle(int64_t ptr) {
    cudaIpcMemHandle_t h;
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&h, reinterpret_cast<void*>(ptr)));
    auto out = at::empty({static_cast<int64_t>(sizeof(h))}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(out.data_ptr(), &h, sizeof(h));
    return out;
}

int64_t open(const at::Tensor& bytes) {
    TORCH_CHECK(bytes.numel() == sizeof(cudaIpcMemHandle_t) && bytes.device().is_cpu(), "64 handle bytes on the host");
    cudaIpcMemHandle_t h;
    std::memcpy(&h, bytes.data_ptr(), sizeof(h));
    void* p = nullptr;
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
    return reinterpret_cast<int64_t>(p);
}

void p2p_sum(const at::Tensor& local, int64_t stage, int64_t peer_stage, int64_t sig, int64_t peer_sig,
             at::Tensor& out, int64_t rank, int64_t round, int64_t base) {
    TORCH_CHECK(local.is_cuda() && local.is_contiguous() && out.is_contiguous() && out.scalar_type() == at::kBFloat16 &&
                out.numel() == local.numel(), "local (contiguous fp32/bf16), out bf16 of the same size");
    c10::cuda::CUDAGuard guard(local.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const long long n = local.numel();
    auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    auto* s = reinterpret_cast<unsigned long long*>(sig);
    auto* ps = reinterpret_cast<unsigned long long*>(peer_sig);
    const auto* b = reinterpret_cast<const unsigned long long*>(base);
    if (local.scalar_type() == at::kFloat)
        p2p_sum_kernel<float><<<BLOCKS, THREADS, 0, stream>>>(local.data_ptr<float>(), reinterpret_cast<float*>(stage),
            reinterpret_cast<const float*>(peer_stage), s, ps, o, n, static_cast<int>(rank), round, b);
    else {
        TORCH_CHECK(local.scalar_type() == at::kBFloat16, "fp32 or bf16 partials");
        p2p_sum_kernel<__nv_bfloat16><<<BLOCKS, THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(local.data_ptr()), reinterpret_cast<__nv_bfloat16*>(stage),
            reinterpret_cast<const __nv_bfloat16*>(peer_stage), s, ps, o, n, static_cast<int>(rank), round, b);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t signal_bytes() { return 2LL * BLOCKS * 2 * sizeof(unsigned long long); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alloc", &alloc, "cudaMalloc'd zeroed bytes (an IPC-shareable base pointer)");
    m.def("handle", &handle, "the 64-byte cudaIpcMemHandle of a base pointer");
    m.def("open", &open, "a peer's pointer from its handle (peer access enabled)");
    m.def("p2p_sum", &p2p_sum, "out (bf16) = rank 0's partial + rank 1's partial (fp32), staged and read over P2P "
          "(base: 0, or a device word added to the round: a graph's calls)");
    m.def("signal_bytes", &signal_bytes, "bytes of a rank's signal flags");
}
