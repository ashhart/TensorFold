// Row-parallel sums of two to eight ranks over peer mappings or shared host memory: rank order in fp32, rounded once.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int BLOCKS = 32, THREADS = 512, MAX_WORLD = 8;
constexpr long long SPIN_LIMIT_NS = 30LL * 1000 * 1000 * 1000;    // a peer gone for 30 s: trap, never hang silently

// every rank's staging buffer and flag region as this process addresses them (mapped peers or shared host memory)
struct Ranks {
    long long stage[MAX_WORLD];
    long long sig[MAX_WORLD];
};

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

// flags: [2 phases][BLOCKS][MAX_WORLD] u64 in each rank's region; rank r marks its arrival at slot r of every peer's
__device__ __forceinline__ void meet(const Ranks& rk, int world, int phase, int rank, unsigned long long round,
                                     unsigned backoff) {
    __syncthreads();
    const int q = threadIdx.x;
    if (q < world && q != rank) {
        __threadfence_system();
        const long long slot = (static_cast<long long>(phase) * BLOCKS + blockIdx.x) * MAX_WORLD;
        release(reinterpret_cast<unsigned long long*>(rk.sig[q]) + slot + rank, round);
        const unsigned long long* wait = reinterpret_cast<const unsigned long long*>(rk.sig[rank]) + slot + q;
        const unsigned long long t0 = now();
        while (acquire(wait) < round) {
            if (backoff) __nanosleep(backoff);        // host flags: poll PCIe gently, the partials share the link
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

// rounds: one counter a block in this process's device memory, so a captured launch replays with the next round
template <typename T>
__global__ void __launch_bounds__(THREADS) p2p_sum_kernel(const T* __restrict__ local, Ranks rk, int world,
                                                          __nv_bfloat16* __restrict__ out, long long n, int rank,
                                                          unsigned long long* __restrict__ rounds, unsigned backoff) {
    __shared__ unsigned long long round;
    if (threadIdx.x == 0) round = rounds[blockIdx.x] + 1;
    __syncthreads();
    T* stage = reinterpret_cast<T*>(rk.stage[rank]);
    const long long step = static_cast<long long>(BLOCKS) * THREADS;
    for (long long i = static_cast<long long>(blockIdx.x) * THREADS + threadIdx.x; i < n; i += step) stage[i] = local[i];
    meet(rk, world, 0, rank, round, backoff);                  // this block's slice is staged on every rank
    for (long long i = static_cast<long long>(blockIdx.x) * THREADS + threadIdx.x; i < n; i += step) {
        float acc = 0.f;
        for (int q = 0; q < world; ++q) {
            const float v = q == rank ? to_float(local[i]) : load_peer(reinterpret_cast<const T*>(rk.stage[q]) + i);
            acc = q == 0 ? v : acc + v;                        // rank order, the first partial as it is
        }
        out[i] = __float2bfloat16_rn(acc);
    }
    meet(rk, world, 1, rank, round, backoff);                  // and read on every rank: the next round may overwrite it
    if (threadIdx.x == 0) rounds[blockIdx.x] = round;
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

// Page-lock a shared host buffer both ranks map, for GPUs without a peer mapping: its device address.
int64_t host_map(int64_t ptr, int64_t bytes) {
    void* host = reinterpret_cast<void*>(ptr);
    C10_CUDA_CHECK(cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
    void* dev = nullptr;
    C10_CUDA_CHECK(cudaHostGetDevicePointer(&dev, host, 0));
    return reinterpret_cast<int64_t>(dev);
}

void p2p_sum(const at::Tensor& local, const at::Tensor& stages, const at::Tensor& sigs, at::Tensor& out,
             int64_t rank, const at::Tensor& rounds, int64_t backoff) {
    TORCH_CHECK(local.is_cuda() && local.is_contiguous() && out.is_contiguous() && out.scalar_type() == at::kBFloat16 &&
                out.numel() == local.numel(), "local (contiguous fp32/bf16), out bf16 of the same size");
    const int world = static_cast<int>(stages.numel());
    TORCH_CHECK(world >= 2 && world <= MAX_WORLD && sigs.numel() == world && !stages.is_cuda() && !sigs.is_cuda() &&
                stages.scalar_type() == at::kLong && sigs.scalar_type() == at::kLong && rank >= 0 && rank < world,
                "stages and sigs: one int64 host address a rank, 2 to 8 ranks");
    TORCH_CHECK(rounds.is_cuda() && rounds.scalar_type() == at::kLong && rounds.numel() >= BLOCKS,
                "rounds: int64 device counters, one a block");
    Ranks rk{};
    for (int q = 0; q < world; ++q) {
        rk.stage[q] = stages.data_ptr<int64_t>()[q];
        rk.sig[q] = sigs.data_ptr<int64_t>()[q];
    }
    c10::cuda::CUDAGuard guard(local.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const long long n = local.numel();
    auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    auto* r = reinterpret_cast<unsigned long long*>(rounds.data_ptr<int64_t>());
    if (local.scalar_type() == at::kFloat)
        p2p_sum_kernel<float><<<BLOCKS, THREADS, 0, stream>>>(local.data_ptr<float>(), rk, world, o, n,
                                                              static_cast<int>(rank), r, static_cast<unsigned>(backoff));
    else {
        TORCH_CHECK(local.scalar_type() == at::kBFloat16, "fp32 or bf16 partials");
        p2p_sum_kernel<__nv_bfloat16><<<BLOCKS, THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(local.data_ptr()), rk, world, o, n, static_cast<int>(rank), r,
            static_cast<unsigned>(backoff));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t signal_bytes() { return 2LL * BLOCKS * MAX_WORLD * sizeof(unsigned long long); }

int64_t blocks() { return BLOCKS; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alloc", &alloc, "cudaMalloc'd zeroed bytes (an IPC-shareable base pointer)");
    m.def("handle", &handle, "the 64-byte cudaIpcMemHandle of a base pointer");
    m.def("open", &open, "a peer's pointer from its handle (peer access enabled)");
    m.def("p2p_sum", &p2p_sum, "out (bf16) = every rank's partial added in rank order (fp32), staged and read");
    m.def("host_map", &host_map, "page-lock shared host bytes for the GPU (mapped): their device address");
    m.def("signal_bytes", &signal_bytes, "bytes of a rank's signal flags");
    m.def("blocks", &blocks, "blocks a sum launches (one round counter each)");
}
