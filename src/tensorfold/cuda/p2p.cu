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

template <typename T>
__device__ __forceinline__ void add_vec(float (&acc)[16 / sizeof(T)], int4 raw, bool first) {
    constexpr int V = 16 / sizeof(T);
    if constexpr (sizeof(T) == 4) {
        const float* f = reinterpret_cast<const float*>(&raw);
#pragma unroll
        for (int i = 0; i < V; ++i) acc[i] = first ? f[i] : acc[i] + f[i];
    } else {
        const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
        for (int i = 0; i < V; ++i) acc[i] = first ? __bfloat162float(h[i]) : acc[i] + __bfloat162float(h[i]);
    }
}

// One device round counter a block (captured launches replay with the next round); stages alternate by round parity.
template <typename T>
__global__ void __launch_bounds__(THREADS) p2p_sum_kernel(const T* __restrict__ local, Ranks rk, int world,
                                                          __nv_bfloat16* __restrict__ out, long long n, int rank,
                                                          unsigned long long* __restrict__ rounds, unsigned backoff,
                                                          long long half) {
    constexpr int V = 16 / sizeof(T);
    __shared__ unsigned long long round;
    if (threadIdx.x == 0) round = rounds[blockIdx.x] + 1;
    __syncthreads();
    const long long off = (round & 1) ? half : 0;
    T* stage = reinterpret_cast<T*>(rk.stage[rank] + off);
    const long long step = static_cast<long long>(BLOCKS) * THREADS, nv = n / V;
    const long long first = static_cast<long long>(blockIdx.x) * THREADS + threadIdx.x;
    for (long long i = first; i < nv; i += step)
        reinterpret_cast<int4*>(stage)[i] = reinterpret_cast<const int4*>(local)[i];
    for (long long i = nv * V + first; i < n; i += step) stage[i] = local[i];
    meet(rk, world, 0, rank, round, backoff);                  // this block's slices are staged on every rank
    for (long long i = first; i < nv; i += step) {
        float acc[V];
        for (int q = 0; q < world; ++q) {
            const int4 raw = q == rank ? reinterpret_cast<const int4*>(local)[i]
                                       : __ldcv(reinterpret_cast<const int4*>(rk.stage[q] + off) + i);
            add_vec<T>(acc, raw, q == 0);                      // rank order, the first partial as it is
        }
#pragma unroll
        for (int j = 0; j < V; ++j) out[i * V + j] = __float2bfloat16_rn(acc[j]);
    }
    for (long long i = nv * V + first; i < n; i += step) {
        float acc = 0.f;
        for (int q = 0; q < world; ++q) {
            const float v = q == rank ? to_float(local[i]) : load_peer(reinterpret_cast<const T*>(rk.stage[q] + off) + i);
            acc = q == 0 ? v : acc + v;
        }
        out[i] = __float2bfloat16_rn(acc);
    }
    if (threadIdx.x == 0) rounds[blockIdx.x] = round;
}

// A stream barrier among the ranks; ``rounds`` (a device counter a phase) keeps a captured barrier's rounds new.
__global__ void meet_kernel(Ranks rk, int world, int rank, int phase, unsigned long long* __restrict__ rounds,
                            unsigned backoff) {
    __shared__ unsigned long long round;
    if (threadIdx.x == 0) round = rounds[phase] + 1;
    __syncthreads();
    const int q = threadIdx.x;
    if (q < world && q != rank) {
        __threadfence_system();
        const long long slot = (2LL * BLOCKS + phase) * MAX_WORLD;           // past the sums' flags
        release(reinterpret_cast<unsigned long long*>(rk.sig[q]) + slot + rank, round);
        const unsigned long long* wait = reinterpret_cast<const unsigned long long*>(rk.sig[rank]) + slot + q;
        const unsigned long long t0 = now();
        while (acquire(wait) < round) {
            if (backoff) __nanosleep(backoff);
            if (now() - t0 > SPIN_LIMIT_NS) __trap();
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) rounds[phase] = round;
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
             int64_t rank, const at::Tensor& rounds, int64_t backoff, int64_t half) {
    TORCH_CHECK(local.is_cuda() && local.is_contiguous() && out.is_contiguous() && out.scalar_type() == at::kBFloat16 &&
                out.numel() == local.numel(), "local (contiguous fp32/bf16), out bf16 of the same size");
    const int world = static_cast<int>(stages.numel());
    TORCH_CHECK(world >= 2 && world <= MAX_WORLD && sigs.numel() == world && !stages.is_cuda() && !sigs.is_cuda() &&
                stages.scalar_type() == at::kLong && sigs.scalar_type() == at::kLong && rank >= 0 && rank < world,
                "stages and sigs: one int64 host address a rank, 2 to 8 ranks");
    TORCH_CHECK(rounds.is_cuda() && rounds.scalar_type() == at::kLong && rounds.numel() >= BLOCKS,
                "rounds: int64 device counters, one a block");
    TORCH_CHECK(half % 16 == 0 && reinterpret_cast<uintptr_t>(local.data_ptr()) % 16 == 0 &&
                local.numel() * local.element_size() <= half, "a 16-byte aligned partial no larger than a buffer");
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
                                                              static_cast<int>(rank), r, static_cast<unsigned>(backoff),
                                                              half);
    else {
        TORCH_CHECK(local.scalar_type() == at::kBFloat16, "fp32 or bf16 partials");
        p2p_sum_kernel<__nv_bfloat16><<<BLOCKS, THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(local.data_ptr()), rk, world, o, n, static_cast<int>(rank), r,
            static_cast<unsigned>(backoff), half);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t signal_bytes() { return (2LL * BLOCKS + 2) * MAX_WORLD * sizeof(unsigned long long); }

void meet_ranks(const at::Tensor& sigs, int64_t rank, int64_t phase, const at::Tensor& rounds, int64_t backoff) {
    const int world = static_cast<int>(sigs.numel());
    TORCH_CHECK(world >= 2 && world <= MAX_WORLD && !sigs.is_cuda() && sigs.scalar_type() == at::kLong &&
                rounds.is_cuda() && rounds.scalar_type() == at::kLong && rounds.numel() >= 2 && phase >= 0 &&
                phase < 2, "sigs: one int64 host address a rank; rounds: two int64 device counters");
    Ranks rk{};
    for (int q = 0; q < world; ++q) rk.sig[q] = sigs.data_ptr<int64_t>()[q];
    c10::cuda::CUDAGuard guard(rounds.device());
    meet_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(rk, world, static_cast<int>(rank),
        static_cast<int>(phase), reinterpret_cast<unsigned long long*>(rounds.data_ptr<int64_t>()),
        static_cast<unsigned>(backoff));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t blocks() { return BLOCKS; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alloc", &alloc, "cudaMalloc'd zeroed bytes (an IPC-shareable base pointer)");
    m.def("handle", &handle, "the 64-byte cudaIpcMemHandle of a base pointer");
    m.def("open", &open, "a peer's pointer from its handle (peer access enabled)");
    m.def("p2p_sum", &p2p_sum, "out (bf16) = every rank's partial added in rank order (fp32), staged and read");
    m.def("host_map", &host_map, "page-lock shared host bytes for the GPU (mapped): their device address");
    m.def("signal_bytes", &signal_bytes, "bytes of a rank's signal flags");
    m.def("blocks", &blocks, "blocks a sum launches (one round counter each)");
    m.def("meet", &meet_ranks, "every rank reaches this point on its stream (phase 0 or 1), for copies around it");
}
