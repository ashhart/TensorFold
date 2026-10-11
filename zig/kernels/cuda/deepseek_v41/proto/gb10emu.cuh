// Emulates a 48-SM, bandwidth-limited GB10-class part on a bigger sm_120 GPU: blocker CTAs park or hog the other SMs.

// Call gb10::eager() before any CUDA call: with lazy loading a first launch waits on the parked blockers (deadlock).
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

namespace gb10 {

constexpr int FREE = 48;                          // the GB10's SMs
constexpr int BLOCK_SMEM = 99 * 1024 - 64;        // one blocker an SM (sm_120: 99 KB a CTA with the static flag)

__device__ __forceinline__ unsigned smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }

// hogs: the first `hogs` blocked SMs stream `buf` (16-byte loads, 512 threads); the others spin
__global__ void __launch_bounds__(512) blocker(volatile int* release, int* ready, const int4* __restrict__ buf,
                                               size_t n16, int hogs, unsigned long long* sink) {
    extern __shared__ int4 pad[];
    const unsigned s = smid();
    if (s < FREE) return;
    if (threadIdx.x == 0) atomicAdd(ready, 1);
    const bool hog = (int)(s - FREE) < hogs;
    int4 acc = make_int4(0, 0, 0, 0);
    size_t i = ((size_t)(s - FREE) * 512 + threadIdx.x) * 8;
    const size_t stride = (size_t)hogs * 512 * 8;
    __shared__ int stop;
    for (;;) {                                    // one thread polls the host flag, every 64 sweeps
        if (threadIdx.x == 0) stop = *release;
        __syncthreads();
        if (stop) break;
        __syncthreads();
        for (int k = 0; k < 64; ++k) {
            if (hog) {
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const int4 v = __ldcs(buf + ((i + u) & (n16 - 1)));   // n16: a power of two
                    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
                }
                i = (i + stride) & (n16 - 1);
            } else {
                __nanosleep(2000);
            }
        }
    }
    if (acc.x == 0x7fffffff) sink[0] = (unsigned long long)(acc.y ^ acc.z ^ acc.w ^ pad[0].x);
}

// a plain stream read of n16 16-byte words on the kernel's SMs: the bandwidth the kernel under test sees
__global__ void __launch_bounds__(512) stream_read(const int4* __restrict__ p, size_t n16, unsigned long long* sink) {
    int4 acc = make_int4(0, 0, 0, 0);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
        const int4 v = __ldcs(p + i);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (acc.x == 0x7fffffff) sink[1] = (unsigned long long)(acc.y ^ acc.z ^ acc.w);
}

inline void eager() { setenv("CUDA_MODULE_LOADING", "EAGER", 1); }

enum class Mode { Full, Sms, Gb10 };

// begin() parks the blockers, end() releases them; wrap a timed region: emu.begin(); time(kernel); emu.end();
struct Emu {
    Mode mode = Mode::Full;
    int sms = 0, hogs = 0;
    cudaStream_t side{};
    int* release_h = nullptr; int* release_d = nullptr; int* ready = nullptr;
    int4* buf = nullptr; size_t n16 = 0; unsigned long long* sink = nullptr;

    void init(Mode m, size_t hog_bytes = (size_t)4 << 30) {
        mode = m;
        cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p, 0)); sms = p.multiProcessorCount;
        CK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
        CK(cudaHostAlloc(&release_h, sizeof(int), cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer(&release_d, release_h, 0));
        CK(cudaMalloc(&ready, sizeof(int)));
        CK(cudaMalloc(&sink, 16));
        if (m == Mode::Gb10) {
            n16 = hog_bytes / 16; CK(cudaMalloc(&buf, n16 * 16)); CK(cudaMemset(buf, 1, n16 * 16));
        }
        CK(cudaFuncSetAttribute(blocker, cudaFuncAttributeMaxDynamicSharedMemorySize, BLOCK_SMEM));
    }
    int grid_sms() const { return mode == Mode::Full ? sms : FREE; }   // what the kernel under test may size for
    void begin() {
        if (mode == Mode::Full) return;
        *release_h = 0; CK(cudaMemset(ready, 0, sizeof(int))); CK(cudaDeviceSynchronize());
        blocker<<<sms, 512, BLOCK_SMEM, side>>>(release_h ? release_d : nullptr, ready, buf, n16,
                                                mode == Mode::Gb10 ? hogs : 0, sink);
        CK(cudaGetLastError());
        int r = 0;
        for (int spin = 0; r < sms - FREE && spin < 200000; ++spin) CK(cudaMemcpy(&r, ready, 4, cudaMemcpyDeviceToHost));
        if (r < sms - FREE) { fprintf(stderr, "gb10emu: only %d of %d blockers resident\n", r, sms - FREE); exit(1); }
    }
    void end() {
        if (mode == Mode::Full) return;
        *release_h = 1; __sync_synchronize(); CK(cudaDeviceSynchronize());
    }
    // GB/s of a plain 16-byte stream read of `bytes` on the kernel's SMs (inside begin / end)
    double stream_gbps(const void* p, size_t bytes, cudaStream_t st) {
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        const int grid = grid_sms() * 4;
        stream_read<<<grid, 512, 0, st>>>((const int4*)p, bytes / 16, sink);
        CK(cudaEventRecord(a, st));
        stream_read<<<grid, 512, 0, st>>>((const int4*)p, bytes / 16, sink);
        CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
        return bytes / (ms * 1e6);
    }
    // Gb10 mode: the hog count that leaves the kernel's SMs closest to `target` GB/s of stream read
    void calibrate(const void* p, size_t bytes, cudaStream_t st, double target) {
        if (mode != Mode::Gb10) return;
        int best = 0; double bestd = 1e30;
        for (int h = 0; h <= sms - FREE; h += 4) {
            hogs = h; begin(); const double g = stream_gbps(p, bytes, st); end();
            printf("  calib hogs %3d -> stream %.1f GB/s\n", h, g);
            if (fabs(g - target) < bestd) { bestd = fabs(g - target); best = h; }
            if (g < target * 0.9) break;
        }
        hogs = best;
        begin(); const double g = stream_gbps(p, bytes, st); end();
        printf("gb10emu: %d hog SMs, the 48 SMs stream %.1f GB/s (target %.0f)\n", hogs, g, target);
    }
};

}  // namespace gb10
