// Runtime test kernels: copies, argument passing, module globals, graphs and launch cost; nothing here serves a model.

#include <cstdint>

struct ProbeView {
    const float* src;
    int n;
    float scale;
};

__device__ int tf_probe_table[4];

extern "C" __global__ void tf_probe_fill(float* out, float base, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = base + static_cast<float>(i);
}

extern "C" __global__ void tf_probe_axpy(float* y, const float* x, float a, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// One thread adds to a counter, so a chain of these is a chain of dependent launches.
extern "C" __global__ void tf_probe_step(unsigned long long* counter, unsigned long long add) {
    *counter += add;
}

// By-value struct, bool, int8, double and int64 parameters, echoed so the host can check its packing.
extern "C" __global__ void tf_probe_args(float* out, ProbeView view, bool flag, int8_t small, double wide, long long big) {
    const int i = threadIdx.x;
    if (i < view.n) out[i] = view.src[i] * view.scale;
    if (i == 0) {
        out[view.n] = flag ? 1.0f : 0.0f;
        out[view.n + 1] = static_cast<float>(small);
        out[view.n + 2] = static_cast<float>(wide);
        out[view.n + 3] = static_cast<float>(big);
    }
}

extern "C" __global__ void tf_probe_read_table(int* out) {
    out[threadIdx.x] = tf_probe_table[threadIdx.x] * 2;
}

// Each block records its rank in its thread-block cluster, so the host sees the cluster launch attribute took.
extern "C" __global__ void tf_probe_cluster_rank(unsigned* out) {
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    if (threadIdx.x == 0) out[blockIdx.x] = rank;
}

// A dependent step for programmatic dependent launch: waits on the previous grid, then lets the next one start early.
extern "C" __global__ void tf_probe_pdl_step(unsigned long long* counter, unsigned long long add) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    *counter += add;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
