#include <hip/hip_runtime.h>

extern "C" __global__ void tf_hip_read_fill(unsigned int* out, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = out[i] ^ 0x01010101U;
}

extern "C" __global__ void tf_hip_probe(unsigned int* out, unsigned int n,
                                        unsigned int bias, unsigned long long tag) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i + bias + static_cast<unsigned int>(tag >> 32);
}

__device__ int tf_hip_table[4];

extern "C" __global__ void tf_hip_fill_f32(float* out, float base, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = base + static_cast<float>(i);
}

extern "C" __global__ void tf_hip_axpy(float* y, const float* x, float a, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// One thread adds to a counter, so a chain of these is a chain of dependent launches.
extern "C" __global__ void tf_hip_step(unsigned long long* counter, unsigned long long add) {
    *counter += add;
}

extern "C" __global__ void tf_hip_read_table(int* out) {
    out[threadIdx.x] = tf_hip_table[threadIdx.x] * 2;
}

// The wavefront width the code object runs at.
extern "C" __global__ void tf_hip_wave(int* out) {
    if (threadIdx.x == 0) out[0] = warpSize;
}
