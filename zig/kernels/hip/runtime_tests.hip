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
