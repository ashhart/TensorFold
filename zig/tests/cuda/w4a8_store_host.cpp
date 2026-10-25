// Compile the production epilogue with instrumented BF16 stores, without CUDA headers or a device.
#include <stdint.h>
#include <string.h>
static int misaligned;
struct __nv_bfloat16 { uint16_t bits; };
struct __nv_bfloat162 {
    __nv_bfloat16 lo, hi;
    __nv_bfloat162& operator=(const __nv_bfloat162& rhs) {
        if (reinterpret_cast<uintptr_t>(this) % 4) ++misaligned;
        memcpy(this, &rhs, sizeof(rhs));
        return *this;
    }
};
static __nv_bfloat16 __float2bfloat16_rn(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return {static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16)};
}
static __nv_bfloat162 __floats2bfloat162_rn(float a, float b) {
    return {__float2bfloat16_rn(a), __float2bfloat16_rn(b)};
}
#include "../../kernels/cuda/w4a8_store.cuh"
extern "C" int tf_w4a8_store_host(uint16_t* dst, int remaining, float a, float b) {
    misaligned = 0;
    tf_w4a8_store(reinterpret_cast<__nv_bfloat16*>(dst), remaining, a, b);
    return misaligned;
}
