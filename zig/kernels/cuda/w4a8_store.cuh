// The output epilogue shared with the host alignment regression.
#ifndef TF_W4A8_STORE_CUH
#define TF_W4A8_STORE_CUH
#include <stdint.h>
#ifndef __CUDACC__
#define TF_W4A8_STORE_INLINE static inline
#else
#define TF_W4A8_STORE_INLINE __device__ __forceinline__
#endif
TF_W4A8_STORE_INLINE void tf_w4a8_store(__nv_bfloat16* dst, int remaining, float a, float b) {
    // Odd output strides put odd rows two bytes off pair alignment. Keep the same rounding per component.
    if (remaining > 1 && (reinterpret_cast<uintptr_t>(dst) & 3) == 0)
        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(a, b);
    else {
        dst[0] = __float2bfloat16_rn(a);
        if (remaining > 1) dst[1] = __float2bfloat16_rn(b);
    }
}
#undef TF_W4A8_STORE_INLINE
#endif
