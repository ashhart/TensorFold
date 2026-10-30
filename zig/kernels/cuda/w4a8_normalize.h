// Shared finite normalization, compiled by CUDA and by the CPU regression without CUDA headers.
#ifndef TF_W4A8_NORMALIZE_H
#define TF_W4A8_NORMALIZE_H
#ifdef __CUDACC__
#define TF_W4A8_INLINE __host__ __device__ __forceinline__
#else
#define TF_W4A8_INLINE static inline
#endif
TF_W4A8_INLINE float tf_w4a8_normalized(float v, float amax, float inv) {
    // Preserve the original multiply whenever its reciprocal is finite.
    return inv > 0x1.fffffep127f ? (v / amax) * 127.0f : v * inv;
}
#undef TF_W4A8_INLINE
#endif
