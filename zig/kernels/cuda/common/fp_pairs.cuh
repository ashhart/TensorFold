// FP4 (E2M1) and FP8 (E4M3) codes widened to bf16 pairs on the bf16x2 FMA, and one E4M3 scale to float.
#pragma once

#include <cuda_fp8.h>
#include <stdint.h>

// -Dnative-fp-decode (TF_NATIVE_FP_DECODE): PTX ISA 9.2's one-instruction FP4/FP8 pair decode, sm_120 family targets.
#if defined(TF_NATIVE_FP_DECODE) && (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ < 1302)
#error "-Dnative-fp-decode needs nvcc 13.2 or newer (PTX ISA 9.2's cvt.rn.bf16x2.e2m1x2 and .e4m3x2)"
#endif
#if defined(TF_NATIVE_FP_DECODE) && defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200
#define TF_FP_NATIVE 1
#else
#define TF_FP_NATIVE 0
#endif

namespace tf_common {

// Two E2M1 codes, bits s..s+3 of each 16-bit half of w, as a bf16x2 pair.
__device__ __forceinline__ uint32_t fp4pair(uint32_t w, int s) {
    const uint32_t v = w >> s;
    const uint32_t t = ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7E807E80u), "r"(0x80008000u));
    return r;
}

// fp4pair with code 0b1000 read as +5 (NVFP4_NZ5): the one code fp4pair widens to 0x8000, -0, becomes 0x40A0.
__device__ __forceinline__ uint32_t fp4pair_nz5(uint32_t w, int s) {
    const uint32_t r = fp4pair(w, s);
    return r ^ (__vcmpeq2(r, 0x80008000u) & 0xC0A0C0A0u);
}

// Two E4M3 codes, the low two bytes of v, as a bf16x2 pair.
__device__ __forceinline__ uint32_t fp8pair(uint32_t v) {
    const uint32_t t = ((v & 0x7Fu) << 4) | ((v & 0x80u) << 8) | ((v & 0x7F00u) << 12) | ((v & 0x8000u) << 16);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7B807B80u), "r"(0x80008000u));
    return r;
}

__device__ __forceinline__ float e4m3f(uint32_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3)));
}

__device__ __forceinline__ uint32_t mul2(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));
    return r;
}

__device__ __forceinline__ uint32_t fp8pair_unit(uint32_t w, uint32_t unit) {
    const uint32_t x = (w & 0xFFu) | ((w & 0xFF00u) << 8);
    const uint32_t t = ((x & 0x007F007Fu) << 4) | ((x & 0x00800080u) << 8);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(unit), "r"(0x80008000u));
    return r;
}

#if TF_FP_NATIVE
// One byte's E2M1 codes (low nibble to the low half) as a bf16x2 pair; equals fp4pair on all 256 bytes, -0 kept.
__device__ __forceinline__ uint32_t e2m1x2(uint32_t b) {
    uint32_t r;
    asm("{ .reg .b8 a; cvt.u8.u32 a, %1; cvt.rn.bf16x2.e2m1x2 %0, a; }" : "=r"(r) : "r"(b));
    return r;
}

// The low two bytes' E4M3 codes as a bf16x2 pair; equals fp8pair on every pair without a NaN byte (0x7F, 0xFF).
__device__ __forceinline__ uint32_t e4m3x2(uint32_t h) {
    uint32_t r;
    asm("{ .reg .b16 a; cvt.u16.u32 a, %1; cvt.rn.bf16x2.e4m3x2 %0, a; }" : "=r"(r) : "r"(h));
    return r;
}

// fp4pair(w, 8q), fp4pair(w, 8q + 4) bit for bit: byte q of x holds nibbles (2q, 2q + 4), of y (2q + 1, 2q + 5)
__device__ __forceinline__ void e2m1_pairs(uint32_t w, int q, uint32_t& b0, uint32_t& b1) {
    const uint32_t lo = w & 0x0F0F0F0Fu, hi = (w >> 4) & 0x0F0F0F0Fu;
    const uint32_t x = lo | (lo >> 12), y = hi | (hi >> 12);
    b0 = e2m1x2(q ? x >> 8 : x);
    b1 = e2m1x2(q ? y >> 8 : y);
}
#endif

}  // namespace tf_common
