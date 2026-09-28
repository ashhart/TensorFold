// The NVFP4 expert weights' own encoding, as the device kernels read it.
//
// The checkpoint stores a value a nibble: four bits of e2m1 -- the magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 with a
// sign bit -- and one fp8 e4m3 scale a block of 16 inputs along K. That is not the affine encoding q4 uses
// (``q * s - b``, which an int8 mma can consume in pairs); e2m1 is a float format, so its nibbles are decoded
// to bf16 and the block's scale multiplies the decoded value. Sharing this header keeps the decode form and
// the prefill form on one definition of the format.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace tf {

// Every e2m1 magnitude, sign included: the code is the index, so no branch and no bit math a value.
__device__ __constant__ float k_e2m1[16] = {0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
                                           -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__device__ __forceinline__ float e2m1(unsigned code) { return k_e2m1[code & 0xFu]; }

// A block's scale is one byte of e4m3: the sign, a 4-bit exponent biased by 7 and a 3-bit mantissa.
__device__ __forceinline__ float e4m3(unsigned byte) {
  return __half2float(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(byte), __NV_E4M3));
}

// One stored byte holds the block's two nibbles: the low one the even input, the high one the odd.
__device__ __forceinline__ __nv_bfloat16 decode_byte(unsigned byte, int half, float scale) {
  const unsigned code = half == 0 ? (byte & 0xFu) : (byte >> 4);
  return __float2bfloat16(e2m1(code) * scale);
}

}  // namespace tf
