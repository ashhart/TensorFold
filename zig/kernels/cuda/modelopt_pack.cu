// ModelOpt tensors as stored into the layouts qmmf, prompt16 and nvfp4_experts read (host twins in zig/src/cuda).

#include <stdint.h>

namespace {

// The k of byte `b` of `lane` in a 64-input FP8 group's 16 bytes a lane (fp8.zig kin).
__device__ __forceinline__ int kin(int lane, int b) {
  const int p = 16 * (b / 4) + 4 * (lane % 4) + b % 4;
  return 16 * (p / 16) + 2 * ((p % 16) / 4) + p % 2 + 8 * ((p % 4) / 2);
}

}  // namespace

// e4m3 codes [n, k] into [npad/64][k/64][8][32][2][8] bytes, rows past n zero (fp8.packCodes).
extern "C" __global__ void __launch_bounds__(256) tf_pack_fp8_lane(const uint8_t* __restrict__ codes, uint8_t* __restrict__ out,
                                                                   int n, int k, long long total) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int b = i % 8, h = (i / 8) % 2, lane = (i / 16) % 32, a = (i / 512) % 8;
  const long long tg = i / 4096;
  const int kg = k / 64, g = tg % kg;
  const long long row = (tg / kg) * 64 + a * 8 + lane / 4;
  out[i] = row < n ? codes[row * k + g * 64 + h * 32 + kin(lane, b)] : 0;
}

// bf16 [n, k] into [npad/64][k/64][8][4][32][4] (n8 tile, k16 step, lane, b0 then b1 halves), rows past n zero.
extern "C" __global__ void __launch_bounds__(256) tf_pack_bf16_lane(const uint16_t* __restrict__ w, uint16_t* __restrict__ out,
                                                                    int n, int k, long long total) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int e = i % 4, lane = (i / 4) % 32, kt = (i / 128) % 4, jj = (i / 512) % 8;
  const long long tg = i / 4096;
  const int kg = k / 64, g = tg % kg;
  const long long row = (tg / kg) * 64 + jj * 8 + lane / 4;
  const int col = g * 64 + kt * 16 + 2 * (lane % 4) + (e & 1) + 8 * (e >> 1);
  out[i] = row < n ? w[row * k + col] : 0;
}

// NVFP4 codes [n, k/2] (low nibble first) into words [npad/64][k/64][8][32][2], rows past n zero (nvfp4.packWords).
extern "C" __global__ void __launch_bounds__(256) tf_pack_fp4_words(const uint8_t* __restrict__ codes, uint32_t* __restrict__ out,
                                                                    int n, int k, long long total) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int v = i & 1, lane = (i >> 1) & 31, j = (i >> 6) & 7;
  const long long tg = i >> 9;
  const int kg = k / 64, g = tg % kg;
  const long long row = (tg / kg) * 64 + j * 8 + (lane >> 2);
  uint32_t word = 0;
  if (row < n) {
    const int offs[8] = {0, 8, 16, 24, 1, 9, 17, 25};
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const int input = g * 64 + 32 * v + 2 * (lane & 3) + offs[p];
      const uint8_t byte = codes[row * (k / 2) + input / 2];
      word |= static_cast<uint32_t>((input & 1) ? byte >> 4 : byte & 0xF) << (4 * p);
    }
  }
  out[i] = word;
}

// e4m3 scales [n, k/16] into [npad/64][k/64][64][4] bytes, rows past n zero (nvfp4.packScales).
extern "C" __global__ void __launch_bounds__(256) tf_pack_fp4_scales(const uint8_t* __restrict__ scales, uint8_t* __restrict__ out,
                                                                     int n, int k, long long total) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int q = i % 4, c = (i / 4) % 64;
  const long long tg = i / 256;
  const int kg = k / 64, g = tg % kg;
  const long long row = (tg / kg) * 64 + c;
  out[i] = row < n ? scales[row * (k / 16) + g * 4 + q] : 0;
}

// Experts' codes [E][n][ks/2] and scales [E][n][ks/16], inputs k0..k0+k, as part m of [E][n/32][k/32][parts][144].
extern "C" __global__ void __launch_bounds__(144) tf_pack_fp4_experts(const uint8_t* __restrict__ codes,
                                                                      const uint8_t* __restrict__ scales,
                                                                      uint32_t* __restrict__ out, int n, int ks, int k0,
                                                                      int k, int parts, int m) {
  const int g = blockIdx.x, cb = blockIdx.y, e = blockIdx.z, w = threadIdx.x, kg = k / 32;
  uint32_t* dst = out + (((static_cast<long long>(e) * (n / 32) + cb) * kg + g) * parts + m) * 144;
  const uint8_t* ce = codes + static_cast<long long>(e) * n * (ks / 2);
  const uint8_t* se = scales + static_cast<long long>(e) * n * (ks / 16);
  if (w < 128) {
    const int gq = w / 16, t = (w / 4) % 4, j = w % 4;
    const long long row = cb * 32 + j * 8 + gq;
    uint32_t word = 0;
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int input = k0 + g * 32 + h * 16 + t * 4 + q;
        const uint8_t byte = ce[row * (ks / 2) + input / 2];
        const uint32_t nib = (input & 1) ? byte >> 4 : byte & 0xF;
        word |= nib << (4 * (2 * h + q / 2 + 4 * (q % 2)));
      }
    dst[w] = word;
  } else {
    uint32_t word = 0;
#pragma unroll
    for (int x = 0; x < 4; ++x) {
      const int s = 4 * (w - 128) + x;  // ((t * 2 + h) * 4 + j) * 2 + c
      const int c = s % 2, j = (s / 2) % 4, h = (s / 8) % 2, t = s / 16;
      const long long row = cb * 32 + j * 8 + t * 2 + c;
      word |= static_cast<uint32_t>(se[row * (ks / 16) + (k0 + g * 32) / 16 + h]) << (8 * x);
    }
    dst[w] = word;
  }
}
