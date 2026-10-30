// Execute the production quantizer with a serial, deterministic CUDA warp-instruction shim.
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
#define __launch_bounds__(n)
struct Dim { int x = 0, y = 0; };
static Dim blockIdx, threadIdx;
struct __nv_bfloat16 { uint16_t bits; };
static float __bfloat162float(__nv_bfloat16 v) {
    union { uint32_t u; float f; } bits{uint32_t(v.bits) << 16};
    return bits.f;
}
static float maxima[6][32], sums[6][32];
static bool any_bad, any_sum_bad;
static int shuffle_at, vote_at, conversions;
static float __shfl_xor_sync(unsigned, float, int offset) {
    const int step = shuffle_at / 2;
    const bool sum = shuffle_at++ % 2;
    return (sum ? sums : maxima)[step][(threadIdx.x & 31) ^ offset];
}
[[maybe_unused]] static bool __any_sync(unsigned, bool) { return vote_at++ == 0 ? any_bad : any_sum_bad; }
[[maybe_unused]] static unsigned atomicOr(unsigned* p, unsigned v) { unsigned old = *p; *p |= v; return old; }
static int __float2int_rn(float v) {
    if (!isfinite(v)) { ++conversions; return 0; }
    return int(nearbyintf(v));
}
#include "../../kernels/cuda/w4a8_quantize.cuh"
using Old = void (*)(const __nv_bfloat16*, uint8_t*, float*, float*, int, int);
using Current = void (*)(const __nv_bfloat16*, uint8_t*, float*, float*, int, int, unsigned*);
[[maybe_unused]] static int arity(Old) { return 6; }
[[maybe_unused]] static int arity(Current) { return 7; }
[[maybe_unused]] static void call(Old fn, const __nv_bfloat16* x, uint8_t* q, float* a, float* s, int k, int ld, unsigned*) { fn(x, q, a, s, k, ld); }
[[maybe_unused]] static void call(Current fn, const __nv_bfloat16* x, uint8_t* q, float* a, float* s, int k, int ld, unsigned* flag) { fn(x, q, a, s, k, ld, flag); }
extern "C" int tf_w4a8_quantize_parameters() { return arity(tf_w4a8_quantize); }
extern "C" int tf_w4a8_quantize_host(const uint16_t* raw, uint8_t* xq, float* xa, float* xs, int rows, int k, int ldx, unsigned* flag) {
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(raw);
    conversions = 0;
    for (blockIdx.x = 0; blockIdx.x < rows; ++blockIdx.x)
        for (blockIdx.y = 0; blockIdx.y < (k / 64 + 3) / 4; ++blockIdx.y)
            for (int warp = 0; warp < 4; ++warp) {
                const int g = blockIdx.y * 4 + warp;
                any_bad = false;
                any_sum_bad = false;
                if (g * 64 < k) {
                    for (int lane = 0; lane < 32; ++lane) {
                        const int step = lane >> 4, s = (2 * lane) & 31;
                        const auto* src = x + blockIdx.x * ldx + g * 64 + step * 32;
                        const float a = __bfloat162float(src[tf_w4a8::slot_input(s)]);
                        const float b = __bfloat162float(src[tf_w4a8::slot_input(s + 1)]);
                        any_bad |= !isfinite(a) || !isfinite(b);
                        maxima[0][lane] = fmaxf(fabsf(a), fabsf(b));
                        sums[0][lane] = a + b;
                    }
                    for (int step = 0; step < 5; ++step) for (int lane = 0; lane < 32; ++lane) {
                        const int peer = lane ^ (16 >> step);
                        maxima[step + 1][lane] = fmaxf(maxima[step][lane], maxima[step][peer]);
                        sums[step + 1][lane] = sums[step][lane] + sums[step][peer];
                    }
                }
                for (int lane = 0; lane < 32 && g * 64 < k; ++lane) any_sum_bad |= !isfinite(sums[5][lane]);
                for (int lane = 0; lane < 32; ++lane) {
                    vote_at = 0;
                    threadIdx.x = warp * 32 + lane;
                    shuffle_at = 0;
                    call(tf_w4a8_quantize, x, xq, xa, xs, k, ldx, flag);
                }
            }
    return conversions;
}
