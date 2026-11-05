// The Flash Next family's EXL3 glue (cuda_exl3_pass.zig): an fp16 row as two bf16 rows whose sum is it exactly.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// hi = bf16(x), lo = bf16(x - hi): an fp16 has 11 significant bits, hi keeps 8 and the remainder has at most 3.
extern "C" __global__ void fn_exl3_split_f16(const __half* __restrict__ x, __nv_bfloat16* __restrict__ hi,
                                             __nv_bfloat16* __restrict__ lo, long long n) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __half2float(x[i]);
    const __nv_bfloat16 h = __float2bfloat16_rn(v);
    hi[i] = h;
    lo[i] = __float2bfloat16_rn(v - __bfloat162float(h));
}
