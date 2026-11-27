// ERNIE 5.5's head: MultiMax on the fp32 logits in place, each step rounded once in the vendor's order.

namespace ernie55h {

// torch.relu as the vendor's kernel writes it: NaN stays NaN, -0 becomes +0.
__device__ __forceinline__ float relu(float v) { return v <= 0.0f ? 0.0f : v; }

// x + t0 relu(r0-x) + t1 relu(x-r1) + t2 relu(r2-x)^2 + t3 relu(x-r3)^2, no step fused (FUSE: the fma control).
template <bool FUSE, bool DROP>
__device__ __forceinline__ float multimax(float x, float r0, float r1, float r2, float r3, float t0, float t1, float t2,
                                          float t3) {
    const float a = relu(__fsub_rn(r0, x));
    const float b = relu(__fsub_rn(x, r1));
    const float c = relu(__fsub_rn(r2, x));
    const float d = relu(__fsub_rn(x, r3));
    const float dd = DROP ? d : __fmul_rn(d, d);
    if (FUSE) return fmaf(t3, dd, fmaf(t2, __fmul_rn(c, c), fmaf(t1, b, fmaf(t0, a, x))));
    float y = __fadd_rn(x, __fmul_rn(t0, a));
    y = __fadd_rn(y, __fmul_rn(t1, b));
    y = __fadd_rn(y, __fmul_rn(t2, __fmul_rn(c, c)));
    return __fadd_rn(y, __fmul_rn(t3, dd));
}

template <bool FUSE, bool DROP>
__device__ __forceinline__ void rows(float* x, long long n, float r0, float r1, float r2, float r3, float t0, float t1,
                                     float t2, float t3) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = multimax<FUSE, DROP>(x[i], r0, r1, r2, r3, t0, t1, t2, t3);
}

}  // namespace ernie55h

// n logits in place, 256 a block.
extern "C" __global__ void __launch_bounds__(256)
    tf_ernie55_multimax(float* __restrict__ x, long long n, float r0, float r1, float r2, float r3, float t0, float t1,
                        float t2, float t3) {
    ernie55h::rows<false, false>(x, n, r0, r1, r2, r3, t0, t1, t2, t3);
}

// The contract's planted control: each term's multiply and add fused into an fma.
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_ernie55_multimax_fma(float* __restrict__ x, long long n, float r0, float r1, float r2, float r3, float t0,
                                float t1, float t2, float t3) {
    ernie55h::rows<true, false>(x, n, r0, r1, r2, r3, t0, t1, t2, t3);
}

// The contract's planted control: the last term's square dropped.
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_ernie55_multimax_drop_square(float* __restrict__ x, long long n, float r0, float r1, float r2, float r3,
                                        float t0, float t1, float t2, float t3) {
    ernie55h::rows<false, true>(x, n, r0, r1, r2, r3, t0, t1, t2, t3);
}
