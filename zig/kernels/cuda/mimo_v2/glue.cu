// MiMo-V2's torch glue on the decode path, each op bit-identical to the Python family's torch op.

// embed: w.embed[ids].float(); add2: the ranks' fp32 partials summed in rank order (forward._sum).

// rope: attention.rope_tables in fp64 with libdevice's pow / cos / sin, then .float().

// Built with the glue flags (no contraction, IEEE division): see zig/build/cuda.zig.

#include <cstdint>
#include <math_constants.h>
#ifndef INFINITY
#define INFINITY CUDART_INF_F
#endif

extern "C" __global__ void mimo_embed_f32(const uint16_t* __restrict__ table, const int64_t* __restrict__ ids,
                                          float* __restrict__ out, int D) {
    const int64_t r = blockIdx.x;
    const uint16_t* row = table + ids[r] * (int64_t)D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) out[r * D + d] = __uint_as_float(((uint32_t)row[d]) << 16);
}

extern "C" __global__ void mimo_add2_f32(const float* __restrict__ a, const float* __restrict__ b,
                                         float* __restrict__ out, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        out[i] = a[i] + b[i];
}

extern "C" __global__ void mimo_rope_tables(float* __restrict__ cos_out, float* __restrict__ sin_out, int context,
                                            double theta, int rope_dim) {
    const int half = rope_dim / 2;
    const int64_t n = (int64_t)context * half;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int j = (int)(i % half);
        const int64_t p = i / half;
        const double e = (double)(2 * j) / (double)rope_dim;   // arange(0, RD, 2, f64) / RD
        const double inv = 1.0 / pow(theta, e);                // 1.0 / theta ** t (torch: reciprocal, then * 1.0)
        const double ang = (double)p * inv;                    // arange(ctx, f64)[:, None] * inv[None, :]
        cos_out[i] = __double2float_rn(cos(ang));
        sin_out[i] = __double2float_rn(sin(ang));
    }
}

// logits.argmax(-1) over the gathered head (rank-major recv [world, n, vr], or one rank's [n, vr] when world 1): the

// first index of the maximum, as torch.argmax; a NaN counts as the maximum (its first index). out int32 [n].
extern "C" __global__ void mimo_argmax(const float* __restrict__ recv, int32_t* __restrict__ out, int world, int n,
                                       int vr) {
    __shared__ float bv[256];
    __shared__ int bi[256];
    const int r = blockIdx.x, tid = threadIdx.x, vocab = world * vr;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    for (int c = tid; c < vocab; c += 256) {
        const int k = c / vr, j = c % vr;
        const float v = recv[((int64_t)k * n + r) * vr + j];
        const bool nan_v = v != v, nan_b = best != best;
        if (idx == 0x7fffffff || (nan_v && !nan_b) || (!nan_b && v > best)) { best = v; idx = c; }
    }
    bv[tid] = best;
    bi[tid] = idx;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (tid < o) {
            const float v = bv[tid + o], b = bv[tid];
            const int i = bi[tid + o];
            const bool nv = v != v, nb = b != b;
            const bool take = i != 0x7fffffff && (bi[tid] == 0x7fffffff || (nv && !nb) || (nv == nb && (nv ? i < bi[tid] : (v > b || (v == b && i < bi[tid])))));
            if (take) { bv[tid] = v; bi[tid] = i; }
        }
        __syncthreads();
    }
    if (tid == 0) out[r] = bi[0];
}
