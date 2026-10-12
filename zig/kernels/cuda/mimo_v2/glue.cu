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

// The prompt path's MoE glue (moe.MoE._prompt), torch's ops bit for bit.

// counts = torch.bincount(pick.flatten(), minlength=E) (counts are exact whatever the atomic order)
extern "C" __global__ void mimo_moe_count(const int32_t* __restrict__ pick, int n, int32_t* __restrict__ counts) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) atomicAdd(&counts[pick[i]], 1);
}

// order = torch.argsort(pick.flatten(), stable=True): a block per expert e places its (row, slot) pairs at starts[e] ..

// in increasing flat index (a ballot scan per 1024-entry tile).
extern "C" __global__ void __launch_bounds__(1024) mimo_moe_order(const int32_t* __restrict__ pick, int n,
                                                                  const int32_t* __restrict__ starts,
                                                                  int32_t* __restrict__ order) {
    __shared__ int warp_n[32];
    __shared__ int base;
    const int e = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) base = starts[e];
    __syncthreads();
    for (int t0 = 0; t0 < n; t0 += blockDim.x) {
        const int i = t0 + threadIdx.x;
        const bool mine = i < n && pick[i] == e;
        const unsigned bal = __ballot_sync(0xffffffffu, mine);
        if (lane == 0) warp_n[warp] = __popc(bal);
        __syncthreads();
        int before = 0;
        for (int w = 0; w < warp; ++w) before += warp_n[w];
        if (mine) order[base + before + __popc(bal & ((1u << lane) - 1u))] = i;
        __syncthreads();
        if (threadIdx.x == 0) {
            int tot = 0;
            for (int w = 0; w < (int)(blockDim.x >> 5); ++w) tot += warp_n[w];
            base += tot;
        }
        __syncthreads();
    }
}

// xs = x.index_select(0, order // k); ws = wts.flatten().index_select(0, order) / scale (torch divides by a CPU scalar

// as a product with its fp32 reciprocal); inv[order] = arange. A block per routed copy j.
extern "C" __global__ void mimo_moe_gather(const uint16_t* __restrict__ x, const int32_t* __restrict__ order,
                                           const float* __restrict__ wts, float inv_scale, int k, int D,
                                           uint16_t* __restrict__ xs, float* __restrict__ ws,
                                           int32_t* __restrict__ inv) {
    const int64_t j = blockIdx.x;
    const int32_t o = order[j];
    const uint16_t* src = x + (int64_t)(o / k) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) xs[j * D + d] = src[d];
    if (threadIdx.x == 0) {
        ws[j] = wts[o] * inv_scale;
        inv[o] = (int32_t)j;
    }
}

// out = d.index_select(0, inv).view(rows, k, D).sum(1) (* scale when scale != 1): torch's reduce over a strided

// dimension of 8 runs one thread an output with 4 accumulators (vt0): a_i = (0 + x_i) + x_{i+4}, then ((a_0 + a_1) +

// a_2) + a_3. k == 8 only.
extern "C" __global__ void mimo_moe_combine(const float* __restrict__ d, const int32_t* __restrict__ inv, int D,
                                            float scale, int apply, float* __restrict__ out) {
    const int64_t r = blockIdx.x;
    int32_t at[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) at[j] = inv[r * 8 + j];
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float a[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = 0.0f + d[(int64_t)at[i] * D + c];
#pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = a[i] + d[(int64_t)at[i + 4] * D + c];
        float y = ((a[0] + a[1]) + a[2]) + a[3];
        if (apply) y = y * scale;
        out[r * D + c] = y;
    }
}
