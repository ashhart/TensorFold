// Flash Next's MTP drafts (cuda_mtp.zig): a score row's best value, its first id and the row's log-sum-exp.

#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>

namespace fn_mtp {

static __device__ __forceinline__ void better(float& v, int& i, float ov, int oi) {
    if (ov > v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

}  // namespace fn_mtp

// Program (row), 1024 threads: out [rows, 4] fp32 holds the best of a row's first V values, its log-sum-exp, its id, 0.
extern "C" __global__ void __launch_bounds__(1024) fn_row_best(const __nv_bfloat16* __restrict__ L, int V, long long ld,
                                                               float* __restrict__ out) {
    using namespace fn_mtp;
    const int r = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const __nv_bfloat16* row = L + static_cast<long long>(r) * ld;
    float v = -CUDART_INF_F;
    int at = 0x7fffffff;
    for (int i = t; i < V; i += blockDim.x) {
        const float x = __bfloat162float(row[i]);
        if (x > v) {
            v = x;
            at = i;
        }
    }
    for (int o = 16; o; o >>= 1) better(v, at, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, at, o));
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float top;
    __shared__ int top_at;
    if (lane == 0) {
        sv[warp] = v;
        si[warp] = at;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < (int)(blockDim.x >> 5) ? sv[lane] : -CUDART_INF_F;
        at = lane < (int)(blockDim.x >> 5) ? si[lane] : 0x7fffffff;
        for (int o = 16; o; o >>= 1) better(v, at, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, at, o));
        if (lane == 0) {
            top = v;
            top_at = at;
        }
    }
    __syncthreads();
    const float m = top;
    float sum = 0.0f;
    for (int i = t; i < V; i += blockDim.x) sum += expf(__bfloat162float(row[i]) - m);
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    __shared__ float ss[32];
    if (lane == 0) ss[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < (int)(blockDim.x >> 5) ? ss[lane] : 0.0f;
        for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) {
            out[r * 4 + 0] = m;
            out[r * 4 + 1] = m + logf(sum);
            out[r * 4 + 2] = __int_as_float(top_at);
            out[r * 4 + 3] = 0.0f;
        }
    }
}

namespace fn_mtp {
// A bf16 value's order as an unsigned key (larger value, larger key); -0 as +0, so equal values tie as drawRow's do.
static __device__ __forceinline__ unsigned int order_key(unsigned short u) {
    if (u == 0x8000u) u = 0;
    return (u & 0x8000u) ? (~u & 0xFFFFu) : (u | 0x8000u);
}
}  // namespace fn_mtp

// A row's K best of its first V values as drawRow ranks them (ties: lowest columns): those above the K-th, then ties.
extern "C" __global__ void __launch_bounds__(1024) fn_row_topk(const __nv_bfloat16* __restrict__ L, int V, long long ld,
                                                               int K, unsigned short* __restrict__ out_v,
                                                               int* __restrict__ out_c) {
    using namespace fn_mtp;
    const int r = blockIdx.x, t = threadIdx.x, nt = blockDim.x;
    const unsigned short* row = reinterpret_cast<const unsigned short*>(L + static_cast<long long>(r) * ld);
    unsigned short* ov = out_v + static_cast<long long>(r) * K;
    int* oc = out_c + static_cast<long long>(r) * K;
    __shared__ unsigned int hist[256];
    __shared__ unsigned int scan[1024], scan_eq[1024];
    __shared__ unsigned int pick_hi, pick_lo, above;
    for (int i = t; i < 256; i += nt) hist[i] = 0;
    __syncthreads();
    for (int i = t; i < V; i += nt) atomicAdd(&hist[order_key(row[i]) >> 8], 1u);
    __syncthreads();
    if (t == 0) {
        unsigned int cum = 0;
        int b = 255;
        for (; b > 0; --b) {
            if (cum + hist[b] >= static_cast<unsigned int>(K)) break;
            cum += hist[b];
        }
        pick_hi = b;
        above = cum;
    }
    __syncthreads();
    const unsigned int hi = pick_hi;
    for (int i = t; i < 256; i += nt) hist[i] = 0;
    __syncthreads();
    for (int i = t; i < V; i += nt) {
        const unsigned int k = order_key(row[i]);
        if ((k >> 8) == hi) atomicAdd(&hist[k & 255u], 1u);
    }
    __syncthreads();
    if (t == 0) {
        unsigned int cum = above;
        int b = 255;
        for (; b > 0; --b) {
            if (cum + hist[b] >= static_cast<unsigned int>(K)) break;
            cum += hist[b];
        }
        pick_lo = b;
        above = cum;
    }
    __syncthreads();
    const unsigned int kth = (hi << 8) | pick_lo, high = above, need = K - high;
    // each thread's contiguous columns: the values above the K-th and those equal to it, counted, scanned in order
    const int per = (V + nt - 1) / nt, lo = t * per, end = min(V, lo + per);
    unsigned int gt = 0, eq = 0;
    for (int i = lo; i < end; ++i) {
        const unsigned int k = order_key(row[i]);
        gt += k > kth;
        eq += k == kth;
    }
    scan[t] = gt;
    scan_eq[t] = eq;
    __syncthreads();
    if (t == 0) {
        unsigned int s = 0, q = 0;
        for (int i = 0; i < nt; ++i) {
            const unsigned int x = scan[i], y = scan_eq[i];
            scan[i] = s;
            scan_eq[i] = q;
            s += x;
            q += y;
        }
    }
    __syncthreads();
    // in column order: the values above the K-th fill the first `high` slots, the first `need` ties the rest
    unsigned int at = scan[t], rank = scan_eq[t];
    for (int i = lo; i < end; ++i) {
        const unsigned int k = order_key(row[i]);
        if (k > kth) {
            ov[at] = row[i];
            oc[at] = i;
            ++at;
        } else if (k == kth && rank < need) {
            ov[high + rank] = row[i];
            oc[high + rank] = i;
            ++rank;
        }
    }
}
