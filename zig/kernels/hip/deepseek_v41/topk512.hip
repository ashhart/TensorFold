// DeepSeek-V4.1's indexer top-512 on HIP, topk512.metal's arithmetic; fault 1-4: NaN high, ties high, last lost, write.

#include <hip/hip_runtime.h>
#include <stdint.h>

#define TF_TOPK512_TG 1024u
#define TF_TOPK512_K 512u
#define TF_TOPK512_BINS 2048u

static_assert(TF_WAVE == 32, "the scans below are wave32's");

// An fp32's bits as a key that sorts like the number, NaN as -inf (a control: as +inf) and -0 as +0.
__device__ __forceinline__ uint32_t tf_topk512_key(uint32_t u, int fault) {
    if ((u & 0x7fffffffu) > 0x7f800000u) u = fault == 1 ? 0x7f800000u : 0xff800000u;
    if (u == 0x80000000u) u = 0u;
    return (u >> 31) ? ~u : (u | 0x80000000u);
}

// A wave's exclusive prefix sum, in lane order.
__device__ __forceinline__ uint32_t tf_topk512_excl(uint32_t v, uint32_t lane) {
    uint32_t x = v;
    for (uint32_t o = 1; o < 32u; o <<= 1) {
        const uint32_t y = __shfl_up(x, o, 32);
        if (lane >= o) x += y;
    }
    return x - v;
}

__device__ __forceinline__ uint32_t tf_topk512_wsum(uint32_t v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

// The bin holding the k-th largest key and the keys above it (thread t owns bins 2047-2t and 2046-2t, from the top).
__device__ __forceinline__ uint2 tf_topk512_pick(const uint32_t* bins, uint32_t k, uint32_t* sc, uint32_t tid,
                                                 uint32_t lane, uint32_t wave) {
    const uint32_t a = bins[TF_TOPK512_BINS - 1u - 2u * tid], b = bins[TF_TOPK512_BINS - 2u - 2u * tid];
    const uint32_t local = a + b, ex = tf_topk512_excl(local, lane);
    if (lane == 31u) sc[wave] = ex + local;
    __syncthreads();
    uint32_t before = ex;
    for (uint32_t w = 0; w < wave; ++w) before += sc[w];
    if (before < k && before + local >= k) {
        const bool first = before + a >= k;
        sc[32] = TF_TOPK512_BINS - (first ? 1u : 2u) - 2u * tid;
        sc[33] = first ? before : before + a;
    }
    __syncthreads();
    const uint2 r = make_uint2(sc[32], sc[33]);
    __syncthreads();
    return r;
}

// Row r's top min(512, visible[r]) by a stable rank (NaN as -inf, -0 as +0): indices ascending then -1 (fault 0 served)
extern "C" __global__ void __launch_bounds__(TF_TOPK512_TG)
    tf_dsv41_topk512(const uint32_t* scores, const int* visible, int* selected, uint32_t width, int fault) {
    __shared__ uint32_t bins[TF_TOPK512_BINS];
    __shared__ uint32_t sc[34], part[64];
    const uint32_t row = blockIdx.x, tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
    const uint32_t n = min(static_cast<uint32_t>(max(visible[row], 0)), width);
    const uint32_t* s = scores + static_cast<uint64_t>(row) * width;
    int* out = selected + static_cast<uint64_t>(row) * TF_TOPK512_K;
    const uint32_t keep = min(n, TF_TOPK512_K);
    const uint32_t m = (fault == 3 && n > 0u) ? n - 1u : n;
    if (fault == 4 && tid == 0u && n > 0u) const_cast<uint32_t*>(s)[0] ^= 1u;
    // the threshold key: three passes of a radix select over 11, 11 and 10 bits from the top
    uint32_t threshold = 0u, above = 0u, ties = 0u;
    if (n > TF_TOPK512_K) {
        uint32_t prefix = 0u, k = TF_TOPK512_K;
        for (uint32_t pass = 0; pass < 3u; ++pass) {
            const uint32_t shift = pass == 0u ? 21u : (pass == 1u ? 10u : 0u);
            const uint32_t mask = pass == 2u ? 0x3ffu : 0x7ffu;
            const uint32_t up = pass == 1u ? 21u : 10u;
            for (uint32_t i = tid; i < TF_TOPK512_BINS; i += TF_TOPK512_TG) bins[i] = 0u;
            __syncthreads();
            if (pass == 0u) {
                uint32_t b3 = 0u;  // bin 3 holds -inf (and NaN): a masked row's many, summed a wave at a time
                for (uint32_t i = tid; i < m; i += TF_TOPK512_TG) {
                    const uint32_t bin = tf_topk512_key(s[i], fault) >> 21u;
                    if (bin == 3u) ++b3;
                    else atomicAdd(bins + bin, 1u);
                }
                const uint32_t sb3 = tf_topk512_wsum(b3);
                if (lane == 0u && sb3 != 0u) atomicAdd(bins + 3u, sb3);
            } else {
                for (uint32_t i = tid; i < m; i += TF_TOPK512_TG) {
                    const uint32_t key = tf_topk512_key(s[i], fault);
                    if ((key >> up) == prefix) atomicAdd(bins + ((key >> shift) & mask), 1u);
                }
            }
            __syncthreads();
            const uint2 r = tf_topk512_pick(bins, k, sc, tid, lane, wave);
            prefix = (prefix << (pass == 2u ? 10u : 11u)) | r.x;
            k -= r.y;
            above += r.y;
        }
        threshold = prefix;
        ties = bins[prefix & 0x3ffu];
    }
    // the kept set in index order: every key above the threshold, then the lowest-indexed keys equal to it
    const uint32_t wanted = keep - above;
    const uint32_t skip = fault == 2 ? ties - wanted : 0u;
    uint32_t gt_run = 0u, eq_run = 0u;
    for (uint32_t base = 0; base < m && gt_run + min(eq_run, wanted) < keep; base += TF_TOPK512_TG) {
        const uint32_t i = base + tid;
        const uint32_t key = i < m ? tf_topk512_key(s[i], fault) : 0u;
        const uint32_t gt = (i < m && (n <= TF_TOPK512_K || key > threshold)) ? 1u : 0u;
        const uint32_t eq = (i < m && n > TF_TOPK512_K && key == threshold) ? 1u : 0u;
        const uint32_t packed = gt | (eq << 16);
        const uint32_t ex = tf_topk512_excl(packed, lane);
        uint32_t* p = part + 32u * ((base / TF_TOPK512_TG) & 1u);  // two halves: one barrier a block of indices
        if (lane == 31u) p[wave] = ex + packed;
        __syncthreads();
        uint32_t before = ex, total = 0u;
        for (uint32_t q = 0; q < 32u; ++q) {
            const uint32_t v = p[q];
            before += q < wave ? v : 0u;
            total += v;
        }
        const uint32_t gt_x = gt_run + (before & 0xffffu), eq_x = eq_run + (before >> 16);
        const uint32_t eq_taken = fault == 2 ? (eq_x > skip ? eq_x - skip : 0u) : min(eq_x, wanted);
        if (gt != 0u || (eq != 0u && (fault == 2 ? eq_x >= skip : eq_x < wanted))) out[gt_x + eq_taken] = static_cast<int>(i);
        gt_run += total & 0xffffu;
        eq_run += total >> 16;
    }
    for (uint32_t i = keep + tid; i < TF_TOPK512_K; i += TF_TOPK512_TG) out[i] = -1;
}
