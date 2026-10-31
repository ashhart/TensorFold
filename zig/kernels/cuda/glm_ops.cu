// Own kernels where the Python family used torch ops (integer work, copies, rounding), with the same bits.

#include <cuda_bf16.h>
#include <stdint.h>

constexpr long long PAGE = 512;  // rows a KV page holds

// sparse.select_tokens' assembly, a block a row: K pools' 4 tokens, then the tail; on pages each token's row in them.
extern "C" __global__ void tf_glm_dsa_tokens(const long long* __restrict__ pools, const int* __restrict__ pos, int K,
                                             int* __restrict__ tokens, int* __restrict__ counts,
                                             const int* __restrict__ pt, long long page_rows) {
    const int r = blockIdx.x;
    const long long q = static_cast<long long>(*pos) + r;
    const long long npool = (q + 1) / 4;
    const int width = 4 * K + 3;
    int* row = tokens + static_cast<long long>(r) * width;
    auto at = [&](long long t) { return static_cast<int>(pt ? pt[t / PAGE] * page_rows + t % PAGE : t); };
    for (int i = threadIdx.x; i < 4 * K; i += blockDim.x)
        row[i] = at(pools[static_cast<long long>(r) * K + i / 4] * 4 + i % 4);
    if (threadIdx.x < 3) {
        const long long t = npool * 4 + threadIdx.x;
        row[4 * K + threadIdx.x] = t <= q ? at(t) : -1;
    }
    if (threadIdx.x == 0) {
        int tail = 0;
        for (int t = 0; t < 3; ++t) tail += npool * 4 + t <= q;
        counts[r] = npool > K ? 4 * K + tail : 0;
    }
}

// pos_dev + a (sparse.select_tokens' at = pos_dev + a for a row block past the first): one int32 on the device.
extern "C" __global__ void tf_glm_add_i32(const int* __restrict__ x, int a, int* __restrict__ out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = *x + a;
}

// forward.shared_expert's b.ey[:, slot].copy_(b.sy): fp32 rows into slot `slot`, as is or bf16 round-to-nearest-even.
extern "C" __global__ void tf_glm_put_slot(const float* __restrict__ src, void* __restrict__ dst, int rows, int d,
                                           int slots, int slot, int bf16) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(rows) * d) return;
    const long long r = i / d, c = i % d;
    const long long at = (r * slots + slot) * d + c;
    if (bf16) reinterpret_cast<__nv_bfloat16*>(dst)[at] = __float2bfloat16_rn(src[i]);
    else reinterpret_cast<float*>(dst)[at] = src[i];
}

// A strided destination's copy_ (Fp4Linear's out= column slice): `rows` rows of row_bytes between pitches, as is.
extern "C" __global__ void tf_glm_copy_rows(const unsigned char* __restrict__ src, long long src_pitch,
                                            unsigned char* __restrict__ dst, long long dst_pitch, int rows,
                                            int row_bytes) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(rows) * row_bytes) return;
    const long long r = i / row_bytes, c = i % row_bytes;
    dst[r * dst_pitch + c] = src[r * src_pitch + c];
}

// TF_GLM_KV_HOST mark: each picked row (first counts[r] of tokens [rows][width]) gets a slot; slot_of -1 if unseen.
extern "C" __global__ void tf_glm_kvh_mark(const int* __restrict__ tokens, const int* __restrict__ counts, int width,
                                           long long total, int* slot_of, int* list, int* n, int cap) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    if (static_cast<int>(i % width) >= counts[i / width]) return;
    const int t = tokens[i];
    if (t < 0 || atomicCAS(slot_of + t, -1, -2) != -1) return;
    const int s = atomicAdd(n, 1);
    if (s < cap) {
        list[s] = t;
        slot_of[t] = s;
    }
}

// copy: slot s gets row list[s] (row_vec 16-byte words a row), one warp a row.
extern "C" __global__ void tf_glm_kvh_copy(const uint4* __restrict__ lc, const int* __restrict__ list,
                                           const int* __restrict__ n, uint4* __restrict__ slots, int row_vec, int cap) {
    const long long s = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (s >= min(*n, cap)) return;
    const long long t = list[s];
    for (int j = threadIdx.x % 32; j < row_vec; j += 32) slots[s * row_vec + j] = lc[t * row_vec + j];
}

// remap: the call's token lists name slots instead of rows (same order, so the attention sums are unchanged).
extern "C" __global__ void tf_glm_kvh_remap(int* tokens, const int* __restrict__ counts, int width, long long total,
                                            const int* __restrict__ slot_of) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    if (static_cast<int>(i % width) >= counts[i / width]) return;
    const int t = tokens[i];
    if (t >= 0) tokens[i] = slot_of[t];
}

// reset: the rows this call marked are unseen again.
extern "C" __global__ void tf_glm_kvh_reset(const int* __restrict__ list, const int* __restrict__ n, int* slot_of, int cap) {
    const long long s = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s < min(*n, cap)) slot_of[list[s]] = -1;
}

// rows: a sequence's rows 0 .. *pos + rows (at most max_rows) into slots in position order, through its page table.
extern "C" __global__ void tf_glm_kvh_rows(const uint4* __restrict__ lc, const int* __restrict__ pt, long long page_rows,
                                           const int* __restrict__ pos, int rows, uint4* __restrict__ slots, int row_vec,
                                           int max_rows) {
    const long long s = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (s >= min(static_cast<long long>(*pos) + rows, static_cast<long long>(max_rows))) return;
    const long long t = pt ? pt[s / PAGE] * page_rows + s % PAGE : s;
    for (int j = threadIdx.x % 32; j < row_vec; j += 32) slots[s * row_vec + j] = lc[t * row_vec + j];
}

// prev_mark: prev_of[row] = its slot in the layer's row cache, except rows this call writes (rejected drafts' rows).
extern "C" __global__ void tf_glm_kvh_prev_mark(const int* __restrict__ prev_rows, const int* __restrict__ prev_n,
                                                int* prev_of, const int* __restrict__ pt, long long page_rows,
                                                const int* __restrict__ pos, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *prev_n) return;
    const int t = prev_rows[i];
    for (int r = 0; r < rows; ++r) {
        const long long q = static_cast<long long>(*pos) + r;
        if (pt[q / PAGE] * page_rows + q % PAGE == t) return;
    }
    prev_of[t] = i;
}

// copy_cached: as copy, from the layer's cache where the row is there, else from host memory.
extern "C" __global__ void tf_glm_kvh_copy_cached(const uint4* __restrict__ lc, const int* __restrict__ list,
                                                  const int* __restrict__ n, uint4* __restrict__ slots, int row_vec,
                                                  int cap, const int* __restrict__ prev_of,
                                                  const uint4* __restrict__ cache) {
    const long long s = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (s >= min(*n, cap)) return;
    const long long t = list[s];
    const int p = prev_of[t];
    const uint4* src = p >= 0 ? cache + static_cast<long long>(p) * row_vec : lc + t * row_vec;
    for (int j = threadIdx.x % 32; j < row_vec; j += 32) slots[s * row_vec + j] = src[j];
}

// keep: the call's gathered rows become the layer's cache (the first cache_rows of them), one warp a row.
extern "C" __global__ void tf_glm_kvh_keep(const uint4* __restrict__ slots, const int* __restrict__ list,
                                           const int* __restrict__ n, uint4* __restrict__ cache, int* prev_rows,
                                           int* prev_n, int row_vec, int cache_rows) {
    const long long s = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const int m = min(*n, cache_rows);
    if (s == 0 && threadIdx.x == 0) *prev_n = m;
    if (s >= m) return;
    if (threadIdx.x % 32 == 0) prev_rows[s] = list[s];
    for (int j = threadIdx.x % 32; j < row_vec; j += 32) cache[s * row_vec + j] = slots[s * row_vec + j];
}

// unmark: the previous call's rows (old_rows, old_n) and this call's (list, n) are unseen again.
extern "C" __global__ void tf_glm_kvh_unmark(const int* __restrict__ old_rows, const int* __restrict__ old_n,
                                             int* prev_of, const int* __restrict__ list, const int* __restrict__ n,
                                             int* slot_of, int cap) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < *old_n) prev_of[old_rows[i]] = -1;
    if (i < min(*n, cap)) slot_of[list[i]] = -1;
}
