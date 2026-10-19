// Batched DFlash glue: stream-local convolution and rotary positions, then 256-key attention chunks.
#pragma once
struct DraftSegment { unsigned long long ring; long long begin, position; };
constexpr int DRAFT_CHUNKS = 9;

extern "C" __global__ void tf_q35d_dyn_conv_batch(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ DYN,
                                              const __nv_bfloat16* __restrict__ BASE, const __nv_bfloat16* __restrict__ RES,
                                              __nv_bfloat16* __restrict__ OUT, int D, int group, int branch, int block_rows) {
    const int t = blockIdx.x;
    const int groups = D / group;
    // grid (rows, channel blocks): a thread a channel, so a 16-row window fills the device instead of 16 blocks
    for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < D; c += gridDim.y * blockDim.x) {
        const int g = c / group;
        float total = 0.0f;
#pragma unroll
        for (int l = 0; l < 2; l++) {
            if (t % block_rows - l < 0) continue;
            const float x = bf(X[(size_t)(t - l) * D + c]);
            total = rbf(total + rbf(bf(BASE[(branch * 2 + l) * D + c]) * x));
            total = rbf(total + rbf(bf(DYN[(size_t)t * (4 * groups) + (branch * 2 + l) * groups + g]) * x));
        }
        if (RES != nullptr) total = rbf(total + bf(RES[(size_t)t * D + c]));
        OUT[(size_t)t * D + c] = __float2bfloat16_rn(total);
    }
}

extern "C" __global__ void __launch_bounds__(32) tf_q35d_norm_rope_batch(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                     __nv_bfloat16* __restrict__ OUT, int heads, int stride, const DraftSegment* segments, int block_rows,
                                                                     float eps, float theta) {
    const int head = blockIdx.x % heads;
    const size_t m = blockIdx.x / heads;
    const int t = threadIdx.x;
    const __nv_bfloat16* x = X + m * stride + head * HD;
    float v[4], acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = bf(x[4 * t + i]);
        acc += v[i] * v[i];
    }
    acc = warp_sum(acc);
    const float inv = rsqrtf(acc / (float)HD + eps);
    float n[4];
#pragma unroll
    for (int i = 0; i < 4; i++) n[i] = rbf(rbf(v[i] * inv) * bf(W[4 * t + i]));
    // the pair of lane element d is d + HD / 2: this lane's partner values come from the lane HD / 8 = 16 away
    const float pos = (float)(segments[m / block_rows].position + (long long)(m % block_rows));
    __nv_bfloat16* out = OUT + (m * heads + head) * HD;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = 4 * t + i;
        const float partner = __shfl_xor_sync(0xffffffffu, n[i], 16);
        const int j = d < HD / 2 ? d : d - HD / 2;
        const float angle = pos * powf(theta, -(float)j / (float)(HD / 2));
        float sn, cs;
        sincosf(angle, &sn, &cs);
        out[d] = __float2bfloat16_rn(d < HD / 2 ? n[i] * cs - partner * sn : n[i] * cs + partner * sn);
    }
}

// Remove each stream's pending row before the output head and selector; rows stay grouped by stream.
extern "C" __global__ void tf_q35d_mask_rows(const __nv_bfloat16* X, __nv_bfloat16* OUT, int D, int block_rows) {
    const int row = blockIdx.x, source = row / (block_rows - 1) * block_rows + 1 + row % (block_rows - 1);
    for (int c = threadIdx.x; c < D; c += blockDim.x) OUT[(size_t)row * D + c] = X[(size_t)source * D + c];
}

// 256 keys of one row, its stream's ring and block rows only: four warps' lanes score them, warp 0 merges the slices.
extern "C" __global__ void __launch_bounds__(128) tf_q35d_attn_partial(const __nv_bfloat16* Q,
        const __nv_bfloat16* KB, const __nv_bfloat16* VB, const DraftSegment* segments,
        int block_rows, int heads, int kv_heads, int W, int window, int layer, float* PM, float* PL, float* PO) {
    constexpr int DPL = HD / 32, CHUNK = 256, WARPS = 4, SLICE = CHUNK / WARPS, PER = SLICE / 32, BATCH = 16;
    __shared__ float qs[HD];
    __shared__ float ps[CHUNK];
    __shared__ float wm[WARPS], wl[WARPS];
    __shared__ float wo[WARPS][HD];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int m = blockIdx.x, h = blockIdx.y, chunk = blockIdx.z;
    const int stream = m / block_rows, local = m % block_rows;
    const DraftSegment sg = segments[stream];
    const int kvh = h / (heads / kv_heads), width = kv_heads * HD;
    const __nv_bfloat16* RK = reinterpret_cast<const __nv_bfloat16*>(sg.ring) + (size_t)layer * 2 * W * width;
    const __nv_bfloat16* RV = RK + (size_t)W * width;
    const long long first = max(sg.begin, sg.position + local - window + 1);
    const int ring_keys = (int)(sg.position - first), keys = ring_keys + block_rows;
    const size_t slot = ((size_t)m * heads + h) * DRAFT_CHUNKS + chunk;
    if (chunk * CHUNK >= keys) {  // past the stream's keys: an empty partial, which the merge skips
        if (threadIdx.x == 0) { PM[slot] = -INFINITY; PL[slot] = 0.0f; }
        return;
    }
    const int start = chunk * CHUNK + warp * SLICE, n = max(min(keys - start, SLICE), 0);  // this warp's keys
    const int ring0 = start < ring_keys ? (int)((first + start) % W) : 0;  // the slice's first ring row, then wrapping
    auto row_at = [&](int e) -> size_t {  // key start + e's element offset in the ring (RK, RV) or the block (KB, VB)
        const int index = start + e, wrapped = ring0 + e;
        const size_t row = index < ring_keys ? (size_t)(wrapped >= W ? wrapped - W : wrapped) : (size_t)(stream * block_rows + index - ring_keys);
        return (row * kv_heads + kvh) * HD;
    };
    for (int i = threadIdx.x; i < HD; i += 32 * WARPS) qs[i] = bf(Q[((size_t)m * heads + h) * HD + i]);
    __syncthreads();
    float mx = -INFINITY, score[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int e = lane + 32 * j;
        score[j] = -INFINITY;
        if (e >= n) continue;
        const int index = start + e;
        const uint4* k = reinterpret_cast<const uint4*>((index < ring_keys ? RK : KB) + row_at(e));
        float dot = 0.0f;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            uint4 kw[HD / 16];
#pragma unroll
            for (int c = 0; c < HD / 16; ++c) kw[c] = k[half * (HD / 16) + c];
#pragma unroll
            for (int c = 0; c < HD / 16; ++c) {
                const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kw[c]);
#pragma unroll
                for (int i = 0; i < 8; ++i) dot = fmaf(qs[8 * (half * (HD / 16) + c) + i], bf(kb[i]), dot);
            }
        }
        score[j] = dot * rsqrtf((float)HD);
        mx = fmaxf(mx, score[j]);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const float p = lane + 32 * j < n ? __expf(score[j] - mx) : 0.0f;
        ps[warp * SLICE + lane + 32 * j] = p;
        sum += p;
    }
    sum = warp_sum(sum);
    __syncwarp();
    float o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) o[i] = 0.0f;
    for (int e0 = 0; e0 < n; e0 += BATCH) {
        uint2 vw[BATCH];
#pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            const int index = start + e0 + b;
            if (e0 + b < n) vw[b] = *reinterpret_cast<const uint2*>((index < ring_keys ? RV : VB) + row_at(e0 + b) + lane * DPL);
        }
#pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            if (e0 + b >= n) break;
            const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vw[b]);
            const float p = ps[warp * SLICE + e0 + b];
#pragma unroll
            for (int i = 0; i < DPL; ++i) o[i] = fmaf(p, bf(vb[i]), o[i]);
        }
    }
    if (lane == 0) { wm[warp] = n > 0 ? mx : -INFINITY; wl[warp] = n > 0 ? sum : 0.0f; }
#pragma unroll
    for (int i = 0; i < DPL; ++i) wo[warp][lane * DPL + i] = o[i];
    __syncthreads();
    if (warp != 0) return;
    float top = -INFINITY;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) if (wl[w] > 0.0f) top = fmaxf(top, wm[w]);
    float total = 0.0f, out[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) out[i] = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        if (wl[w] == 0.0f) continue;
        const float scale = __expf(wm[w] - top);
        total = fmaf(wl[w], scale, total);
#pragma unroll
        for (int i = 0; i < DPL; ++i) out[i] = fmaf(wo[w][lane * DPL + i], scale, out[i]);
    }
    if (lane == 0) { PM[slot] = total > 0.0f ? top : -INFINITY; PL[slot] = total; }
#pragma unroll
    for (int i = 0; i < DPL; ++i) PO[slot * HD + lane * DPL + i] = out[i];
}

extern "C" __global__ void __launch_bounds__(32) tf_q35d_attn_merge(const float* PM, const float* PL,
        const float* PO, int heads, __nv_bfloat16* OUT) {
    constexpr int DPL = HD / 32;
    const int lane = threadIdx.x, m = blockIdx.x, h = blockIdx.y;
    const size_t base = ((size_t)m * heads + h) * DRAFT_CHUNKS;
    float mx = -INFINITY, sum = 0.0f, out[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) out[i] = 0.0f;
    for (int chunk = 0; chunk < DRAFT_CHUNKS; ++chunk) {
        const size_t slot = base + chunk;
        if (PL[slot] == 0.0f) continue;
        const float next = fmaxf(mx, PM[slot]), a = __expf(mx - next), b = __expf(PM[slot] - next);
        sum = sum * a + PL[slot] * b;
#pragma unroll
        for (int i = 0; i < DPL; ++i) out[i] = fmaf(b, PO[slot * HD + lane * DPL + i], out[i] * a);
        mx = next;
    }
#pragma unroll
    for (int i = 0; i < DPL; ++i) OUT[((size_t)m * heads + h) * HD + lane * DPL + i] = __float2bfloat16_rn(out[i] / sum);
}
