// Batched DFlash glue: stream-local convolution and rotary positions, then 256-key attention chunks.
#pragma once
struct DraftSegment { unsigned long long ring; long long begin, position; };
constexpr int DRAFT_CHUNKS = 9;

extern "C" __global__ void tf_q35d_dyn_conv_batch(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ DYN,
                                              const __nv_bfloat16* __restrict__ BASE, const __nv_bfloat16* __restrict__ RES,
                                              __nv_bfloat16* __restrict__ OUT, int D, int group, int branch, int block_rows) {
    const int t = blockIdx.x;
    const int groups = D / group;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
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

// Each warp accumulates 256 logical keys of one row, reading only that stream's ring and block rows.
extern "C" __global__ void __launch_bounds__(32) tf_q35d_attn_partial(const __nv_bfloat16* Q,
        const __nv_bfloat16* KB, const __nv_bfloat16* VB, const DraftSegment* segments,
        int block_rows, int heads, int kv_heads, int W, int window, int layer, float* PM, float* PL, float* PO) {
    constexpr int DPL = HD / 32;
    const int lane = threadIdx.x, m = blockIdx.x, h = blockIdx.y, chunk = blockIdx.z;
    const int stream = m / block_rows, local = m % block_rows;
    const DraftSegment sg = segments[stream];
    const int kvh = h / (heads / kv_heads), width = kv_heads * HD;
    const __nv_bfloat16* RK = reinterpret_cast<const __nv_bfloat16*>(sg.ring) + (size_t)layer * 2 * W * width;
    const __nv_bfloat16* RV = RK + (size_t)W * width;
    const long long first = max(sg.begin, sg.position + local - window + 1);
    const int ring_keys = (int)(sg.position - first), keys = ring_keys + block_rows;
    const size_t slot = ((size_t)m * heads + h) * DRAFT_CHUNKS + chunk;
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) {
        q[i] = bf(Q[((size_t)m * heads + h) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, sum = 0.0f;
    const int end = min(keys, (chunk + 1) * 256);
    for (int index = chunk * 256; index < end; ++index) {
        const bool ring = index < ring_keys;
        const size_t row = ring ? (size_t)((first + index) % W) : (size_t)(stream * block_rows + index - ring_keys);
        const size_t at = (row * kv_heads + kvh) * HD + lane * DPL;
        const __nv_bfloat16* k = (ring ? RK : KB) + at;
        const __nv_bfloat16* v = (ring ? RV : VB) + at;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; ++i) dot = fmaf(q[i], bf(k[i]), dot);
        const float score = warp_sum(dot) * rsqrtf((float)HD);
        const float next = fmaxf(mx, score), a = __expf(mx - next), b = __expf(score - next);
        sum = sum * a + b;
#pragma unroll
        for (int i = 0; i < DPL; ++i) o[i] = fmaf(b, bf(v[i]), o[i] * a);
        mx = next;
    }
    if (lane == 0) { PM[slot] = mx; PL[slot] = sum; }
#pragma unroll
    for (int i = 0; i < DPL; ++i) PO[slot * HD + lane * DPL + i] = o[i];
}

// Merge nonempty chunks in logical order with FP32 maxima, sums and value accumulators, then cast once to BF16.
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
