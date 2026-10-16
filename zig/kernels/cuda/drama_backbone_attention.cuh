// Backbone full attention: Q/K norm and RoPE prologue, KV append and causal GQA; included by drama_backbone.cu.
#pragma once

// Gemma Q/K norm (bf16), NeoX RoPE on the first 64 dims with fp32 cos/sin; k and v go to the caches at pos0 + row.
extern "C" __global__ void __launch_bounds__(32) drama_attn_prep(const __nv_bfloat16* qkv, int ld_qkv,
                                                                const __nv_bfloat16* qn_w, const __nv_bfloat16* kn_w,
                                                                const float* inv_freq, __nv_bfloat16* q_out,
                                                                __nv_bfloat16* kc, __nv_bfloat16* vc, int pos0, int H,
                                                                int HK, float eps) {
    constexpr int D = 256, ROT = 64, HALF = 32;
    const int r = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    const int pos = pos0 + r;
    const __nv_bfloat16* row = qkv + static_cast<size_t>(r) * ld_qkv;
    const bool is_q = h < H;
    const int kh = h - H;
    const __nv_bfloat16* src = is_q ? row + h * 2 * D : row + H * 2 * D + kh * D;
    const __nv_bfloat16* w = is_q ? qn_w : kn_w;
    float v[8];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = bf(src[lane * 8 + i]);
        ss += v[i] * v[i];
    }
    ss = warp_sum(ss);
    const float rcp = rsqrtf(ss / static_cast<float>(D) + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = round_bf(v[i] * rcp * (1.0f + bf(w[lane * 8 + i])));
    // lanes 0..3 hold dims 0..31, lanes 4..7 dims 32..63: partner lane ^ 4
    float o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float other = __shfl_xor_sync(0xffffffffu, v[i], 4);
        const int d = lane * 8 + i;
        if (d < ROT) {
            const float ang = static_cast<float>(pos) * inv_freq[d % HALF];
            const float c = cosf(ang), s = sinf(ang);
            o[i] = d < HALF ? v[i] * c - other * s : v[i] * c + other * s;
        } else {
            o[i] = v[i];
        }
    }
    __nv_bfloat16* dst = is_q ? q_out + (static_cast<size_t>(r) * H + h) * D
                              : kc + (static_cast<size_t>(pos) * HK + kh) * D;
#pragma unroll
    for (int i = 0; i < 8; ++i) dst[lane * 8 + i] = tobf(o[i]);
    if (!is_q) {
        const __nv_bfloat16* vs = row + H * 2 * D + HK * D + kh * D;
        __nv_bfloat16* vd = vc + (static_cast<size_t>(pos) * HK + kh) * D;
#pragma unroll
        for (int i = 0; i < 8; ++i) vd[lane * 8 + i] = vs[lane * 8 + i];
    }
}

// Causal GQA attention with fp32 online softmax and bf16(p) in P.V, as Triton's tl.dot, then the sigmoid gate.
extern "C" __global__ void __launch_bounds__(256) drama_attention(const __nv_bfloat16* q, const __nv_bfloat16* kc,
                                                                 const __nv_bfloat16* vc, const __nv_bfloat16* qkv,
                                                                 int ld_qkv, __nv_bfloat16* out, int pos0, int H,
                                                                 int HK, float scale) {
    constexpr int D = 256, NW = 8, U = 8;
    __shared__ float sm_m[NW], sm_l[NW];
    __shared__ float sm_o[NW][D];
    const int r = blockIdx.x, h = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kh = h / (H / HK);
    const int last = pos0 + r;
    float qv[8], o[8];
    const __nv_bfloat16* qr = q + (static_cast<size_t>(r) * H + h) * D + lane * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        qv[i] = bf(qr[i]);
        o[i] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    // Keys go U at a time with the next batch's loads in flight; the softmax update stays in key order.
    const __nv_bfloat16* kp = kc + kh * D + lane * 8;
    const __nv_bfloat16* vp = vc + kh * D + lane * 8;
    uint4 kw[U], vw[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const size_t at = static_cast<size_t>(min(warp + u * NW, last)) * HK * D;
        kw[u] = *reinterpret_cast<const uint4*>(kp + at);
        vw[u] = *reinterpret_cast<const uint4*>(vp + at);
    }
    for (int j0 = warp; j0 <= last; j0 += NW * U) {
        float sc[U];
        uint4 vb[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const __nv_bfloat16* k = reinterpret_cast<const __nv_bfloat16*>(&kw[u]);
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) dot += qv[i] * bf(k[i]);
            sc[u] = dot;
            vb[u] = vw[u];
            const size_t at = static_cast<size_t>(min(j0 + (U + u) * NW, last)) * HK * D;
            kw[u] = *reinterpret_cast<const uint4*>(kp + at);
            vw[u] = *reinterpret_cast<const uint4*>(vp + at);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) sc[u] = warp_sum(sc[u]) * scale;
        const int n = min(U, (last - j0) / NW + 1);
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (u >= n) break;
            const float mn = fmaxf(m, sc[u]);
            const float corr = expf(m - mn);  // m = -inf on the first key: 0
            const float p = expf(sc[u] - mn);
            l = l * corr + p;
            const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&vb[u]);
#pragma unroll
            for (int i = 0; i < 8; ++i) o[i] = o[i] * corr + round_bf(p) * bf(v[i]);
            m = mn;
        }
    }
    if (lane == 0) {
        sm_m[warp] = m;
        sm_l[warp] = l;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) sm_o[warp][lane * 8 + i] = o[i];
    __syncthreads();
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < NW; ++w) M = fmaxf(M, sm_m[w]);
    float L = 0.f;
    float f[NW];
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        f[w] = sm_l[w] > 0.f ? expf(sm_m[w] - M) : 0.f;
        L += sm_l[w] * f[w];
    }
    const int d = threadIdx.x;  // 256 threads, one dim each
    float acc = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) acc += sm_o[w][d] * f[w];
    const float attn = round_bf(acc / L);
    const float gate = bf(qkv[static_cast<size_t>(r) * ld_qkv + h * 2 * D + D + d]);
    out[(static_cast<size_t>(r) * H + h) * D + d] = tobf(attn * sigmoid(gate));
}
