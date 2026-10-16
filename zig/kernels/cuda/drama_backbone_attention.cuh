// Backbone full attention: Q/K norm and RoPE prologue, KV append and causal GQA; included by drama_backbone.cu.
#pragma once

// fused_qk_gemma_rmsnorm_rope_gate for one head: bf16 Gemma norm, NeoX RoPE on 64 dims in fp32, rounded at the store.
__device__ __forceinline__ void qk_norm_rope(const __nv_bfloat16* src, const __nv_bfloat16* w, const float* inv_freq,
                                             int pos, float eps, int lane, float (&o)[8]) {
    constexpr int D = 256, ROT = 64, HALF = 32;
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
}

// Q/K prologue for each head of a row: q to q_out [rows][H][256], k and v to the caches at pos0 + row.
extern "C" __global__ void __launch_bounds__(32) drama_attn_prep(const __nv_bfloat16* qkv, int ld_qkv,
                                                                const __nv_bfloat16* qn_w, const __nv_bfloat16* kn_w,
                                                                const float* inv_freq, __nv_bfloat16* q_out,
                                                                __nv_bfloat16* kc, __nv_bfloat16* vc, int pos0, int H,
                                                                int HK, float eps) {
    constexpr int D = 256;
    const int r = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    const int pos = pos0 + r;
    const __nv_bfloat16* row = qkv + static_cast<size_t>(r) * ld_qkv;
    const bool is_q = h < H;
    const int kh = h - H;
    const __nv_bfloat16* src = is_q ? row + h * 2 * D : row + H * 2 * D + kh * D;
    float o[8];
    qk_norm_rope(src, is_q ? qn_w : kn_w, inv_freq, pos, eps, lane, o);
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

// Causal GQA attention as Triton's (bf16(p) in P.V) with the sigmoid gate; given qn_w, one row runs the prologue too.
extern "C" __global__ void __launch_bounds__(256) drama_attention(const __nv_bfloat16* q, __nv_bfloat16* kc,
                                                                 __nv_bfloat16* vc, const __nv_bfloat16* qkv,
                                                                 int ld_qkv, __nv_bfloat16* out, int pos0, int H,
                                                                 int HK, float scale, const __nv_bfloat16* qn_w,
                                                                 const __nv_bfloat16* kn_w, const float* inv_freq,
                                                                 float eps) {
    constexpr int D = 256, NW = 8, U = 8;
    __shared__ float sm_m[NW], sm_l[NW];
    __shared__ float sm_o[NW][D];
    const int r = blockIdx.x, h = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kh = h / (H / HK);
    const int last = pos0 + r;
    const bool prologue = qn_w != nullptr;
    const __nv_bfloat16* row = qkv + static_cast<size_t>(r) * ld_qkv;
    // Cached keys go U at a time with loads in flight; with no cached key (the fused row at 0) nothing is loaded.
    const int cached = prologue ? last - 1 : last;
    const __nv_bfloat16* kp = kc + kh * D + lane * 8;
    const __nv_bfloat16* vp = vc + kh * D + lane * 8;
    uint4 kw[U], vw[U];
    if (cached >= 0) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const size_t at = static_cast<size_t>(min(warp + u * NW, cached)) * HK * D;
            kw[u] = *reinterpret_cast<const uint4*>(kp + at);
            vw[u] = *reinterpret_cast<const uint4*>(vp + at);
        }
    }
    float qv[8], o[8];
    if (prologue) {
        qk_norm_rope(row + h * 2 * D, qn_w, inv_freq, last, eps, lane, qv);
#pragma unroll
        for (int i = 0; i < 8; ++i) qv[i] = round_bf(qv[i]);
    } else {
        const __nv_bfloat16* qr = q + (static_cast<size_t>(r) * H + h) * D + lane * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) qv[i] = bf(qr[i]);
    }
    // The row's own key is the last key of warp last % NW; scored now, while the first loads are in flight.
    const bool own = prologue && warp == last % NW;
    float own_score = 0.f;
    uint4 own_value{};
    if (own) {
        float kv[8];
        qk_norm_rope(row + H * 2 * D + kh * D, kn_w, inv_freq, last, eps, lane, kv);
        own_value = *reinterpret_cast<const uint4*>(row + H * 2 * D + HK * D + kh * D + lane * 8);
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) dot += qv[i] * round_bf(kv[i]);
        own_score = warp_sum(dot) * scale;
        if (h % (H / HK) == 0) {
            const size_t at = (static_cast<size_t>(last) * HK + kh) * D + lane * 8;
            store8(kc + at, kv);
            *reinterpret_cast<uint4*>(vc + at) = own_value;
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = 0.f;
    float m = -INFINITY, l = 0.f;
    const auto step = [&](float score, const uint4& value) {
        const float mn = fmaxf(m, score);
        const float corr = expf(m - mn);  // m = -inf on the first key: 0
        const float p = expf(score - mn);
        l = l * corr + p;
        const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&value);
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = o[i] * corr + round_bf(p) * bf(v[i]);
        m = mn;
    };
    for (int j0 = warp; j0 <= cached; j0 += NW * U) {
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
            const size_t at = static_cast<size_t>(min(j0 + (U + u) * NW, cached)) * HK * D;
            kw[u] = *reinterpret_cast<const uint4*>(kp + at);
            vw[u] = *reinterpret_cast<const uint4*>(vp + at);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) sc[u] = warp_sum(sc[u]) * scale;
        const int n = min(U, (cached - j0) / NW + 1);
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (u >= n) break;
            step(sc[u], vb[u]);
        }
    }
    if (own) step(own_score, own_value);
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
