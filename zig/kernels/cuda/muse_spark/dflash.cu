// Muse Spark's DFlash2 drafter glue in fp32 (block embed, taps, q/k norm + RoPE, context rows, attention, argmax).

#include <cuda_fp16.h>
#include <cstdint>

// x [S * B, D] fp32: row s * B = embed[tokens[s]] (bf16 -> fp32), rows s * B + 1 .. = mask_emb (fp32)
extern "C" __global__ void dr_block_embed(const uint16_t* __restrict__ embed, const float* __restrict__ mask_emb,
                                          const int64_t* __restrict__ tokens, float* __restrict__ x, int B, int D) {
    const int r = blockIdx.x, s = r / B, b = r % B;
    const uint16_t* row = embed + tokens[s] * (int64_t)D;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        x[(int64_t)r * D + d] = b == 0 ? __uint_as_float(((uint32_t)row[d]) << 16) : mask_emb[d];
}

// res fp32 [n, D] -> taps fp16 [n, NT * D] column block j (torch.cat(taps, -1).to(fp16))
extern "C" __global__ void dr_tap(const float* __restrict__ res, __half* __restrict__ taps, int D, int NT, int j) {
    const int r = blockIdx.x;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        taps[((int64_t)r * NT + j) * D + d] = __float2half_rn(res[(int64_t)r * D + d]);
}

// _head_rms then _rope: a block per (row, head), 128 threads (head_dim 128). x = in.float(); y = x * rsqrt(mean(x^2) +

// eps) * w; the first ROT dims rotate NEOX-style with cos / sin at the row's position. Out: fp32 [n, NH, 128] when

// out32, else fp16 into the cache at (slot, pos % ring).
extern "C" __global__ void dr_norm_rope(const __half* __restrict__ in, const float* __restrict__ w,
                                        const float* __restrict__ cos_t, const float* __restrict__ sin_t,
                                        const int32_t* __restrict__ pos, const int32_t* __restrict__ slot,
                                        float* __restrict__ out32, __half* __restrict__ cache, int NH, int ROT,
                                        int ring, float eps) {
    __shared__ float y[128];
    __shared__ float part[4];
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float x = __half2float(in[((int64_t)r * NH + h) * 128 + d]);
    float sq = x * x;
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, o);
    if ((d & 31) == 0) part[d >> 5] = sq;
    __syncthreads();
    const float mean = (part[0] + part[1] + part[2] + part[3]) / 128.0f;
    y[d] = x * rsqrtf(mean + eps) * w[d];
    __syncthreads();
    const int half = ROT / 2;
    const int p = pos[r];
    float v = y[d];
    if (d < ROT) {
        const int j = d % half;
        const float c = cos_t[(int64_t)p * half + j], s = sin_t[(int64_t)p * half + j];
        v = d < half ? y[d] * c - y[d + half] * s : y[d] * c + y[d - half] * s;
    }
    if (out32) out32[((int64_t)r * NH + h) * 128 + d] = v;
    else cache[(((int64_t)slot[r] * ring + p % ring) * NH + h) * 128 + d] = __float2half_rn(v);
}

// v rows fp16 [n, NH * 128] into the cache at (slot, pos % ring)
extern "C" __global__ void dr_store(const __half* __restrict__ in, const int32_t* __restrict__ pos,
                                    const int32_t* __restrict__ slot, __half* __restrict__ cache, int NH, int ring) {
    const int r = blockIdx.x;
    const int64_t at = ((int64_t)slot[r] * ring + pos[r] % ring) * NH * 128;
    for (int i = threadIdx.x; i < NH * 128; i += blockDim.x) cache[at + i] = in[(int64_t)r * NH * 128 + i];
}

// Draft attention: a block per (row r = s * B + b, KV head), one warp per query head of the group (H / KV, <= 8). Keys

// pos0[s] - (W - 1) + t, t < W - 1 + B, visible when >= 0 and > the query's position - W (and within the block: all

// are); scores q.K * scale; the head's sink logit as one more column; softmax; out fp16 [S * B, H * 128] = (p @

// V).to(fp16). Keys and values come through shared memory 32 at a time, read once for the group's heads; the softmax

// runs online (max / sum rescaled per tile), so its rounding differs from torch's two-pass softmax.
#define DR_TILE 32
extern "C" __global__ void __launch_bounds__(256) dr_attn(const float* __restrict__ q, const __half* __restrict__ kc,
                                   const __half* __restrict__ vc, const float* __restrict__ sinks,
                                   const int32_t* __restrict__ pos0, const int32_t* __restrict__ slot,
                                   __half* __restrict__ out, int B, int W, int ring, int H, int KV, float scale) {
    __shared__ float qs[8][128];
    __shared__ __half ks[DR_TILE][130];   // padded rows: lane j reads row j, conflict-free
    __shared__ __half vs[DR_TILE][128];
    const int r = blockIdx.x, kvh = blockIdx.y, s = r / B, b = r % B;
    const int G = H / KV, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int span = W - 1 + B;
    const int p0 = pos0[s], qpos = p0 + b;
    const int64_t base = (int64_t)slot[s] * ring;
    const int h = kvh * G + warp;
    const bool active = warp < G;
    for (int i = threadIdx.x; i < G * 128; i += blockDim.x) qs[i / 128][i % 128] = q[((int64_t)r * H + kvh * G + i / 128) * 128 + i % 128];
    float m = -INFINITY, l = 0.0f;
    float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // the first key any query of this row sees: past kp < 0 and past qpos - W
    int t_first = (W - 1) - p0;               // kp >= 0
    if (t_first < 0) t_first = 0;
    const int t_vis = qpos - W + 1 - (p0 - (W - 1));   // kp > qpos - W
    if (t_vis > t_first) t_first = t_vis;
    for (int t0 = t_first - (t_first % DR_TILE); t0 < span; t0 += DR_TILE) {
        __syncthreads();
        for (int i = threadIdx.x; i < DR_TILE * 64; i += blockDim.x) {   // 64 half2 a row
            const int j = i / 64, d2 = i % 64;
            const int kp = p0 - (W - 1) + t0 + j;
            __half2 kv = __float2half2_rn(0.0f), vv = __float2half2_rn(0.0f);
            if (t0 + j < span && kp >= 0) {
                const int64_t at = ((base + kp % ring) * KV + kvh) * 128 + 2 * d2;
                kv = *reinterpret_cast<const __half2*>(kc + at);
                vv = *reinterpret_cast<const __half2*>(vc + at);
            }
            ks[j][2 * d2] = __low2half(kv);
            ks[j][2 * d2 + 1] = __high2half(kv);
            *reinterpret_cast<__half2*>(&vs[j][2 * d2]) = vv;
        }
        __syncthreads();
        if (!active) continue;
        const int t = t0 + lane;
        const int kp = p0 - (W - 1) + t;
        float sc = -INFINITY;
        if (t < span && kp >= 0 && kp > qpos - W) {
            float acc = 0.0f;
#pragma unroll 8
            for (int d = 0; d < 128; ++d) acc += qs[warp][d] * __half2float(ks[lane][d]);
            sc = acc * scale;
        }
        float tm = sc;
        for (int k = 16; k > 0; k >>= 1) tm = fmaxf(tm, __shfl_xor_sync(0xffffffff, tm, k));
        if (tm == -INFINITY) continue;        // nothing visible in this tile for this query
        const float mn = fmaxf(m, tm);
        const float corr = m == -INFINITY ? 0.0f : expf(m - mn);
        const float p = sc == -INFINITY ? 0.0f : expf(sc - mn);
        float ps = p;
        for (int k = 16; k > 0; k >>= 1) ps += __shfl_xor_sync(0xffffffff, ps, k);
        l = l * corr + ps;
#pragma unroll
        for (int e = 0; e < 4; ++e) o[e] *= corr;
        for (int j = 0; j < DR_TILE; ++j) {
            const float pj = __shfl_sync(0xffffffff, p, j);
            if (pj != 0.0f) {
#pragma unroll
                for (int e = 0; e < 4; ++e) o[e] += pj * __half2float(vs[j][lane * 4 + e]);
            }
        }
        m = mn;
    }
    if (!active) return;
    if (sinks) {
        const float sk = sinks[h];
        const float mn = fmaxf(m, sk);
        const float corr = m == -INFINITY ? 0.0f : expf(m - mn);
        l = l * corr + expf(sk - mn);
#pragma unroll
        for (int e = 0; e < 4; ++e) o[e] *= corr;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) out[((int64_t)r * H + h) * 128 + lane * 4 + e] = __float2half_rn(o[e] / l);
}

// rows s * B + 1 .. s * B + B - 1 of x fp16 [S * B, D] -> [S * (B - 1), D]
extern "C" __global__ void dr_take(const __half* __restrict__ x, __half* __restrict__ out, int B, int D) {
    const int r = blockIdx.x, s = r / (B - 1), b = r % (B - 1) + 1;
    for (int d = threadIdx.x; d < D; d += blockDim.x) out[(int64_t)r * D + d] = x[((int64_t)s * B + b) * D + d];
}
