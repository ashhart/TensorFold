// fn_qsa_scores' sums without shuffles: warp_sum's tree in the thread, leaves bit-reversed, R rows a block (same bits).
namespace qsa_tile {
constexpr int DI = 128;
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ int rev5(int c) { return __brev(c) >> 27; }
// a binary counter of partial sums: level k holds a closed subtree of 2^k leaves until its sibling closes
struct Stack {
    float v[5];
    __device__ __forceinline__ void push(int c, float x) {
#pragma unroll
        for (int k = 0; k < 5; ++k) {
            if (((c >> k) & 1) == 0) { v[k] = x; return; }
            x = v[k] + x;
        }
        v[0] = x;                                         // c == 31: the root, kept in v[0]
    }
};
// PAGED: POOLED is the arena's slot in page 0, a block's row in its page (fn_attn's Pg): one lookup a thread.
template <int R, int HI, bool PAGED>
__device__ __forceinline__ void scores_body(const __nv_bfloat16* __restrict__ IQ,
                                            const __nv_bfloat16* __restrict__ POOLED, const int* __restrict__ POS0,
                                            float* __restrict__ SC, int NB, int RATIO, int TOP, int rows,
                                            const Pg& pg) {
    __shared__ float4 qs[R][HI][DI / 4];
    const int r0 = blockIdx.x * R, b = blockIdx.y * 256 + threadIdx.x;
    const int pos0 = POS0[0];
    const int last = min(r0 + R, rows) - 1;
    const int cmax = (pos0 + last + 1) / RATIO;
    if (cmax <= TOP || blockIdx.y * 256 >= cmax) return;
    for (int e = threadIdx.x; e < R * HI * DI / 4; e += blockDim.x) {
        const int ri = e / (HI * DI / 4), rem = e % (HI * DI / 4), r = min(r0 + ri, rows - 1);
        const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(IQ + (static_cast<long long>(r) * HI * DI + rem * 4));
        const float2 a = __bfloat1622float2(src[0]), c = __bfloat1622float2(src[1]);
        reinterpret_cast<float4*>(qs)[e] = make_float4(a.x, a.y, c.x, c.y);
    }
    __syncthreads();
    if (b >= cmax) return;
    Stack st[R][HI];
    const __nv_bfloat16* kb = POOLED + (PAGED ? page_row(pg, b, pg.ptok / RATIO, DI) : static_cast<long long>(b) * DI);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        float k[16][4];                                   // the 16 lanes l (bit 0 == half) a half's leaves read
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            const int l = rev5(c + 16 * half);
            const uint2 raw = *reinterpret_cast<const uint2*>(kb + 4 * l);
            const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
            for (int e = 0; e < 4; ++e) k[c][e] = __bfloat162float(p[e]);
        }
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            const int cc = c + 16 * half, l = rev5(cc);
#pragma unroll
            for (int ri = 0; ri < R; ++ri)
#pragma unroll
                for (int h = 0; h < HI; ++h) {
                    const float4 q4 = qs[ri][h][l];
                    float part = 0.0f;
                    part = __fmaf_rn(k[c][0], q4.x, part);
                    part = __fmaf_rn(k[c][1], q4.y, part);
                    part = __fmaf_rn(k[c][2], q4.z, part);
                    part = __fmaf_rn(k[c][3], q4.w, part);
                    st[ri][h].push(cc, part);
                }
        }
    }
    const float scale = __fsqrt_rn(static_cast<float>(DI));
#pragma unroll
    for (int ri = 0; ri < R; ++ri) {
        const int r = r0 + ri;
        if (r >= rows) break;
        const int complete = (pos0 + r + 1) / RATIO;
        if (complete <= TOP || b >= complete) continue;
        float total = 0.0f;
#pragma unroll
        for (int h = 0; h < HI; ++h) total = __fadd_rn(total, fmaxf(st[ri][h].v[0], 0.0f));
        SC[static_cast<long long>(r) * NB + b] = __fdiv_rn(total, scale);
    }
}
template <int R, int HI>
__global__ void __launch_bounds__(256) scores(const __nv_bfloat16* __restrict__ IQ,
                                              const __nv_bfloat16* __restrict__ POOLED, const int* __restrict__ POS0,
                                              float* __restrict__ SC, int NB, int RATIO, int TOP, int rows) {
    scores_body<R, HI, false>(IQ, POOLED, POS0, SC, NB, RATIO, TOP, rows, Pg{});
}
}  // namespace qsa_tile
