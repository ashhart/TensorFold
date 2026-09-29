// ROCm prompt attention for head size 256 on WMMA (v_wmma_f32_16x16x16_bf16): a block per 16 query rows and KV
// head, a wave per query head of the group, each 16-key tile of K and V staged once in shared memory for the group.
// S^T = K Q^T puts a query's 16 scores in its lane pair, so its row max and sum take one shuffle and P is already the
// A operand of P V. Keys fold in 16-key tiles by absolute position and a row's online softmax reads no other row
// (a row is rescaled only when its own factor is not 1), so any chunking of the prompt gives a row the same bits.

#ifndef __HIPCC__
#error "attention_rocm.cu is ROCm's prompt attention; NVIDIA builds use the Triton kernel"
#endif

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

namespace {

typedef short short8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

constexpr int D = 256;
constexpr int KPAD = 8;                     // bf16 padding a K row: the 16 rows a step reads miss each other's banks

__device__ __forceinline__ unsigned short bf16_round(float f) {
    const unsigned u = __float_as_uint(f);
    return static_cast<unsigned short>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ unsigned pack2(float a, float b) {
    return static_cast<unsigned>(bf16_round(a)) | (static_cast<unsigned>(bf16_round(b)) << 16);
}

__device__ __forceinline__ float8 wmma(uint4 a, uint4 b, float8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(short8, a),
                                                              __builtin_bit_cast(short8, b), c);
}

// q (W, H, D), caches (T, HK, D) holding keys [0, p0 + W), out (W, H, D); G = H / HK query heads a KV head; KT keys
// a softmax step, each 16 of them one WMMA's worth (16 is launched: 32 measured a third slower on an R9700).
template <int G, int KT>
__global__ void __launch_bounds__(32 * G) attention_kernel(
        const unsigned short* __restrict__ q, const unsigned short* __restrict__ kc,
        const unsigned short* __restrict__ vc, unsigned short* __restrict__ out, int p0, int w, int h, int hk,
        float scale) {
    constexpr int SUB = KT / 16, TILE = KT * D / 8, PER = (TILE + 32 * G - 1) / (32 * G);
    constexpr int VPAD = KT == 16 ? 0 : 8;                       // a V^T row's padding: B reads miss each other's banks
    __shared__ __align__(16) unsigned short ks[KT][D + KPAD];
    __shared__ __align__(16) unsigned short vt[D][KT + VPAD];
    constexpr float NEG = -__builtin_huge_valf();
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int c = lane & 15, half = lane >> 4;
    const int r0 = (gridDim.x - 1 - blockIdx.x) * 16;          // longest causal blocks first
    const int kvh = blockIdx.y, head = kvh * G + wave;
    const int row = r0 + c;                                      // this lane's query in S^T and P
    const int pos = p0 + row;
    const int keys = p0 + w;                                     // cached keys
    const int tiles = (p0 + min(r0 + 16, w) - 1) / KT + 1;
    // Q^T as the B operand of step t: query row, d = 16t + 8 half .. + 7
    // (held in registers: reloading it a tile from cache measured 22% slower)
    const size_t qat = (static_cast<size_t>(min(row, w - 1)) * h + head) * D;
    const uint4* qrow = reinterpret_cast<const uint4*>(q + qat) + half;
    uint4 qb[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) qb[t] = row < w ? qrow[2 * t] : make_uint4(0, 0, 0, 0);
    float8 o[16];
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n] = float8{0, 0, 0, 0, 0, 0, 0, 0};
    float m = NEG, l = 0.0f;
    for (int kt = 0; kt < tiles; ++kt) {
        __syncthreads();                                         // the previous tile is consumed
        // a tile, zeros past the cache: K by rows, V keys first so a wave's transposed stores hit consecutive keys;
        // every load issued before the first store
        uint4 kr[PER], vr[PER];
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int i = threadIdx.x + j * 32 * G;
            const int kkey = kt * KT + i / 32, vkey = kt * KT + i % KT;
            const size_t kat = (static_cast<size_t>(kkey) * hk + kvh) * D + (i % 32) * 8;
            const size_t vat = (static_cast<size_t>(vkey) * hk + kvh) * D + (i / KT) * 8;
            kr[j] = i < TILE && kkey < keys ? *reinterpret_cast<const uint4*>(kc + kat) : make_uint4(0, 0, 0, 0);
            vr[j] = i < TILE && vkey < keys ? *reinterpret_cast<const uint4*>(vc + vat) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int i = threadIdx.x + j * 32 * G;
            if (i < TILE) {
                *reinterpret_cast<uint4*>(&ks[i / 32][(i % 32) * 8]) = kr[j];
                const int key = i % KT, col = (i / KT) * 8;
                const unsigned e[4] = {vr[j].x, vr[j].y, vr[j].z, vr[j].w};
#pragma unroll
                for (int u = 0; u < 4; ++u) {                    // V transposed: B operand rows are keys
                    vt[col + 2 * u][key] = static_cast<unsigned short>(e[u]);
                    vt[col + 2 * u + 1][key] = static_cast<unsigned short>(e[u] >> 16);
                }
            }
        }
        __syncthreads();
        float8 s[SUB];                                           // S^T: keys 16 u + 8 half + i of the tile, query row
        float mt = NEG;
#pragma unroll
        for (int u = 0; u < SUB; ++u) {
            s[u] = float8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int t = 0; t < 16; ++t)
                s[u] = wmma(*reinterpret_cast<const uint4*>(&ks[16 * u + c][16 * t + 8 * half]), qb[t], s[u]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int key = kt * KT + 16 * u + 8 * half + i;
                s[u][i] = key <= pos && row < w ? s[u][i] * scale : NEG;
                mt = fmaxf(mt, s[u][i]);
            }
        }
        mt = fmaxf(mt, __shfl_xor(mt, 16));
        const bool active = mt != NEG;
        const float next = active ? fmaxf(m, mt) : m;
        const float alpha = active ? (m == NEG ? 0.0f : __expf(m - next)) : 1.0f;
        uint4 pa[SUB];
        float sum = 0.0f;
#pragma unroll
        for (int u = 0; u < SUB; ++u) {
            float p[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                p[i] = active && s[u][i] != NEG ? __expf(s[u][i] - next) : 0.0f;
                sum = sum + p[i];
            }
            pa[u] = make_uint4(pack2(p[0], p[1]), pack2(p[2], p[3]), pack2(p[4], p[5]), pack2(p[6], p[7]));
        }
        sum = sum + __shfl_xor(sum, 16);                         // the query's two halves (addition commutes)
        l = l * alpha + sum;
        m = next;
        if (__ballot(alpha != 1.0f) != 0) {                      // O rows 8 half + i: rescale those whose factor moved
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float a = __shfl(alpha, 8 * half + i);
                if (a != 1.0f) {
#pragma unroll
                    for (int n = 0; n < 16; ++n) o[n][i] = o[n][i] * a;
                }
            }
        }
#pragma unroll
        for (int n = 0; n < 16; ++n)
#pragma unroll
            for (int u = 0; u < SUB; ++u)
                o[n] = wmma(pa[u], *reinterpret_cast<const uint4*>(&vt[16 * n + c][16 * u + 8 * half]), o[n]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float li = __shfl(l, 8 * half + i);
        const int r = r0 + 8 * half + i;
        if (r < w) {
#pragma unroll
            for (int n = 0; n < 16; ++n)
                out[(static_cast<size_t>(r) * h + head) * D + 16 * n + c] = bf16_round(o[n][i] / li);
        }
    }
}

}  // namespace

bool attention_supported(int heads, int kv_heads, int dim) {
    const int g = kv_heads > 0 && heads % kv_heads == 0 ? heads / kv_heads : 0;
    return dim == D && (g == 1 || g == 2 || g == 4 || g == 6 || g == 8);
}

// q (W, H, 256), k_cache and v_cache (T, HK, 256) holding keys through p0 + W - 1, out (W, H, 256); bf16, contiguous.
void prompt_attention(const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache, at::Tensor& out,
                      int p0, double scale) {
    const int w = q.size(0), h = q.size(1), hk = k_cache.size(1);
    TORCH_CHECK(attention_supported(h, hk, q.size(2)), "ROCm prompt attention: head size 256, 1-8 heads a KV head");
    if (w == 0) return;
    const dim3 grid((w + 15) / 16, hk);
    auto stream = at::hip::getCurrentHIPStream();
    const auto* qp = reinterpret_cast<const unsigned short*>(q.data_ptr());
    const auto* kp = reinterpret_cast<const unsigned short*>(k_cache.data_ptr());
    const auto* vp = reinterpret_cast<const unsigned short*>(v_cache.data_ptr());
    auto* op = reinterpret_cast<unsigned short*>(out.data_ptr());
    const float sc = static_cast<float>(scale);
#define LAUNCH(G) \
    hipLaunchKernelGGL((attention_kernel<G, 16>), grid, dim3(32 * (G)), 0, stream, qp, kp, vp, op, p0, w, h, hk, sc)
    switch (h / hk) {
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        case 4: LAUNCH(4); break;
        case 6: LAUNCH(6); break;
        default: LAUNCH(8); break;
    }
#undef LAUNCH
}
