// MiMo-V2's full-attention layers on an 8-bit key/value cache (--policy kv=fp8); not bit-equal to the 16-bit cache.

// Keys [slots, LEN, HK, 192] and values [slots, LEN, HK, 128] in e4m3, an fp32 scale (absmax / 448) a (token, head).

// mimo_qkv8: attention._qkv for a full layer, RoPE on q and k, q to QO in fp16, k and v quantised at pos % LEN.

// mimo_rows8: attention._rows for a full layer, a program (row run, key head, chunk); partials in Triton's layout.

// mimo_merge8 folds the partials; mimo_kv8_dequant gives the prompt kernel fp16 keys; quant and fill serve tests.

// Fragment layouts and the fold follow prefill_attention.cu (tf_prefill_attention::pattn_kernel), fp16 not bf16.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

namespace mimo_kv8 {

constexpr int D = 192, DV = 128, BN = 64, G = 16;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float FP8_MAX = 448.0f;

__device__ __forceinline__ uint32_t smem(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x * LOG2E));
    return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ float rowsum(const float (&c)[8]) {
    float w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float x = c[j] + __shfl_xor_sync(0xffffffffu, c[j], 2);
        w[j] = x + __shfl_xor_sync(0xffffffffu, x, 1);
    }
    return ((w[0] + w[4]) + (w[2] + w[6])) + ((w[1] + w[5]) + (w[3] + w[7]));
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int x = 16; x >= 1; x /= 2) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, x));
    return v;
}

__device__ __forceinline__ uint8_t to_fp8(float v) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

// 16 e4m3 bytes times scale -> 16 halves (two 16-byte chunks)
__device__ __forceinline__ void dequant16(uint4 raw, float scale, uint4& lo, uint4& hi) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(&raw);
    uint32_t out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(p[i]), __NV_E4M3);
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
        out[i] = pack(f.x * scale, f.y * scale);
    }
    lo = make_uint4(out[0], out[1], out[2], out[3]);
    hi = make_uint4(out[4], out[5], out[6], out[7]);
}

} // namespace mimo_kv8

using namespace mimo_kv8;

// Grid (rows, H + 2 HK), one warp. Q / K / V rows of 192 halves a head (V: 128 real, 64 padding).
extern "C" __global__ void __launch_bounds__(32) mimo_qkv8(
    const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
    const float* __restrict__ COS, const float* __restrict__ SIN, const int32_t* __restrict__ POS,
    const int32_t* __restrict__ SLOT, __half* __restrict__ QO, uint8_t* __restrict__ KC, float* __restrict__ KS,
    uint8_t* __restrict__ VC, float* __restrict__ VS, int LEN, int H, int HK) {
    const int64_t r = blockIdx.x;
    const int hd = blockIdx.y, lane = threadIdx.x;
    const int pos = POS[r];
    const int64_t at = static_cast<int64_t>(SLOT[r]) * LEN + pos % LEN;
    if (hd < H + HK) {
        const bool is_q = hd < H;
        const __half* src = is_q ? Q + (r * H + hd) * D : K + (r * HK + (hd - H)) * D;
        const float x0 = __half2float(src[lane]), x1 = __half2float(src[lane + 32]);
        const float c = COS[static_cast<int64_t>(pos) * 32 + lane], s = SIN[static_cast<int64_t>(pos) * 32 + lane];
        const __half y0 = __float2half_rn(x0 * c - x1 * s), y1 = __float2half_rn(x1 * c + x0 * s);
        __half rest[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) rest[i] = src[64 + lane + 32 * i];
        if (is_q) {
            __half* dst = QO + (r * H + hd) * D;
            dst[lane] = y0;
            dst[lane + 32] = y1;
#pragma unroll
            for (int i = 0; i < 4; ++i) dst[64 + lane + 32 * i] = rest[i];
            return;
        }
        float v[6] = {__half2float(y0), __half2float(y1)};
        float amax = fmaxf(fabsf(v[0]), fabsf(v[1]));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[2 + i] = __half2float(rest[i]);
            amax = fmaxf(amax, fabsf(v[2 + i]));
        }
        amax = warp_max(amax);
        const float scale = amax > 0.0f ? amax / FP8_MAX : 0.0f, inv = amax > 0.0f ? FP8_MAX / amax : 0.0f;
        const int64_t e = at * HK + (hd - H);
        uint8_t* dst = KC + e * D;
        dst[lane] = to_fp8(v[0] * inv);
        dst[lane + 32] = to_fp8(v[1] * inv);
#pragma unroll
        for (int i = 0; i < 4; ++i) dst[64 + lane + 32 * i] = to_fp8(v[2 + i] * inv);
        if (lane == 0) KS[e] = scale;
    } else {
        const int k = hd - H - HK;
        const __half* src = V + (r * HK + k) * D;
        float v[4], amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i] = __half2float(src[lane + 32 * i]);
            amax = fmaxf(amax, fabsf(v[i]));
        }
        amax = warp_max(amax);
        const float scale = amax > 0.0f ? amax / FP8_MAX : 0.0f, inv = amax > 0.0f ? FP8_MAX / amax : 0.0f;
        const int64_t e = at * HK + k;
#pragma unroll
        for (int i = 0; i < 4; ++i) VC[e * DV + lane + 32 * i] = to_fp8(v[i] * inv);
        if (lane == 0) VS[e] = scale;
    }
}

// Tiles stay 8-bit in shared memory, two stages in flight; a key's scale multiplies its score and its probability.
namespace mimo_kv8 {
constexpr int KROW = D + 16, VROW = DV + 16; // padded rows (bytes): fewer bank conflicts on the pair reads
constexpr int STAGE = BN * KROW + BN * VROW + 2 * BN * 4;

__device__ __forceinline__ void cp16z(void* dst, const void* src, bool in) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem(dst)), "l"(src), "r"(in ? 16 : 0));
}
__device__ __forceinline__ void cp4z(void* dst, const void* src, bool in) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem(dst)), "l"(src), "r"(in ? 4 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ uint32_t cvt2(uint16_t pair) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
    return *reinterpret_cast<const uint32_t*>(&h);
}
} // namespace mimo_kv8

extern "C" __global__ void __launch_bounds__(128) mimo_rows8(
    const __half* __restrict__ Q, const uint8_t* __restrict__ KC, const float* __restrict__ KS,
    const uint8_t* __restrict__ VC, const float* __restrict__ VS, const int32_t* __restrict__ POS,
    const int32_t* __restrict__ SLOT, float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL, int LEN,
    int NCH, int H, int HK, float scale, int CKK) {
    __shared__ __align__(16) uint8_t st[2][STAGE];
    const int r = blockIdx.x, hk = blockIdx.y, c = blockIdx.z, n = gridDim.x;
    const int slot = SLOT[r];
    if (r % 4 != 0 && SLOT[r - 1] == slot) return;
    int R = 1;
    while (R < 4 && r + R < n && (r + R) % 4 != 0 && SLOT[r + R] == slot) ++R;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4;
    const bool live = warp < R;
    const int row = r + (live ? warp : 0);
    const int pos = POS[row];
    int maxpos = POS[r];
    for (int i = 1; i < R; ++i) maxpos = max(maxpos, POS[r + i]);
    const int start = c * CKK;
    const int64_t base = static_cast<int64_t>(slot) * LEN;
    float o[DV / 8][4];
#pragma unroll
    for (int i = 0; i < DV / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;
    const int head0 = hk * G + g;
    const int tiles = start <= maxpos ? min(CKK / BN, (maxpos - start) / BN + 1) : 0;
    auto stage = [&](int t) {
        uint8_t* s = st[t & 1];
        const int key0 = start + t * BN;
        for (int i = threadIdx.x; i < BN * (D / 16); i += blockDim.x) {
            const int key = i / (D / 16), cc = i % (D / 16), kidx = key0 + key;
            const bool in = kidx <= maxpos;
            cp16z(s + key * KROW + cc * 16, KC + ((base + (in ? kidx : 0)) * HK + hk) * D + cc * 16, in);
        }
        for (int i = threadIdx.x; i < BN * (DV / 16); i += blockDim.x) {
            const int key = i / (DV / 16), cc = i % (DV / 16), kidx = key0 + key;
            const bool in = kidx <= maxpos;
            cp16z(s + BN * KROW + key * VROW + cc * 16, VC + ((base + (in ? kidx : 0)) * HK + hk) * DV + cc * 16, in);
        }
        for (int i = threadIdx.x; i < 2 * BN; i += blockDim.x) {
            const int key = i % BN, kidx = key0 + key;
            const bool in = kidx <= maxpos;
            const float* src = (i < BN ? KS : VS) + (base + (in ? kidx : 0)) * HK + hk;
            cp4z(s + BN * KROW + BN * VROW + i * 4, src, in);
        }
    };
    uint32_t qa[D / 16][4];
    if (tiles > 0) {
        stage(0);
        commit();
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(Q + (static_cast<int64_t>(row) * H + head0) * D);
        const uint32_t* q1 = q0 + 8 * D / 2;
#pragma unroll
        for (int k = 0; k < D / 16; ++k) {
            qa[k][0] = q0[8 * k + qd];
            qa[k][1] = q1[8 * k + qd];
            qa[k][2] = q0[8 * k + qd + 4];
            qa[k][3] = q1[8 * k + qd + 4];
        }
    }
    for (int t = 0; t < tiles; ++t) {
        if (t + 1 < tiles) stage(t + 1);
        commit();
        wait_group<1>();
        __syncthreads();
        const uint8_t* s = st[t & 1];
        const uint8_t* ks8 = s;
        const uint8_t* vs8 = s + BN * KROW;
        const float* ksc = reinterpret_cast<const float*>(s + BN * KROW + BN * VROW);
        const float* vsc = ksc + BN;
        const int key0 = start + t * BN;
        if (live && key0 <= pos) {
            float sc[8][4];
#pragma unroll
            for (int i = 0; i < 8; ++i) sc[i][0] = sc[i][1] = sc[i][2] = sc[i][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < D / 16; ++k) {
#pragma unroll
                for (int nn = 0; nn < 8; ++nn) {
                    const uint8_t* kr = ks8 + (8 * nn + g) * KROW + 16 * k + 2 * qd;
                    mma(sc[nn], qa[k], cvt2(*reinterpret_cast<const uint16_t*>(kr)),
                        cvt2(*reinterpret_cast<const uint16_t*>(kr + 8)));
                }
            }
            const int kb = 2 * qd;
            float tm0 = -CUDART_INF_F, tm1 = -CUDART_INF_F;
#pragma unroll
            for (int nn = 0; nn < 8; ++nn) {
                const int kk = kb + 8 * nn, cidx = key0 + kk;
                const float s0 = ksc[kk] * scale, s1 = ksc[kk + 1] * scale;
                sc[nn][0] = cidx <= pos ? sc[nn][0] * s0 : -CUDART_INF_F;
                sc[nn][1] = cidx + 1 <= pos ? sc[nn][1] * s1 : -CUDART_INF_F;
                sc[nn][2] = cidx <= pos ? sc[nn][2] * s0 : -CUDART_INF_F;
                sc[nn][3] = cidx + 1 <= pos ? sc[nn][3] * s1 : -CUDART_INF_F;
                tm0 = fmaxf(tm0, fmaxf(sc[nn][0], sc[nn][1]));
                tm1 = fmaxf(tm1, fmaxf(sc[nn][2], sc[nn][3]));
            }
            for (int x = 1; x <= 2; x *= 2) {
                tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
                tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
            }
            const float n0 = fmaxf(m0, tm0), n1 = fmaxf(m1, tm1);
            const float a0 = m0 == -CUDART_INF_F ? 0.0f : texp(m0 - n0);
            const float a1 = m1 == -CUDART_INF_F ? 0.0f : texp(m1 - n1);
            float c0[8], c1[8];
#pragma unroll
            for (int nn = 0; nn < 8; ++nn) {
                const int kk = kb + 8 * nn, cidx = key0 + kk;
                sc[nn][0] = cidx <= pos ? texp(sc[nn][0] - n0) : 0.0f;
                sc[nn][1] = cidx + 1 <= pos ? texp(sc[nn][1] - n0) : 0.0f;
                sc[nn][2] = cidx <= pos ? texp(sc[nn][2] - n1) : 0.0f;
                sc[nn][3] = cidx + 1 <= pos ? texp(sc[nn][3] - n1) : 0.0f;
                c0[nn] = sc[nn][0] + sc[nn][1];
                c1[nn] = sc[nn][2] + sc[nn][3];
                // the value scale rides on the probability: p v = (p s_v) v8
                const float v0 = vsc[kk], v1 = vsc[kk + 1];
                sc[nn][0] *= v0;
                sc[nn][1] *= v1;
                sc[nn][2] *= v0;
                sc[nn][3] *= v1;
            }
            l0 = l0 * a0 + rowsum(c0);
            l1 = l1 * a1 + rowsum(c1);
            m0 = n0;
            m1 = n1;
#pragma unroll
            for (int i = 0; i < DV / 8; ++i) {
                o[i][0] *= a0;
                o[i][1] *= a0;
                o[i][2] *= a1;
                o[i][3] *= a1;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t pa[4] = {pack(sc[2 * j][0], sc[2 * j][1]), pack(sc[2 * j][2], sc[2 * j][3]),
                                        pack(sc[2 * j + 1][0], sc[2 * j + 1][1]),
                                        pack(sc[2 * j + 1][2], sc[2 * j + 1][3])};
                const uint8_t* v0r = vs8 + (16 * j + 2 * qd) * VROW;
#pragma unroll
                for (int i = 0; i < DV / 8; ++i) {
                    const int dcol = 8 * i + g;
                    const uint16_t lo = static_cast<uint16_t>(v0r[dcol]) | (static_cast<uint16_t>(v0r[VROW + dcol]) << 8);
                    const uint16_t hi = static_cast<uint16_t>(v0r[8 * VROW + dcol]) |
                                        (static_cast<uint16_t>(v0r[9 * VROW + dcol]) << 8);
                    mma(o[i], pa, cvt2(lo), cvt2(hi));
                }
            }
        }
        __syncthreads(); // the stage is read: the next iteration's copy may overwrite it
    }
    if (!live) return;
    const int64_t part0 = (static_cast<int64_t>(row) * NCH + c) * H + head0, part1 = part0 + 8;
    if (qd == 0) {
        PM[part0] = m0;
        PL[part0] = l0;
        PM[part1] = m1;
        PL[part1] = l1;
    }
#pragma unroll
    for (int i = 0; i < DV / 8; ++i) {
        const int d = 8 * i + 2 * qd;
        *reinterpret_cast<float2*>(PO + part0 * DV + d) = make_float2(o[i][0], o[i][1]);
        *reinterpret_cast<float2*>(PO + part1 * DV + d) = make_float2(o[i][2], o[i][3]);
    }
}

// attention._merge without sinks, a block per (row, head): eight warps fold every eighth chunk, summed in order.
extern "C" __global__ void __launch_bounds__(256) mimo_merge8(const float* __restrict__ PO,
                                                             const float* __restrict__ PM,
                                                             const float* __restrict__ PL, __half* __restrict__ OUT,
                                                             int NCH, int H) {
    __shared__ float red[8];
    __shared__ float4 acc_s[8][32];
    __shared__ float l_s[8];
    const int64_t r = blockIdx.x;
    const int h = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float mx = -CUDART_INF_F;
    for (int c = threadIdx.x; c < NCH; c += blockDim.x) mx = fmaxf(mx, PM[(r * NCH + c) * H + h]);
    mx = warp_max(mx);
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    mx = red[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) mx = fmaxf(mx, red[w]);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    float l = 0.0f;
    for (int c = warp; c < NCH; c += 8) {
        const int64_t at = (r * NCH + c) * H + h;
        const float cm = PM[at];
        if (cm == -CUDART_INF_F) continue;
        const float f = texp(cm - mx);
        l += PL[at] * f;
        const float4 v = reinterpret_cast<const float4*>(PO + at * DV)[lane];
        acc.x += v.x * f;
        acc.y += v.y * f;
        acc.z += v.z * f;
        acc.w += v.w * f;
    }
    acc_s[warp][lane] = acc;
    if (lane == 0) l_s[warp] = l;
    __syncthreads();
    if (warp == 0) {
        float4 s = acc_s[0][lane];
        float ls = l_s[0];
#pragma unroll
        for (int w = 1; w < 8; ++w) {
            const float4 v = acc_s[w][lane];
            s.x += v.x;
            s.y += v.y;
            s.z += v.z;
            s.w += v.w;
            ls += l_s[w];
        }
        const float inv = 1.0f / ls;
        __half2* out = reinterpret_cast<__half2*>(OUT + (r * H + h) * DV + 4 * lane);
        out[0] = __floats2half2_rn(s.x * inv, s.y * inv);
        out[1] = __floats2half2_rn(s.z * inv, s.w * inv);
    }
}

// One stream's keys [0, keys): KC / VC rows at base + key -> KD [keys, HK, 192], VD [keys, HK, 128] fp16.
extern "C" __global__ void mimo_kv8_dequant(const uint8_t* __restrict__ KC, const float* __restrict__ KS,
                                            const uint8_t* __restrict__ VC, const float* __restrict__ VS,
                                            __half* __restrict__ KD, __half* __restrict__ VD, int64_t base, int keys,
                                            int HK) {
    const int64_t nk = static_cast<int64_t>(keys) * HK * (D / 16), nv = static_cast<int64_t>(keys) * HK * (DV / 16);
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < nk + nv;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const bool is_k = i < nk;
        const int64_t j = is_k ? i : i - nk;
        const int per = is_k ? D / 16 : DV / 16;
        const int64_t e = j / per, cc = j % per; // e = key * HK + head
        const int64_t src = base * HK + e;
        uint4 lo, hi;
        if (is_k) {
            dequant16(*reinterpret_cast<const uint4*>(KC + src * D + cc * 16), KS[src], lo, hi);
            uint4* dst = reinterpret_cast<uint4*>(KD + e * D + cc * 16);
            dst[0] = lo;
            dst[1] = hi;
        } else {
            dequant16(*reinterpret_cast<const uint4*>(VC + src * DV + cc * 16), VS[src], lo, hi);
            uint4* dst = reinterpret_cast<uint4*>(VD + e * DV + cc * 16);
            dst[0] = lo;
            dst[1] = hi;
        }
    }
}

// Tests: an fp16 cache of `entries` (token, head) rows into the 8-bit one; grid (entries), one warp.
extern "C" __global__ void __launch_bounds__(32) mimo_kv8_quant(const __half* __restrict__ K16,
                                                               const __half* __restrict__ V16,
                                                               uint8_t* __restrict__ KC, float* __restrict__ KS,
                                                               uint8_t* __restrict__ VC, float* __restrict__ VS) {
    const int64_t e = blockIdx.x;
    const int lane = threadIdx.x;
    float k[6], v[4], ak = 0.0f, av = 0.0f;
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        k[i] = __half2float(K16[e * D + lane + 32 * i]);
        ak = fmaxf(ak, fabsf(k[i]));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = __half2float(V16[e * DV + lane + 32 * i]);
        av = fmaxf(av, fabsf(v[i]));
    }
    ak = warp_max(ak);
    av = warp_max(av);
    const float ik = ak > 0.0f ? FP8_MAX / ak : 0.0f, iv = av > 0.0f ? FP8_MAX / av : 0.0f;
#pragma unroll
    for (int i = 0; i < 6; ++i) KC[e * D + lane + 32 * i] = to_fp8(k[i] * ik);
#pragma unroll
    for (int i = 0; i < 4; ++i) VC[e * DV + lane + 32 * i] = to_fp8(v[i] * iv);
    if (lane == 0) {
        KS[e] = ak / FP8_MAX;
        VS[e] = av / FP8_MAX;
    }
}

// Tests: hashed values in [-amp, amp] (fp16).
extern "C" __global__ void mimo_fill_half(__half* __restrict__ p, int64_t n, uint32_t seed, float amp) {
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        uint32_t x = static_cast<uint32_t>(i) * 2654435761u ^ seed ^ static_cast<uint32_t>(i >> 32) * 40503u;
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        p[i] = __float2half_rn((static_cast<float>(x) / 4294967296.0f * 2.0f - 1.0f) * amp);
    }
}
