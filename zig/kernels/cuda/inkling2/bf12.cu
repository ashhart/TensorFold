// Inkling 2 Mini lossless BF12 dense linears (Stage 3 spec, section 1).
// Ported 1:1 from tf-wt-dflash fn_bf12.cu (patch 0033): only the namespace
// (bf12 -> ink2bf12) and the extern "C" entry names (fn_bf12_* ->
// tf_ink2_bf12_*) change, so both objects link without collision.
// Serves Inkling-2-Mini BF16 dense projections with k % 64 == 0 (qkv, wo,
// rel/gate/sink, attn/mlp convs, dyn rows, controllers) in lossless BF12:
// per-row base + 12-bit (sign, 7-bit mantissa, 4-bit exp-delta) tiles of
// 768 B + (col<<16|bits) escape list. Bit layout is exactly what
// sm121a_pack.zig packBf12 writes and its tests decode.
// 14-variant kernel table (BM by row count on DRAM matrices, L2-cached
// weights for the small plans), split-K partials + CSR escape FMAs,
// M-invariant bits. Built for compute_121a/sm_121a only.
#include <cuda_bf16.h>
#include <cstdint>

namespace ink2bf12 {
constexpr int INK2BF12_TILE_BYTES = 768;
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}

__device__ __forceinline__ void cp16_ca(void* dst, const void* src) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}

__device__ __forceinline__ void cp16z(void* dst, const void* src, bool pred) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    int src_size = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(src_size));
}

__device__ __forceinline__ void cp16z_ca(void* dst, const void* src, bool pred) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    int src_size = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(src_size));
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <int CHUNKS>
__device__ __forceinline__ int swz(int r, int ch) {
    return ch ^ (r & (CHUNKS - 1));
}

__device__ __forceinline__ void ldmatrix4(uint32_t (&a)[4], const void* src) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4], const uint32_t (&a)[4],
                                                   uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t unpack_bf12_pair_lo(uint32_t sm_word, uint32_t e_pair, uint32_t base_x2) {
    uint32_t sm_pos = (sm_word & 0x007F007Fu) | ((sm_word & 0x00800080u) << 8);
    uint32_t exp_bits = base_x2 + (e_pair << 7);
    uint32_t is_esc = (e_pair + 0x00010001u) & 0x00100010u;
    uint32_t esc_mask = (is_esc << 11) - (is_esc >> 4);
    return (sm_pos | exp_bits) & ~esc_mask;
}

__device__ __forceinline__ uint32_t unpack_bf12_pair_hi(uint32_t sm_word, uint32_t e_pair, uint32_t base_x2) {
    uint32_t sm_pos = ((sm_word >> 8) & 0x007F007Fu) | (sm_word & 0x80008000u);
    uint32_t exp_bits = base_x2 + (e_pair << 7);
    uint32_t is_esc = (e_pair + 0x00010001u) & 0x00100010u;
    uint32_t esc_mask = (is_esc << 11) - (is_esc >> 4);
    return (sm_pos | exp_bits) & ~esc_mask;
}

__device__ __forceinline__ uint32_t get_u4_comp(const uint4& v, int kt) {
    return kt == 0 ? v.x : (kt == 1 ? v.y : (kt == 2 ? v.z : v.w));
}

// ============================================================================
// GPU 100% Bit-Exact Unpacker (BF12 -> row-major uint16_t[N, K])
// Used for prefill fallback (M > 128 / _hc_up_mix) and GPU round-trip tests.
// ============================================================================

extern "C" __global__ void tf_ink2_bf12_unpack_tiles_kernel(const uint8_t* __restrict__ w12,
                                                       const uint8_t* __restrict__ row_base,
                                                       uint16_t* __restrict__ out_bf16,
                                                       int N, int K) {
    const int total_n8 = (N + 7) / 8;
    const int KG = K / 64;
    const size_t total_tiles = static_cast<size_t>(total_n8) * static_cast<size_t>(KG);

    const size_t warp_id = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= total_tiles) return;

    const int tile_n8 = static_cast<int>(warp_id / KG);
    const int kg = static_cast<int>(warp_id % KG);
    const int b = tile_n8 >> 3;
    const int t = tile_n8 & 7;
    const int block_n8 = min(8, total_n8 - b * 8);

    const size_t tile_off = static_cast<size_t>(b) * static_cast<size_t>(KG) * 8ULL * INK2BF12_TILE_BYTES +
                            static_cast<size_t>(kg) * static_cast<size_t>(block_n8) * INK2BF12_TILE_BYTES +
                            static_cast<size_t>(t) * INK2BF12_TILE_BYTES;

    const int n = tile_n8 * 8 + (lane >> 2);
    const int tid4 = lane & 3;
    if (n >= N) return;

    const uint4 sm = *reinterpret_cast<const uint4*>(w12 + tile_off + lane * 16);
    const uint2 ec = *reinterpret_cast<const uint2*>(w12 + tile_off + 512 + lane * 8);
    const uint32_t base = static_cast<uint32_t>(row_base[n]);
    const uint32_t base_x2 = (base << 23) | (base << 7);

    uint32_t* dst_u32 = reinterpret_cast<uint32_t*>(out_bf16 + static_cast<size_t>(n) * static_cast<size_t>(K));

#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
        uint32_t sm_w = get_u4_comp(sm, kt);
        uint32_t ec_w = (kt < 2) ? ec.x : ec.y;
        int sh = (kt & 1) * 8;
        uint32_t e_b0 = (ec_w >> sh) & 0x000F000Fu;
        uint32_t e_b1 = (ec_w >> (sh + 4)) & 0x000F000Fu;
        uint32_t p0 = unpack_bf12_pair_lo(sm_w, e_b0, base_x2);
        uint32_t p1 = unpack_bf12_pair_hi(sm_w, e_b1, base_x2);

        int k_pair_base = (kg * 64 + kt * 16) >> 1;
        dst_u32[k_pair_base + tid4] = p0;
        dst_u32[k_pair_base + 4 + tid4] = p1;
    }
}

extern "C" __global__ void tf_ink2_bf12_scatter_outliers_kernel(const uint32_t* __restrict__ row_ptr,
                                                           const uint32_t* __restrict__ entries,
                                                           uint16_t* __restrict__ out_bf16,
                                                           int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    uint32_t p0 = row_ptr[n];
    uint32_t p1 = row_ptr[n + 1];
    if (p0 == p1) return;
    uint16_t* row = out_bf16 + static_cast<size_t>(n) * static_cast<size_t>(K);
    for (uint32_t idx = p0; idx < p1; ++idx) {
        uint32_t e = entries[idx];
        uint32_t col = e >> 16;
        uint16_t val = static_cast<uint16_t>(e & 0xFFFFu);
        row[col] = val;
    }
}

// Deterministic split-K reduction WITH fused parallel CSR-4B FP32 outlier accumulation.
// 1 thread per (m, n): ordered slice sum 0..SK-1, then ascending-col CSR FMAs,
// then single rounding to BF16 (or FP32 store). Strictly M-invariant and CUDA-graph safe.
template <bool F32>
__device__ __forceinline__ void tf_ink2_bf12_reduce_csr_body(const float* __restrict__ part,
                                          const __nv_bfloat16* __restrict__ X,
                                          int x_stride,
                                          const uint32_t* __restrict__ row_ptr,
                                          const uint32_t* __restrict__ entries,
                                          void* __restrict__ out,
                                          int M, int N, int SK, int round_f32) {
    const size_t total = static_cast<size_t>(M) * static_cast<size_t>(N);
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // the slices' loads issued 8 at a time (independent), summed in slice order 0..SK-1 as before: the same bits
    float sum = part[idx];
    int s = 1;
    for (; s + 8 <= SK; s += 8) {
        float v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = part[static_cast<size_t>(s + u) * total + idx];
#pragma unroll
        for (int u = 0; u < 8; ++u) sum += v[u];
    }
    for (; s < SK; ++s) {
        sum += part[static_cast<size_t>(s) * total + idx];
    }

    const int m = static_cast<int>(idx / static_cast<size_t>(N));
    const int n = static_cast<int>(idx % static_cast<size_t>(N));
    const uint32_t p0 = __ldg(row_ptr + n);
    const uint32_t p1 = __ldg(row_ptr + n + 1);
    if (p0 < p1) {
        const __nv_bfloat16* x_row = X + static_cast<size_t>(m) * static_cast<size_t>(x_stride);
        for (uint32_t k_idx = p0; k_idx < p1; ++k_idx) {
            uint32_t e = __ldg(entries + k_idx);
            uint32_t col = e >> 16;
            uint16_t wb = static_cast<uint16_t>(e & 0xFFFFu);
            float wv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&wb));
            float xv = __bfloat162float(__ldg(x_row + col));
            sum = __fmaf_rn(xv, wv, sum);
        }
    }

    if constexpr (F32) {
        // round_f32: the bf16 value widened back (tf_ink2_loop_round_bf16's bits, in the store)
        reinterpret_cast<float*>(out)[idx] = round_f32 ? __bfloat162float(__float2bfloat16_rn(sum)) : sum;
    } else {
        reinterpret_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(sum);
    }
}

// Helper: fused CSR-4B FP32 outlier accumulation in SK==1 MMA epilogue (guarded by active row!)
template <int MT, int BM>
__device__ __forceinline__ void apply_csr_outliers_colpair(
    float (&acc)[MT][4],
    const __nv_bfloat16* __restrict__ X,
    int x_stride,
    const uint32_t* __restrict__ row_ptr,
    const uint32_t* __restrict__ entries,
    int M, int N, int m0, int lane, int c0) {
    const int gr = lane >> 2;
    const int r_first = m0 + gr;
    if (r_first >= M || c0 >= N) return;

    uint32_t p0 = __ldg(row_ptr + c0);
    uint32_t p1 = __ldg(row_ptr + c0 + 1);
    if (p0 < p1) {
        for (uint32_t idx = p0; idx < p1; ++idx) {
            uint32_t e = __ldg(entries + idx);
            uint32_t col = e >> 16;
            uint16_t wb = static_cast<uint16_t>(e & 0xFFFFu);
            float wv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&wb));
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                int r0 = r_first + i * 16;
                int r1 = r0 + 8;
                if (r0 < M) {
                    float xv = __bfloat162float(__ldg(X + static_cast<size_t>(r0) * static_cast<size_t>(x_stride) + col));
                    acc[i][0] = __fmaf_rn(xv, wv, acc[i][0]);
                }
                if (r1 < M) {
                    float xv = __bfloat162float(__ldg(X + static_cast<size_t>(r1) * static_cast<size_t>(x_stride) + col));
                    acc[i][2] = __fmaf_rn(xv, wv, acc[i][2]);
                }
            }
        }
    }
    int c1 = c0 + 1;
    if (c1 < N) {
        uint32_t p2 = __ldg(row_ptr + c1 + 1);
        if (p1 < p2) {
            for (uint32_t idx = p1; idx < p2; ++idx) {
                uint32_t e = __ldg(entries + idx);
                uint32_t col = e >> 16;
                uint16_t wb = static_cast<uint16_t>(e & 0xFFFFu);
                float wv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&wb));
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    int r0 = r_first + i * 16;
                    int r1 = r0 + 8;
                    if (r0 < M) {
                        float xv = __bfloat162float(__ldg(X + static_cast<size_t>(r0) * static_cast<size_t>(x_stride) + col));
                        acc[i][1] = __fmaf_rn(xv, wv, acc[i][1]);
                    }
                    if (r1 < M) {
                        float xv = __bfloat162float(__ldg(X + static_cast<size_t>(r1) * static_cast<size_t>(x_stride) + col));
                        acc[i][3] = __fmaf_rn(xv, wv, acc[i][3]);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Unified sm_121 BF12 MMA Kernel
// - Grid: (cdiv(M, BM), cdiv(N, BN), SK) -- M varies fastest for L2 weight sharing.
// - BN in {64, 32}, WN = 4, WM in {1, 2} (128 or 256 threads/CTA), STAGES in {3, 4}, KG_STEP in {1, 2}.
//   WM = 2 (BM 128, wide prefill) splits the rows between two warp rows: each (m, n) still runs the same sequence.
// - Strict M-invariance: every (m, n) executes the exact same m16n8k16 sequence,
//   ordered split-K reduction, and ascending-col CSR FMAs regardless of M.
// - 64-bit byte indexing throughout (safe for lm_head > 635M elems / > 1.9e9).
// ============================================================================

template <int BM, int BN, int WM, int WN, int STAGES, int KG_STEP, bool F32, bool SPLITK, bool CACHE_W>
__device__ __forceinline__ void
tf_ink2_bf12_mm_body(const __nv_bfloat16* __restrict__ X,
                  int x_stride,
                  const uint8_t* __restrict__ W12,
                  const uint8_t* __restrict__ row_base,
                  const uint32_t* __restrict__ row_ptr,
                  const uint32_t* __restrict__ entries,
                  void* __restrict__ OUT,
                  float* __restrict__ PART,
                  int M, int N, int K, int SK, int round_f32) {
    static_assert(BN == 64 || BN == 32);
    static_assert(KG_STEP == 1 || KG_STEP == 2);
    static_assert(BM % (WM * 16) == 0 && BN % (WN * 8) == 0);

    constexpr int THREADS = WM * WN * 32;
    constexpr int MT = BM / (WM * 16);
    constexpr int NT = BN / (WN * 8);
    constexpr int TILE_N8 = BN / 8;
    constexpr int X_ROW_BYTES = 128;
    constexpr int X_CHUNKS = 8;
    constexpr int X_GROUP_BYTES = BM * X_ROW_BYTES;
    constexpr int W_GROUP_BYTES = TILE_N8 * INK2BF12_TILE_BYTES;
    constexpr int SUB_BYTES = ((X_GROUP_BYTES + W_GROUP_BYTES + 127) / 128) * 128;
    constexpr int STAGE_BYTES = KG_STEP * SUB_BYTES;

    extern __shared__ __align__(128) unsigned char smem[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wn = WM == 1 ? tid >> 5 : (tid >> 5) % WN;  // WM == 1: the 128-thread kernels' code exactly
    const int wm = WM == 1 ? 0 : (tid >> 5) / WN;

    const int m0 = blockIdx.x * BM;
    const int wm0 = m0 + wm * MT * 16;  // this warp's first row
    const int tile_bn = blockIdx.y;
    const int n0 = tile_bn * BN;
    const int slice = SPLITK ? blockIdx.z : 0;

    const int total_n8 = (N + 7) / 8;
    const int first_n8 = tile_bn * TILE_N8;
    const int b = first_n8 >> 3;
    const int sub_t0 = first_n8 & 7;
    const int block_n8 = min(8, total_n8 - b * 8);
    const int active_n8 = min(TILE_N8, block_n8 - sub_t0);

    const int KG = K / 64;
    const int per_kg = SPLITK ? (KG / SK) : KG;
    const int per_steps = per_kg / KG_STEP;
    const int kg0 = SPLITK ? (slice * per_kg) : 0;
    const size_t block_base = static_cast<size_t>(b) * static_cast<size_t>(KG) * 8ULL * INK2BF12_TILE_BYTES;
    const size_t kg_stride = static_cast<size_t>(block_n8) * INK2BF12_TILE_BYTES;

    uint32_t base_x2[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        int sub_t = wn * NT + j;
        if (sub_t < active_n8) {
            int n_row = n0 + sub_t * 8 + (lane >> 2);
            uint32_t rb = static_cast<uint32_t>(__ldg(row_base + n_row));
            base_x2[j] = (rb << 23) | (rb << 7);
        } else {
            base_x2[j] = (112u << 23) | (112u << 7);
        }
    }

    auto load_sub = [&](unsigned char* sub_p, int kg) {
        const int k0 = kg * 64;
        for (int c = tid; c < BM * X_CHUNKS; c += THREADS) {
            int r = c / X_CHUNKS;
            int ch = c % X_CHUNKS;
            int row = m0 + r;
            bool ok = row < M;
            const __nv_bfloat16* src = X + static_cast<size_t>(ok ? row : 0) * static_cast<size_t>(x_stride) + k0 + ch * 8;
            cp16z_ca(sub_p + r * X_ROW_BYTES + swz<X_CHUNKS>(r, ch) * 16, src, ok);
        }
        unsigned char* pw = sub_p + X_GROUP_BYTES;
        const uint8_t* wsrc = W12 + block_base + static_cast<size_t>(kg) * kg_stride +
                              static_cast<size_t>(sub_t0) * INK2BF12_TILE_BYTES;
        if (active_n8 == TILE_N8) {
            constexpr int W_CHUNKS = W_GROUP_BYTES / 16;
#pragma unroll
            for (int c = tid; c < W_CHUNKS; c += THREADS) {
                if constexpr (CACHE_W) cp16_ca(pw + c * 16, wsrc + c * 16);
                else                   cp16(pw + c * 16, wsrc + c * 16);
            }
        } else {
            int w_chunks = active_n8 * (INK2BF12_TILE_BYTES / 16);
            for (int c = tid; c < w_chunks; c += THREADS) {
                if constexpr (CACHE_W) cp16_ca(pw + c * 16, wsrc + c * 16);
                else                   cp16(pw + c * 16, wsrc + c * 16);
            }
        }
    };

    auto load_stage = [&](int s, int step) {
        unsigned char* st_p = smem + s * STAGE_BYTES;
#pragma unroll
        for (int g = 0; g < KG_STEP; ++g) {
            load_sub(st_p + g * SUB_BYTES, kg0 + step * KG_STEP + g);
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per_steps) load_stage(s, s);
        cp_commit();
    }

    for (int it = 0; it < per_steps; ++it) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        int next = it + STAGES - 1;
        if (next < per_steps) load_stage(next % STAGES, next);
        cp_commit();

        const unsigned char* st_p = smem + (it % STAGES) * STAGE_BYTES;

#pragma unroll
        for (int g = 0; g < KG_STEP; ++g) {
            const unsigned char* p = st_p + g * SUB_BYTES;
            const unsigned char* pw = p + X_GROUP_BYTES;

            uint4 sm_q[NT];
            uint2 ec_q[NT];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                int sub_t = wn * NT + j;
                if (sub_t < active_n8) {
                    const unsigned char* tptr = pw + sub_t * INK2BF12_TILE_BYTES;
                    sm_q[j] = *reinterpret_cast<const uint4*>(tptr + lane * 16);
                    ec_q[j] = *reinterpret_cast<const uint2*>(tptr + 512 + lane * 8);
                } else {
                    sm_q[j] = make_uint4(0u, 0u, 0u, 0u);
                    ec_q[j] = make_uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
                }
            }

#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                uint32_t a[MT][4];
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    int r = wm * MT * 16 + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    int ch = kt * 2 + (lane >> 4);
                    ldmatrix4(a[i], p + r * X_ROW_BYTES + swz<X_CHUNKS>(r, ch) * 16);
                }
                const int sh = (kt & 1) * 8;
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    uint32_t sm_w = get_u4_comp(sm_q[j], kt);
                    uint32_t ec_w = (kt < 2) ? ec_q[j].x : ec_q[j].y;
                    uint32_t e_b0 = (ec_w >> sh) & 0x000F000Fu;
                    uint32_t e_b1 = (ec_w >> (sh + 4)) & 0x000F000Fu;
                    uint32_t b0 = unpack_bf12_pair_lo(sm_w, e_b0, base_x2[j]);
                    uint32_t b1 = unpack_bf12_pair_hi(sm_w, e_b1, base_x2[j]);
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        mma_m16n8k16_bf16(acc[i][j], a[i], b0, b1);
                    }
                }
            }
        }
    }

    cp_wait<0>();

    const int gr = lane >> 2;
    const int gc = (lane & 3) * 2;

    if constexpr (SPLITK) {
        const size_t slice_off = static_cast<size_t>(slice) * static_cast<size_t>(M) * static_cast<size_t>(N);
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            int r0 = wm0 + i * 16 + gr;
            int r1 = r0 + 8;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                int c0 = n0 + (wn * NT + j) * 8 + gc;
                if (r0 < M) {
                    if (c0 < N) PART[slice_off + static_cast<size_t>(r0) * static_cast<size_t>(N) + c0] = acc[i][j][0];
                    if (c0 + 1 < N) PART[slice_off + static_cast<size_t>(r0) * static_cast<size_t>(N) + c0 + 1] = acc[i][j][1];
                }
                if (r1 < M) {
                    if (c0 < N) PART[slice_off + static_cast<size_t>(r1) * static_cast<size_t>(N) + c0] = acc[i][j][2];
                    if (c0 + 1 < N) PART[slice_off + static_cast<size_t>(r1) * static_cast<size_t>(N) + c0 + 1] = acc[i][j][3];
                }
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            int c0 = n0 + (wn * NT + j) * 8 + gc;
            float col_acc[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int e = 0; e < 4; ++e) col_acc[i][e] = acc[i][j][e];
            apply_csr_outliers_colpair<MT, BM>(col_acc, X, x_stride, row_ptr, entries, M, N, wm0, lane, c0);
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = col_acc[i][e];
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            int r0 = wm0 + i * 16 + gr;
            int r1 = r0 + 8;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                int c0 = n0 + (wn * NT + j) * 8 + gc;
                if constexpr (F32) {
                    float* dst = reinterpret_cast<float*>(OUT);
                    // round_f32: each sum through bf16 and back (tf_ink2_loop_round_bf16's bits, in the store)
                    float v[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        v[e] = round_f32 ? __bfloat162float(__float2bfloat16_rn(acc[i][j][e])) : acc[i][j][e];
                    if (r0 < M) {
                        if (c0 < N) dst[static_cast<size_t>(r0) * static_cast<size_t>(N) + c0] = v[0];
                        if (c0 + 1 < N) dst[static_cast<size_t>(r0) * static_cast<size_t>(N) + c0 + 1] = v[1];
                    }
                    if (r1 < M) {
                        if (c0 < N) dst[static_cast<size_t>(r1) * static_cast<size_t>(N) + c0] = v[2];
                        if (c0 + 1 < N) dst[static_cast<size_t>(r1) * static_cast<size_t>(N) + c0 + 1] = v[3];
                    }
                } else {
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(OUT);
                    if (r0 < M) {
                        if (c0 < N) dst[static_cast<size_t>(r0) * static_cast<size_t>(N) + c0] = __float2bfloat16_rn(acc[i][j][0]);
                        if (c0 + 1 < N) dst[static_cast<size_t>(r0) * static_cast<size_t>(N) + c0 + 1] = __float2bfloat16_rn(acc[i][j][1]);
                    }
                    if (r1 < M) {
                        if (c0 < N) dst[static_cast<size_t>(r1) * static_cast<size_t>(N) + c0] = __float2bfloat16_rn(acc[i][j][2]);
                        if (c0 + 1 < N) dst[static_cast<size_t>(r1) * static_cast<size_t>(N) + c0 + 1] = __float2bfloat16_rn(acc[i][j][3]);
                    }
                }
            }
        }
    }
}


} // namespace ink2bf12

// -- exported entry points (cuda_bf12.zig looks them up by these names) --

#define INK2BF12_REDUCE(F)                                                                                                   \
    extern "C" __global__ void tf_ink2_bf12_reduce_csr_f##F(const float* __restrict__ part, const __nv_bfloat16* __restrict__ X, \
                                                       int x_stride, const uint32_t* __restrict__ row_ptr,               \
                                                       const uint32_t* __restrict__ entries, void* __restrict__ out,     \
                                                       int M, int N, int SK, int round_f32) {                            \
        ink2bf12::tf_ink2_bf12_reduce_csr_body<F>(part, X, x_stride, row_ptr, entries, out, M, N, SK, round_f32);                 \
    }
INK2BF12_REDUCE(0)
INK2BF12_REDUCE(1)

// tf_ink2_bf12_mm_<BM>_<BN>_<STAGES>_<KG_STEP>_<F32>_<SPLITK>_<CACHE_W>
#define INK2BF12_MM(BM, BN, ST, KS, F, SK, CW)                                                                               \
    extern "C" __global__ void __launch_bounds__(128) tf_ink2_bf12_mm_##BM##_##BN##_##ST##_##KS##_##F##_##SK##_##CW(          \
        const __nv_bfloat16* __restrict__ X, int x_stride, const uint8_t* __restrict__ W12,                            \
        const uint8_t* __restrict__ row_base, const uint32_t* __restrict__ row_ptr,                                     \
        const uint32_t* __restrict__ entries, void* __restrict__ OUT, float* __restrict__ PART, int M, int N, int K,     \
        int SKn, int round_f32) {                                                                                        \
        ink2bf12::tf_ink2_bf12_mm_body<BM, BN, 1, 4, ST, KS, (F != 0), (SK != 0), (CW != 0)>(X, x_stride, W12, row_base,      \
                                                                                   row_ptr, entries, OUT, PART, M, N, K, SKn, round_f32); \
    }
// Two warp rows (WM = 2, 256 threads): the wide-prefill tile
#define INK2BF12_MM_W2(BM, BN, ST, KS, F, SK, CW)                                                                            \
    extern "C" __global__ void __launch_bounds__(256) tf_ink2_bf12_mm_##BM##_##BN##_##ST##_##KS##_##F##_##SK##_##CW(          \
        const __nv_bfloat16* __restrict__ X, int x_stride, const uint8_t* __restrict__ W12,                            \
        const uint8_t* __restrict__ row_base, const uint32_t* __restrict__ row_ptr,                                     \
        const uint32_t* __restrict__ entries, void* __restrict__ OUT, float* __restrict__ PART, int M, int N, int K,     \
        int SKn, int round_f32) {                                                                                        \
        ink2bf12::tf_ink2_bf12_mm_body<BM, BN, 2, 4, ST, KS, (F != 0), (SK != 0), (CW != 0)>(X, x_stride, W12, row_base,      \
                                                                                   row_ptr, entries, OUT, PART, M, N, K, SKn, round_f32); \
    }
#define INK2BF12_MM_F(BM, BN, ST, KS, SK, CW) INK2BF12_MM(BM, BN, ST, KS, 0, SK, CW) INK2BF12_MM(BM, BN, ST, KS, 1, SK, CW)
// L2-resident matrices (BM 16, cached weights)
INK2BF12_MM_F(16, 32, 4, 2, 0, 1)
INK2BF12_MM_F(16, 32, 3, 1, 0, 1)
INK2BF12_MM_F(16, 32, 2, 1, 0, 1)
INK2BF12_MM_F(16, 64, 3, 1, 1, 1)
INK2BF12_MM_F(16, 64, 3, 1, 0, 1)
// DRAM-resident matrices, BM by the row count
INK2BF12_MM_F(16, 64, 4, 1, 1, 0)
INK2BF12_MM_F(16, 64, 3, 1, 1, 0)
INK2BF12_MM_F(32, 64, 3, 1, 1, 0)
INK2BF12_MM_F(48, 64, 3, 1, 1, 0)
INK2BF12_MM_F(64, 64, 3, 1, 1, 0)
INK2BF12_MM_F(16, 64, 4, 1, 0, 0)
INK2BF12_MM_F(32, 64, 4, 1, 0, 0)
INK2BF12_MM_F(48, 64, 3, 1, 0, 0)
INK2BF12_MM_F(64, 64, 3, 1, 0, 0)
// Wide prefill (m >= 128): BM 128 over 2 x 4 warps, MT 4, NT 2; two stages (44 KB) keep two CTAs on an SM
INK2BF12_MM_W2(128, 64, 2, 1, 0, 0, 0)
INK2BF12_MM_W2(128, 64, 2, 1, 1, 0, 0)
