// Hy4-flash compressed indexer CUDA kernels: pooling, M-RoPE, mma.m16n8k16 scoring, and top-511 selection.

#include "index_paged_pool.cuh"
#include "index_paged_score.cuh"
#include "index_paged_select.cuh"

using tf_hy4_idx::bf16;
using tf_hy4_idx::kDim;
using tf_hy4_idx::kNone;

// RoPE / 3D M-RoPE cos/sin table for [first, first + npos) into CS[first..].
extern "C" __global__ void __launch_bounds__(256)
    tf_hy4_idx_rope_table(float2* CS, int first, int npos, const float* INV, const int* POS3, int ld3, int sec_h,
                          int sec_w, int round_bf16) {
    tf_hy4_idx::rope_table_body<kNone>(CS, first, npos, INV, POS3, ld3, sec_h, sec_w, round_bf16);
}

// 1-CTA non-cluster 4:1 group pooling (sm_80+, 16 jobs/CTA, 49,152 B dynamic SMEM).
extern "C" __global__ void __launch_bounds__(128, 2)
    tf_hy4_idx_pool_1cta(const int* JS, const int* JG, int g0, int njobs, const bf16* TKC, int ctx, const float* WV,
                         const float* WG, const float* APE, const float* KNW, const float2* RT, int rt_stream_stride,
                         int rt_group_stride, float eps, bf16* GKC) {
    tf_hy4_idx::pool_1cta_body<kNone>(JS, JG, g0, njobs, TKC, ctx, WV, WG, APE, KNW, RT, rt_stream_stride,
                                      rt_group_stride, eps, GKC);
}

// 4-CTA DSMEM cluster 4:1 group pooling for small batches (sm_120+, njobs <= 48).
extern "C" __global__ void __launch_bounds__(128)
    tf_hy4_idx_pool_small(const int* JS, const int* JG, int g0, int njobs, const bf16* TKC, int ctx, const float* WV,
                          const float* WG, const float* APE, const float* KNW, const float2* RT, int rt_stream_stride,
                          int rt_group_stride, float eps, bf16* GKC) {
    tf_hy4_idx::pool_small_body<kNone>(JS, JG, g0, njobs, TKC, ctx, WV, WG, APE, KNW, RT, rt_stream_stride,
                                       rt_group_stride, eps, GKC);
}

// Persistent 4-CTA DSMEM cluster 4:1 group pooling for prefill (sm_120+, njobs > 48, 49,408 B dynamic SMEM).
extern "C" __global__ void __launch_bounds__(256, 2)
    tf_hy4_idx_pool_persist(const int* JS, const int* JG, int g0, int njobs, const bf16* TKC, int ctx, const float* WV,
                            const float* WG, const float* APE, const float* KNW, const float2* RT,
                            int rt_stream_stride, int rt_group_stride, float eps, bf16* GKC) {
    tf_hy4_idx::pool_persist_body<kNone>(JS, JG, g0, njobs, TKC, ctx, WV, WG, APE, KNW, RT, rt_stream_stride,
                                         rt_group_stride, eps, GKC);
}

// Query M-RoPE on channels 64..127 into QI[n_sel, 16, 128] bf16.
extern "C" __global__ void __launch_bounds__(256)
    tf_hy4_idx_q_mrope(const int* SEL, int n_sel, const bf16* IQ, const int* ROW_POS, const int* POS3, int R,
                       const float2* CS, bf16* QI) {
    tf_hy4_idx::q_mrope_body<kNone>(SEL, n_sel, IQ, ROW_POS, POS3, R, CS, QI);
}

// 16-head ReLU-weighted pool scoring (decode/verify, pre-rotated QI, up to 4 rows/CTA).
extern "C" __global__ void __launch_bounds__(128, 3)
    tf_hy4_idx_score_decode(const int* SEL, int n_sel, const bf16* QI, const float* HW, const int* ROW_STREAM,
                            const int* ROW_POS, const bf16* GKC, int gpool, float* SC, int gcap) {
    __shared__ __align__(16) bf16 tiles[2 * 64 * kDim];
    __shared__ int meta[3 * 4];
    tf_hy4_idx::score_rows<4, 4, 2, 4, kNone, false>(tiles, meta, SEL, n_sel, QI, HW, ROW_STREAM, ROW_POS, GKC, gpool,
                                                     SC, gcap);
}

// 16-head ReLU-weighted pool scoring with fused query M-RoPE (decode, n_sel <= 4).
extern "C" __global__ void __launch_bounds__(128, 3)
    tf_hy4_idx_score_decode_rope(const int* SEL, int n_sel, const bf16* IQ, const float* HW, const int* ROW_STREAM,
                                 const int* ROW_POS, const bf16* GKC, int gpool, float* SC, int gcap, const float2* CS,
                                 const int* POS3, int R) {
    __shared__ __align__(16) bf16 tiles[2 * 64 * kDim];
    __shared__ int meta[3 * 4];
    tf_hy4_idx::score_rows<4, 4, 2, 4, kNone, true>(tiles, meta, SEL, n_sel, IQ, HW, ROW_STREAM, ROW_POS, GKC, gpool,
                                                    SC, gcap, tf_hy4_idx::RopeIn{CS, POS3, R});
}

// 16-head ReLU-weighted pool scoring (prefill, 8 rows/CTA).
extern "C" __global__ void __launch_bounds__(256, 3)
    tf_hy4_idx_score_prefill(const int* SEL, int n_sel, const bf16* QI, const float* HW, const int* ROW_STREAM,
                             const int* ROW_POS, const bf16* GKC, int gpool, float* SC, int gcap) {
    __shared__ __align__(16) bf16 tiles[2 * 64 * kDim];
    __shared__ int meta[3 * 8];
    tf_hy4_idx::score_rows<8, 4, 2, 8, kNone, false>(tiles, meta, SEL, n_sel, QI, HW, ROW_STREAM, ROW_POS, GKC, gpool,
                                                     SC, gcap);
}

// Single-CTA radix top-511 + own-group selection (own <= 4096) with optional 2,048-token expansion.
extern "C" __global__ void __launch_bounds__(tf_hy4_idx::kTSel)
    tf_hy4_idx_select_1cta(const int* SEL, const int* ROW_POS, const float* SC, int gcap, int* groups_out,
                           int* tokens_out) {
    tf_hy4_idx::select_1cta_body<kNone>(SEL, ROW_POS, SC, gcap, groups_out, tokens_out);
}

// Multi-CTA candidate floor scan for long contexts (own > 4096).
extern "C" __global__ void __launch_bounds__(tf_hy4_idx::kTScan)
    tf_hy4_idx_select_scan(const int* SEL, const int* ROW_POS, const float* SC, int gcap, uint32_t* bits, int words) {
    tf_hy4_idx::scan_body<kNone>(SEL, ROW_POS, SC, gcap, bits, words);
}

// Single-CTA merge of scanned candidates into top-511 + own-group and optional 2,048-token expansion.
extern "C" __global__ void __launch_bounds__(tf_hy4_idx::kTSel)
    tf_hy4_idx_select_merge(const int* SEL, const int* ROW_POS, const float* SC, int gcap, const uint32_t* bits,
                            int words, int* groups_out, int* tokens_out, int* PATH) {
    __shared__ __align__(16) tf_hy4_idx::Merge<tf_hy4_idx::kTSel> sm;
    __shared__ int out_smem[tf_hy4_idx::kListW];
    tf_hy4_idx::merge_body<kNone>(sm, out_smem, SEL, ROW_POS, SC, gcap, bits, words, groups_out, tokens_out, PATH);
}

extern "C" __global__ void __launch_bounds__(128, 2)
    tf_hy4_idx_pool_1cta_paged(const int* JS, const int* JG, int g0, int njobs, const bf16* TKC, int ctx,
                               const float* WV, const float* WG, const float* APE, const float* KNW, const float2* RT,
                               int rt_stream_stride, int rt_group_stride, float eps, bf16* GKC, const uint32_t* PT,
                               int pt_stride, int g_base) {
    tf_hy4_idx::pool_1cta_body<kNone>(JS, JG, g0, njobs, TKC, ctx, WV, WG, APE, KNW, RT, rt_stream_stride,
                                      rt_group_stride, eps, GKC, PT, pt_stride, g_base);
}

extern "C" __global__ void __launch_bounds__(128)
    tf_hy4_idx_pool_small_paged(const int* JS, const int* JG, int g0, int njobs, const bf16* TKC, int ctx,
                                const float* WV, const float* WG, const float* APE, const float* KNW, const float2* RT,
                                int rt_stream_stride, int rt_group_stride, float eps, bf16* GKC, const uint32_t* PT,
                                int pt_stride, int g_base) {
    tf_hy4_idx::pool_small_body<kNone>(JS, JG, g0, njobs, TKC, ctx, WV, WG, APE, KNW, RT, rt_stream_stride,
                                       rt_group_stride, eps, GKC, PT, pt_stride, g_base);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    tf_hy4_idx_pool_persist_paged(const int* JS, const int* JG, int g0, int njobs, const bf16* TKC, int ctx,
                                  const float* WV, const float* WG, const float* APE, const float* KNW,
                                  const float2* RT, int rt_stream_stride, int rt_group_stride, float eps, bf16* GKC,
                                  const uint32_t* PT, int pt_stride, int g_base) {
    tf_hy4_idx::pool_persist_body<kNone>(JS, JG, g0, njobs, TKC, ctx, WV, WG, APE, KNW, RT, rt_stream_stride,
                                         rt_group_stride, eps, GKC, PT, pt_stride, g_base);
}

extern "C" __global__ void __launch_bounds__(128, 3)
    tf_hy4_idx_score_decode_paged(const int* SEL, int n_sel, const bf16* QI, const float* HW, const int* ROW_STREAM,
                                  const int* ROW_POS, const bf16* GKC, int gpool, float* SC, int gcap,
                                  const uint32_t* PT, int pt_stride, int g_base) {
    __shared__ __align__(16) bf16 tiles[2 * 64 * kDim];
    __shared__ int meta[3 * 4];
    tf_hy4_idx::score_rows<4, 4, 2, 4, kNone, false>(tiles, meta, SEL, n_sel, QI, HW, ROW_STREAM, ROW_POS, GKC, gpool,
                                                     SC, gcap, tf_hy4_idx::RopeIn{nullptr, nullptr, 0}, PT, pt_stride,
                                                     g_base);
}

extern "C" __global__ void __launch_bounds__(128, 3)
    tf_hy4_idx_score_decode_rope_paged(const int* SEL, int n_sel, const bf16* IQ, const float* HW,
                                       const int* ROW_STREAM, const int* ROW_POS, const bf16* GKC, int gpool, float* SC,
                                       int gcap, const float2* CS, const int* POS3, int R, const uint32_t* PT,
                                       int pt_stride, int g_base) {
    __shared__ __align__(16) bf16 tiles[2 * 64 * kDim];
    __shared__ int meta[3 * 4];
    tf_hy4_idx::score_rows<4, 4, 2, 4, kNone, true>(tiles, meta, SEL, n_sel, IQ, HW, ROW_STREAM, ROW_POS, GKC, gpool,
                                                    SC, gcap, tf_hy4_idx::RopeIn{CS, POS3, R}, PT, pt_stride, g_base);
}

extern "C" __global__ void __launch_bounds__(256, 3)
    tf_hy4_idx_score_prefill_paged(const int* SEL, int n_sel, const bf16* QI, const float* HW, const int* ROW_STREAM,
                                   const int* ROW_POS, const bf16* GKC, int gpool, float* SC, int gcap,
                                   const uint32_t* PT, int pt_stride, int g_base) {
    __shared__ __align__(16) bf16 tiles[2 * 64 * kDim];
    __shared__ int meta[3 * 8];
    tf_hy4_idx::score_rows<8, 4, 2, 8, kNone, false>(tiles, meta, SEL, n_sel, QI, HW, ROW_STREAM, ROW_POS, GKC, gpool,
                                                     SC, gcap, tf_hy4_idx::RopeIn{nullptr, nullptr, 0}, PT, pt_stride,
                                                     g_base);
}
