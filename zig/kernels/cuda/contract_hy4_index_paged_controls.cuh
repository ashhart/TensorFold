// Planted contract controls for hy4_index.cu: each kernel instantiated with each applicable Fault.
#pragma once

#include "hy4_flash/index_paged_pool.cuh"
#include "hy4_flash/index_paged_score.cuh"
#include "hy4_flash/index_paged_select.cuh"

#define TF_HY4_IDX_CTL_ROPE(tag, fault)                                                                               \
    extern "C" __global__ void __launch_bounds__(256) tf_ctl_hy4_idx_rope_table_##tag(                                \
        float2* CS, int first, int npos, const float* INV, const int* POS3, int ld3, int sec_h, int sec_w,            \
        int round_bf16) {                                                                                             \
        tf_hy4_idx::rope_table_body<fault>(CS, first, npos, INV, POS3, ld3, sec_h, sec_w, round_bf16);                \
    }

#define TF_HY4_IDX_CTL_POOL_1CTA(tag, fault)                                                                          \
    extern "C" __global__ void __launch_bounds__(128, 2) tf_ctl_hy4_idx_pool_1cta_##tag(                              \
        const int* JS, const int* JG, int g0, int njobs, const tf_hy4_idx::bf16* TKC, int ctx, const float* WV,       \
        const float* WG, const float* APE, const float* KNW, const float2* RT, int rt_stream_stride,                  \
        int rt_group_stride, float eps, tf_hy4_idx::bf16* GKC) {                                                      \
        tf_hy4_idx::pool_1cta_body<fault>(JS, JG, g0, njobs, TKC, ctx, WV, WG, APE, KNW, RT, rt_stream_stride,        \
                                          rt_group_stride, eps, GKC);                                                 \
    }

#define TF_HY4_IDX_CTL_POOL_SMALL(tag, fault)                                                                         \
    extern "C" __global__ void __launch_bounds__(128) tf_ctl_hy4_idx_pool_small_##tag(                                \
        const int* JS, const int* JG, int g0, int njobs, const tf_hy4_idx::bf16* TKC, int ctx, const float* WV,       \
        const float* WG, const float* APE, const float* KNW, const float2* RT, int rt_stream_stride,                  \
        int rt_group_stride, float eps, tf_hy4_idx::bf16* GKC) {                                                      \
        tf_hy4_idx::pool_small_body<fault>(JS, JG, g0, njobs, TKC, ctx, WV, WG, APE, KNW, RT, rt_stream_stride,       \
                                           rt_group_stride, eps, GKC);                                                \
    }

#define TF_HY4_IDX_CTL_POOL_PERSIST(tag, fault)                                                                       \
    extern "C" __global__ void __launch_bounds__(256, 2) tf_ctl_hy4_idx_pool_persist_##tag(                           \
        const int* JS, const int* JG, int g0, int njobs, const tf_hy4_idx::bf16* TKC, int ctx, const float* WV,       \
        const float* WG, const float* APE, const float* KNW, const float2* RT, int rt_stream_stride,                  \
        int rt_group_stride, float eps, tf_hy4_idx::bf16* GKC) {                                                      \
        tf_hy4_idx::pool_persist_body<fault>(JS, JG, g0, njobs, TKC, ctx, WV, WG, APE, KNW, RT, rt_stream_stride,     \
                                             rt_group_stride, eps, GKC);                                              \
    }

#define TF_HY4_IDX_CTL_QMROPE(tag, fault)                                                                             \
    extern "C" __global__ void __launch_bounds__(256) tf_ctl_hy4_idx_q_mrope_##tag(                                   \
        const int* SEL, int n_sel, const tf_hy4_idx::bf16* IQ, const int* ROW_POS, const int* POS3, int R,            \
        const float2* CS, tf_hy4_idx::bf16* QI) {                                                                     \
        tf_hy4_idx::q_mrope_body<fault>(SEL, n_sel, IQ, ROW_POS, POS3, R, CS, QI);                                    \
    }

#define TF_HY4_IDX_CTL_SCORE_DEC(tag, fault)                                                                          \
    extern "C" __global__ void __launch_bounds__(128, 3) tf_ctl_hy4_idx_score_decode_##tag(                           \
        const int* SEL, int n_sel, const tf_hy4_idx::bf16* QI, const float* HW, const int* ROW_STREAM,                \
        const int* ROW_POS, const tf_hy4_idx::bf16* GKC, int gpool, float* SC, int gcap) {                            \
        __shared__ __align__(16) tf_hy4_idx::bf16 tiles[2 * 64 * tf_hy4_idx::kDim];                                   \
        __shared__ int meta[3 * 4];                                                                                   \
        tf_hy4_idx::score_rows<4, 4, 2, 4, fault, false>(tiles, meta, SEL, n_sel, QI, HW, ROW_STREAM, ROW_POS, GKC,   \
                                                         gpool, SC, gcap);                                            \
    }

#define TF_HY4_IDX_CTL_SCORE_DEC_ROPE(tag, fault)                                                                     \
    extern "C" __global__ void __launch_bounds__(128, 3) tf_ctl_hy4_idx_score_decode_rope_##tag(                      \
        const int* SEL, int n_sel, const tf_hy4_idx::bf16* IQ, const float* HW, const int* ROW_STREAM,                \
        const int* ROW_POS, const tf_hy4_idx::bf16* GKC, int gpool, float* SC, int gcap, const float2* CS,            \
        const int* POS3, int R) {                                                                                     \
        __shared__ __align__(16) tf_hy4_idx::bf16 tiles[2 * 64 * tf_hy4_idx::kDim];                                   \
        __shared__ int meta[3 * 4];                                                                                   \
        tf_hy4_idx::score_rows<4, 4, 2, 4, fault, true>(tiles, meta, SEL, n_sel, IQ, HW, ROW_STREAM, ROW_POS, GKC,    \
                                                        gpool, SC, gcap, tf_hy4_idx::RopeIn{CS, POS3, R});            \
    }

#define TF_HY4_IDX_CTL_SCORE_PRE(tag, fault)                                                                          \
    extern "C" __global__ void __launch_bounds__(256, 3) tf_ctl_hy4_idx_score_prefill_##tag(                          \
        const int* SEL, int n_sel, const tf_hy4_idx::bf16* QI, const float* HW, const int* ROW_STREAM,                \
        const int* ROW_POS, const tf_hy4_idx::bf16* GKC, int gpool, float* SC, int gcap) {                            \
        __shared__ __align__(16) tf_hy4_idx::bf16 tiles[2 * 64 * tf_hy4_idx::kDim];                                   \
        __shared__ int meta[3 * 8];                                                                                   \
        tf_hy4_idx::score_rows<8, 4, 2, 8, fault, false>(tiles, meta, SEL, n_sel, QI, HW, ROW_STREAM, ROW_POS, GKC,   \
                                                         gpool, SC, gcap);                                            \
    }

#define TF_HY4_IDX_CTL_SEL_1CTA(tag, fault)                                                                           \
    extern "C" __global__ void __launch_bounds__(tf_hy4_idx::kTSel) tf_ctl_hy4_idx_select_1cta_##tag(                 \
        const int* SEL, const int* ROW_POS, const float* SC, int gcap, int* groups_out, int* tokens_out) {            \
        tf_hy4_idx::select_1cta_body<fault>(SEL, ROW_POS, SC, gcap, groups_out, tokens_out);                          \
    }

#define TF_HY4_IDX_CTL_SEL_SCAN(tag, fault)                                                                           \
    extern "C" __global__ void __launch_bounds__(tf_hy4_idx::kTScan) tf_ctl_hy4_idx_select_scan_##tag(                \
        const int* SEL, const int* ROW_POS, const float* SC, int gcap, uint32_t* bits, int words) {                   \
        tf_hy4_idx::scan_body<fault>(SEL, ROW_POS, SC, gcap, bits, words);                                            \
    }

#define TF_HY4_IDX_CTL_SEL_MERGE(tag, fault)                                                                          \
    extern "C" __global__ void __launch_bounds__(tf_hy4_idx::kTSel) tf_ctl_hy4_idx_select_merge_##tag(                \
        const int* SEL, const int* ROW_POS, const float* SC, int gcap, const uint32_t* bits, int words,               \
        int* groups_out, int* tokens_out, int* PATH) {                                                                \
        __shared__ __align__(16) tf_hy4_idx::Merge<tf_hy4_idx::kTSel> sm;                                             \
        __shared__ int out_smem[tf_hy4_idx::kListW];                                                                  \
        tf_hy4_idx::merge_body<fault>(sm, out_smem, SEL, ROW_POS, SC, gcap, bits, words, groups_out, tokens_out,      \
                                      PATH);                                                                          \
    }

#define TF_HY4_IDX_CTL_3(K)              \
    K(bad_nan, tf_hy4_idx::kBadNan)      \
    K(bad_repeat, tf_hy4_idx::kBadRepeat)\
    K(bad_tail, tf_hy4_idx::kBadTail)

#define TF_HY4_IDX_CTL_4(K)              \
    TF_HY4_IDX_CTL_3(K)                  \
    K(bad_sums, tf_hy4_idx::kBadSums)

#define TF_HY4_IDX_CTL_SEL(K)            \
    TF_HY4_IDX_CTL_3(K)                  \
    K(bad_ties, tf_hy4_idx::kBadTies)

TF_HY4_IDX_CTL_3(TF_HY4_IDX_CTL_ROPE)
TF_HY4_IDX_CTL_4(TF_HY4_IDX_CTL_POOL_1CTA)
TF_HY4_IDX_CTL_4(TF_HY4_IDX_CTL_POOL_SMALL)
TF_HY4_IDX_CTL_4(TF_HY4_IDX_CTL_POOL_PERSIST)
TF_HY4_IDX_CTL_3(TF_HY4_IDX_CTL_QMROPE)
TF_HY4_IDX_CTL_4(TF_HY4_IDX_CTL_SCORE_DEC)
TF_HY4_IDX_CTL_4(TF_HY4_IDX_CTL_SCORE_DEC_ROPE)
TF_HY4_IDX_CTL_4(TF_HY4_IDX_CTL_SCORE_PRE)
TF_HY4_IDX_CTL_SEL(TF_HY4_IDX_CTL_SEL_1CTA)
TF_HY4_IDX_CTL_3(TF_HY4_IDX_CTL_SEL_SCAN)
TF_HY4_IDX_CTL_SEL(TF_HY4_IDX_CTL_SEL_MERGE)
