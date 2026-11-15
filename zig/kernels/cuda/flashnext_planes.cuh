// Flash Next attention launches for a stream whose K/V rows sit in frames and whose indexer keys sit by logical page.
#pragma once

// The same two through a host tier's planes (PgP): the K/V rows by frame, the indexer keys by logical page.
extern "C" __global__ void __launch_bounds__(256)
    fn_attn_prep_planes(const __nv_bfloat16* __restrict__ P, const int* __restrict__ POS0, const float* __restrict__ QW,
                        const float* __restrict__ KW, const float* __restrict__ IW, const float* __restrict__ INV,
                        __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KC, __nv_bfloat16* __restrict__ VC,
                        __nv_bfloat16* __restrict__ IQ, __nv_bfloat16* __restrict__ IKC, float eps, int PW, int NQ,
                        int NKV, int NI, int IHD, int HALF, PgP pg) {
    fn_attn_prep_body<true>(P, POS0, QW, KW, IW, INV, Q, KC, VC, IQ, IKC, eps, PW, NQ, NKV, NI, IHD, HALF, nullptr, 0,
                            0, pg);
}

extern "C" __global__ void __launch_bounds__(256)
    fn_attn_prep_media_planes(const __nv_bfloat16* __restrict__ P, const int* __restrict__ POS0,
                              const float* __restrict__ QW, const float* __restrict__ KW, const float* __restrict__ IW,
                              const float* __restrict__ INV, __nv_bfloat16* __restrict__ Q,
                              __nv_bfloat16* __restrict__ KC, __nv_bfloat16* __restrict__ VC,
                              __nv_bfloat16* __restrict__ IQ, __nv_bfloat16* __restrict__ IKC, float eps, int PW,
                              int NQ, int NKV, int NI, int IHD, int HALF, const int* rope3, int n, int delta, PgP pg) {
    fn_attn_prep_body<true>(P, POS0, QW, KW, IW, INV, Q, KC, VC, IQ, IKC, eps, PW, NQ, NKV, NI, IHD, HALF, rope3, n,
                            delta, pg);
}

extern "C" __global__ void __launch_bounds__(128)
    fn_qsa_pool_planes(const __nv_bfloat16* __restrict__ IKC, __nv_bfloat16* __restrict__ POOLED,
                       const int* __restrict__ POS0, const float* __restrict__ W, const float* __restrict__ INV,
                       float eps, int R, int DI, int HALF, int RATIO, PgP pg) {
    fn_qsa_pool_body<true>(IKC, POOLED, POS0, W, INV, eps, R, DI, HALF, RATIO, nullptr, 0, 0, pg);
}

extern "C" __global__ void __launch_bounds__(128)
    fn_qsa_pool_media_planes(const __nv_bfloat16* __restrict__ IKC, __nv_bfloat16* __restrict__ POOLED,
                             const int* __restrict__ POS0, const float* __restrict__ W, const float* __restrict__ INV,
                             float eps, int R, int DI, int HALF, int RATIO, const int* rope3, int n, int delta,
                             PgP pg) {
    fn_qsa_pool_body<true>(IKC, POOLED, POS0, W, INV, eps, R, DI, HALF, RATIO, rope3, n, delta, pg);
}

// The same sums through a host tier's planes: the pooled rows by logical page.
extern "C" __global__ void __launch_bounds__(256)
    fn_qsa_scores_planes(const __nv_bfloat16* __restrict__ IQ, const __nv_bfloat16* __restrict__ POOLED,
                         const int* __restrict__ POS0, float* __restrict__ SC, int NB, int RATIO, int TOP, int rows,
                         PgP pg) {
    qsa_tile::scores_body<8, 4, true>(IQ, POOLED, POS0, SC, NB, RATIO, TOP, rows, pg);
}
