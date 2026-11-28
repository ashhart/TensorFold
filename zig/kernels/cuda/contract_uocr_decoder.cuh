// The contract's planted controls for uocr_decoder.cuh's steps: each kernel with one step changed (its PLANT > 0).
#pragma once
#include "uocr_dec_attend.cuh"
#include "uocr_dec_moe.cuh"

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_norm(bf16* R, const bf16* A, const bf16* w, bf16* Y, int add,
                                                                   float eps) {
  tf_uocr::ud_norm<1>(R, A, w, Y, add, eps);
}

extern "C" __global__ void __launch_bounds__(128) tf_ctl_uocr_qk(const bf16* P, const bf16* RP, const bf16* COS, const bf16* SIN,
                                                                 const bf16* qn, const bf16* kn, bf16* Q, bf16* K, bf16* V, bf16* RQ,
                                                                 bf16* RK, int pos0, float eps) {
  tf_uocr::ud_qk<1>(P, RP, COS, SIN, qn, kn, Q, K, V, RQ, RK, pos0, eps);
}

extern "C" __global__ void __launch_bounds__(32) tf_ctl_uocr_block_keys(const bf16* RK, const bf16* hc, float* KB, int c0) {
  tf_uocr::ud_block_keys<1>(RK, hc, KB, c0);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_route(const bf16* RQ, const float* KB, const unsigned* masks,
                                                                    const int* qmask, unsigned* SEL, float* BIAS, int pos0, int words) {
  tf_uocr::ud_route<1>(RQ, KB, masks, qmask, SEL, BIAS, pos0, words);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_attend(const bf16* Q, const bf16* K, const bf16* V, const unsigned* SEL,
                                                                     const float* BIAS, bf16* O, int pos0, int bf16_bias) {
  tf_uocr::ud_attend<1>(Q, K, V, SEL, BIAS, O, pos0, bf16_bias);
}

extern "C" __global__ void __launch_bounds__(1024) tf_ctl_uocr_moe_gate(const bf16* X, const bf16* G, unsigned* IDS, float* WTS) {
  tf_uocr::ud_moe_gate<1>(X, G, IDS, WTS);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_silu_mul(const bf16* G, const bf16* U, bf16* Y, int width, int gstride,
                                                                       int ustride) {
  tf_uocr::ud_silu_mul<1>(G, U, Y, width, gstride, ustride);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_expert_gu(const bf16* x, const unsigned long long* E, const unsigned* ids,
                                                                        bf16* GU, int width) {
  tf_uocr::ud_expert_gu<1>(x, E, ids, GU, width);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_expert_down(const bf16* A, const unsigned long long* E,
                                                                          const unsigned* ids, bf16* D, int width) {
  tf_uocr::ud_expert_down<1>(A, E, ids, D, width);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_combine(const bf16* D, const unsigned* slot, const float* WTS,
                                                                      const bf16* S, bf16* Y) {
  tf_uocr::ud_combine<1>(D, slot, WTS, S, Y);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_norm_mean(bf16* R, const bf16* A, const bf16* w, bf16* Y, int add,
                                                                   float eps) {
  tf_uocr::ud_norm<2>(R, A, w, Y, add, eps);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_norm_sign(bf16* R, const bf16* A, const bf16* w, bf16* Y, int add,
                                                                   float eps) {
  tf_uocr::ud_norm<3>(R, A, w, Y, add, eps);
}

extern "C" __global__ void __launch_bounds__(128) tf_ctl_uocr_qk_turn(const bf16* P, const bf16* RP, const bf16* COS, const bf16* SIN,
                                                                 const bf16* qn, const bf16* kn, bf16* Q, bf16* K, bf16* V, bf16* RQ,
                                                                 bf16* RK, int pos0, float eps) {
  tf_uocr::ud_qk<2>(P, RP, COS, SIN, qn, kn, Q, K, V, RQ, RK, pos0, eps);
}

extern "C" __global__ void __launch_bounds__(128) tf_ctl_uocr_qk_clamp(const bf16* P, const bf16* RP, const bf16* COS, const bf16* SIN,
                                                                 const bf16* qn, const bf16* kn, bf16* Q, bf16* K, bf16* V, bf16* RQ,
                                                                 bf16* RK, int pos0, float eps) {
  tf_uocr::ud_qk<3>(P, RP, COS, SIN, qn, kn, Q, K, V, RQ, RK, pos0, eps);
}

extern "C" __global__ void __launch_bounds__(32) tf_ctl_uocr_block_keys_sum(const bf16* RK, const bf16* hc, float* KB, int c0) {
  tf_uocr::ud_block_keys<2>(RK, hc, KB, c0);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_route_mean(const bf16* RQ, const float* KB, const unsigned* masks,
                                                                    const int* qmask, unsigned* SEL, float* BIAS, int pos0, int words) {
  tf_uocr::ud_route<2>(RQ, KB, masks, qmask, SEL, BIAS, pos0, words);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_attend_bias(const bf16* Q, const bf16* K, const bf16* V, const unsigned* SEL,
                                                                     const float* BIAS, bf16* O, int pos0, int bf16_bias) {
  tf_uocr::ud_attend<2>(Q, K, V, SEL, BIAS, O, pos0, bf16_bias);
}

extern "C" __global__ void __launch_bounds__(1024) tf_ctl_uocr_moe_gate_mass(const bf16* X, const bf16* G, unsigned* IDS, float* WTS) {
  tf_uocr::ud_moe_gate<2>(X, G, IDS, WTS);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_silu_mul_swap(const bf16* G, const bf16* U, bf16* Y, int width, int gstride,
                                                                       int ustride) {
  tf_uocr::ud_silu_mul<2>(G, U, Y, width, gstride, ustride);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_expert_gu_ids(const bf16* x, const unsigned long long* E, const unsigned* ids,
                                                                        bf16* GU, int width) {
  tf_uocr::ud_expert_gu<2>(x, E, ids, GU, width);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_expert_down_ids(const bf16* A, const unsigned long long* E,
                                                                          const unsigned* ids, bf16* D, int width) {
  tf_uocr::ud_expert_down<2>(A, E, ids, D, width);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_combine_shared(const bf16* D, const unsigned* slot, const float* WTS,
                                                                      const bf16* S, bf16* Y) {
  tf_uocr::ud_combine<2>(D, slot, WTS, S, Y);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_gather(const bf16* x, const unsigned* rows, bf16* y) {
  tf_uocr::ud_gather<1>(x, rows, y);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_matvec(const bf16* x, const bf16* W, bf16* y, int n, int k) {
  tf_uocr::ud_matvec<1>(x, W, y, n, k);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_matvec_half(const bf16* x, const bf16* W, bf16* y, int n, int k) {
  tf_uocr::ud_matvec<2>(x, W, y, n, k);
}
