// Ripple's attention prep on CUDA: the Metal family's rc_prep step for step, every rounding point kept.

#include "ricursive_prep.cuh"

// q and k heads normed (w, not 1 + w), NeoX-rotated if rope != 0; k and v at ring slot (pos + row) % ring, or in pages.
extern "C" __global__ void __launch_bounds__(tf_rcp::NT)
    tf_rc_prep(const tf_rcp::Rows XQ, const tf_rcp::Rows XK, const tf_rcp::Rows XV, const float* __restrict__ inv,
               const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ kw, __nv_bfloat16* __restrict__ Q,
               __nv_bfloat16* __restrict__ KC, __nv_bfloat16* __restrict__ VC, int rows, int heads, int kv_heads, int pos,
               int ring, int rope, float eps, const int* __restrict__ tab, long long page, int tok) {
  tf_rcp::prep(XQ, XK, XV, inv, qw, kw, Q, KC, VC, rows, heads, kv_heads, pos, ring, rope, eps, tab, page, tok, tf_rcp::ALL);
}

// tf_rc_prep with row r at pos[r] of stream slot[r] (device int32, so graphs replay new values); tf_rc_prep's bits.
extern "C" __global__ void __launch_bounds__(tf_rcp::NT)
    tf_rc_prep_rows(const tf_rcp::Rows XQ, const tf_rcp::Rows XK, const tf_rcp::Rows XV, const float* __restrict__ inv,
                    const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ kw,
                    __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KC, __nv_bfloat16* __restrict__ VC, int rows,
                    int heads, int kv_heads, const int* __restrict__ pos, const int* __restrict__ slot, int ring, int rope,
                    float eps, const int* __restrict__ tab, long long page, int tok, int stride) {
  tf_rcp::prep_rows(XQ, XK, XV, inv, qw, kw, Q, KC, VC, rows, heads, kv_heads, pos, slot, ring, rope, eps, tab, page, tok,
                    stride, tf_rcp::ALL);
}
