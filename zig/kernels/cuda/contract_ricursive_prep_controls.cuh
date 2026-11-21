// Gate 3: ricursive_prep.cu's tf_rc_prep keeping only the rounding points in `keep`; the shapes gate must go red.
#include "ricursive_prep.cuh"

extern "C" __global__ void __launch_bounds__(tf_rcp::NT) tf_ctl_rc_prep(
    const tf_rcp::Rows XQ, const tf_rcp::Rows XK, const tf_rcp::Rows XV, const float* inv, const __nv_bfloat16* qw,
    const __nv_bfloat16* kw, __nv_bfloat16* Q, __nv_bfloat16* KC, __nv_bfloat16* VC, int rows, int heads, int kv_heads, int pos,
    int ring, int rope, float eps, const int* tab, long long page, int tok, int keep) {
    tf_rcp::prep(XQ, XK, XV, inv, qw, kw, Q, KC, VC, rows, heads, kv_heads, pos, ring, rope, eps, tab, page, tok, keep);
}

// Gate 3: tf_rc_prep_rows with each row's slot dropped (all rows in stream 0's cache), `keep` as above; shapes go red.
extern "C" __global__ void __launch_bounds__(tf_rcp::NT) tf_ctl_rc_prep_rows(
    const tf_rcp::Rows XQ, const tf_rcp::Rows XK, const tf_rcp::Rows XV, const float* inv, const __nv_bfloat16* qw,
    const __nv_bfloat16* kw, __nv_bfloat16* Q, __nv_bfloat16* KC, __nv_bfloat16* VC, int rows, int heads, int kv_heads,
    const int* pos, const int* slot, int ring, int rope, float eps, const int* tab, long long page, int tok, int stride,
    int keep) {
    (void)slot; // the planted fault: no row's own stream
    tf_rcp::prep_rows(XQ, XK, XV, inv, qw, kw, Q, KC, VC, rows, heads, kv_heads, pos, nullptr, ring, rope, eps, tab, page,
                      tok, stride, keep);
}
