// Ripple's DFlash2 drafter glue on CUDA: dflash2.metal's norm, prep, conv and silu step for step, and planted controls.

#include "ricursive_dflash2.cuh"

namespace tf_rcd {

// Add-norm: grid rows, 256 threads; Y null is norm16's form (R only read); width % 8 == 0, at most 8,192, 16-byte rows.
#define TF_RCD_ADD_NORM(NAME, K)                                                                                         \
  extern "C" __global__ void __launch_bounds__(NT) NAME(bf16* __restrict__ R, const bf16* __restrict__ Y,               \
                                                        const bf16* __restrict__ w, bf16* __restrict__ X, int rows,      \
                                                        int width, float eps) {                                          \
    add_norm<K>(R, Y, w, X, rows, width, eps);                                                                           \
  }

// Prep: grid (ceil((heads + kv_heads) / 8), rows), 256 threads; segs [nsegs] on the device; Q null when heads is 0.
#define TF_RCD_PREP(NAME, K)                                                                                             \
  extern "C" __global__ void __launch_bounds__(NT)                                                                       \
      NAME(const bf16* __restrict__ X, const bf16* __restrict__ qw, const bf16* __restrict__ kw,                        \
           const float* __restrict__ inv, const Segment* __restrict__ segs, int nsegs, bf16* __restrict__ Q, int rows,  \
           int heads, int kv_heads, int stride, int q_off, int k_off, int v_off, int ring, int rope, float eps) {       \
    prep<K>(X, qw, kw, inv, segs, nsegs, Q, rows, heads, kv_heads, stride, q_off, k_off, v_off, ring, rope, eps);        \
  }

// Conv: grid (ceil(width / 256), rows), 256 threads; rows grouped in blocks of `block` (each stream's whole blocks).
#define TF_RCD_CONV(NAME, K)                                                                                             \
  extern "C" __global__ void __launch_bounds__(NT) NAME(const bf16* __restrict__ X, const bf16* __restrict__ D,          \
                                                        const bf16* __restrict__ base, bf16* __restrict__ Y, int rows,   \
                                                        int width, int block, int groups, int side, int taps) {          \
    conv<K>(X, D, base, Y, rows, width, block, groups, side, taps);                                                      \
  }

// SiLU: grid (ceil(width / 256), rows), 256 threads; GU rows are 2 x width (gate | up).
#define TF_RCD_SILU(NAME, K)                                                                                             \
  extern "C" __global__ void __launch_bounds__(NT) NAME(const bf16* __restrict__ GU, bf16* __restrict__ A, int rows,    \
                                                        int width) {                                                     \
    silu<K>(GU, A, rows, width);                                                                                         \
  }

TF_RCD_ADD_NORM(tf_rcd_add_norm, NORM_ALL)
TF_RCD_PREP(tf_rcd_prep, PREP_ALL)
TF_RCD_CONV(tf_rcd_conv, CONV_ALL)
TF_RCD_SILU(tf_rcd_silu, GS)

// Controls (each must fail the contract's shapes gate): the served kernel without one rounding point, or with one edit.
TF_RCD_ADD_NORM(tf_rcd_ctl_add_norm_resid, NORM_ALL & ~RESID)
TF_RCD_ADD_NORM(tf_rcd_ctl_add_norm_x_row, NORM_ALL & ~X_ROW)
TF_RCD_PREP(tf_rcd_ctl_prep_norm_row, PREP_ALL & ~NORM_ROW)
TF_RCD_PREP(tf_rcd_ctl_prep_norm_out, PREP_ALL & ~NORM_OUT)
TF_RCD_PREP(tf_rcd_ctl_prep_cos, PREP_ALL & ~COS)
TF_RCD_PREP(tf_rcd_ctl_prep_sin, PREP_ALL & ~SIN)
TF_RCD_PREP(tf_rcd_ctl_prep_rot_c, PREP_ALL & ~ROT_C)
TF_RCD_PREP(tf_rcd_ctl_prep_rot_s, PREP_ALL & ~ROT_S)
TF_RCD_PREP(tf_rcd_ctl_prep_out, PREP_ALL & ~OUT)
TF_RCD_PREP(tf_rcd_ctl_prep_nope_ropes, PREP_ALL | NOPE_ROPES)
TF_RCD_PREP(tf_rcd_ctl_prep_mask, PREP_ALL | MASK)
TF_RCD_CONV(tf_rcd_ctl_conv_base, CONV_ALL & ~C_BASE)
TF_RCD_CONV(tf_rcd_ctl_conv_add, CONV_ALL & ~C_ADD)
TF_RCD_CONV(tf_rcd_ctl_conv_dyn, CONV_ALL & ~C_DYN)
TF_RCD_CONV(tf_rcd_ctl_conv_two_round, CONV_ALL | TWO_ROUND)
TF_RCD_CONV(tf_rcd_ctl_conv_no_edge, CONV_ALL | NO_EDGE)
TF_RCD_CONV(tf_rcd_ctl_conv_swap, CONV_ALL | SWAP)
TF_RCD_SILU(tf_rcd_ctl_silu_gs, 0)

// The embed's control, as tf_rc_embed's arguments at scale 1: row r reads table row r, not row ids[r].
extern "C" __global__ void __launch_bounds__(NT) tf_rcd_ctl_embed_own_row(const unsigned* __restrict__ ids,
                                                                          const bf16* __restrict__ E, bf16* __restrict__ R,
                                                                          bf16* __restrict__ V0, int rows, int width,
                                                                          float scale) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (x >= width || row >= rows) return;
  const size_t at = (size_t)row * width + x;
  const bf16 v = __float2bfloat16_rn(__fmul_rn(ld(E + at), scale));
  R[at] = v;
  V0[at] = v;
}

#undef TF_RCD_ADD_NORM
#undef TF_RCD_PREP
#undef TF_RCD_CONV
#undef TF_RCD_SILU

}  // namespace tf_rcd
