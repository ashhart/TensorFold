// Ripple's glue on CUDA: glue.metal step for step (bf16 residual, fp32 sums), and its planted controls.

#include "ricursive.cuh"

namespace tf_rc {

// Embed: grid (ceil(width / 256), rows), 256 threads; ids are u32.
#define TF_RC_EMBED(NAME, BF_SCALE)                                                                                      \
  extern "C" __global__ void __launch_bounds__(NT) NAME(const unsigned* __restrict__ ids, const bf16* __restrict__ E,    \
                                                        bf16* __restrict__ R, bf16* __restrict__ V0, int rows, int width, \
                                                        float scale) {                                                   \
    embed<BF_SCALE>(ids, E, R, V0, rows, width, scale);                                                                  \
  }

// Add-norm: grid rows, 256 threads; Y null skips the add, V0 null the lerp; width % 8 == 0, <= 8,192, 16-byte rows.
#define TF_RC_ADD_NORM(NAME, K)                                                                                          \
  extern "C" __global__ void __launch_bounds__(NT)                                                                       \
      NAME(bf16* __restrict__ R, const bf16* __restrict__ Y, const bf16* __restrict__ w_post,                            \
           const bf16* __restrict__ w_next, const bf16* __restrict__ V0, bf16* __restrict__ X, bf16* __restrict__ VX,    \
           int rows, int width, float eps, float blend) {                                                                \
    add_norm<K>(R, Y, w_post, w_next, V0, X, VX, rows, width, eps, blend);                                               \
  }

// Gate: grid (ceil(width / 256), rows), 256 threads, A in place.
#define TF_RC_GATE(NAME, K)                                                                                              \
  extern "C" __global__ void __launch_bounds__(NT) NAME(bf16* __restrict__ A, const bf16* __restrict__ G, int rows,     \
                                                        int width) {                                                     \
    gate<K>(A, G, rows, width);                                                                                          \
  }

// SiLU-clamp: grid (ceil(width / 256), rows), 256 threads; GU rows are 2 x width (gate | up).
#define TF_RC_SILU_CLAMP(NAME, K)                                                                                        \
  extern "C" __global__ void __launch_bounds__(NT) NAME(const bf16* __restrict__ GU, bf16* __restrict__ A, int rows,    \
                                                        int width, float limit) {                                        \
    silu_clamp<K>(GU, A, rows, width, limit);                                                                            \
  }

TF_RC_EMBED(tf_rc_embed, false)
TF_RC_ADD_NORM(tf_rc_add_norm, NORM_ALL)
TF_RC_GATE(tf_rc_gate, SIG)
TF_RC_SILU_CLAMP(tf_rc_silu_clamp, SILU | LOW)

// Controls, one a rounding point (each must fail the contract's shapes gate): the served kernel without it.
TF_RC_EMBED(tf_rc_ctl_embed_bf16_scale, true)
TF_RC_ADD_NORM(tf_rc_ctl_add_norm_y_row, NORM_ALL & ~Y_ROW)
TF_RC_ADD_NORM(tf_rc_ctl_add_norm_y_out, NORM_ALL & ~Y_OUT)
TF_RC_ADD_NORM(tf_rc_ctl_add_norm_resid, NORM_ALL & ~RESID)
TF_RC_ADD_NORM(tf_rc_ctl_add_norm_x_row, NORM_ALL & ~X_ROW)
TF_RC_ADD_NORM(tf_rc_ctl_add_norm_x_lerp, NORM_ALL & ~X_LERP)
TF_RC_GATE(tf_rc_ctl_gate_sigmoid, 0)
TF_RC_SILU_CLAMP(tf_rc_ctl_silu_round, LOW)
TF_RC_SILU_CLAMP(tf_rc_ctl_silu_no_lower, SILU)

#undef TF_RC_EMBED
#undef TF_RC_ADD_NORM
#undef TF_RC_GATE
#undef TF_RC_SILU_CLAMP

// Plug-in point: the prep (per-head norm, RoPE, KV-ring write) and attention kernels go below, named tf_rc_*.

}  // namespace tf_rc
