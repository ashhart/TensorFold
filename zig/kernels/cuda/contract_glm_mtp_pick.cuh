// Real rank-scan controls: reverse the tie, or let a later rank displace a first-rank NaN.
#pragma once
#include "glm_mtp_pick.cuh"

extern "C" __global__ void __launch_bounds__(32) tf_ctl_glm_mtp_pick_tie(const unsigned* words, unsigned world,
                                                                      unsigned* out) {
  if (threadIdx.x == 0) *out = glm_mtp_pick_id<1>(words, world);
}
extern "C" __global__ void __launch_bounds__(32) tf_ctl_glm_mtp_pick_nan(const unsigned* words, unsigned world,
                                                                      unsigned* out) {
  if (threadIdx.x == 0) *out = glm_mtp_pick_id<2>(words, world);
}
