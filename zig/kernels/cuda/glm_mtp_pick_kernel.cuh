// One gathered MTP top-1 per rank to the next draft's device ID; every row's original words stay intact.
#pragma once
#include "glm_pdl.cuh"
#include "glm_mtp_pick.cuh"

extern "C" __global__ void __launch_bounds__(32) tf_glm_mtp_pick(const unsigned* words, unsigned world,
                                                              unsigned* out) {
  glm_pdl::enter();
  if (threadIdx.x == 0) *out = glm_mtp_pick_id(words, world);
}
