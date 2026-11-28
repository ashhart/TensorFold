// Compile the actual integer picker and kernel body as host C++ for the no-device exactness fixture.
#define __device__
#define __forceinline__ inline
#define __global__
#define __launch_bounds__(x)
static struct { unsigned x; } threadIdx;
#include "kernels/cuda/glm_mtp_pick_kernel.cuh"

extern "C" unsigned glm_mtp_pick_host(const unsigned* words, unsigned world) {
  unsigned out = 0;
  threadIdx.x = 0;
  tf_glm_mtp_pick(words, world, &out);
  return out;
}

extern "C" unsigned glm_mtp_pick_control_host(const unsigned* words, unsigned world, unsigned control) {
  return control == 1 ? glm_mtp_pick_id<1>(words, world) : glm_mtp_pick_id<2>(words, world);
}

extern "C" unsigned glm_mtp_pick_idle_host(const unsigned* words, unsigned world) {
  unsigned out = 0xfeedfaceu;
  threadIdx.x = 31;
  tf_glm_mtp_pick(words, world, &out);
  return out;
}
