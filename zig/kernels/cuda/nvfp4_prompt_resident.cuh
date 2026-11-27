// Opt-in R3 prompt entries: same tile arithmetic, two 32-input stages, three CTAs an SM.
extern "C" __global__ void __launch_bounds__(128, 3) tf_nvfp4_prompt3_swiglu(
    const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
    const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {
  tf_nvfp4_experts::prompt_body<2, 2, 4, 1, 4, false, false, 1, 2>(X, x_stride, slots, W, scale, KG, NB, items, counts, members,
                                                                  out, N, limit, skip, nullptr, nullptr, 0);
}
extern "C" __global__ void __launch_bounds__(256, 3) tf_nvfp4_prompt3_down(
    const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
    const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {
  tf_nvfp4_experts::prompt_body<1, 3, 4, 1, 8, false, false, 1, 2>(X, x_stride, slots, W, scale, KG, NB, items, counts, members,
                                                                  out, N, limit, skip, nullptr, nullptr, 0);
}
