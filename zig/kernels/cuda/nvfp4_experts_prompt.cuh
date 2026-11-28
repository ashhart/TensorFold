// The prompt form's entries, 4 row tiles a warp in one row of WN warps: relu^2 up, SwiGLU gate-up, bf16 sums down.
#define TF_NVFP4_PROMPT(NAME, M, EPI, WN)                                                                              \
  extern "C" __global__ void __launch_bounds__(WN * 32) NAME(                                                        \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,                     \
      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts, \
      const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {                       \
    tf_nvfp4_experts::prompt_body<M, EPI, 4, 1, WN>(X, x_stride, slots, W, scale, KG, NB, items, counts, members,    \
                                                    out, N, limit, skip, nullptr, nullptr, 0);                       \
  }
TF_NVFP4_PROMPT(tf_nvfp4_prompt_relu2, 1, 1, 8)
TF_NVFP4_PROMPT(tf_nvfp4_prompt_swiglu, 2, 2, 4)
TF_NVFP4_PROMPT(tf_nvfp4_prompt_down, 1, 3, 8)

// Paired prompt chunks: global member IDs select either allocation, with each pair's arithmetic unchanged.
#define TF_NVFP4_PAIR_PROMPT(NAME, M, EPI, WN)                                                                       \
  extern "C" __global__ void __launch_bounds__(2 * WN * 32) NAME(                                                    \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,                     \
      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts, \
      const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip,                         \
      const __nv_bfloat16* __restrict__ X_b, void* __restrict__ out_b, int split) {                                  \
    tf_nvfp4_experts::prompt_body<M, EPI, 4, 2, WN, false, true>(X, x_stride, slots, W, scale, KG, NB, items, counts, \
                                                                 members, out, N, limit, skip, X_b, out_b, split);     \
  }
TF_NVFP4_PAIR_PROMPT(tf_nvfp4_pair_prompt_swiglu, 2, 2, 4)
TF_NVFP4_PAIR_PROMPT(tf_nvfp4_pair_prompt_down, 1, 3, 8)

// The decode form's factored sums on the prompt tiles, bit for bit: SwiGLU gate-up to bf16, down to fp32 slot rows.
#define TF_NVFP4_FACT(NAME, M, EPI, WN)                                                                                \
  extern "C" __global__ void __launch_bounds__(WN * 32) NAME(                                                        \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,                     \
      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts, \
      const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {                       \
    tf_nvfp4_experts::prompt_body<M, EPI, 4, 1, WN, true, false>(X, x_stride, slots, W, scale, KG, NB, items, counts,  \
                                                          members, out, N, limit, skip, nullptr, nullptr, 0);              \
  }
TF_NVFP4_FACT(tf_nvfp4_fact_swiglu, 2, 2, 4)
TF_NVFP4_FACT(tf_nvfp4_fact_down_f32, 1, 0, 8)

// Optional 128-row FACT items: two M-warps share each staged W tile; per-row FACT arithmetic is unchanged.
#define TF_NVFP4_FACT128(NAME, M, EPI, WN)                                                                             \
  extern "C" __global__ void __launch_bounds__(2 * WN * 32) NAME(                                                      \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,                     \
      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts, \
      const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {                       \
    tf_nvfp4_experts::prompt_body<M, EPI, 4, 2, WN, true, false>(X, x_stride, slots, W, scale, KG, NB, items, counts,  \
                                                          members, out, N, limit, skip, nullptr, nullptr, 0);              \
  }
TF_NVFP4_FACT128(tf_nvfp4_fact128_swiglu, 2, 2, 4)
TF_NVFP4_FACT128(tf_nvfp4_fact128_down_f32, 1, 0, 4)
