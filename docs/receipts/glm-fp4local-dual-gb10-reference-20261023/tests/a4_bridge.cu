#include <cuda_runtime.h>
#include "glm_a4_prompt.cuh"
extern "C" int run_a4_tile(int tile, int parts, const void* x, int ldx, int slots, const void* w, const void* global,
 const void* input, int K, int N, const void* items, const void* counts, const void* members,
 void* out, float limit, int skip, int max_items, void* stream) {
    if (tile != 64 && tile != 128) return (int)cudaErrorInvalidValue;
    const dim3 grid(max_items * ((N + tile - 1) / tile));
    if (tile == 128) {
      if (parts == 2) tf_glm_a4_prompt_up_n128<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)x, ldx, slots, (const uint32_t*)w, (const float*)global, (const float*)input,
        K, N, (const int*)items, (const int*)counts, (const int*)members, (__nv_bfloat16*)out, limit, skip);
      else tf_glm_a4_prompt_down_n128<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)x, ldx, slots, (const uint32_t*)w, (const float*)global, (const float*)input,
        K, N, (const int*)items, (const int*)counts, (const int*)members, (__nv_bfloat16*)out, limit, skip);
      return (int)cudaGetLastError();
    }
    if (parts == 2) tf_glm_a4_prompt_up<<<grid, 128, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)x, ldx, slots, (const uint32_t*)w, (const float*)global, (const float*)input,
      K, N, (const int*)items, (const int*)counts, (const int*)members, (__nv_bfloat16*)out, limit, skip);
    else tf_glm_a4_prompt_down<<<grid, 128, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)x, ldx, slots, (const uint32_t*)w, (const float*)global, (const float*)input,
      K, N, (const int*)items, (const int*)counts, (const int*)members, (__nv_bfloat16*)out, limit, skip);
    return (int)cudaGetLastError();
}
extern "C" int run_a4(int parts, const void* x, int ldx, int slots, const void* w, const void* global,
 const void* input, int K, int N, const void* items, const void* counts, const void* members,
 void* out, float limit, int skip, int max_items, void* stream) {
    return run_a4_tile(64, parts, x, ldx, slots, w, global, input, K, N, items, counts, members,
                      out, limit, skip, max_items, stream);
}
