#ifndef TF_PREFILL_ATTENTION_TUNED_INCLUDED
#define TF_PREFILL_ATTENTION_TUNED_INCLUDED
// Geometry-only candidates reuse TensorFold's current attention body without changing a floating-point operation.
#ifndef TF_PREFILL_ATTENTION_BASE_INCLUDED
#include "prefill_attention.cu"
#endif

namespace tf_exact_attention {
using namespace tf_prefill_attention;
template<int WARPS, int HPC, int NS, bool HPAGED>
__device__ __forceinline__ void rows(const __nv_bfloat16* q, const uint8_t* kc, const uint8_t* vc,
    const float* ks, const float* vs, __nv_bfloat16* out, int p0, int W, int H, int HK, int G,
    float scale, int CAP, const int* PT, long long PS, int PTOK) {
    constexpr int D = 256;
    constexpr bool causal = true, F8 = true, HM = true, P256 = false, IDS = false;
    constexpr int PN = 0;
    constexpr int skip = 0;
    const int HO = H;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#include "prefill_attention_rows.cuh"
}
}
#define TF_EXACT_ARGS \
    const __nv_bfloat16* q, const uint8_t* k, const uint8_t* v, const float* ks, const float* vs, \
    __nv_bfloat16* out, int p0, int rows, int heads, int kvheads, int group, float scale, int capacity
#define TF_EXACT_VALUES q,k,v,ks,vs,out,p0,rows,heads,kvheads,group,scale,capacity
#define TF_EXACT_DEFINE(TAG,W,H,S) \
extern "C" __global__ __launch_bounds__(32*W,1) void tf_exact_##TAG##_whole(TF_EXACT_ARGS) { \
    tf_exact_attention::rows<W,H,S,false>(TF_EXACT_VALUES,nullptr,0,1); \
} \
extern "C" __global__ __launch_bounds__(32*W,1) void tf_exact_##TAG##_paged(TF_EXACT_ARGS, \
    const int* pages, long long page_bytes, int page_tokens) { \
    tf_exact_attention::rows<W,H,S,true>(TF_EXACT_VALUES,pages,page_bytes,page_tokens); \
}
TF_EXACT_DEFINE(w4h2s1,4,2,1)
TF_EXACT_DEFINE(w2h2s1,2,2,1)
TF_EXACT_DEFINE(w4h1s1,4,1,1)
TF_EXACT_DEFINE(w8h2s1,8,2,1)
TF_EXACT_DEFINE(w4h2s2,4,2,2)

#endif
