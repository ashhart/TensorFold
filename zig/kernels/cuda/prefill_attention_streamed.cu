#ifndef TF_PREFILL_ATTENTION_STREAMED_INCLUDED
#define TF_PREFILL_ATTENTION_STREAMED_INCLUDED
// NS1 scheduling keeps the next FP8 item out of registers until the previous item has been consumed.
#ifndef TF_PREFILL_ATTENTION_BASE_INCLUDED
#include "prefill_attention.cu"
#endif

namespace tf_streamed_attention {
using namespace tf_prefill_attention;
template<int WARPS, int HPC, int NS, bool HPAGED>
__device__ __forceinline__ void rows(const __nv_bfloat16* q, const uint8_t* kc, const uint8_t* vc,
    const float* ks, const float* vs, __nv_bfloat16* out, int p0, int W, int H, int HK, int G,
    float scale, int CAP, const int* PT, long long PS, int PTOK) {
    constexpr int D = 256;
    constexpr bool causal = true, F8 = true, HM = true;
    constexpr int skip = 0;
    const int HO = H;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#include "prefill_attention_streamed_rows.cuh"
}
}
#define TF_STREAMED_ARGS \
    const __nv_bfloat16* q, const uint8_t* k, const uint8_t* v, const float* ks, const float* vs, \
    __nv_bfloat16* out, int p0, int rows, int heads, int kvheads, int group, float scale, int capacity
#define TF_STREAMED_VALUES q,k,v,ks,vs,out,p0,rows,heads,kvheads,group,scale,capacity
#define TF_STREAMED_DEFINE(TAG,W,H,S) \
extern "C" __global__ __launch_bounds__(32*W,1) void tf_streamed_##TAG##_whole(TF_STREAMED_ARGS) { \
    tf_streamed_attention::rows<W,H,S,false>(TF_STREAMED_VALUES,nullptr,0,1); \
} \
extern "C" __global__ __launch_bounds__(32*W,1) void tf_streamed_##TAG##_paged(TF_STREAMED_ARGS, \
    const int* pages, long long page_bytes, int page_tokens) { \
    tf_streamed_attention::rows<W,H,S,true>(TF_STREAMED_VALUES,pages,page_bytes,page_tokens); \
}
TF_STREAMED_DEFINE(w4h2s1,4,2,1)
TF_STREAMED_DEFINE(w2h2s1,2,2,1)
TF_STREAMED_DEFINE(w4h1s1,4,1,1)
TF_STREAMED_DEFINE(w8h2s1,8,2,1)

#endif
