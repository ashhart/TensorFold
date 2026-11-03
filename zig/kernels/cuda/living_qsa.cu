#define TF_LW_CONTROLS 1
#include <cuda_runtime.h>
#include <math.h>
#define LW_DEV
#define LW_INLINE __device__ __forceinline__
#define LW_BITS_FLOAT(x) __uint_as_float(x)
#define LW_FLOAT_BITS(x) __float_as_uint(x)
#define LW_TANH(x) tanhf(x)
#define LW_EXP(x) expf(x)
#define LW_ABS(x) fabsf(x)
#define LW_SQRT(x) sqrtf(x)
#define LW_LOG(x) logf(x)
#define LW_SIN(x) sinf(x)
#define LW_COS(x) cosf(x)
#define LW_INFINITY INFINITY
#define LW_ISNAN(x) isnan(x)
#define LW_FMA(a,b,c) __fmaf_rn(a,b,c)
#include "../common/living_rows.h"
#include "../common/living_qsa.h"
#define LW_KERNEL(name,op,control) extern "C" __global__ void __launch_bounds__(128) name( \
    const unsigned char* a,const unsigned char* b,const unsigned char* c,const unsigned char* d, \
    const unsigned char* e,const unsigned char* f,const unsigned char* g,const unsigned char* h, \
    unsigned char* x,unsigned char* y,unsigned char* z,LwParams p) { \
    const unsigned id=blockIdx.x*blockDim.x+threadIdx.x; p.u[14]=control; \
    if(control==3 && id+1==p.u[15])return; \
    lw_qsa(op,id,p,a,b,c,d,e,f,g,h,x,y,z); }
LW_KERNEL(tf_lw_qsa_scores,0,0)
LW_KERNEL(tf_lw_qsa_softmax_back,1,0)
LW_KERNEL(tf_lw_qsa_query_back,2,0)
LW_KERNEL(tf_lw_qsa_kv_back,3,0)

#if TF_LW_CONTROLS
LW_KERNEL(tf_lw_qsa_scores_bad_nan,0,1)
LW_KERNEL(tf_lw_qsa_scores_bad_repeat,0,2)
LW_KERNEL(tf_lw_qsa_scores_bad_tail,0,3)
LW_KERNEL(tf_lw_qsa_scores_bad_sums,0,4)
LW_KERNEL(tf_lw_qsa_softmax_back_bad_nan,1,1)
LW_KERNEL(tf_lw_qsa_softmax_back_bad_repeat,1,2)
LW_KERNEL(tf_lw_qsa_softmax_back_bad_tail,1,3)
LW_KERNEL(tf_lw_qsa_softmax_back_bad_sums,1,4)
LW_KERNEL(tf_lw_qsa_query_back_bad_nan,2,1)
LW_KERNEL(tf_lw_qsa_query_back_bad_repeat,2,2)
LW_KERNEL(tf_lw_qsa_query_back_bad_tail,2,3)
LW_KERNEL(tf_lw_qsa_query_back_bad_sums,2,4)
LW_KERNEL(tf_lw_qsa_kv_back_bad_nan,3,1)
LW_KERNEL(tf_lw_qsa_kv_back_bad_repeat,3,2)
LW_KERNEL(tf_lw_qsa_kv_back_bad_tail,3,3)
LW_KERNEL(tf_lw_qsa_kv_back_bad_sums,3,4)
#endif
