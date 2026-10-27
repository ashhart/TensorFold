// Shared hyper-connection activation only; the original Flash Next image remains unchanged.
#include "common/glue_math.cuh"
#include <cuda_fp16.h>
#define TF_INLINE __device__ __forceinline__
#define TF_DATA
#define TF_INDEX unsigned long long
#define TF_FROM_BITS __uint_as_float
#define TF_TO_BITS __float_as_uint
#define TF_LDEXP ldexpf
#define TF_FMA __fmaf_rn
#define TF_HALF(bits) __half2float(__ushort_as_half(bits))
#include "../metal/core/weight_only.h"
__device__ __forceinline__ float hc_sigmoid(float x) {
    const float e = tf_glue::tf_exp(-fabsf(x));
    return x < 0 ? __fdiv_rn(e, __fadd_rn(1.0f,e)) : __fdiv_rn(1.0f, __fadd_rn(1.0f,e));
}
__device__ __forceinline__ float hc_silu(float x) {
    const float e = tf_glue::tf_exp(-fabsf(x));
    return __fdiv_rn(x < 0 ? __fmul_rn(x,e) : x, __fadd_rn(1.0f,e));
}
#define TF_HC_EXP tf_glue::tf_exp
#define TF_HC_SIGMOID hc_sigmoid
#define TF_HC_SILU hc_silu
#include "../metal/core/hc_activation.h"
template<int CONTROL>
__device__ __forceinline__ void hc_run(const unsigned short* x,unsigned short* out,int rows,int cols,int streams,int injection,int repeat) {
    const unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long count=(unsigned long long)rows*cols;
    if(i>=count || (CONTROL==3 && i==count-1))return;
    unsigned short bits=tf_hc_activation(x[i],streams,injection);
    if(CONTROL==1 && isnan(tf_wo_bf(bits))) bits=0;
    if(CONTROL==2) bits=tf_wo_round(tf_wo_bf(bits)+float(repeat));
    out[i]=bits;
}
#define TF_HC_KERNEL(NAME,CONTROL) \
extern "C" __global__ void __launch_bounds__(256) NAME(const unsigned short* x,unsigned short* out,int rows,int cols,int streams,int injection,int repeat) {hc_run<CONTROL>(x,out,rows,cols,streams,injection,repeat);}
TF_HC_KERNEL(qwen4_hc_activation,0)
TF_HC_KERNEL(qwen4_hc_activation_bad_nan,1)
TF_HC_KERNEL(qwen4_hc_activation_bad_repeat,2)
TF_HC_KERNEL(qwen4_hc_activation_bad_tail,3)

template<int CONTROL>
__device__ __forceinline__ void hc_pointwise(const unsigned short* x,const unsigned short* branch,const unsigned short* gate,
    unsigned short* out,int rows,int hidden,int streams,int op,int round_mid,int gate_stride,int gate_dim,int repeat) {
    const unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long count=(unsigned long long)rows*hidden*(op==0?streams:1);
    if(i>=count || (CONTROL==3 && i==count-1))return;
    unsigned short bits;
    if(op==0) {
        const unsigned long long row=i/(hidden*streams),d=i%hidden,s=(i/hidden)%streams;
        bits=tf_hc_branch(x[i],branch[row*hidden+d],gate[row*streams+s],round_mid);
    } else {
        const unsigned long long row=i/hidden,d=i%hidden;
        const unsigned long long at=gate_dim>0?row*gate_stride+(d/gate_dim)*(2*gate_dim)+gate_dim+d%gate_dim:i;
        bits=tf_hc_output_gate(x[i],branch[at],round_mid);
    }
    if(CONTROL==1 && isnan(tf_wo_bf(bits)))bits=0;
    if(CONTROL==2)bits=tf_wo_round(tf_wo_bf(bits)+float(repeat));
    out[i]=bits;
}
#define TF_HC_POINTWISE(NAME,CONTROL) \
extern "C" __global__ void __launch_bounds__(256) NAME(const unsigned short* x,const unsigned short* branch,const unsigned short* gate,unsigned short* out,int rows,int hidden,int streams,int op,int round_mid,int gate_stride,int gate_dim,int repeat) {hc_pointwise<CONTROL>(x,branch,gate,out,rows,hidden,streams,op,round_mid,gate_stride,gate_dim,repeat);}
TF_HC_POINTWISE(qwen4_hc_pointwise,0)
TF_HC_POINTWISE(qwen4_hc_pointwise_bad_nan,1)
TF_HC_POINTWISE(qwen4_hc_pointwise_bad_repeat,2)
TF_HC_POINTWISE(qwen4_hc_pointwise_bad_tail,3)

// Dense activation preserves the BF16 SiLU node before the up product.
template<int CONTROL>
__device__ __forceinline__ void qwen4_dense_run(const unsigned short* gate,const unsigned short* up,unsigned short* out,int width,int repeat) {
    const int row=blockIdx.x,col=blockIdx.y*blockDim.x+threadIdx.x;
    if(col>=width || (CONTROL==3 && col==width-1))return;
    const unsigned long long at=(unsigned long long)row*width+col;
    unsigned short bits=tf_wo_round(tf_wo_bf(tf_wo_round(hc_silu(tf_wo_bf(gate[at]))))*tf_wo_bf(up[at]));
    if(CONTROL==1 && isnan(tf_wo_bf(bits)))bits=0;
    if(CONTROL==2)bits=tf_wo_round(tf_wo_bf(bits)+float(repeat));
    out[at]=bits;
}
extern "C" __global__ void __launch_bounds__(256) qwen4_dense_act(const unsigned short* gate,const unsigned short* up,unsigned short* out,int width,int repeat) {
    qwen4_dense_run<0>(gate,up,out,width,repeat);
}
extern "C" __global__ void __launch_bounds__(256) qwen4_dense_act_bad_nan(const unsigned short* gate,const unsigned short* up,unsigned short* out,int width,int repeat) {
    qwen4_dense_run<1>(gate,up,out,width,repeat);
}
extern "C" __global__ void __launch_bounds__(256) qwen4_dense_act_bad_repeat(const unsigned short* gate,const unsigned short* up,unsigned short* out,int width,int repeat) {
    qwen4_dense_run<2>(gate,up,out,width,repeat);
}
extern "C" __global__ void __launch_bounds__(256) qwen4_dense_act_bad_tail(const unsigned short* gate,const unsigned short* up,unsigned short* out,int width,int repeat) {
    qwen4_dense_run<3>(gate,up,out,width,repeat);
}
