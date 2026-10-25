// BF16 hyper-connection activation uses the same scalar seams and output type on every row count.
#include <metal_stdlib>
using namespace metal;
#define TF_INLINE inline
#define TF_DATA device
#define TF_INDEX ulong
#define TF_FROM_BITS as_type<float>
#define TF_TO_BITS as_type<uint>
#define TF_LDEXP ldexp
#define TF_FMA fma
#define TF_HALF(bits) float(as_type<half>(ushort(bits)))
// TF_WEIGHT_ONLY_HEADER
inline float tf_hc_sigmoid(float x){const float e=metal::precise::exp(-metal::abs(x));return (x<0?e:1.0f)/(1.0f+e);}
inline float tf_hc_silu(float x){const float e=metal::precise::exp(-metal::abs(x));return (x<0?x*e:x)/(1.0f+e);}
#define TF_HC_EXP metal::precise::exp
#define TF_HC_SIGMOID tf_hc_sigmoid
#define TF_HC_SILU tf_hc_silu
// TF_HC_ACTIVATION_HEADER
struct TfHcActivation {int rows,cols,streams,injection,repeat;};
#define TF_HC_METAL(NAME,CONTROL) \
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(device const ushort* x [[buffer(0)]],device ushort* out [[buffer(1)]],constant TfHcActivation& p [[buffer(2)]],uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) { \
    const ulong i=ulong(group)*256+lane,count=ulong(p.rows)*p.cols; \
    if(i>=count || (CONTROL==3 && i==count-1))return; \
    ushort bits=tf_hc_activation(x[i],p.streams,p.injection); \
    if(CONTROL==1 && isnan(tf_wo_bf(bits)))bits=0; \
    if(CONTROL==2)bits=tf_wo_round(tf_wo_bf(bits)+float(p.repeat)); \
    out[i]=bits; \
}
TF_HC_METAL(qwen4_hc_activation,0)
#ifdef TF_KERNEL_CONTRACT
TF_HC_METAL(qwen4_hc_activation_bad_nan,1)
TF_HC_METAL(qwen4_hc_activation_bad_repeat,2)
TF_HC_METAL(qwen4_hc_activation_bad_tail,3)
#endif
