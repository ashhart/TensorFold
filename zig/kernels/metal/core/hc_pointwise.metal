// Branch writeback and attention output gates retain their explicit checkpoint rounding nodes.
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
inline float tf_hc_pointwise_sigmoid(float x){const float e=metal::precise::exp(-metal::abs(x));return (x<0?e:1.0f)/(1.0f+e);}
#define TF_HC_EXP metal::precise::exp
#define TF_HC_SIGMOID tf_hc_pointwise_sigmoid
#define TF_HC_SILU(x) ((x)*tf_hc_pointwise_sigmoid(x))
// TF_HC_ACTIVATION_HEADER
struct TfHcPointwise {int rows,hidden,streams,op,round_mid,gate_stride,gate_dim,repeat;};
#define TF_HC_POINTWISE(NAME,CONTROL) \
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(device const ushort* x [[buffer(0)]],device const ushort* branch [[buffer(1)]],device const ushort* gate [[buffer(2)]],device ushort* out [[buffer(3)]],constant TfHcPointwise& p [[buffer(4)]],uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) { \
    const ulong i=ulong(group)*256+lane,count=ulong(p.rows)*p.hidden*(p.op==0?p.streams:1); \
    if(i>=count || (CONTROL==3 && i==count-1))return; \
    ushort bits; \
    if(p.op==0) {const ulong row=i/(p.hidden*p.streams),d=i%p.hidden,s=(i/p.hidden)%p.streams;bits=tf_hc_branch(x[i],branch[row*p.hidden+d],gate[row*p.streams+s],p.round_mid);} \
    else {const ulong row=i/p.hidden,d=i%p.hidden,at=p.gate_dim>0?row*p.gate_stride+(d/p.gate_dim)*(2*p.gate_dim)+p.gate_dim+d%p.gate_dim:i;bits=tf_hc_output_gate(x[i],branch[at],p.round_mid);} \
    if(CONTROL==1 && isnan(tf_wo_bf(bits)))bits=0; \
    if(CONTROL==2)bits=tf_wo_round(tf_wo_bf(bits)+float(p.repeat)); \
    out[i]=bits; \
}
TF_HC_POINTWISE(qwen4_hc_pointwise,0)
#ifdef TF_KERNEL_CONTRACT
TF_HC_POINTWISE(qwen4_hc_pointwise_bad_nan,1)
TF_HC_POINTWISE(qwen4_hc_pointwise_bad_repeat,2)
TF_HC_POINTWISE(qwen4_hc_pointwise_bad_tail,3)
#endif
