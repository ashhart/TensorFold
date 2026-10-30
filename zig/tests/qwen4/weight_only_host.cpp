// Execute the shared device arithmetic over owned CPU buffers with the kernels' exact shuffle tree.
#include <stdint.h>
#include <math.h>
#include <string.h>
static float from_bits(unsigned x) { float out; memcpy(&out,&x,4);return out; }
static unsigned to_bits(float x) { unsigned out;memcpy(&out,&x,4);return out; }
static float half_value(unsigned short bits) {
    const unsigned sign=unsigned(bits&0x8000)<<16, exp=(bits>>10)&31, mant=bits&1023;
    if(exp==31) return from_bits(sign|0x7f800000u|(mant?0x400000u:0));
    const float v=exp?ldexpf(float(1024+mant),int(exp)-25):ldexpf(float(mant),-24);
    return (bits&0x8000)?-v:v;
}
#define TF_INLINE inline
#define TF_DATA
#define TF_INDEX uint64_t
#define TF_FROM_BITS from_bits
#define TF_TO_BITS to_bits
#define TF_LDEXP ldexpf
#define TF_FMA fmaf
#define TF_HALF half_value
#include "../../kernels/metal/core/weight_only.h"
extern "C" int tf_weight_only_params_size() {return sizeof(TfWeightOnly);}
extern "C" void tf_weight_only_cpu(const unsigned short* x,const unsigned char* w,const unsigned char* s,
    const unsigned char* b,unsigned short* out,TfWeightOnly p,int control,int repetition) {
    for(int row=0;row<p.rows;++row) for(int col=0;col<p.n;++col) {
        float total=0;
        for(int first=0;first<p.k;first+=p.group) {
            float partial[32];
            for(int lane=0;lane<32;++lane) partial[lane]=tf_wo_partial(x,w,s,b,p,row,col,first,lane,control);
            for(int offset=16;offset;offset>>=1) {
                float next[32];for(int lane=0;lane<32;++lane) {
                    next[lane]=partial[lane]+partial[lane^offset];
                    if(control==4) next[lane]=tf_wo_bf(tf_wo_round(next[lane]));
                }
                memcpy(partial,next,sizeof(partial));
            }
            total+=partial[0];if(control==4) total=tf_wo_bf(tf_wo_round(total));
        }
        if(control==1&&isnan(total)) total=0;
        if(control==2) total+=float(repetition);
        out[uint64_t(row)*p.n+col]=tf_wo_round(total);
    }
}

static float hc_sigmoid(float x) {const float e=expf(-fabsf(x));return (x<0?e:1.0f)/(1.0f+e);}
static float hc_silu(float x) {const float e=expf(-fabsf(x));return (x<0?x*e:x)/(1.0f+e);}
#define TF_HC_EXP expf
#define TF_HC_SIGMOID hc_sigmoid
#define TF_HC_SILU hc_silu
#include "../../kernels/metal/core/hc_activation.h"
extern "C" void tf_hc_activation_cpu(const unsigned short* x,unsigned short* out,int count,int streams,int injection,int control,int repeat) {
    for(int i=0;i<count;++i) {
        if(control==3&&i==count-1)continue;
        unsigned short bits=tf_hc_activation(x[i],streams,injection);
        if(control==1&&isnan(tf_wo_bf(bits)))bits=0;
        if(control==2)bits=tf_wo_round(tf_wo_bf(bits)+float(repeat));
        out[i]=bits;
    }
}
extern "C" void tf_hc_pointwise_strided_cpu(const unsigned short* x,const unsigned short* branch,const unsigned short* gate,
    unsigned short* out,int rows,int hidden,int streams,int op,int round_mid,int gate_stride,int gate_dim,int control,int repeat) {
    const uint64_t count=uint64_t(rows)*hidden*(op==0?streams:1);
    for(uint64_t i=0;i<count;++i) {
        if(control==3&&i==count-1)continue;
        unsigned short bits;
        if(op==0){const uint64_t row=i/(hidden*streams),d=i%hidden,s=(i/hidden)%streams;bits=tf_hc_branch(x[i],branch[row*hidden+d],gate[row*streams+s],round_mid);}
        else {const uint64_t row=i/hidden,d=i%hidden,at=gate_dim>0?row*gate_stride+(d/gate_dim)*(2*gate_dim)+gate_dim+d%gate_dim:i;bits=tf_hc_output_gate(x[i],branch[at],round_mid);}
        if(control==1&&isnan(tf_wo_bf(bits)))bits=0;
        if(control==2)bits=tf_wo_round(tf_wo_bf(bits)+float(repeat));out[i]=bits;
    }
}

extern "C" void tf_hc_pointwise_cpu(const unsigned short* x,const unsigned short* branch,const unsigned short* gate,
    unsigned short* out,int rows,int hidden,int streams,int op,int round_mid,int control,int repeat) {
    tf_hc_pointwise_strided_cpu(x,branch,gate,out,rows,hidden,streams,op,round_mid,0,0,control,repeat);
}
