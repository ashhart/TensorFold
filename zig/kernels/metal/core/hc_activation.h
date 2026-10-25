// HC activation seams shared by Metal and CUDA: BF16 division, then SiLU or twice a BF16 sigmoid.
#ifndef TF_HC_ACTIVATION_BODY
#define TF_HC_ACTIVATION_BODY
TF_INLINE unsigned short tf_hc_activation(unsigned short x, int streams, int injection) {
    const float divided = tf_wo_bf(tf_wo_round(tf_wo_bf(x) / float(streams)));
    const float value = injection ? 2.0f * tf_wo_bf(tf_wo_round(TF_HC_SIGMOID(divided))) : TF_HC_SILU(divided);
    return tf_wo_round(value);
}
#endif

#ifndef TF_HC_POINTWISE_BODY
#define TF_HC_POINTWISE_BODY
TF_INLINE unsigned short tf_hc_branch(unsigned short h,unsigned short branch,unsigned short injection,int round_product) {
    const float x=tf_wo_bf(branch),g=tf_wo_bf(injection),residual=tf_wo_bf(h);
    return tf_wo_round(round_product ? residual+tf_wo_bf(tf_wo_round(x*g)) : TF_FMA(x,g,residual));
}
TF_INLINE unsigned short tf_hc_output_gate(unsigned short input,unsigned short gate,int round_sigmoid) {
    const float x=tf_wo_bf(input),g=tf_wo_bf(gate);
    if(round_sigmoid)return tf_wo_round(x*tf_wo_bf(tf_wo_round(TF_HC_SIGMOID(g))));
    if(g<0) {
        const float e=TF_HC_EXP(0.5f*g);
        return tf_wo_round(((x*e)*e)/(1.0f+e*e));
    }
    return tf_wo_round(x*TF_HC_SIGMOID(g));
}
#endif
