#define LW_KERNEL(name,op,control) [[max_total_threads_per_threadgroup(128)]] kernel void name( \
    device const uchar* a [[buffer(0)]],device const uchar* b [[buffer(1)]], \
    device const uchar* c [[buffer(2)]],device const uchar* d [[buffer(3)]], \
    device const uchar* e [[buffer(4)]],device const uchar* f [[buffer(5)]], \
    device const uchar* g [[buffer(6)]],device const uchar* h [[buffer(7)]], \
    device uchar* x [[buffer(8)]],device uchar* y [[buffer(9)]],device uchar* z [[buffer(10)]], \
    constant LwParams& p [[buffer(11)]],uint id [[thread_position_in_grid]]) { \
    LwParams call=p; call.u[14]=control; \
    if(control==3 && id+1==call.u[15])return; \
    lw_qsa(op,id,call,a,b,c,d,e,f,g,h,x,y,z); }
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
