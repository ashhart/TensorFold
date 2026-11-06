#define LW_KERNEL(name,op,control) [[max_total_threads_per_threadgroup(128)]] kernel void name( \
    device const uchar* a [[buffer(0)]],device const uchar* b [[buffer(1)]], \
    device const uchar* c [[buffer(2)]],device const uchar* d [[buffer(3)]], \
    device const uchar* e [[buffer(4)]],device const uchar* f [[buffer(5)]], \
    device const uchar* g [[buffer(6)]],device const uchar* h [[buffer(7)]], \
    device uchar* x [[buffer(8)]],device uchar* y [[buffer(9)]],device uchar* z [[buffer(10)]], \
    constant LwParams& p [[buffer(11)]],uint id [[thread_position_in_grid]]) { \
    LwParams call=p; call.u[14]=control; \
    if(control==3 && id+1==call.u[15])return; \
    lw_gemma(op,id,call,a,b,c,d,e,f,g,h,x,y,z); }
LW_KERNEL(tf_lw_g4_copy,0,0)
LW_KERNEL(tf_lw_g4_zero,1,0)
LW_KERNEL(tf_lw_g4_move,2,0)
LW_KERNEL(tf_lw_g4_rms_back,3,0)
LW_KERNEL(tf_lw_g4_geglu_back,4,0)
LW_KERNEL(tf_lw_g4_softcap_back,5,0)
LW_KERNEL(tf_lw_g4_scale,6,0)
LW_KERNEL(tf_lw_g4_rope_back,7,0)

#if TF_LW_CONTROLS
LW_KERNEL(tf_lw_g4_copy_bad_nan,0,1)
LW_KERNEL(tf_lw_g4_copy_bad_repeat,0,2)
LW_KERNEL(tf_lw_g4_copy_bad_tail,0,3)
LW_KERNEL(tf_lw_g4_copy_bad_sums,0,4)
LW_KERNEL(tf_lw_g4_zero_bad_nan,1,1)
LW_KERNEL(tf_lw_g4_zero_bad_repeat,1,2)
LW_KERNEL(tf_lw_g4_zero_bad_tail,1,3)
LW_KERNEL(tf_lw_g4_zero_bad_sums,1,4)
LW_KERNEL(tf_lw_g4_move_bad_nan,2,1)
LW_KERNEL(tf_lw_g4_move_bad_repeat,2,2)
LW_KERNEL(tf_lw_g4_move_bad_tail,2,3)
LW_KERNEL(tf_lw_g4_move_bad_sums,2,4)
LW_KERNEL(tf_lw_g4_rms_back_bad_nan,3,1)
LW_KERNEL(tf_lw_g4_rms_back_bad_repeat,3,2)
LW_KERNEL(tf_lw_g4_rms_back_bad_tail,3,3)
LW_KERNEL(tf_lw_g4_rms_back_bad_sums,3,4)
LW_KERNEL(tf_lw_g4_geglu_back_bad_nan,4,1)
LW_KERNEL(tf_lw_g4_geglu_back_bad_repeat,4,2)
LW_KERNEL(tf_lw_g4_geglu_back_bad_tail,4,3)
LW_KERNEL(tf_lw_g4_geglu_back_bad_sums,4,4)
LW_KERNEL(tf_lw_g4_softcap_back_bad_nan,5,1)
LW_KERNEL(tf_lw_g4_softcap_back_bad_repeat,5,2)
LW_KERNEL(tf_lw_g4_softcap_back_bad_tail,5,3)
LW_KERNEL(tf_lw_g4_softcap_back_bad_sums,5,4)
LW_KERNEL(tf_lw_g4_scale_bad_nan,6,1)
LW_KERNEL(tf_lw_g4_scale_bad_repeat,6,2)
LW_KERNEL(tf_lw_g4_scale_bad_tail,6,3)
LW_KERNEL(tf_lw_g4_scale_bad_sums,6,4)
LW_KERNEL(tf_lw_g4_rope_back_bad_nan,7,1)
LW_KERNEL(tf_lw_g4_rope_back_bad_repeat,7,2)
LW_KERNEL(tf_lw_g4_rope_back_bad_tail,7,3)
LW_KERNEL(tf_lw_g4_rope_back_bad_sums,7,4)
#endif
