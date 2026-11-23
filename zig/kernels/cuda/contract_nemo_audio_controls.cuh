// The contract's controls for nemo_audio.cu: each step built with CTL set, dropping what nemo_audio.cuh names.
#pragma once

#include "nemo_audio.cuh"

extern "C" __global__ void __launch_bounds__(256) tf_ctl_nemo_mel(const float* __restrict__ X, long lo, long n, const tf_nemo_audio::bf16* __restrict__ win, const tf_nemo_audio::bf16* __restrict__ fb, const float* __restrict__ cs, const float* __restrict__ sn, tf_nemo_audio::bf16* __restrict__ out, int first, int valid, float preemph, float guard) {
    tf_nemo_audio::mel<true>(X, lo, n, win, fb, cs, sn, out, first, valid, preemph, guard);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_nemo_norm(const tf_nemo_audio::bf16* __restrict__ x, tf_nemo_audio::bf16* __restrict__ y, int frames, int valid, float eps) {
    tf_nemo_audio::norm<true>(x, y, frames, valid, eps);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_nemo_qk(tf_nemo_audio::bf16* __restrict__ qkv, const tf_nemo_audio::bf16* __restrict__ qw, const tf_nemo_audio::bf16* __restrict__ qb, const tf_nemo_audio::bf16* __restrict__ kw, const tf_nemo_audio::bf16* __restrict__ kb, const tf_nemo_audio::bf16* __restrict__ cs, const tf_nemo_audio::bf16* __restrict__ sn, int rows, int heads, int dim, float eps) {
    tf_nemo_audio::qk<true>(qkv, qw, qb, kw, kb, cs, sn, rows, heads, dim, eps);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_nemo_attn64(const tf_nemo_audio::bf16* __restrict__ qkv, tf_nemo_audio::bf16* __restrict__ out, int rows, int heads, float scale) {
    tf_nemo_audio::attend<64, true>(qkv, out, rows, heads, scale);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_nemo_taps(const tf_nemo_audio::bf16* __restrict__ x, tf_nemo_audio::bf16* __restrict__ y, int rows, int width) {
    tf_nemo_audio::taps<true>(x, y, rows, width);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_nemo_relu(tf_nemo_audio::bf16* __restrict__ x, long count) {
    tf_nemo_audio::relu<true>(x, count);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_nemo_preds(const tf_nemo_audio::bf16* __restrict__ logits, float* __restrict__ out, int rows, int spk, int sub) {
    tf_nemo_audio::preds<true>(logits, out, rows, spk, sub);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_nemo_fuse(const tf_nemo_audio::bf16* __restrict__ an, const float* __restrict__ preds, const tf_nemo_audio::bf16* __restrict__ nw, const tf_nemo_audio::bf16* __restrict__ nb, const tf_nemo_audio::bf16* __restrict__ K, tf_nemo_audio::bf16* __restrict__ out, tf_nemo_audio::bf16* __restrict__ sn, int width, float thr, float eps) {
    tf_nemo_audio::fuse<true>(an, preds, nw, nb, K, out, sn, width, thr, eps);
}
