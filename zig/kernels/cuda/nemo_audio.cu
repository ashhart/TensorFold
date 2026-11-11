// Nemotron 3.5 Transcribe's audio encoder kernels: nemo_audio.cuh's steps as served.
#include "nemo_audio.cuh"

using tf_nemo_audio::bf16;

extern "C" __global__ void __launch_bounds__(256) tf_nemo_mel(const float* __restrict__ X, long lo, long n, const bf16* __restrict__ win, const bf16* __restrict__ fb, const float* __restrict__ cs, const float* __restrict__ sn, bf16* __restrict__ out, int first, int valid, float preemph, float guard) {
    tf_nemo_audio::mel<false>(X, lo, n, win, fb, cs, sn, out, first, valid, preemph, guard);
}

extern "C" __global__ void __launch_bounds__(256) tf_nemo_norm(const bf16* __restrict__ x, bf16* __restrict__ y, int frames, int valid, float eps) {
    tf_nemo_audio::norm<false>(x, y, frames, valid, eps);
}

extern "C" __global__ void __launch_bounds__(256) tf_nemo_qk(bf16* __restrict__ qkv, const bf16* __restrict__ qw, const bf16* __restrict__ qb, const bf16* __restrict__ kw, const bf16* __restrict__ kb, const bf16* __restrict__ cs, const bf16* __restrict__ sn, int rows, int heads, int dim, float eps) {
    tf_nemo_audio::qk<false>(qkv, qw, qb, kw, kb, cs, sn, rows, heads, dim, eps);
}

extern "C" __global__ void __launch_bounds__(256) tf_nemo_attn64(const bf16* __restrict__ qkv, bf16* __restrict__ out, int rows, int heads, float scale) {
    tf_nemo_audio::attend<64, false>(qkv, out, rows, heads, scale);
}

extern "C" __global__ void __launch_bounds__(256) tf_nemo_taps(const bf16* __restrict__ x, bf16* __restrict__ y, int rows, int width) {
    tf_nemo_audio::taps<false>(x, y, rows, width);
}

extern "C" __global__ void __launch_bounds__(256) tf_nemo_relu(bf16* __restrict__ x, long count) {
    tf_nemo_audio::relu<false>(x, count);
}

extern "C" __global__ void __launch_bounds__(256) tf_nemo_preds(const bf16* __restrict__ logits, float* __restrict__ out, int rows, int spk, int sub) {
    tf_nemo_audio::preds<false>(logits, out, rows, spk, sub);
}

extern "C" __global__ void __launch_bounds__(256) tf_nemo_fuse(const bf16* __restrict__ an, const float* __restrict__ preds, const bf16* __restrict__ nw, const bf16* __restrict__ nb, const bf16* __restrict__ K, bf16* __restrict__ out, bf16* __restrict__ sn, int width, float thr, float eps) {
    tf_nemo_audio::fuse<false>(an, preds, nw, nb, K, out, sn, width, thr, eps);
}
