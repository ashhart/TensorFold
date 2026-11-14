// TF_GEMMA4_V_ONLY's planted control: tf_vonly_keys without the bf16 rounding of w * V (the bit check must go red).

#include "window_vonly.cuh"

extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_vonly_keys_unrounded(const __nv_bfloat16* vc, __nv_bfloat16* ks, const int* tab, long long page, int tok,
                                int hk, int k0, const __nv_bfloat16* kw, const float* inv) {
    constexpr int D = 512;
    const int key = k0 + blockIdx.x, kh = blockIdx.y, d = threadIdx.x;
    const int64_t at = static_cast<int64_t>(tab[key / tok]) * page + (static_cast<int64_t>(key % tok) * hk + kh) * D;
    __nv_bfloat16 e[2] = {vc[at + d], vc[at + d + D / 2]};
    tf_wrows::vk_pair<D, false>(&e[0], &e[1], d, key, tf_wrows::Vk{kw, inv});
    ks[at + d] = e[0];
    ks[at + d + D / 2] = e[1];
}
