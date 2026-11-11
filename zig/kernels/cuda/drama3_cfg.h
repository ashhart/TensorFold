// Drama 3's semantic CFG mix of one candidate logit, shared by d3_cfg_mix and its host test.
#pragma once

#if defined(__CUDACC__)
#define D3_CFG_HD __host__ __device__
#else
#define D3_CFG_HD
#endif

// Element i of n candidate logits (semantic tokens and EOS): c + (gamma - 1)(c - u); gamma 1 keeps c's bits, any u.
D3_CFG_HD static inline void d3_cfg_mix_at(float* conditional, const float* unconditional, int n, float gamma, int i) {
    if (i >= n || gamma == 1.0f) return;
    float c = conditional[i];
    conditional[i] = c + (gamma - 1.0f) * (c - unconditional[i]);
}

#undef D3_CFG_HD
