// Ling-3.1-flash's dense SwiGLU rows in its clamp order, min(silu(g), L) * clip(u, -L, L), on glm_swiglu.cuh.
#include "glm_swiglu.cuh"

TF_GLM_SWIGLU_512(tf_ling_swiglu_512, true)
TF_GLM_SWIGLU_64(tf_ling_swiglu_64, true)
