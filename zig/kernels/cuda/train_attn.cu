// Gated attention's input gradients (Qwen3-Next's: a head's q then its output gate, RMS-normed q and k, partial RoPE).

// fp32 gradients from bf16 kept projections, as train.cu, whose tf_train_attn_kv takes the keys' and values' part.

#define TF_ATTN_BACK(name) tf_gattn_##name
#include "attn_back.cuh"
