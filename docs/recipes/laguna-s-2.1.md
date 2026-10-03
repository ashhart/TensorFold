# Laguna-S-2.1 (`laguna`)

poolside's Laguna-S-2.1 on Apple Silicon, tested with `mlx-community/Laguna-S-2.1-oQ4e` (oMLX oQ4e: 4-bit experts
in groups of 128, attention at 5 or 8 bits in groups of 64, shared experts at 8 bits in groups of 128, the router
in bf16). Packages: `src/tensorfold/families/laguna/` and `src/tensorfold/kernels/laguna/v1/`.

## The model

- 48 layers, hidden size 3,072. 12 full-attention layers (48 query heads, YaRN RoPE over half of each 128-dim
  head) and 36 sliding-window layers (window 512, 72 query heads, default RoPE over the whole head), 8 KV heads
  everywhere. Q and K are RMS-normed per head; the attention output is scaled per head by `softplus(g_proj(x))`.
- Layer 0 has a dense SwiGLU MLP; the other 47 a MoE block: 256 experts of width 1,024, top 10 by sigmoid score
  plus a correction bias (the weights are the unbiased scores, normalised), scaled by 2.5, plus one shared expert.
- Vocabulary 100,352 with an untied head.

## Decode

Prompts run the mlx-lm forward (`families/laguna/mlx_model.py`, mlx-lm's `laguna` vendored, since mlx-lm 0.31
does not have it). Every decode row runs `kernels/laguna/v1`:

- projections of every affine width through the M5 lane matmul (`--lane-kernels auto`; groups of 128 are read as
  two groups of 64 with the same scale and bias, which dequantizes to the same weights), or the packed affine row
  kernel on earlier chips. Both give a row the same bits at any row count;
- q/k norms and partial RoPE in one kernel a head; Gemma 4's decode attention with the simdgroup split halved
  until 9 query heads a KV head fit a threadgroup; Gemma 4's 4-bit expert kernels with SiLU and fp32 routing
  weights; a sigmoid top-k kernel (ties to the lower id) and a bf16 router matvec of fixed order a row.

The engine interface, ring and full KV caches and load-time checks are Gemma 4's. KV stays bf16.

## Drafts

`--drafter poolside/Laguna-S-2.1-DFlash --drafter-bits 8` (`tensorfold pull` it first) drafts chains of up to 15
tokens with poolside's DFlash model, run as vLLM's `laguna_dflash` runs it: six sliding-window Laguna layers over
the target's rows after layers 1, 10, 19, 29, 38 and 47, each tap RMS-normed before `fc`, the context through each
layer's input norm, a causal block. Replies stay equal to `"draft": false` ones. 8-bit drafts are accepted as
often as bf16 ones; 4-bit ones slightly less.

## Exactness

Every reply hashes equal across serial, 2- and 4-stream and `"draft": false` runs, and seeded sampled replies equal
their `"draft": false` runs. Teacher-forced, the decode kernels' NLL and top-1 agreement with mlx-lm's prompt
forward sit within mlx-lm's own step-against-prompt spread.
