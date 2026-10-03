# Gemma 4 12B unified (`gemma4_unified`)

Served as an oQe checkpoint (for example the uniform 8-bit `unigilby/gemma-4-12B-it-oQ8e`) with the
`gemma-4-12B-it-qat-assistant-4bit` MTP assistant. Packages: `src/tensorfold/families/gemma4_unified/` and
`src/tensorfold/kernels/gemma/dense/v1/`. Text only: the unified checkpoint's patch embedder and audio
projection are dropped at load.

## The model

- `Gemma4UnifiedForConditionalGeneration`; its text model (`gemma4_unified_text`) is Gemma 4's dense decoder:
  48 layers, 40 sliding (window 1,024; 16 query heads over 8 KV heads, head dim 256) and 8 full attention
  (16 query heads over 1 KV head, head dim 512, keys reused as values, proportional RoPE). Hidden 3,840,
  GeGLU MLP width 15,360, per-layer scalar, vocabulary 262,144 tied and soft-capped at 30.
- The 12B has no per-layer inputs and no shared-KV layers (`check` refuses checkpoints that have them, and
  MoE ones, which the `gemma4` family serves).
- mlx-lm's `gemma4` wrapper loads it (both the oQ layout and the mlx-vlm layout sanitize to `gemma4_text`).

## Kernels

Prompts run mlx-lm's forward on the planned chunks. Decode rows run Gemma 4's v1 attention, head norm/RoPE
and KV caches with a dense tail (`kernels/gemma/dense/v1/glue.py`) and row-exact affine projections at any
oQ width in groups of 64 (4-bit also 32): the M5 lane matmul (`--lane-kernels auto` on tensor-unit GPUs) or
`row_matmul`'s simd kernels (`--lane-kernels off`, any Mac). Stacked q|k|v and gate|up split into runs of one
width when oQ boosts one member. The lane matmul keeps tiled copies of the decode weights beside mlx-lm's,
so resident weights are about twice the checkpoint.

## Drafts

`--drafter` takes a `gemma4_unified_assistant` (MTP) checkpoint: four layers that attend over the target's
last sliding and last full layer's keys from the last kept row's position, fed the target's scaled embedding
of the pending token next to the final-normed hidden row, then their own projected output. Up to 6 drafts a
round (the depth learner picks); shared rounds draft every stream in one assistant forward a step. The
assistant runs MLX's stock kernels: drafts only propose. Ordered (centroid) embeddings are refused.

## Checkpoint

oQe in groups of 64: oMLX's `oq` quantizer, enhanced (imatrix), with every quantized tensor pinned to groups of
64 and the widths left as oQ picks them. Level 8 from `google/gemma-4-12B-it` gives a uniform 8-bit oQ8e; level 6
gives an oQ6e (6-bit with a few 8-bit boosts) the family also reads.

## Exactness

Drafted replies equal `"draft": false`, and concurrent replies equal serial ones, by token hash. Windows are exact
to 16 rows; shared forwards up to 64.
