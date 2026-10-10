# MiniMax H3 / FastH3 transformer blocks on CUDA (GB10)

This is a video model's denoising transformer, not a token model. It has no lane round and no drafted tokens, and
it runs its projections and attention in int8, which trades precision for speed. It sits beside the engine as a
library a host pipeline calls; nothing in the lane core or the server changes.

## What it is

`zig build tf-h3 -Dnvcc=/usr/local/cuda/bin/nvcc` builds `zig-out/lib/libtf_h3.so`. The host owns the text encoder,
the sampler, the patch projections, the final layer and the decoders, and shares its CUDA context. Each denoising
pass it hands over the packed stream (rows of the model's hidden width, bf16) and gets all 50 blocks applied in place.

| Call | Does |
|---|---|
| `tf_h3_open(path)` | Maps the one-file checkpoint, quantizes the block projections to int8 on the GPU, or reads the saved copy |
| `tf_h3_forward(model, inputs)` | Every block over the stream; returns the seconds taken |
| `tf_h3_profile`, `tf_h3_spent` | Per-stage seconds, each stage waited for |
| `tf_h3_close(model)` | Frees the model |

The checkpoint is FastVideo's `fastvideo_fasth3_8step_v2_pruned_bf16.safetensors` (50 blocks, hidden 5376, attention
width 7168, MLP 14336). The int8 copy is written beside it as `<checkpoint>.tf-int8` (21 GB) on first use.

## Arithmetic

- Projections: int8 weights with a scale per output channel; activations int8 with a scale per row (per row and 1024
  columns for the MLP's wide rows). Products on `mma.sync m16n8k32`, accumulated in int32.
- Attention: FastH3's video sparse attention. Rows are laid out in 64-row tiles; a video tile attends the prefix tiles
  and its top 20% of video tiles by pooled score, and the pooled branch is gated in. Scores and values are int8,
  softmax weights 8-bit per key tile with an online softmax across tiles.
- Rounding to bf16 where the reference's bf16 arithmetic rounds.

## Receipt

Measured 2026-10-09 on one DGX Spark (GB10, compute capability 12.1, CUDA 13.0 toolkit, DGX OS, driver's bundled
runtime), host ComfyUI at commit `a4b5a045`, 124 frames, 8 passes, seed fixed. The kernels are the ones in this
change; the Zig around them was split into smaller files afterwards and renders a bit-identical clip.

| Size | Rows | This family, a pass | ComfyUI's own blocks in bf16, a pass |
|---|---|---|---|
| 864x480 | 15,452 | 7.1 s | 10.5 s (dense attention) |
| 1280x736 | 34,507 | 18 s | not measured |
| 1344x768 | 37,763 | 20.5 s | 29.4 s (its block-sparse attention) |

Checks, same input stream on both sides:

| Check | Result |
|---|---|
| int8 product kernel against a float64 host sum, sampled outputs, four shapes | worst relative difference 0.004 |
| Tile attention kernel against exact float64 attention over the same int8 inputs | relative error 0.002 |
| One block, dense attention, against ComfyUI's bf16 block | cosine 0.9999, relative error 0.015 |
| All 50 blocks, dense attention, 864x480 | cosine 0.998, relative error 0.065 |
| All 50 blocks, dense attention, 1344x768 | cosine 0.996, relative error 0.090 |
| Same seed twice | bit-identical clip |

Raw tensor-core rate on the GB10, registers only: int8 245 TOPS, bf16 122 TFLOPS. The product kernel reaches 150 to
175 TOPS on these shapes.

## Not done

- No FP64 reference per projection class in the tree; the checks above were run from a bench harness and a host
  plug-in that are not part of this change.
- One GPU model tested (GB10). Two checkpoints' worth of shapes are not covered: only FastH3 8-Step V2.
- The Metal family for the same model lives on a fork and is not in this change.
