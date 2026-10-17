# MiniMax H3 / FastH3 transformer on Metal (M5 tensor units)

This is a video model's denoising transformer, not a token model. It has no lane round and no drafted tokens. It
keeps the checkpoint's precision: bf16 weights read in place from the checkpoint, bf16 activations, fp32 sums. It is
a test tool beside the engine; nothing in the lane core, the server or another family changes.

## What it is

`zig build tf-h3-dit -Dcpu=apple_m1 -Doptimize=ReleaseFast` builds `zig-out/bin/tf-h3-dit`. It reads the FastH3
checkpoint's shards and a case file, runs every denoising forward for the packed [text | keyframe | audio | video]
rows, prints the first forward's velocities against the reference velocities the case holds, and writes the final
video and audio rows (and the first forward's velocities) as raw float32.

```sh
zig-out/bin/tf-h3-dit <transformer_dir> <shards> case.safetensors out_prefix [dense | profile | DEFINE, ...]
```

It needs the M5's tensor units and refuses to load without them. `dense` makes every query tile attend to every tile
and leaves the pooled branch out (not how the checkpoint was trained; for kernel work only). `profile` times one more
forward stage by stage.

## Arithmetic

- Block projections: the prefill kernels' `tfp::gemm_nax` (MLX 0.32.3's NAX GEMM), bf16 x w^T as the checkpoint
  stores w, fp32 sums, bf16 out. The SwiGLU's first projection is one product over its stored [value; gate] rows.
- Norms, rotary, the gated adds and the SwiGLU round to bf16 where the reference's bf16 MLX arithmetic rounds.
- Attention is FastH3's video sparse attention over 64-row tiles: a text, keyframe or audio tile attends every
  tile; a video tile attends the prefix tiles and its top 20% of video tiles by pooled score; the pooled branch is
  gated in. Tile means, scores, the top-k and the pooled branch are fp32. The routed attention is bf16 q, k and v
  on the tensor units with fp32 scores and an online softmax across key tiles; its weights are rounded to bf16
  before the value product, as MLX's attention rounds them.

## Kernel contract

Thirteen kernels are registered (`zig/tests/contract/metal_h3*.zig`), each against its formula in float64, and their
launches in `registry.zig`. The two GEMM entry points are the prefill source's own and are registered as launches.
Kernels whose outputs pass through intermediate bf16 roundings declare them as `mids`; their test inputs are of one
sign where a sum could cancel, so no rounding hides behind a small result.

```sh
zig build tf-kernel-contract -Dcpu=apple_m1 -Doptimize=ReleaseFast
zig-out/bin/tf-kernel-contract --only h3
```

## The case file

The prompt's rows, the schedule and the tile map come in one safetensors file. The exporter that writes one is not in
this change: it uses the Python 0.6 H3 family, which lives on a fork (`drowzeys/TensorFold`, `tools/zig/h3_case.py` on
branch `h3-1.0.4`).

| Tensor | Holds |
|---|---|
| `text` | The prompt's rows after the condition projection and the token refiner, bf16 |
| `video`, `audio`, `condition` | Starting noise rows; a first frame's rows when the clip starts from an image |
| `cos`, `sin` | Rotary tables for every packed row |
| `adaln`, `times`, `tables.N`, `final` | Each row's line in a block's six modulation tables per step, and the tables |
| `video_step`, `audio_step` | Per step: the sigma the model sees and the Euler ratio |
| `tile_slot`, `tile_sizes`, `geometry` | FastVideo's tile map: each row's padded slot, each tile's rows, the counts |
| `video_in.*`, `audio_in.*`, `video_out.*`, `audio_out.*`, `final_norm.weight` | The small float projections |
| `video_velocity`, `audio_velocity` | The reference's first forward, for the parity line the tool prints |

## Receipt

Mac Studio, Apple M5 Ultra, 256 GB, macOS 27.0.1, Zig 0.17.0. Checkpoint `FastVideo/FastVideo-FastH3-8-Step-V2`,
864x480, 124 frames, 8 steps, 15,435 packed rows (36 text, 414 audio, 14,985 video), 288 tiles. Reference: the
Python 0.6 family in bf16 on MLX 0.32.3 with FastVideo's reference routing; with FastVideo's fp Metal sparse attention
it takes 27.9 s a forward on the same Mac. Peak process footprint 4.3 GB, the checkpoint mapped read-only beside it.
The pull request has the full tables.

| Run | A forward (median) | Video velocity against the reference | Audio velocity against the reference |
|---|---|---|---|
| Step 1 of 8 | 10.1 s | cosine 0.98981, relative error 0.143 | cosine 0.99987, relative error 0.017 |
| Step 5 of 8 | 10.1 s | cosine 0.99838, relative error 0.057 | cosine 0.99980, relative error 0.020 |

At step 1 every video tile is noise, so FastH3's top-k among them is decided by differences far below bf16. The same
binary with only the fp32 summation order of the tile scores reversed differs from itself by video cosine 0.99021 at
step 1 and 0.99834 at step 5: the comparison with the reference is at that floor.

## A clip that starts from an image

A case exported with a first frame (`h3_case.py --first-frame IMAGE`) has a seventh geometry count: the keyframe's
rows, between the text (which then also holds the image's vision tokens) and the audio. They enter like video rows
on every pass, at the keyframe noise level the case's tables give them, and are never stepped.

Same Mac, checkpoint and size, a 864x480 first frame: 16,253 packed rows (449 text, 405 keyframe, 414 audio,
14,985 video), 302 tiles (22 prefix).

| Run | Video velocity against the reference, step 1 | Audio velocity against the reference, step 1 |
|---|---|---|
| From a first frame | cosine 0.99711, relative error 0.076 | cosine 0.99986, relative error 0.017 |

Text to video is bit-identical with and without this change (final video and audio rows, 8 steps).

## Other step counts

With `tables` or `steps=N` the tool makes the run's schedule itself instead of reading the case's. The rungs come from
the checkpoint's `fastvideo_inference.json`, or N spread evenly from its first rung, with the video and audio shifts
(a preview checkpoint's from its scheduler folders). Then come every row's timestep at each step and the timestep
embedder (on the host, float64 sums of the checkpoint's float32 values). Each block's and the final layer's AdaLN
tables come from `h3_modulation` on the GPU, bf16 inputs and weights with fp32 sums. `from=K` starts at step K of
a case exported mid-run. The case still supplies the prompt's rows, the noise, the rotary tables and the tile map.

## The released MiniMax-H3, and re-voicing a clip

The tool also loads the released MiniMax-H3 transformer (its FL2VA or Ref2VA folder, `model-*` shards): each head's
fused [q, k, v] rows in one product, the SwiGLU's [gate; value] rows, dense attention with no routing gates, and its
own timestep embedder and AdaLN names. `revoice=N` re-voices a finished picture as the Python sampler's `revoice`
does in its whole-sequence form: every row but the generated audio stays at the keyframe level for every step, the
audio rows go down the released model's grid of N + 1 points (audio shift 3), and the video rows are not moved.
A case for it comes from `h3_revoice_case.py` on the fork's exporter branch, from a finished clip's rows.

## Not done

- No exporter in the tree, so the tool cannot be exercised from this repository alone.
- One machine (M5 Ultra). Chips without tensor units are refused.
- No `tensorfold generate`, no text encoder, decoders or sampler: the forward only.
