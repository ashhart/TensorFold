# MiniMax-H3-Image's transformer on Metal (M5 tensor units)

MiniMax-H3-Image makes pictures from text, and edits pictures from instructions. Its transformer is MiniMax H3's,
with the audio layers removed: 50 blocks, hidden 5,376, 56 heads of 128, an FFN of 14,336. It is distilled to six
sampling steps with no guidance, and each step has its own output head in place of the transformer's.

`tf-h3-image` runs that transformer on Metal from a picture case, using the H3 family's own blocks
([minimax-h3-metal.md](minimax-h3-metal.md)). It is a denoiser beside the engine, not a token family. Nothing in the
lane engine, the server or another family changes, and a clip's case runs exactly as before.

## What runs here, and what the case brings

Made natively, from the request and the model folder:

- **The canvas.** A request below 2048 x 2048 is drawn at the same aspect ratio on a canvas of about that area. Every
  side is snapped to 32 pixels, rounding half to even, as Python's `round` does. One transformer row covers 32 x 32
  pixels: the VAE's 16, then a 2 x 2 patch.
- **The schedule.** `pdd_config.json` holds the step count and its steps + 1 sigmas, which fall strictly from 1 to 0.
  Each sigma goes through a float32 timestep (u = f32(1 - sigma), then sigma' = 1 - u), so the timestep the model sees
  plus its sigma is exactly 1. Each step moves the target rows by plain Euler, x + (sigma'_k - sigma'_k+1) v, in
  float32. That is the form the heads were distilled with, not the clip tool's ratio form. The tool refuses a run with
  any other step count.
- **The step heads.** `pdd_fused_heads.safetensors` holds six float32 heads, [6, 96, 5,376] and [6, 96]. Step k's
  head replaces the final layer's `video_out`. A picture's case may hold such per-step heads; a clip's case may not.
- **The rows' positions.** Each row gets a (t, h, w) position:
  - text row i sits at (i, 0, 0);
  - reference k sits at t = text + k;
  - the target sits at t = text + the reference count.
  
  A block's h and w are centred on 16 and normalised by the block's own area, in float64, then rounded once to
  float32. A square block spans [0, 32): a 2048-pixel target in steps of 0.5, a 1024-pixel reference in steps of 1.
- **The rotary table.** Each axis has 16 frequencies, read from the checkpoint's `rope.inv_freq`. That gives 48 angles
  a row, whose cosines and sines rotate the first 96 of each head's 128 dims. The heads kernel rounds them to bf16, as
  for clips.
- **The tables.** A row's timestep is the step's, except references, which are held at 0.999 for every step. Its
  AdaLN line is that timestep's index times three plus its modality (text 1, picture 0). The AdaLN tables come from
  the checkpoint as for the released MiniMax-H3 clips (`modulation.zig`).
- **Attention.** Dense over every row, in tiles of 64: the prefix (text and references), then the target.

The picture case brings the rest, made by the reference pipeline for now:

| Tensor | Holds |
|---|---|
| `canvas` | I32 [2]: the canvas's width and height |
| `text` | BF16 [text, 5,376]: the prompt's rows as they enter block 0 (after `condition_proj` and the two-block token refiner) |
| `text_tags` | I32 [text]: 1 for text tokens, 0 for the tokens of a reference picture's vision block (all 1 when absent) |
| `references`, `anchors` | An edit's reference grids, I32 [n, 2] in latent cells (rows, columns), and their anchor rows, F32 [rows, 96] |
| `noise` | F32 [target rows, 96]: the starting target rows |
| `step_in`, `velocity` | F32 [6, target rows, 96]: the reference's input and velocity at each step |
| `latent` | F32 [target rows, 96]: the reference's rows after the sixth step |
| `positions`, `rotary_cos`, `rotary_sin`, `timesteps`, `deltas` | The reference's own, checked against ours |

Rows are 96 wide: 24 latent channels times a 2 x 2 patch, channel-major, then the patch's row, then its column.
Patches run in row-major order. The case must be written aligned (Python's `safetensors` writer does this).

## Run a case

```sh
zig build tf-h3-image -Dcpu=apple_m1 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image MODEL_DIR picture.safetensors out/case [once | free]
```

`MODEL_DIR` is the MiniMax-H3-Image folder: `model_index.json` (the image pipeline's class and partition),
`transformer/` (its config must say `audio_enabled: false`), `pdd_config.json` and the heads file. The tool:

1. checks the case's positions, rotary table, timesteps and Euler steps against ours;
2. runs each step on the reference's own input rows (teacher-forced) and compares the target's velocity;
3. runs all six steps from the noise with our own rows.

It writes `out/case.latent.f32`, the normalised latent [24, rows, columns] in float32, ready for the reference's
decoder. `once` runs the first teacher-forced step alone; `free` runs only the free run. It needs the M5's tensor
units, as the clip tool does.

## Gates, fixed before any run

These are the bounds `tf-h3-image` checks (`picture.gates`). It exits 1 when one fails, and 2 when the case holds no
reference to gate against.

- Positions, timesteps and Euler steps: bit-identical to the reference's.
- The rotary table: at most 1 bf16 ULP from the reference's.
- Each teacher-forced step: target velocity cosine at least 0.999 and relative error at most 0.03.
- The free run's final latent: cosine at least 0.99 against the reference's.

Attention here is dense, so no routing floor applies. FastH3's 0.990 at step 1 came from its top-k tiles.

## Checkpoint

`MiniMaxAI/MiniMax-H3-Image`. `transformer/` is bf16, with the patch projection, time embedder and step heads in
float32. The text encoder is MiniMax-H3's Qwen3-VL; its 50th layer's rows are the prompt's. The weights are under the
MiniMax-H3-Image Community License, which excludes the European Union, the United Kingdom, the Republic of Korea and
the United States of America. Read it before use.

## Not done

- The prompt's rows come from the case. The text encoder (Qwen3-VL, 50 of 64 layers), `condition_proj` and the token
  refiner are not run here.
- No image decoder: the latent is decoded by the reference pipeline. That decoder is a 2D ViT (24 layers of 1,024) in
  256-pixel tiles.
- No edit encoder: an edit's anchor rows come from the case. That encoder is MiniMax H3's CNN, on one frame.
- No `/v1/images/generations` or edits route, and no CUDA path.
- One machine class (M5). Chips without tensor units are refused.

## Host checks

`zig build test-h3-geometry` runs the picture rules without a GPU:

- the canvas, including 1280x720 drawn at 2720x1536 and 4000x3000 at 4000x3008;
- the schedule's float32 round trip and its refusals;
- positions, the rotary layout, the packing and the tile map;
- the model folder's checks;
- the case reader's picture admission.

`zig build compile-h3-host` compiles both H3 tools without the Metal compiler.
