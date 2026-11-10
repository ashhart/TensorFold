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

## Prompt

`tf-h3-image-prompt` makes a picture case's `text` and `text_tags` from a prompt. It runs on any Apple GPU: no tensor
units are needed.

```sh
zig build tf-h3-image-prompt -Dcpu=apple_m1 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-prompt TEXT_ENCODER_DIR MODEL_DIR prompt.txt out/prompt.safetensors
```

`TEXT_ENCODER_DIR` is MiniMax-H3's `text_encoder/` (Qwen3-VL). `MODEL_DIR` is the MiniMax-H3-Image folder, for its
`tokenizer/` and `transformer/`. The prompt is used verbatim, except for one trailing line ending: no chat template and
no system prompt.

1. **Ids.** The prompt is tokenised on its own, with no special tokens added, and cut to its first 4,096 tokens.
   Every row is tagged 1 (text).
2. **The text encoder.** Qwen3-VL's text model runs 50 of its 64 layers:
   - hidden 5,120, with 64 query and 8 KV heads of 128;
   - q and k RMSNorm, and a SwiGLU of 25,600;
   - causal attention, with rotary at theta 5,000,000.

   Text positions count up, equal on M-RoPE's three axes. The rows leaving layer index 49 are kept unnormalised. The
   final norm and layers 50 to 63 are never mapped or run.
3. **The front end.** `condition_proj` and its bias map each row to 5,376. The two token-refiner blocks then run over
   every row as one document, with no rotary and no AdaLN, and `token_refiner.final_norm` ends it.

It writes `text` BF16 [rows, 5,376], `text_tags` I32 [rows] and `ids` I32 [rows], aligned. `tf-h3-image` reads the
first two as a case's own.

**Precision.** Weights are bf16, read in place. Products run in simdgroup matrices with fp32 sums. Norms keep fp32
statistics, and the softmax and its sums stay in fp32. Each step rounds to bf16 where the reference's bf16 arithmetic
does:

- the text model's norms round x r before the weight;
- its rotary turns by bf16 cos and sin;
- its SiLU rounds once;
- the refiner rounds as H3's blocks do: x r w at once, then the sigmoid, then the product.

**Edits.** The encoder places a reference's vision rows at its image pads and adds DeepStack's features after layers
0, 1 and 2. The labels (`Picture i: `), vision blocks, tags (0 for a vision block's tokens) and M-RoPE positions are
made here. Qwen3-VL's vision tower is not, so an edit's prompt rows still come from the case.

**Gates, fixed before the first run except the whole-run gate below.** `tf-h3-image-prompt selftest [WORK_DIR]`
writes a tiny synthetic checkpoint and loads it as a real run does. The checkpoint is a 256-wide text model with 4
query and 2 KV heads, three layers written and two run, and a 384-wide refiner of 2 heads. The selftest checks 70 rows
against a float64 host reference with the same bf16 rounding points:

- **Each kernel alone:** at least 99.9% of values bit-equal, none further than 1/32 of its row's RMS. Kernels: the
  product, both norms, heads with rotary, fused heads, causal and full attention, both SiLU gates, the residual add.
- **Whole runs:** a text prompt; a reference's vision and DeepStack rows at M-RoPE positions; the refiner on the GPU's
  encoder rows. Each must be within twice a float32-sum host run's distance from float64, in both 1 - cosine and
  relative error, with cosine at least 0.999. This gate replaced a fixed cosine bound of 0.99999, which the refiner
  missed at 0.9999884 on a second seed while every kernel alone was bit-exact.
- **A second run:** byte for byte the same.

On an M3 Ultra, every kernel alone is 99.97% to 100% bit-equal. The whole runs sit 0.7 to 1.03 times as far from
float64 as the float32-sum run.

`zig build test-h3-image-prompt` runs the prompt rules without a GPU:

- ids per segment, the 4,096 cut, tags, pad rows and vision token counts;
- M-RoPE positions and the interleaved rotary table;
- the Qwen copy's size and LANCZOS pixels;
- the encoder's and refiner's configs, and the text model's tensor prefix.

`zig build compile-h3-image-prompt` compiles the tool without running a shader.

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

## Edit encoder

An edit's reference reaches the transformer as clean latent rows made by the VAE's CNN encoder. `tf-h3-image-encoder`
makes them on Metal in fp32, on any Apple GPU; no tensor units are needed. Its kernels are compiled from source when it
starts, so it builds without the Metal compiler.

- **The VAE copy.** The reference is read upright in RGB, with any alpha dropped. A short edge already in 1024..2048
  keeps the original pixels. Otherwise the picture is scaled to the nearer bound, each side snapped to 32 (halves to
  even) and resampled with Pillow's LANCZOS: 512x768 becomes 1024x1536, 3000x4000 becomes 2048x2720, and 1536x2048 is
  kept. A side under 32 pixels, or a copy over 64 MP, is refused.
- **The input.** Bytes / 255, then ImageNet's mean and std, in fp32, cropped top-left to multiples of 16 pixels.
- **The network.**
  - Each 3x3x3 causal convolution runs as the 2D convolution of its last time slice, reflect-padded by one.
  - `conv_in` maps 3 channels to 128.
  - Six levels of two ResNet blocks follow, at widths 128, 256, 256, 512, 512 and 1,024, with `nin_shortcut` where the
    width changes.
  - Levels 0 to 3 end with a stride-2 downsample that reflects the right and bottom edges by one pixel.
  - Then GroupNorm(32, eps 1e-6) with SiLU, `conv_out` (1,024 to 48) and `quant_conv`.
  - Every tensor name and shape is checked against the checkpoint's header before anything loads.
- **Tiles.** 256-pixel tiles overlapping by at least 64, split as the decoder splits. Each tile's 48 moment channels
  are blended in latent cells (overlap / 16) with the decoder's rule.
- **The rows.**
  - The posterior sample, mean + exp(0.5 clamp(logvar, -30, 20)) x noise, is rounded to fp16.
  - It is normalised with the latent statistics and cropped top-left to the even grid: 1024x1024 gives 64 x 64, and
    1050x1000 gives 62 rows by 64 columns.
  - Then it is patchified.
  - Anchor rows are 0.999 x clean + (1 - 0.999 in fp32) x the anchor noise.
- **The noise.** One CPU torch generator per request (`moss_image/noise.zig`, its draws chained), drawn in order: the
  target, each reference's anchor noise, then each reference's posterior noise on its full grid.

```sh
zig build tf-h3-image-encoder -Dcpu=apple_m1 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-encoder MODEL_DIR reference.png SEED out/ref [CANVAS]
zig-out/bin/tf-h3-image-encoder selftest
```

It writes:

- `out/ref.rows.f32`: the clean rows [rows, 96];
- `out/ref.moments.f32`: the blended moments [48, h, w], before sampling;
- `out/ref.grid.json`.

With `CANVAS` (WxH), the generator first draws the target's noise and this reference's anchor noise, as a
one-reference edit does, and `out/ref.anchors.f32` is written too. Without it, the posterior's noise is the seed's
first draw.

`selftest` runs each kernel on small synthetic weights against a float64 host reference and exits 1 on a miss. It
covers:

- a reflect-padded 3x3 convolution and the stride-2 downsample;
- a 1x1 convolution with a residual;
- GroupNorm with SiLU;
- both kinds of ResNet block;
- a small encode over uneven tiles, run twice.

Its gates, measured on an M3 Ultra:

| Check | Gate | Measured |
|---|---|---|
| Each layer, worst error over the largest value | 1e-5 | 2e-7 to 9e-7 |
| The tiled moments | 1e-4 | 1.6e-6 |
| The fp16 samples | one fp16 step at the largest value | within it |
| Repeat runs | bit-identical | bit-identical |

A 1024 x 1024 reference at the released widths encodes in 0.6 s.

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
