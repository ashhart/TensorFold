# MiniMax-H3-Image's transformer on Metal (M5 tensor units)

MiniMax-H3-Image makes pictures from text, and edits pictures from instructions. Its transformer is MiniMax H3's,
with the audio layers removed: 50 blocks, hidden 5,376, 56 heads of 128, an FFN of 14,336. It is distilled to six
sampling steps with no guidance, and each step has its own output head in place of the transformer's.

`tf-h3-image` runs that transformer on Metal from a picture case, using the H3 family's own blocks
([minimax-h3-metal.md](minimax-h3-metal.md)). It is a denoiser beside the engine, not a token family. Nothing in the
lane engine, the server or another family changes, and a clip's case runs exactly as before.

## Text to image, end to end

Three tools, each on Metal. The prompt and decoder run on any Apple GPU; the transformer needs an M5's tensor units:

```sh
zig build tf-h3-image-prompt tf-h3-image tf-h3-image-decoder -Dcpu=apple_m1 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-prompt TEXT_ENCODER_DIR MODEL_DIR prompt.txt out/case.safetensors 2048x2048 42
zig-out/bin/tf-h3-image MODEL_DIR out/case.safetensors out/case free
zig-out/bin/tf-h3-image-decoder MODEL_DIR out/case.latent.f32 128 128 out/case.png 2048x2048
```

1. With a size and a seed, `tf-h3-image-prompt` writes a whole text-to-image case:
   - the prompt's rows;
   - the canvas, grown to 2048 x 2048 at the size's ratio when smaller, with sides snapped to 32;
   - the requested size;
   - the starting rows: the seed's first draw of torch's CPU `randn` over the latent [24, height / 16, width / 16],
     made natively (`moss_image/noise.zig`) and patchified.
2. `tf-h3-image ... free` runs the six steps from those rows. It writes the latent, and exits 0 with nothing to gate.
3. `tf-h3-image-decoder` decodes the latent. Its rows and columns are the canvas over 16. It resizes the picture back
   to the requested size when that differs from the canvas.

## Edit, end to end

An edit adds two tools before the prompt: the edit encoder makes each reference's latent rows, and the vision tool
makes its rows for the text encoder. The transformer needs an M5; the other tools run on any Apple GPU:

```sh
zig build tf-h3-image-encoder tf-h3-image-vision tf-h3-image-prompt tf-h3-image tf-h3-image-decoder -Dcpu=apple_m1 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-encoder MODEL_DIR reference.png 42 out/ref 2048x2048
zig-out/bin/tf-h3-image-vision TEXT_ENCODER_DIR MODEL_DIR reference.png out/ref.vision.safetensors
zig-out/bin/tf-h3-image-prompt TEXT_ENCODER_DIR MODEL_DIR prompt.txt out/case.safetensors 2048x2048 42 \
  --reference out/ref.vision.safetensors --anchors out/ref
zig-out/bin/tf-h3-image MODEL_DIR out/case.safetensors out/case free
zig-out/bin/tf-h3-image-decoder MODEL_DIR out/case.latent.f32 128 128 out/case.png 2048x2048
```

1. `tf-h3-image-encoder` makes the reference's clean and anchor rows ([Edit encoder](#edit-encoder)). Give it the
   request's seed and its canvas, which the prompt tool prints (a 2048 x 2048 request's canvas is itself). It draws the
   target's noise and then the anchor's, as the edit's one generator does.
2. `tf-h3-image-vision` makes the reference's vision and DeepStack rows ([Vision tower](#vision-tower)).
3. `tf-h3-image-prompt --reference` puts `Picture 1: ` and the reference's vision block before the prompt, runs the
   text encoder with the reference's rows, and tags the block's rows 0. `--anchors` writes the encoder's grid and
   anchor rows as the case's `references` and `anchors`, beside the canvas and the target's starting rows.
4. `tf-h3-image` steps only the target rows, holding the reference's at 0.999. The decoder runs as for text to image.

`--reference` may be given more than once, in upload order, each with its own `--anchors`. The encoder draws each
reference's noise as a one-reference edit does, so only a one-reference edit draws exactly the reference pipeline's
noise.

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

The picture case brings the rest. The prompt tool writes `canvas`, `text`, `text_tags`, `noise`, and with an edit's
`--anchors` `references` and `anchors`; a reference pipeline's case adds its own rows for the gates:

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

**Edits.** Each `--reference` file (from `tf-h3-image-vision`) adds its label, `Picture i: `, and its vision block
before the prompt: vision start, one image pad per token, vision end, all tagged 0. The encoder places the reference's
rows at its image pads and adds its DeepStack rows there after layers 0, 1 and 2. M-RoPE gives the pads (t, h, w) over
the merged grid, and the text after a block resumes past its larger side.

**Gates, fixed before the first run except the whole-run gate below.** `tf-h3-image-prompt selftest [WORK_DIR]`
writes a tiny synthetic checkpoint and loads it as a real run does. The checkpoint is a 256-wide text model with 4
query and 2 KV heads, three layers written and two run, and a 384-wide refiner of 2 heads. The selftest checks 70 rows
against a float64 host reference with the same bf16 rounding points:

- **Each kernel alone:** at least 99.9% of values bit-equal, none further than 1/32 of its row's RMS. Kernels: the
  product, both norms, heads with rotary, fused heads, causal and full attention, both SiLU gates, the residual add.
- **Whole runs:** a text prompt; a reference's vision and DeepStack rows at M-RoPE positions, read from a file as
  `--reference` reads one; the refiner on the GPU's encoder rows. Each must be within twice a float32-sum host run's
  distance from float64, in both 1 - cosine and relative error, with cosine at least 0.999. This gate replaced a fixed
  cosine bound of 0.99999, which the refiner missed at 0.9999884 on a second seed while every kernel alone was
  bit-exact.
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

## Decoder

`tf-h3-image-decoder` turns `tf-h3-image`'s latent into a picture on Metal. It runs the VAE's 2D ViT decoder (24 blocks
of 1,024, 16 heads of 64, 4 register tokens) on any Apple GPU. Its kernels compile from source when it loads and use no
tensor units.

**What runs:**

1. The latent file (normalised, [24, rows, columns] float32) is mapped back with `latents_mean` and `latents_std` in
   float32.
2. It is cut into tiles of 16 x 16 cells (256 pixels) overlapping by at least 64 pixels, as `picture_vae.zig` splits
   it.
3. Each tile runs `post_quant_conv`, then its cells become tokens through `x_embedder`, with the 4 registers and one
   zero token appended.
4. The 2D rotary table is built, and the 24 blocks run, then `norm_out` and `proj_out`.
5. Each token's 768 values are unpatchified to its 16 x 16 pixels. Up to 16 tiles run in one command buffer.
6. Tiles are blended in float32: the top band against the tile above, then the left band against the tile to the left,
   both as decoded.
7. The result is mapped to [0, 1] with ImageNet's statistics, clamped and truncated to bytes.
8. If the request's size differs from the canvas, the picture is resized with Pillow's LANCZOS. Then it is written as
   a PNG.

**Precision:**

- Every parameter is rounded from the file's F32 to bf16 when loaded, and the latent is rounded to bf16 at
  `post_quant_conv`.
- Products are bf16 with float32 sums, the bias added before the one rounding.
- RMSNorm's statistics and weight are float32, rounded once; q and k are RMS-normed with no weight.
- The rotary ids and 2π·id are in bf16 arithmetic, the angles in float32, cos and sin in bf16.
- Rotation, LayerScale, the residuals and SiLU are bf16 elementwise.
- Attention takes bf16 q, k and v, with float32 scores, softmax, probabilities and sums. The reference's fused
  attention may round the probabilities to bf16 before the value product. A reference case will say which.
- `norm_out` is float32; `proj_out` takes bf16 in and gives bf16 out, held in float32. The blend is float32.

```sh
zig build tf-h3-image-decoder -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-decoder MODEL_DIR out/case.latent.f32 ROWS COLUMNS out/case.png [WIDTHxHEIGHT]
zig-out/bin/tf-h3-image-decoder selftest
```

`ROWS` and `COLUMNS` are the latent's grid in cells: the canvas's height and width over 16, so a 2048 x 2048 canvas is
`128 128`. `WIDTHxHEIGHT` is the requested size.

**Selftest.** `selftest` runs the kernels on synthetic weights against a float64 host reference with the same
rounding points, and exits 1 past a bound. On an M3 Ultra:

| Check | Result |
|---|---|
| The embedding and `proj_out` | within 1 ULP |
| Each block, on the GPU's own input | 93-96% identical, worst 4 ULPs |
| The output after two chained blocks | relative error 3e-3 |
| The output after 24 chained blocks | relative error 1.5e-2 |
| Decodes in different batch sizes | bit-identical |
| The picture's bytes | at most 1 apart |

The host's own float32 sums spread as far from its float64 sums as the GPU's do.

All 24 blocks take 13.4 ms a 16 x 16 tile, so about 1.6 s of GPU time for a 2048 x 2048 canvas (121 tiles).
`zig build test-h3-geometry` covers the decoder's host rules. `zig build compile-h3-image-decoder-host` compiles the
tool without the Metal compiler.

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

## Vision tower

`tf-h3-image-vision` makes an edit reference's rows for the text encoder: Qwen3-VL's vision tower with DeepStack, on any
Apple GPU. Its kernels compile from source when it loads, and it uses no tensor units.

```sh
zig build tf-h3-image-vision -Dcpu=apple_m1 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-vision TEXT_ENCODER_DIR MODEL_DIR reference.png out/ref.vision.safetensors
zig-out/bin/tf-h3-image-vision selftest [WORK_DIR]
```

**What runs:**

1. The picture (PNG, JPEG or WebP) is read upright in RGB. Its Qwen copy has a short edge of exactly 1,024: the
   original when it already has one, else both sides scaled, snapped to 32 (halves to even) and resampled with Pillow's
   LANCZOS (`picture_prompt.zig`'s rule, in `qwen_copy.zig`).
2. The processor in `MODEL_DIR/processor/preprocessor_config.json` (Qwen2VLImageProcessorFast) cuts it into patches
   (`core/picture/patches.zig`): a smart resize to multiples of 32 inside its pixel bounds, a no-op on a snapped copy;
   bytes / 255, then mean and std 0.5; patches of 16 x 16 with the frame repeated, in merge order. A 1024 x 1024
   reference gives 64 x 64 patches and 1,024 tokens.
3. The tower is core's Qwen-VL tower (`core/qwen_vl`), the one Qwen3.5 serves with:
   - the patch projection, plus four bilinear taps of the learned 48 x 48 position grid;
   - 27 blocks of 1,152 with 16 heads of 72, a 2D axial rotary and a gelu_pytorch_tanh MLP of 4,304;
   - the merger: LayerNorm per patch, the 2 x 2 patches side by side, then fc1, exact GELU and fc2 to 5,120.
4. **DeepStack.** After blocks 8, 16 and 24 (`deepstack_visual_indexes`), each tap's own merger reads the residual.
   Its LayerNorm runs after the shuffle, over a merged row's 4 x 1,152 = 4,608 values; then fc1, exact GELU and fc2
   to 5,120, each with its bias.

It writes `grid` I32 [3] (1, rows and columns in patches), `rows` BF16 [tokens, 5,120] and `deepstack` BF16 [3,
tokens, 5,120], aligned. `tf-h3-image-prompt --reference` reads it.

**The checkpoint.** `TEXT_ENCODER_DIR` is MiniMax-H3's `text_encoder/`, an HF `Qwen3VLForConditionalGeneration`. The
tower's prefix is whatever precedes `blocks.0.norm1.weight` in its index, such as `model.visual.`. The loader reads
exactly the 351 bf16 tensors of a Qwen3-VL tower with three DeepStack mergers (`deepstack_merger_list.{0,1,2}` with
`norm`, `linear_fc1` and `linear_fc2`), and refuses a checkpoint with more or fewer. The patch kernel is read in
torch's [1,152, 3, 2, 16, 16] layout. These names and shapes come from the HF model's code and config. No header of the
released checkpoint has been read here, so the first real load is their check.

**Precision.** As for Qwen3.5's images:

- bf16 weights and patches;
- products in simdgroup matrices with fp32 sums, each rounded once to bf16;
- the residual stream in fp32, where the reference's bf16 tower keeps bf16;
- LayerNorm with fp32 statistics and eps 1e-6, rounded once;
- q and k turned in fp32, then rounded;
- fp32 scores, softmax and sums.

**Qwen3.5 is unchanged.** DeepStack is an opt-in. `Config.parse` still refuses a tower with taps, and only
`Config.parseDeepStack` reads them. A tower without taps loads the same tensors under `vision_tower.`, into the same
buffer, and runs the same kernels in the same order.

**Selftest, gates fixed before the first run.** `selftest` writes a tiny Qwen3-VL tower in HF's names and layouts under
`model.visual.`, beside a text-model tensor the loader must leave alone:

- 4 blocks of 144, with 2 heads of 72 and an MLP of 272;
- a 64-position grid, a 2 x 2 merge to 200, and DeepStack taps after blocks 0, 2 and 3.

It parses its config and processor, finds its prefix and loads it as a run does. A 288 x 192 picture goes through the
processor's patches: 12 x 18 patches, 54 tokens. A float64 host reference runs the whole tower at the same bf16
rounding points: the patch projection, the learned grid's taps, the rotary, attention, the MLP, the merger and each
DeepStack merger. The gates:

- **Each output** (the merged rows and each tap's rows): within twice a float32-sum host run's distance from float64,
  in both 1 - cosine and relative error, with cosine at least 0.999.
- **A second run:** byte for byte the same.
- **No taps asked:** the same merged rows.
- **The Qwen3.5 path:** the same tower as MLX holds a Qwen3.5 one (`vision_tower.*`, the patch kernel [O, T, Y, X, C],
  no DeepStack), loaded by `Tower.load` and run by `image()`, gives the same merged rows.

On an M3 Ultra:

| Output | Cosine | Relative error | Float32-sum host run |
|---|---|---|---|
| The merged rows | 0.9999825 | 5.92e-3 | 0.9999828, 5.87e-3 |
| DeepStack after block 0 | 0.9999949 | 3.20e-3 | 0.9999950, 3.17e-3 |
| DeepStack after block 2 | 0.9999838 | 5.69e-3 | 0.9999832, 5.80e-3 |
| DeepStack after block 3 | 0.9999841 | 5.64e-3 | 0.9999837, 5.71e-3 |

Each output sits 0.96 to 1.02 times as far from float64 as the float32-sum run. The second run, the untapped run and
the Qwen3.5 path are byte-identical. The gates passed on the first run, unchanged.

`zig build test-h3-image-prompt` also runs the tower's host tests: both parses of the config, the 351-tensor plan
beside the 333 of Qwen3.5's, the prefix search, the scratch plan and the reference's norm and GELUs.
`zig build compile-h3-image-vision` compiles the tool without running a shader.

## Checkpoint

`MiniMaxAI/MiniMax-H3-Image`. `transformer/` is bf16, with the patch projection, time embedder and step heads in
float32. The text encoder is MiniMax-H3's Qwen3-VL; its 50th layer's rows are the prompt's. The weights are under the
MiniMax-H3-Image Community License, which excludes the European Union, the United Kingdom, the Republic of Korea and
the United States of America. Read it before use.

## Not done

- **Checked against the release:** nothing yet. The reference cases and the gates above wait for the weights.
- **Edits:** they run end to end in five tools, not one. With several references, each reference's noise is drawn
  as a one-reference edit draws it, not on the edit's one chain. The vision tower's tensor names are inferred, and its
  speed at the released size is not measured here.
- **Serving:** no `/v1/images/generations` or `/v1/images/edits` route for it, and no CUDA path.
- **Machines:** the transformer runs on an M5 only; chips without tensor units are refused.

## Host checks

`zig build test-h3-geometry` runs the picture rules without a GPU:

- the canvas, including 1280x720 drawn at 2720x1536 and 4000x3000 at 4000x3008;
- the schedule's float32 round trip and its refusals;
- positions, the rotary layout, the packing and the tile map;
- the model folder's checks;
- the case reader's picture admission.

`zig build compile-h3-host` compiles both H3 tools without the Metal compiler.

## Kernel contract

The edit encoder's, prompt path's and image decoder's kernels are registered with the kernel contract
([kernel-contract.md](../kernel-contract.md)): 31 registrations with 34 planted controls, and every launch's limits.
They are checked against float64 at their rounding points, beside each tool's own `selftest`. `zig build
h3-image-contract` runs them on any Mac, with no Metal compiler; `tf-kernel-contract` runs them with every other kernel.
On an M3 Ultra all 31 pass gates 1 to 4 and every control goes red as planted.

The contract found one fault, now fixed: the prompt path's attention filled a row's last block of 4 keys with its last
key at weight 0, so an infinite value there gave 0 x Inf = NaN where float64 keeps the infinity. A padded key now adds
nothing; finite outputs keep their bytes.
