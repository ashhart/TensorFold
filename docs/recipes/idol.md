# Idol on Metal

Idol (speridlabs) turns one image into a 3D Gaussian splat. A 0.37B flow transformer denoises 8,192 latent rows of 16
channels, conditioned on the image's DINOv3 and FLUX.2 VAE rows; two decoders then grow an octree from that latent and
emit the Gaussians. `zig/src/families/idol` runs the denoiser and both decoders on the Metal runtime from the released
checkpoint ([Decoders](#decoders) below). The denoiser:

- once an image: the condition embedding (DINOv3's 1,280 channels and the VAE's 128, each projected to 1,024 and
  summed) and the two condition-refiner blocks (no modulation, affine LayerNorms);
- at each step: the two noise-refiner blocks over the latent rows, then the 24 joint blocks over 12,294 rows (8,192
  latent, 4,101 condition, one camera token), then the output head and the camera velocity.

Every block is 16 heads of 64 at width 1,024 with a tanh-GELU MLP of 4,096. Its rotary is RePo: a learned map from
each row's LayerNorm to per-head 3D positions, angles from 10, 10 and 12 frequencies per axis.

The sampler's defaults: 50 Euler steps, `t = 3u / (1 + 2u)`, guidance `v = 4 v_cond - 3 v_uncond` (two passes a step:
with the condition, and with zeros).

Both image encoders run natively too: the FLUX.2 VAE's ([below](#vae-encoder)) and DINOv3's
([below](#dinov3-encoder)). `tools/zig/idol_case.py` runs the publisher's SDK in torch, a development reference.
It is a denoiser, not a token family: it does not register with the lane engine, the server or `families.detect`.

It runs on the M5's tensor units alone, the only chip it was checked on: `Model.load` refuses a GPU without them.

## Checkpoint

`speridlabs/idol` at `2731c2291fc65a797c5cb627a5e5526fc1490902`, Apache 2.0, ungated. `config.json` must name
`LatentSeqMMFlowModel` at the released geometry, and `denoiser.safetensors` must hold its 560 tensors; anything else is
refused. The reference is idol-sdk at `4702a7f00850e34b6e83a2170cc123c3e9570744` (Apache 2.0).

## Precision

The SDK loads the checkpoint's fp32 weights and converts the blocks' linear layers to fp16; everything else stays fp32.
This family keeps that split:

- fp16 block projections with fp32 sums; the bias joins the sums before one fp16 rounding, GELU after it.
- LayerNorms in fp32, rounded to fp16; the modulation's shift and scale in fp16 ops, as torch's fp16 tensors run them.
- RePo in fp32: its LayerNorm, both 1,024 to 128 maps, the 128 to 48 map and the angles.
- Attention: fp16 q, k and v, fp32 scores and sums. q and k are rotated, then L2-normalized and scaled by the
  checkpoint's gamma times 8. The tensor units take an fp32 P short, which left outputs 3.5e-4 low on average, so
  `id_attention` rounds P to fp16 itself and sums the softmax's denominator over those same values.
- The condition embedding and the output projection in fp32; the time MLP, modulation and camera MLPs on the host.

### The tensor units' fp32 matmul is not fp32

On the M5, a GEMM whose inputs are fp32 on the tensor units (`tfp::gemm_nax<float>`, the Qwen-VL tower's
`qwv_gemm_f32`, MLX's fp32 matmul) truncates each input to 10 mantissa bits, toward zero (#1772): its error against
float64 is about 7.5e-4 of the result, against about 1e-7 for an fp32 FMA. RePo's maps at that precision move
block 8's attention out of its gate.

`id_norm_split` therefore writes each RePo input row as fp16 `[hi | hi | lo]` (`lo = x - hi`), and the map's weights are
stored `[hi | lo | hi]`, scaled by a power of two so `lo` stays normal. One fp16 split-K GEMM with K = 3,072 then sums
`hi*hi + hi*lo + lo*hi` in fp32; `id_repo` undoes the scale. With it every block passes its gate. `id_gemm32`,
an fp32 FMA GEMM on simdgroup matrices, is exact but takes 0.47 s a forward if it runs RePo; it serves the condition
embedding (once an image) and the output projection.

## Kernel contract

`zig/tests/contract/metal_idol*.zig` registers every kernel of `idol.metal` and `idol_attention.metal`:

- `id_linear`'s eight forms (plain and GELU, each tile tail) and the split-K halves `id_split_0` and `id_split_1`;
- `id_norm` (plain, affine, modulated), `id_norm32` (affine, modulated) and `id_norm_split`;
- `id_repo`, `id_qk`, `id_gate_add`, `id_embed`, `id_cond`, `id_bias32` and `id_gemm32`;
- the attention's four forms.

Each output must lie in the range its fp32 steps allow at the SDK's fp16 rounding points. Nineteen controls drop or
add one rounding point or sum in fp16, and each must go red at its gate. One runs the tensor units' fp32 GEMM in
`id_gemm32`'s place: gate 4 misses it by 13 million times the exact-sum bound. `id_qk` keeps a NaN norm as
`F.normalize` does. Forms no engine call reaches (`id_linear_00` and `_10`, attention `01` and `10`, plain `id_norm`,
affine `id_norm32`) are checked at made-up shapes. The tensor-unit kernels do not build for M1 or M2 offline, as
MOSS-Image's do not; the rest build at their reserved thread counts.

`tf-kernel-contract --only "metal idol"` on the M5 Ultra:

```
SUMMARY 27 kernels: 27 pass gates 1-4, 0 fail, 0 known findings; 19 controls: 19 red as planted, 0 missed; 0 limit problems
```

## Run a case

`idol_case.py` imports the SDK's `dit.py`, `layers.py` and `sampler.py` only when each file's sha256 is the pinned
commit's, and the snapshot only when it is the pinned revision.

```sh
hf download speridlabs/idol --revision 2731c2291fc65a797c5cb627a5e5526fc1490902 --local-dir WEIGHTS
git clone https://github.com/speridlabs/idol-sdk SDK && git -C SDK checkout 4702a7f00850e34b6e83a2170cc123c3e9570744
python3 tools/zig/idol_case.py case WEIGHTS SDK case/ --features dino.safetensors vae.safetensors
python3 tools/zig/idol_case.py exact WEIGHTS SDK case/             # the same steps with an fp32 denoiser
python3 tools/zig/idol_case.py blocks WEIGHTS SDK case/ --step 0   # each block's input and output, fp16 and fp32
zig build tf-idol -Dcpu=apple_m1
zig-out/bin/tf-idol WEIGHTS case/
zig-out/bin/tf-idol WEIGHTS case/ blocks 0
```

The features are the SDK's own `feature1` [1, 4101, 1280] and `feature2` [1, 4101, 128] for an image. The case holds the
seeded noise (torch's CPU generator), the schedule, the SDK's input, both velocities and camera velocities at every
step, and its final sample. `exact` adds fp32 velocities on the same inputs; `blocks` adds every block's input and
output at one step, and each joint block run alone in fp32 on the SDK's fp16 input.

`tf-idol` gates:

- **Each step**, both branches on the SDK's own input: cosine against the fp32 velocity within 2x the SDK's own
  distance (`1 - cos`). Worst row over the run within 2x the SDK's worst row. Fifty fp16 steps make single rows chaotic,
  so the per-row gate runs per block instead:
- **Each block** alone on the SDK's fp16 input, against that block in fp32: worst row and cosine within 2x the SDK's.
- **Repeat**: the same step twice, another between, byte for byte.

It then runs the whole guided schedule from the noise, times it, and writes `sample.f32`. `reference.zig`'s tests plant
a 5% error in one row and require the gate to fail it.

## Measured

Setup:

- Mac Studio M5 Ultra 256 GB, macOS 27; this commit's `tf-idol`.
- The SDK on torch 2.13.0 (MPS, fp16 blocks, `ATTN_BACKEND=sdpa`).
- The SDK's avocado example: its own DINOv3 and VAE features, seed 42, 50 steps, guidance 3.

Results:

- **Velocity**, both branches of all 50 steps on the SDK's own input (100 of 100 pass):
  - against the SDK: cosine 0.999998 or better;
  - against fp32: our worst row over the run 0.0474, the SDK's 0.0398; per step our `1 - cos` is the SDK's within
    the 2x bound at every step, and the smaller at 75 of the 100.
- **Blocks**, at steps 1 and 41, each of the 23 joint blocks after the first on the SDK's input: all pass, worst rows
  0.0008 to 0.0017 against fp32, where the SDK's are 0.0007 to 0.0025.
- **Sample:** our 50 guided steps from the same noise end at cosine 0.999993 to the SDK's sample.
- **Repeat:** identical bytes.
- **Speed**, one session: 0.275 s a forward (27.5 s for 50 guided steps), 3.4x the SDK's 0.939 s (the median of its
  100). Load 0.2 to 1.6 s; the condition, once an image, 0.07 s.
- **Memory:** 1.84 GB peak process footprint (`/usr/bin/time -l`, the whole `tf-idol` run with its case), the
  denoiser's fp16 weights in shared Metal buffers.

## Decoders

`IdolPipeline.decode` turns the denoiser's final latent [8192, 16] into Gaussians. Both decoders run on Metal from
`decoder.safetensors` and `decoder_gs.safetensors`; the steps between them run on the host
(`zig/src/families/idol/decode.zig`):

- **Octree decoder** (4 blocks): eight levels, each from the occupied parent cells' centres to 8 child logits. Each
  block is cross attention from the cells to the latent, then the MLP, both modulated by the level's embedding
  (`adaLN_modulation(l_embedder(2^level))`, computed once on the host). The host takes the softmax, splits each
  parent's points among its children by `systematic_sample`, and keeps the children that got points. After level 8
  each leaf becomes its count of anchors, jittered inside the cell.
- **Gaussian decoder** (16 blocks): self attention over the anchors, cross attention to the latent, then the MLP.
  Each anchor gets 480 outputs: 32 Gaussians of offset, colour, scale, rotation, opacity and offset scale.
- **Host**: `gaussian.py`'s activations, the 3DGS PLY and the camera JSON (`gaussian.zig`).

A request asks for 32,768 to 1,048,576 Gaussians, 32 an anchor. `Decoder.load` sizes its scratch for the largest it
will serve.

### Precision

The SDK converts the blocks' linear layers to fp16. The stem, the head and the latent's rows before the blocks stay
fp32. This port keeps that split:

- Stem in fp32: `in_proj` of the 3 coordinates as one FMA chain, plus the point embedding (`dec_embed`: sin and cos
  of torch's fp32 angle `fp32(fp32(x f) pi)`, the frequencies `torch.pow`'s on the CPU, bit for bit). Then
  `input_layer` by the denoiser's `id_gemm32` and its bias, rounded once to fp16 (`dec_half`).
- Blocks in fp16, through the denoiser's kernels: `id_linear` (fp32 sums, bias, one fp16 rounding, GELU after it),
  `id_norm` and `id_gate_add`.
- Cross attention: `dec_kv` computes `to_kv` of the fp16 latent as 16-deep fp32 FMA sums, then the bias, rounded to
  fp16. `dec_heads` applies MultiHeadRMSNorm in fp32 (`F.normalize`, times gamma times 8) and rounds to fp16.
- Attention:
  - Scores and sums are fp32. P meets V on the tensor units as two fp16 halves, `hi = fp16(P)` and
    `lo = fp16(P - hi)`, so P keeps about 22 bits. A single fp32 P would lose bits on the units: each pass would come
    out about 2^-12 low, and the blocks would compound it.
  - At 8 query rows or fewer (the octree's first levels), torch on MPS runs its fp32 vector kernel, and
    `dec_attention_few` runs fp32 throughout as well.
- Head: the denoiser's `id_norm32` LayerNorm on the stream cast to fp32, `id_gemm32` and `id_bias32`.
- Host: softmax and resampling in fp32, as `systematic_sample` runs them.

### Gates

`tf-idol-decode` gates a case from `idol_case.py decode` on the SDK's own inputs:

- **Counts.** Every octree level's split, run on the SDK's probabilities and draws, must give the SDK's counts row for
  row.
- **Probabilities and raw rows.** Each level's probabilities and the Gaussian decoder's rows are compared against the
  same decoder in fp32 on the same inputs. They must stay within twice the worst of three distances:
  - the SDK's own distance to fp32;
  - the fp16 floor: fp32 with only its last rows rounded once;
  - the SDK rerun with 1% of its first block's fp16 inputs nudged one ulp, three seeds.

  Including the nudged reruns matters. They show how far the SDK itself moves under a change no fp16 stream can
  avoid. At level 8, one ulp on 1% of the inputs moves the SDK's worst row from 0.0415 to 0.0488, more than the gap
  between this port and the SDK.
- **Blocks.** Levels 1 to 3, block by block on the SDK's fp16 rows: each block's output and its residual update,
  within twice the SDK's own distance to fp32.
- **Output.** The activated fields to 1e-5 relative plus 1e-6, the PLY's header, rows and columns against the SDK's
  `reference.ply`, and the camera.
- **Repeat.** A second decode on the SDK's draws, and two on the native generator, must give identical bytes.

The free run on the SDK's draws is reported, not gated. Its topology leaves the SDK's at level 6 of the 262k case (5
of the 32k case), and that is expected:

- `systematic_sample` is discontinuous. A stratified draw lands on one side of a cumulative-probability edge or the
  other, so a probability 1e-5 off can move one point to a sibling.
- Fed our own probabilities, a few rows a level differ by one point: 1 row at level 1, 20 rows at level 8.
- Once a child's count reaches or leaves zero, the set of occupied cells changes, and every later level follows.
- The counts gate isolates this: given the SDK's probabilities, every count matches.

### Run a case

`idol_case.py decode` decodes a denoiser case's final sample with the SDK. It imports `decoder.py`, `gaussian.py` and
`camera.py` only at their pinned sha256. It records each level's parents, probabilities, draws and counts, the anchors,
the raw rows, the activated Gaussians, the PLY and the camera. `--ruler` adds the fp32 runs, the fp16 floor, the
nudged reruns and levels 1 to 3 block by block.

```sh
python3 tools/zig/idol_case.py decode WEIGHTS SDK case/ decode262k/ --gaussians 262144 --ruler
zig build tf-idol-decode -Dcpu=apple_m1
zig-out/bin/tf-idol-decode WEIGHTS decode262k/                   # the gates; writes .json, .ply, camera.json
zig-out/bin/tf-idol-decode WEIGHTS case/ time 262144 out.ply     # the native generator, cold then warm
```

### Measured

Setup:

- The same Mac and SDK as above.
- The avocado case's final sample, seed 42.
- 262,144 Gaussians unless noted.

Results:

- **Counts:** identical at all 8 levels, given the SDK's probabilities and draws.
- **Octree probabilities**, worst row against fp32: level 1 0.00019 (the SDK 0.00009), level 8 0.0453 (the SDK
  0.0415, nudged 0.0488).
- **Gaussian decoder**, worst row against fp32: 0.133, `1 - cos` 4.5e-6. The SDK's are 0.211 and 8.1e-6; its nudged
  reruns reach 0.195. At 32,768 Gaussians this port's worst row is 0.0213 and the SDK's 0.0072, within the bound
  (nudged 0.0305).
- **PLY:** header and all rows equal to the SDK's. Opacity and scale columns are within 1.9e-6; the other columns are
  bit-equal.
- **Repeat:** identical bytes.
- **Speed and memory**, warm, `tf-idol-decode ... time` against the SDK's warm decode. The footprint is the peak
  process footprint (`/usr/bin/time -l`).

  | Gaussians | ours | SDK | our peak footprint | SDK's |
  | ---: | ---: | ---: | ---: | ---: |
  | 32,768 | 0.071 s | 0.170 s | 1.06 GB | 2.97 GB |
  | 262,144 | 0.343 s | 0.739 s | 1.33 GB | — |
  | 1,048,576 | 1.963 s | 4.653 s | 2.43 GB | 5.64 GB |

  At 1M, the Gaussian decoder's self attention over 32,768 anchors takes most of the time.

### Kernel contract

`zig/tests/contract/metal_idol_decoder*.zig` registers the 9 pipelines of `idol_decoder.metal`, named
`metal idol decoder ...`, so `--only "metal idol"` runs them with the denoiser's:

- `dec_embed`, `dec_half`, `dec_kv` and `dec_heads`;
- the attention's four tiled forms and `dec_attention_few`.

Form `10` is never served, because every call has Lq = Lk or Lk = 8,192; it is checked at made-up shapes.

Gate 3 compares each step against float64 at the SDK's fp32 and fp16 points, as a range over its fp32 steps.
Attention's allowance is:

- 2^-17 of `sum P |v| / l`, for the fp32 scores, exp2 and sums;
- each P's lo-half floor (2^-25, or P where P is smaller) times |v|;
- O's own fp16 rounding.

Half the heads are built so O is P's fp16 remainder alone. On those heads, a P V without P's lo half, or with P rounded
to fp16, gives 0. Fifteen controls must each go red at their gate:

- **Shapes:** add an fp16 rounding point (`pe`, `y` before the bias, `ctx w` before the bias, `x / n`), drop P's lo
  half, round the short form's P to fp16, leave keys past Lk unmasked.
- **Sums:** sum in fp16 (`in_proj`, `to_kv`, the heads' squares, P V in the short and the tiled forms).
- **Repeat:** `dec_half` writing its input.
- **Nonfinite:** the two NaN and infinity rules below, undone.

Gate 1 found two rules the first version broke, and both kernels are fixed:

- `dec_heads` clamped its norm by `max`, which drops a NaN; it now keeps it, as `clamp_min` does.
- The tiled attention gave NaN for an infinite value: an fp16 `lo` of 0, or of the opposite sign, times infinity. The
  lo half now meets the values clamped to fp16's finite range, while the hi half still carries the infinity.
- Finite outputs are unchanged. The 262k and 32k cases give byte-identical PLYs and gate lines before and after.
- The clamp costs about 2% at 262k and 6 to 7% at 32k and 1M.

The tensor-unit kernels do not build for M1 or M2 offline; the rest build at their reserved thread counts.

`tf-kernel-contract --only "metal idol"` on the M5 Ultra, the denoiser's 27 kernels and 19 controls with these:

```
SUMMARY 36 kernels: 36 pass gates 1-4, 0 fail, 0 known findings; 34 controls: 34 red as planted, 0 missed; 0 limit problems
```

## VAE encoder

`zig/src/families/idol/vae*.zig` (with `idol_vae.metal`) builds the denoiser's `feature2` [4101, 128] from the 1,024 x
1,024 picture `IdolPipeline.preprocess` returns, the way the SDK's `ImageEncoder._flux2_tokens` does:

- FLUX.2's VAE encoder: `conv_in`, four `DownEncoderBlock2D` stages of two ResNet blocks (128, 256, 512, 512 wide;
  GroupNorm 32 and SiLU) with a stride-2 3x3 downsample padded right and below after the first three, the mid block
  (ResNet, single-head attention of 512 over the 16,384 positions, ResNet), `conv_norm_out`, SiLU, `conv_out` and
  `quant_conv`, on `2 x - 1` of the picture;
- the posterior's **sample**, not its mean: `mean + exp(logvar / 2) * noise` with logvar clamped to [-30, 20]. Some
  channels' standard deviations reach 1, so the noise matters. The SDK draws it from torch's MPS generator in bf16 after
  `torch.manual_seed(seed)`, which `encode` cannot reproduce; it takes the noise [128, 128, 32] as an input, and the
  case records the SDK's;
- each 2 x 2 patch of the 32 latent channels as channel `c * 4 + dy * 2 + dx` (128 channels on a 64 x 64 grid), less
  the batch norm's `running_mean` over `sqrt(running_var + 1e-4)`;
- five zero rows, then the 4,096 patches in row order: 4,101 rows.

### Weights and precision

`black-forest-labs/FLUX.2-klein-4B` `vae/` at `e7b7dc27f91deacad38e78976d1f2b499d76a294` (Apache 2.0, ungated, bf16).
The SDK loads the VAE with `torch_dtype=torch.bfloat16`. Klein's file is FLUX.2-dev's fp32 VAE rounded to bf16, so both
give the same weights. The loader takes either: it rounds an fp32 file to bf16 as torch does. It reads the 106 encoder
tensors, `quant_conv` and the two batch-norm statistics, and refuses any other geometry.

The SDK then runs the whole encoder in bf16. This port computes in fp32 on those bf16 weights: 3x3 convolutions by
Winograd F(2x2, 3x3) with `G g G^T` made in float64 and rounded once, the stride-2 downsamples and 1x1s as implicit
GEMMs, GroupNorm statistics in two passes (the mean, then the squared deviations), fp32 attention, `precise::exp`. None
of it uses the tensor units, whose fp32 GEMM truncates its inputs to 10 mantissa bits (above).

### Gates

```sh
hf download black-forest-labs/FLUX.2-klein-4B --revision e7b7dc27f91deacad38e78976d1f2b499d76a294 --include 'vae/*' --local-dir KLEIN
python3 tools/zig/idol_case.py vae KLEIN/vae SDK picture.png case/ --stages
zig build tf-idol-vae -Dcpu=apple_m1
zig-out/bin/tf-idol-vae KLEIN/vae case/ [features.safetensors]
```

`idol_case.py vae` imports `idol/conditioning.py` and diffusers 0.42.1's VAE files only when their sha256 values are
the pinned ones, and the weights only when their sha256 is klein's or FLUX.2-dev's. It runs the SDK's own
`_flux2_tokens` on MPS, seeded as `IdolPipeline` seeds it, and records the noise it draws. Then it runs the same steps
in float64 on the CPU with that noise: the ruler, from the exact picture rather than its bf16 rounding. `--stages` adds
float64's output of each down block, the mid block and `quant_conv`.

`tf-idol-vae` gates our rows against float64. Worst row (the largest relative L2 error of a row) and `1 - cosine`
must both be within twice the SDK's own distance, and the padding rows exactly zero. Five more encodes must repeat the
first byte for byte. `vae.zig`'s test plants a 2% error in one row, then a non-zero padding value, and requires the
gate to fail each.

### Kernel contract

`metal_idol_vae*.zig` registers all eight kernels in 13 forms, each against float64: `iv_conv_in`, `iv_conv` (1x1,
1x1 plus residual, stride-2 3x3), `iv_wino` (plain, plus residual), `iv_norm` (SiLU, plain), `iv_group_part` (sums,
squared deviations), `iv_group_finish`, `iv_attention` and `iv_tokens`.

There are twelve controls:

- inputs staged in half (`iv_conv_in`, the 1x1, Winograd's patch);
- symmetric padding in the stride-2 3x3;
- a dropped term of Winograd's output transform;
- a half SiLU;
- half sums (Winograd, both GroupNorm passes), caught at gate 4;
- attention without its rescale;
- swapped patch axes;
- a NaN logvar lost to Metal's `clamp`. `torch.clamp` keeps NaN, so `iv_tokens` does too.

Winograd's float64 reference is its own transform, so a non-finite input reaches its whole 2 x 2 tile. The pictures
and 1x1 row counts are smaller than the engine's, and rows only tile. These kernels use no tensor ops; their records
reserve their thread counts in the source.

```
SUMMARY 13 kernels: 13 pass gates 1-4, 0 fail, 0 known findings; 12 controls: 12 red as planted, 0 missed; 0 limit problems
```

### Measured

On the M5 Ultra, with the avocado example the SDK preprocessed, seed 42:

| against float64 | worst row | 1 - cosine |
| --- | --- | --- |
| ours | 5.1e-5 | 5.0e-11 |
| the SDK (bf16, MPS) | 2.73e-2 | 5.05e-5 |

- **Gate:** passes, by about 500 times on worst row.
- **Against the SDK:** cosine 0.999949, worst row 0.0275. The case's SDK rows equal the SDK's own recorded
  `feature2` for this picture bit for bit, so its noise is the pipeline's.
- **Stages:** each down block, the mid block and the moments sit within 4e-5 (worst pixel) of float64.
- **Repeat:** identical bytes.
- **Speed:** 0.185 s an encode (median of five, 0.185 s of it on the GPU), where the SDK takes 0.095 s on torch MPS
  (median of five, after a first call at 0.19 s). The SDK's bf16 runs on the tensor units; this fp32 path runs at 9
  to 14 TFLOP/s, and the attention, which reads K and V once every 8 queries, takes about 40 ms of it. Load: 0.3 to
  0.4 s.
- **Memory:** 3.28 GB peak footprint (`/usr/bin/time -l`, the whole `tf-idol-vae` run with its case and stage
  copies), 2 GB of it four 512 MB picture tensors for the 1,024 x 1,024 x 128 stage.
- **FLUX.2-dev's fp32 file** in klein's place gives the same rows bit for bit, as rounding it to bf16 should.

## DINOv3 encoder

`dino*.zig` (kernels in `idol_dino.metal`) turns the SDK's 1024 x 1024 preprocessed image into the denoiser's
`feature1` [4101, 1280]: DINOv3 ViT-H+/16's class token, four registers and 64 x 64 patches. idol-sdk's `ImageEncoder`
(`conditioning.py`) loads `facebook/dinov3-vith16plus-pretrain-lvd1689m` through transformers' `DINOv3ViTModel` in
bf16 (`dinov3_dtype='bf16'`, the pipeline's default), normalizes the pixels with ImageNet's mean and std in fp32, and
returns `F.layer_norm(last_hidden_state.float())`: the model's own final LayerNorm (affine, bf16), then a plain one in
fp32 (eps 1e-5, no affine).

- Patch embedding: the 16 x 16 conv as one GEMM, [4096, 768] patches (channel, row, column) against [1280, 768].
- 32 blocks of width 1,280, 20 heads of 64; LayerNorm (eps 1e-5), attention, LayerScale; LayerNorm, SwiGLU MLP of
  5,120 (`silu(gate) * up`; gate and up run as one 10,240-column GEMM), LayerScale.
- RoPE on the 4,096 patches alone, not the five prefix rows: transformers' periods `1 / 100^(4i / 64)` for i < 16,
  generated in fp32 (not the bf16 periods of Meta's checkpoint, which timm's card mentions); patch centres in
  [-1, 1]; angles `2 pi y f` then `2 pi x f`, tiled twice, in `rotate_half` form; cos and sin in fp32, then bf16. The
  host builds these tables at load; they equal torch's bit for bit (524,288 of 524,288 values).

### Weights and licence

`timm/vit_huge_plus_patch16_dinov3.lvd1689m` at `dd0addc09788111fa893d24799a44744ba022eee`, ungated: `config.json`
(`vit_huge_plus_patch16_dinov3`, 1,280 features) and `model.safetensors` (sha256
`0d54131c846c91a78ec7641482f32e45b27cb4895f83653c038f305f18c7e61c`, 486 fp32 tensors), each rounded to bf16 at load
(ties to even, as `from_pretrained(dtype=bfloat16)` rounds). **These weights are under Meta's DINOv3 License, not
Apache 2.0**: its `LICENSE.md` ships with them, and using or redistributing them or a derivative binds you to it.

The SDK reads Meta's gated checkpoint. `idol_case.py dino --official DIR` checks that timm's is the same model: the
names map as transformers' converter maps them (`cls_token`, `reg_token`, `patch_embed.proj`, `norm` to `embeddings.*`
and `norm.*`; `blocks.N.attn.qkv` split in q, k, v thirds; `attn.proj` to `attention.o_proj`; `gamma_1`, `gamma_2` to
`layer_scale1`, `layer_scale2`; `mlp.fc1_g`, `fc1_x`, `fc2` to `gate_proj`, `up_proj`, `down_proj`). On the avocado:

- all 550 mapped tensors (the 32 fused qkv weights as 96 thirds) equal Meta's byte for byte; Meta's q and v biases
  are all zero, and its only other unmapped tensor is the masked-input `mask_token`;
- feature1 from timm's weights in that graph equals feature1 from Meta's `from_pretrained` bit for bit on MPS, and
  both equal the features the SDK pipeline recorded for the image.

### Precision

Where torch on MPS rounds, measured on random bf16 operands against float64 (the share of outputs each candidate
matches): a linear or the conv rounds once, `bf16(x w + b)` (99.98%; rounding before the bias, 69%); nn.LayerNorm
once (99.995%); `silu(g) * u` twice; the rotary three times (`bf16(bf16(q cos) + bf16(rotate_half(q) sin))`); and
LayerScale with the residual twice (`bf16(x + bf16(g y))`), each 100%. The kernels round at those points, with fp32
sums and fp32 statistics.

Attention: torch's bf16 SDPA on MPS matches no candidate scheme (fp32 or bf16 P, its denominator before or after,
bf16 scores) and sits about three times a bf16 rounding of the exact output from it. `dn_attention` keeps fp32 scores
and sums; the tensor units take P in V's type, so it rounds P to bf16 itself and sums the denominator over those
values; the output rounds to bf16.

### Kernel contract

`zig/tests/contract/metal_idol_dino*.zig` registers every kernel of `idol_dino.metal`: `dn_linear_0` and
`dn_linear_1` (the 4,101 rows and the 4,096 patches), `dn_norm`, `dn_final`, `dn_heads`, `dn_scale_add`, `dn_swiglu`
and `dn_attention`, each at transformers' bf16 points. Eleven controls each break one: a bf16 rounding added or
dropped (gate 3), fp16 sums in `dn_norm` and `dn_final`, P V rounded to bf16 each key block (gate 4), the attention's
tail keys unmasked.

```
SUMMARY 8 kernels: 8 pass gates 1-4, 0 fail, 0 known findings; 11 controls: 11 red as planted, 0 missed; 0 limit problems
```

### Run it

```sh
hf download timm/vit_huge_plus_patch16_dinov3.lvd1689m --revision dd0addc09788111fa893d24799a44744ba022eee --local-dir DINO
python3 tools/zig/idol_case.py dino DINO SDK case/ --image preprocessed.png [--official META_DIR] [--sdk-features F]
zig build tf-idol-dino -Dcpu=apple_m1
zig-out/bin/tf-idol-dino DINO case/
```

The image is the SDK's 1024 x 1024 preprocessed image (`IdolOutput.preprocessed_image`). The tool pins idol-sdk's
`conditioning.py` and transformers 5.22.1's DINOv3 files by sha256, the weights by theirs, and writes `dino.safetensors`
(the pixels, the SDK's feature1 on MPS, a float32 CPU ruler's, last_hidden_state and the RoPE tables),
`dino_blocks.safetensors` (torch's rows into each block) and `dino_blocks_f32.safetensors` (the embedding, each block
and the last norm in float32 on torch's own input rows). The ruler runs on the CPU: the M5's fp32 GEMM is not fp32.

`tf-idol-dino` gates, each against the float32 ruler within twice torch bf16's own distance, worst row and `1 - cos`:
feature1 from the image; the embedding; each block alone on torch's input; the last norm alone. The encode must repeat
byte for byte. `dino_reference.zig`'s test plants a 2% error in one row and requires the gate to fail it.

### Measured

The avocado example, the same machine and torch as above (`ATTN_BACKEND=sdpa`):

- **feature1** against float32: our worst row 0.1353, torch's 0.1686 (both row 4038, the bottom-left corner patch);
  `1 - cos` 5.37e-5, torch's 5.57e-5. Against the SDK: cosine 0.999954, worst row 0.109 (row 4099). Pass.
- **Embedding**: worst row 0.0089 and `1 - cos` equal to torch's at the printed digits. Pass.
- **Blocks**, each of the 32 on torch's bf16 input: all pass; our worst row is torch's times 0.82 to 1.24, our
  `1 - cos` torch's times 0.84 to 1.06. Worst rows 0.0045 to 0.0138, except block 2's row 1810 at 0.1905 (torch 0.1905).
- **Last norm**: equal to torch's at the printed digits. Pass.
- **Repeat**: identical bytes over 10 encodes.
- **Speed**: 0.120 s an encode (median of 10; 0.118 s on the GPU), the first 0.150 s, against the SDK's 0.261 s on MPS
  (median of 10 warm; its first 0.385 s): 2.2x. Load 0.2 to 0.6 s.
- **Memory**: 2.28 GB peak process footprint (`/usr/bin/time -l`, `tf-idol-dino` with the case's feature rows), the bf16
  weights 1.68 GB of it. The SDK's MPS encode holds 2.24 GB of MPS driver memory.

## End to end

`tf-idol-generate` (`zig/tests/idol_generate.zig`) chains the pieces above, from one preprocessed picture to a PLY:

1. DINOv3's feature1;
2. the VAE's feature2;
3. the denoiser's condition and its 50 guided steps;
4. the octree and Gaussian decoders;
5. the PLY and the camera.

Each model is loaded when its stage starts and freed when it ends. Background removal and cropping (BiRefNet, the SDK's
`preprocess`) are not ported, so the input is the picture `preprocess` returns, as a `pixels` or `image` tensor that
`idol_case.py dino` or `vae` writes.

```sh
zig build tf-idol-generate -Dcpu=apple_m1
zig-out/bin/tf-idol-generate WEIGHTS KLEIN/vae DINO case/vae.safetensors out/   # seed 42, 50 steps, cfg 3, 262,144
zig-out/bin/tf-idol-generate WEIGHTS KLEIN/vae DINO case/vae.safetensors out/ --noise case/vae.safetensors
zig-out/bin/tf-idol-generate WEIGHTS KLEIN/vae DINO - out/ --features dino.safetensors --features vae.safetensors
```

Noise:

- **The starting noise** is `torch.Generator("cpu").manual_seed(seed)`'s randn: [8192, 16], then [5], as
  `idol_case.py case` draws it. The tool reimplements torch's mt19937 and its Box-Muller. The camera's 5 values are
  bit-equal to torch's. 28 of the 131,072 latent values are 1 ulp off, from torch's float cos and sin.
- **The VAE posterior's noise and the octree's draws** come from native seeded generators.
- **`--noise FILE`** gives the posterior's noise (`noise`, as `idol_case.py vae` records it). If FILE also holds
  `latent` and `camera`, they replace the starting noise.

It writes `generate.ply`, `camera.json`, `start.f32`, `sample.f32` and `tf-idol-generate.json`, each stage's seconds.

### Measured

Setup:

- The avocado example and the same M5 Ultra and torch 2.13.
- Seed 42, 50 steps, guidance 3, 262,144 Gaussians.
- The SDK runs its own `IdolPipeline.__call__` on MPS once (`optimize` is CUDA-only), each stage timed by wrapping
  the SDK's methods with `torch.mps.synchronize()`. Its PLY is byte-identical to an earlier run.
- Ours runs twice, each in a fresh process. The weights were in the page cache both times, so cold and warm differ
  only by the process.

| stage | ours, cold | ours, warm | SDK, MPS |
| --- | ---: | ---: | ---: |
| load (all models) | 0.67 s | 0.67 s | 2.11 s (and 2.75 s of imports) |
| preprocess (BiRefNet) | not ported | not ported | 0.51 s |
| DINOv3 | 0.158 s | 0.155 s | 0.275 s |
| VAE | 0.245 s | 0.241 s | 0.119 s |
| condition | 0.072 s | 0.071 s | in the sampling |
| denoise, 50 steps x 2 | 27.49 s | 27.50 s | 93.89 s |
| octree | 0.145 s | 0.143 s | 0.612 s |
| Gaussian decoder (+ activation) | 0.222 s (+ 0.001) | 0.222 s (+ 0.001) | 0.491 s (+ 0.013) |
| PLY and camera | 0.004 s | 0.004 s | 0.018 s |
| total | 29.09 s | 29.07 s | 100.81 s (95.41 s generating, less preprocess) |

Memory, the whole process under `/usr/bin/time -l`:

| | peak footprint | max RSS | GPU |
| --- | ---: | ---: | --- |
| ours | 2.71 GB | 1.94 GB | 2.40 GB Metal allocated at most |
| SDK | 12.88 GB | 6.20 GB | 5.01 GB MPS allocated, 12.64 GB driver (sampled) |

### Quality

The PLYs cannot be equal:

- **Starting noise:** the SDK draws it from MPS's generator after the VAE's draw, which cannot be reproduced natively.
- **VAE posterior noise:** MPS's generator too, drawn in bf16.
- **The octree's draws:** torch's multinomial and jitter in the SDK, a native generator here. One count's difference
  changes the topology from that level down.
- **The denoiser:** fp16 over 50 guided steps, and our encoders are not bit-equal to the SDK's.

The starting noise matters most. With the same features, the SDK's own sample from MPS noise and its sample from the
CPU-seeded noise have cosine 0.479. Their PLYs are as far apart as ours is from the SDK's.

So the PLYs are compared as distributions. Statistics are per PLY: xyz in PLY units, opacity and scale activated. The
NN columns, against the SDK's pipeline PLY, are:

- the mean symmetric nearest-neighbour distance between 4,096 points sampled from each;
- the same over all 262,144 points.

Two independent 4,096-point samples of the SDK's own PLY lie 0.00896 apart.

| run | xyz std | opacity mean (q50) | scale q50 | NN, sampled | NN, all points |
| --- | --- | --- | --- | ---: | ---: |
| SDK pipeline | 0.189, 0.078, 0.304 | 0.885 (0.9986) | 0.00096 | 0.00906 | 0 |
| ours, native noise | 0.176, 0.073, 0.320 | 0.423 (0.181) | 0.00113 | 0.04623 | 0.0365 |
| ours, SDK's VAE noise | 0.176, 0.074, 0.320 | 0.425 (0.186) | 0.00113 | 0.04627 | 0.0366 |
| ours, SDK's VAE and starting noise | 0.189, 0.079, 0.304 | 0.887 (0.9986) | 0.00096 | 0.00922 | 0.00166 |
| SDK, same features, CPU-seeded noise | 0.176, 0.074, 0.319 | 0.422 (0.179) | 0.00113 | 0.04617 | 0.0368 |

The table shows:

- **Every PLY:** 262,144 Gaussians, all finite.
- **The SDK's VAE noise alone** changes little: the starting noise still differs.
- **With both of the SDK's noises:** our sample has cosine 0.99835 to the SDK's. The PLYs are as close as the sampling
  can resolve: 0.00922 against the 0.00896 baseline, 0.0017 over all points. The camera is yaw 1.6956, pitch 0.5397
  and fov 0.1094, against 1.7006, 0.5413 and 0.1096.
- **Repeat:** our cold and warm PLYs are byte-identical.

### Parity chain

This run gives the SDK's own recorded feature1 and feature2 (`--features`) and the CPU-seeded noise. The SDK's
reference is the same features and noise: the `idol_case.py case` sample, and its `decode` PLY.

- **Sample:** cosine 0.999991 to the SDK's.
- **Camera:** its latent's 5 values within 0.0022 of the SDK's.
- **PLY** against the SDK's decode of its own sample:
  - mean NN 0.00981 between 4,096-point samples (the baseline 0.00944);
  - 0.0018 over all points;
  - xyz mean 6e-5 apart.

From the picture, our encoders move the sample to cosine 0.99437 with native VAE noise. With the SDK's noise, they
move it to 0.99953. The PLY stays 0.0021 over all points from the SDK's decode.
