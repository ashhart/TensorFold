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

The image encoders are not ported yet. `tools/zig/idol_case.py` runs the publisher's SDK in torch, a development
reference. It is a denoiser, not a token family: it does not register with the lane
engine, the server or `families.detect`.

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
