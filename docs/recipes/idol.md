# Idol's denoiser on Metal

Idol (speridlabs) turns one image into a 3D Gaussian splat. A 0.37B flow transformer denoises 8,192 latent rows of 16
channels, conditioned on the image's DINOv3 and FLUX.2 VAE rows; two decoders then grow an octree from that latent and
emit the Gaussians. `zig/src/families/idol` runs the denoiser on the Metal runtime from the released checkpoint:

- once an image: the condition embedding (DINOv3's 1,280 channels and the VAE's 128, each projected to 1,024 and
  summed) and the two condition-refiner blocks (no modulation, affine LayerNorms);
- at each step: the two noise-refiner blocks over the latent rows, then the 24 joint blocks over 12,294 rows (8,192
  latent, 4,101 condition, one camera token), then the output head and the camera velocity.

Every block is 16 heads of 64 at width 1,024 with a tanh-GELU MLP of 4,096. Its rotary is RePo: a learned map from
each row's LayerNorm to per-head 3D positions, angles from 10, 10 and 12 frequencies per axis.

The sampler's defaults: 50 Euler steps, `t = 3u / (1 + 2u)`, guidance `v = 4 v_cond - 3 v_uncond` (two passes a step:
with the condition, and with zeros).

The image encoders, the decoders and the PLY writer are not ported yet. `tools/zig/idol_case.py` runs the publisher's
SDK in torch, a development reference. It is a denoiser, not a token family: it does not register with the lane
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
  checkpoint's gamma times 8.
- The condition embedding and the output projection in fp32; the time MLP, modulation and camera MLPs on the host.

### The tensor units' fp32 matmul is not fp32

On the M5, a GEMM whose inputs are fp32 on the tensor units (`tfp::gemm_nax<float>`, the Qwen-VL tower's
`qwv_gemm_f32`, MLX's fp32 matmul) rounds its inputs to about fp16: its error against float64 is about 7.5e-4 of the
result, against about 1e-7 for an fp32 FMA. RePo's maps at that precision move block 8's attention out of its gate.

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
  - against the SDK: cosine 0.999997 or better;
  - against fp32: our worst row over the run 0.0456, the SDK's 0.0398; per step our `1 - cos` is the SDK's within
    the 2x bound at every step, and the smaller at 40 of the 100.
- **Blocks**, at steps 1 and 41, each of the 23 joint blocks after the first on the SDK's input: all pass, worst rows
  0.0007 to 0.0017 against fp32, where the SDK's are 0.0007 to 0.0025.
- **Sample:** our 50 guided steps from the same noise end at cosine 0.99999 to the SDK's sample.
- **Repeat:** identical bytes.
- **Speed**, one session: 0.273 s a forward (27.3 s for 50 guided steps), 3.4x the SDK's 0.939 s (the median of its
  100). Load 0.2 to 1.6 s; the condition, once an image, 0.07 s.
- **Memory:** 1.84 GB peak process footprint (`/usr/bin/time -l`, the whole `tf-idol` run with its case), the
  denoiser's fp16 weights in shared Metal buffers.
