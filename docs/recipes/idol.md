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

The DINOv3 image encoder runs natively too ([DINOv3 encoder](#dinov3-encoder)); the FLUX.2 VAE encoder, the decoders
and the PLY writer are not ported yet. `tools/zig/idol_case.py` runs the publisher's SDK in torch, a development
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
