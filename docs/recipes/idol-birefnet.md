# Idol background removal on Metal

See [Idol on Metal](idol.md) for the checkpoint and denoiser.

## Background removal and preprocessing

`tf-idol-generate` takes a raw photo too. `preprocess.zig`, `photo.zig` and `birefnet*.zig` (kernels in
`idol_birefnet.metal`) do what idol-sdk's `IdolPipeline.preprocess` (`preprocess.py`) does:

1. Resize the photo so its short side is 1,024 (Pillow LANCZOS).
2. Unless the photo has real alpha (RGBA with some alpha below 255), predict an alpha with BiRefNet
   (`remove_background`). The input is `x / 255`, resized to 1024 x 1024 with torch's CPU bilinear (align_corners) and
   normalized by ImageNet's mean and std. The mask's logits pass through a sigmoid, are resized back (bilinear,
   align_corners), clamped and scaled by 255, then truncated to uint8.
3. Erode the alpha (`MinFilter(3)`).
4. Crop a square of `crop_margin` times the object's box: 1.2 for `synthetic`, 1.7 for `real`.
5. Resize it to 1,024 (LANCZOS, through Pillow's premultiplied RGBa) and paste it on black.

The SDK applies no EXIF orientation, and neither does the port.

The network is BiRefNet's Swin-L (embedding 192; depths 2, 2, 18, 2; windows of 12, shifted by 6), run twice: on the
1,024 input and on a 512 copy. Its features are concatenated, squeezed, and decoded by four ASPP-deformable blocks
with the image-patch convolutions, the gradient gates, the laterals and the head.

### Weights, licence and identity

`birefnet.safetensors` comes in the `speridlabs/idol` snapshot above. It is byte for byte `ZhengPeng7/BiRefNet` at
`e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` (MIT), stored in fp16: all 742 tensors are equal, and 12 training-only
tensors are dropped. The loader requires exactly those 742 tensors at Swin-L's geometry. Every
`relative_position_index` must equal `(dy + 11) * 23 + dx + 11`, the formula `br_window` uses. It widens fp16 to fp32
exactly and folds each batch norm as `(x - mean) * (weight / sqrt(var + 1e-5)) + bias`, the scale computed in float64.
The batch norms' `num_batches_tracked` counters are read and ignored.

### Precision

On a Mac, the SDK runs BiRefNet in fp32 on MPS; it calls `.half()` only on CUDA. A probe of MPS's fp32 matmul and conv
found true fp32: relative error about 7e-7, the CPU's. torchvision's `deform_conv2d` runs on MPS.

The port therefore computes everything in fp32:

- every convolution and linear layer as an fp32 simdgroup-matrix implicit GEMM, never the tensor units, whose fp32
  GEMM truncates its operands to 10 mantissa bits (#1772);
- sums in blocks of 256 terms added into the total, because one running sum over the squeeze's 51,840 terms drifts
  to twice torch's distance from float64;
- LayerNorm in two passes;
- the exact `erf` GELU (musl's `erff`, within 0.76 ulp);
- `precise::exp`;
- the deformable columns as torchvision forms them.

The host's steps are exact:

- PNG and WebP decoding as Pillow 13.0.0 decodes them;
- LANCZOS with Pillow's double-precision coefficients and 22-bit fixed point;
- the RGBa premultiply and its inverse;
- `MinFilter`;
- the paste's `MULDIV255`;
- Python's `round` and `int`;
- torch's CPU bilinear as it computes it, `fma(t0, w0, t1 * w1)` along each axis;
- the alpha's fp32 sigmoid, bilinear and truncation as torch computes them on MPS.

JPEG photos need the pinned libjpeg-turbo 3.1.4.1 and are refused without it.

### Gates

```sh
python3 tools/zig/idol_birefnet_case.py WEIGHTS SDK photo.png case/ [--mode real] [--stages]
zig build tf-idol-birefnet -Dcpu=apple_m1
zig-out/bin/tf-idol-birefnet WEIGHTS case/ photo.png
```

`idol_birefnet_case.py` imports the SDK's `birefnet.py`, `preprocess.py` and `pipeline.py` only at their pinned
sha256. It also pins the weights' sha256 and Pillow 13.0.0. It runs `IdolPipeline.preprocess` on MPS and records:

- the resized photo, BiRefNet's input, logits and alpha;
- the picture;
- a ruler: the same network in float64 on the CPU from the same resized photo.

With `--stages`, it also records each stage's torch input and output and a float64 run of that stage on torch's own
input. The stages are each Swin layer of both passes, the encoder, the squeeze, each decoder block and the decoder.

`tf-idol-birefnet` gates:

- **Host steps:** the resized photo and BiRefNet's input are byte for byte the SDK's. The alpha step on the SDK's
  own logits is byte for byte the SDK's alpha.
- **Logits** from our input: worst image row and `1 - cos` within twice torch's own distance from the ruler. The
  cosine comes from the unit vectors' distance, since a dot product cannot resolve 1e-12.
- **Stages**, each on torch's own input: the same 2x gate.
- **uint8 alpha** against the ruler's: differing pixels within twice torch's, largest difference within torch's (or
  one level).
- **Picture:** from the SDK's alpha and from the ruler's, byte for byte the SDK's picture, so Pillow's steps match
  exactly. From our own alpha, differing pixels and largest difference within twice the SDK's. One alpha level moves
  a pasted, resized pixel by up to two; the SDK's own picture sits two levels off the ruler's on the opaque photo.
- **Repeat:** three more forwards byte for byte.

`birefnet_reference.zig`'s tests plant a 2% error in one logits row and mismatches in the alpha and the picture, and
require each gate to fail them. `preprocess.zig`'s and `photo.zig`'s tests cover the resampler, the PNG writer and
photo detection.

### Measured

The same M5 Ultra and torch 2.13. The photos:

- the avocado reference input (1024 x 1024 PNG);
- the demo's `food-avocado.webp` (768 x 768);
- an opaque 1280 x 960 PNG of the avocado on a gradient, run in both modes;
- an RGBA PNG with real alpha.

All pass. On every photo, the resized photo and BiRefNet's input are byte for byte the SDK's, and so is the alpha from
the SDK's logits. The picture from the SDK's alpha is byte for byte the SDK's.

| photo | logits worst row, ours / torch | `1 - cos`, ours / torch | alpha vs ruler, ours / torch | picture vs SDK |
| --- | --- | --- | --- | --- |
| avocado input | 3.84e-6 / 4.15e-6 | 3.9e-13 / 4.0e-13 | 13 px max 1 / 11 px max 1 | 3 px, max 2 |
| food-avocado.webp | 8.15e-6 / 1.14e-5 | 5.8e-13 / 6.6e-13 | 5 px max 1 / 6 px max 1 | 2 px, max 1 |
| opaque, synthetic | 1.90e-5 / 1.65e-5 | 2.7e-12 / 2.2e-12 | 6 px max 1 / 5 px max 1 | 2 px, max 1 |
| opaque, real | 1.90e-5 / 1.65e-5 | 2.7e-12 / 2.2e-12 | 6 px max 1 / 5 px max 1 | 0 px |
| RGBA, real alpha | no BiRefNet | | | 0 px |

- **Stages** (avocado input): all 24 pass. Our worst row is torch's times 0.2 to 1.13: layer 0's PatchMerging at
  1.13, the squeeze at 0.22.
- **Repeat:** identical bytes.
- **Speed:** BiRefNet's forward takes 0.172 s (median, 0.172 s on the GPU), the first 0.18 to 0.22 s. The host's
  steps take 0.02 to 0.03 s for the resize and 0.02 s for the finish. The SDK's whole `preprocess` takes 0.26 to 0.28 s
  warm and 0.48 to 0.52 s in a generation. Load: 0.7 to 0.9 s.
- **Memory:** `tf-idol-birefnet` peaks at a 2.33 GB footprint with its case; the fp32 weights are 0.89 GB of it.

### In `tf-idol-generate`

New to its command line, all else unchanged:

- PICTURE may be a `.png`, `.jpg`, `.jpeg` or `.webp` photo (any case), preprocessed as above;
- `--mode synthetic|real` (default `synthetic`): the SDK's mode, crop margin 1.2 or 1.7;
- `--crop-margin X` overrides the mode's margin;
- a photo also writes `OUT_DIR/preprocessed.png`, the 1024 x 1024 RGB the encoders saw;
- a photo's `tf-idol-generate.json` adds `preprocess_s` (all of preprocessing but BiRefNet's load, which `load_s`
  counts) and `birefnet_s` (BiRefNet's forward and alpha).

BiRefNet loads from `SNAPSHOT_DIR/birefnet.safetensors` and is freed before DINOv3 loads.

```sh
zig-out/bin/tf-idol-generate WEIGHTS KLEIN/vae DINO photo.png out/ [--mode real] [--crop-margin 1.5]
```

End to end against the SDK's `IdolPipeline.__call__` on MPS:

- the same photo, seed 42, 50 steps, guidance 3, 262,144 Gaussians;
- each run in a fresh process under `/usr/bin/time -l`;
- the SDK's stages timed as in [End to end](idol-generate.md#end-to-end).

| stage | ours, opaque | SDK, opaque | ours, webp | SDK, webp |
| --- | ---: | ---: | ---: | ---: |
| load (all models) | 1.36 s (BiRefNet 0.72) | 2.10 s | 1.35 s (BiRefNet 0.72) | 2.16 s |
| preprocess (BiRefNet) | 0.295 s (0.224) | 0.519 s | 0.264 s (0.221) | 0.481 s |
| DINOv3 | 0.155 s | 0.276 s | 0.156 s | 0.271 s |
| VAE | 0.244 s | 0.117 s | 0.244 s | 0.107 s |
| denoise | 27.49 s | 93.94 s | 27.51 s | 93.92 s |
| octree | 0.139 s | 0.750 s | 0.142 s | 0.600 s |
| Gaussian decoder | 0.222 s | 0.491 s | 0.223 s | 0.492 s |
| total | 30.07 s | 101.27 s (96.11 s generating) | 30.04 s | 100.69 s (95.89 s generating) |
| peak footprint | 2.77 GB | 12.88 GB | 2.76 GB | 12.92 GB |
| max RSS | 2.27 GB | 6.19 GB | 2.25 GB | 6.24 GB |

Our `preprocessed.png` differs from the SDK's preprocessed image in 2 pixels by at most 1 on each photo.

The PLYs, as in [Quality](#quality). The NN columns are against the SDK's PLY of the same photo; the sampled baseline
is 0.00922 for the opaque photo and 0.00896 for the webp.

| run | xyz std | opacity mean (q50) | NN, sampled | NN, all points | sample cosine |
| --- | --- | --- | ---: | ---: | ---: |
| SDK, opaque | 0.188, 0.066, 0.305 | 0.547 (0.674) | | 0 | 1 |
| ours, opaque, native noise | 0.175, 0.068, 0.318 | 0.448 (0.251) | 0.04518 | 0.0367 | 0.510 |
| ours, opaque, SDK's noises | 0.188, 0.066, 0.305 | 0.553 (0.700) | 0.00951 | 0.00175 | 0.99642 |
| SDK, webp | 0.189, 0.078, 0.304 | 0.885 (0.9986) | | 0 | 1 |
| ours, webp, native noise | 0.176, 0.073, 0.320 | 0.414 (0.166) | 0.04647 | 0.0365 | 0.478 |
| ours, webp, SDK's noises | 0.189, 0.078, 0.304 | 0.880 (0.9985) | 0.00937 | 0.00165 | 0.99906 |

- **Every PLY:** 262,144 Gaussians, all finite.
- **Native noise:** the PLYs differ from the SDK's as much as with a preprocessed picture: its starting noise is
  MPS's.
- **Parity** (`--noise` with the SDK's VAE and starting noise): the PLYs are as close as sampling resolves.
  - Opaque: camera yaw 1.6946, pitch 0.5193, fov 41.45, against the SDK's 1.6887, 0.5208 and 41.60.
  - Webp: 1.6990, 0.5358 and 0.1090, against 1.7006, 0.5413 and 0.1096.

### Kernel contract

`zig/tests/contract/metal_idol_birefnet*.zig` registers every kernel of `idol_birefnet.metal`:

- `br_conv`'s nine forms (1x1 plain, GELU, residual, residual by map, BN ReLU; 3x3 plain, BN ReLU, BN; 7x7);
- `br_conv3`;
- `br_norm`'s three modes;
- `br_window` with and without the shift;
- `br_resize`;
- `br_tiles`' two modes;
- `br_deform` at 1x1, 3x3 and 7x7;
- `br_pool`, `br_gate` and `br_head`.

Each is checked against float64 at its fp32 bound. Twenty-one controls each break one kernel, and each must go red at
its gate. They include:

- sums or staging in fp16;
- a tap read a row low;
- the GELU's tanh form;
- ReLU by `max`, which drops a NaN;
- unbiased variance;
- the shifted windows' mask dropped;
- the bias table's axes swapped;
- the deformable modulator as `sigmoid` instead of `2 sigmoid`.

`br_window`, `br_gate` and `br_head` declare inexact sums: fp16 sums would miss by only 61x, 84x and 3x.

```
SUMMARY 24 kernels: 24 pass gates 1-4, 0 fail, 0 known findings; 21 controls: 21 red as planted, 0 missed; 0 limit problems
```

With the rest of Idol (`--only "metal idol"`):

```
SUMMARY 81 kernels: 81 pass gates 1-4, 0 fail, 0 known findings; 78 controls: 78 red as planted, 0 missed; 0 limit problems
```
