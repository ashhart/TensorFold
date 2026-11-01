# Idol on CUDA

Idol's denoiser (see `idol.md`) on the CUDA runtime, beside the Metal family and driven by the same case driver
(`zig/tests/idol_run.zig`): `tf-idol` on Metal and on CUDA reads the same case and runs the same gates. The other
components are ported too, each with its own tool sharing its Metal driver: the DINOv3 image encoder
([DINOv3 encoder](#dinov3-encoder)), the FLUX.2 VAE image encoder ([VAE encoder](#vae-encoder)), the decoders
([Decoders](#decoders)) and background removal ([BiRefNet](#background-removal-birefnet)), and `tf-idol-generate`
runs them all ([End to end](#end-to-end)); [GB10 qualification](#gb10-qualification) is one script. Nothing registers
with the lane engine or the server.

**Not run on a GPU yet.** It compiles for sm_120 and sm_121 and its host tests pass; no CUDA kernel of this family has
run on a device. Until a GB10 or an RTX 50 card has run the kernel contract and a case, treat it as unqualified.

`Model.load` admits sm_121 (GB10) and sm_120 (RTX 50, RTX PRO Blackwell) and refuses any other GPU by name.

## Precision

The same contract as on Metal, in CUDA's own terms:

- fp16 block projections on cuBLASLt (`CUBLAS_COMPUTE_32F`, fp16 inputs, fp32 output); `idol.cu` adds the bias to
  those fp32 sums and rounds once to fp16, GELU after it.
- RePo's maps, the condition embedding and the output projection in fp32 on cuBLASLt with `CUBLAS_COMPUTE_32F` and
  fp32 inputs: true fp32 (no TF32), so no hi/lo split is needed here.
- LayerNorms in fp32, rounded to fp16; the modulation in fp16 ops.
- Attention: `idol_attention.cuh`, head 64, fp16 q, k and v, fp32 scores, softmax and sums, scale 1/8, every query row
  to every key row. It is Idol's own kernel: no shared attention template gains an instance.
- `idol.cu` builds with the glue flags (`--fmad=false`, IEEE division and square root). `F.normalize`'s clamp keeps a
  NaN norm, as on Metal.

## Kernel contract

`tf-cuda-test idol-contract` runs gates 1-4 over every `idol.cu` kernel and the attention, with three controls that
each drop one fp16 rounding point; the full contract runs it among its family sets. Gate-5 records
(`zig/src/core/contract/idol_cuda_records.zig`) are nvcc 13.0's for sm_80, 86, 89, 120 and 121. No SUMMARY line yet:
it needs a GPU.

## Run a case

```sh
zig build idol-cuda
zig-out/bin/tf-idol WEIGHTS case/
zig-out/bin/tf-idol WEIGHTS case/ blocks 0
```

The case comes from `tools/zig/idol_case.py`, as in `idol.md`.

## DINOv3 encoder

`dino_cuda*.zig` (kernels in `idol_dino.cu`) runs the DINOv3 ViT-H+/16 encoder of `idol.md`'s
[DINOv3 encoder](idol.md#dinov3-encoder) on CUDA: the SDK's 1024 x 1024 preprocessed image to `feature1` [4101, 1280],
from the same timm weights. **These weights are under Meta's DINOv3 License, not Apache 2.0**; see that section's
"Weights and licence". `tf-idol-dino` on CUDA shares its case driver with Metal (`zig/tests/idol_dino_run.zig`): the
same case, the same gates. It admits sm_121 and sm_120 and refuses other GPUs by name, as the denoiser does.

**Not run on a GPU yet.** It compiles for sm_120 and sm_121 and its host tests pass; none of its kernels has run on a
device. GB10 qualification is pending.

### Precision

Transformers' bf16 rounding points, as on Metal, in CUDA's terms:

- Every linear (the patch embedding as one [4096, 768] by [1280, 768] GEMM, qkv, the attention's output, fc1 as one
  10,240-column gate | up GEMM, fc2) on cuBLASLt: bf16 inputs, `CUBLAS_COMPUTE_32F`, fp32 sums out. `idol_dino.cu`
  adds the bias to those sums and rounds once to bf16 (`bf16(x w + b)`), in the step after it: `tf_idol_dino_bias`
  (the patches), `tf_idol_dino_heads` (qkv has no bias), `tf_idol_dino_scale_add` (then `bf16(x + bf16(g y))`) and
  `tf_idol_dino_swiglu` (then `bf16(bf16(silu(g)) u)`).
- LayerNorms with fp32 statistics, `bf16(n w + b)`; the last one into `last_hidden_state`, then ImageEncoder's plain
  F.layer_norm in fp32 (eps 1e-5), in one kernel (`tf_idol_dino_final`).
- RoPE on the patches alone, `bf16(bf16(x cos) + bf16(rotate_half(x) sin))`, from the host's fp32 tables rounded to
  bf16 (the Metal port's, equal to torch's bit for bit).
- Attention: `idol_attention.cuh`'s `full<64, 8>` instanced for `__nv_bfloat16` (`tf_idol_dino_attention`): bf16 q, k
  and v, fp32 scores, softmax and sums, P rounded to nearest bf16 for the P V mma, the output rounded to bf16. Its
  denominator sums the fp32 P, as FlashAttention does (the Metal kernel sums the rounded P; both are RN of the same
  values). The header gained the element type as a template parameter defaulting to `__half`, so `idol.cu` is
  unchanged; that its `tf_idol_attention` keeps the same SASS (sm_80, 86, 89, 120, 121) is not yet checked.
- `idol_dino.cu` builds with the glue flags (`--fmad=false`, IEEE division and square root).

### Kernel contract

`tf-cuda-test idol-dino-contract` runs gates 1-4 over every `idol_dino.cu` kernel (the bias, the norm, the final norm's
two outputs, the heads, the scale-add, SwiGLU and the attention), with six controls: four drop or add a bf16 rounding
(gate 3: the bias after a rounded sum, RoPE without its rounded products, LayerScale without `bf16(g y)`, SwiGLU
without `bf16(silu(g))`) and two sum the LayerNorm's or F.layer_norm's statistics in fp16 (gate 4). The full contract
runs it among its family sets. Gate-5 records (`zig/src/core/contract/idol_dino_cuda_records.zig`) are nvcc 13.0's
for sm_80, 86, 89, 120 and 121. No SUMMARY line yet: it needs a GPU. `contract_idol_dino_emulate.zig` replays each
kernel's fp32 steps and each control's on the host, in `test-cuda-host`: the kernels land in their ranges, and the
controls leave them at their gates.

### Run a case

```sh
zig build idol-dino-cuda
zig-out/bin/tf-idol-dino DINO case/
```

The case comes from `tools/zig/idol_case.py dino`, as in `idol.md`; with `dino_blocks.safetensors` and
`dino_blocks_f32.safetensors` in it, the tool also gates the embedding, each block and the last norm.

## VAE encoder

**Not run on a GPU yet.** It compiles and passes host tests; it has never run on a GPU, and GB10 qualification is
pending.

`cuda_vae.zig` runs the encoder of `idol.md`'s [VAE encoder](idol.md#vae-encoder) section on CUDA: the picture
`IdolPipeline.preprocess` returns to the denoiser's `feature2` [4101, 128], with the posterior's noise as an input.
`tf-idol-vae` shares its case driver with Metal (`zig/tests/idol_vae_run.zig`), so it reads the same case and runs the
same float64 gate. `Encoder.load` admits sm_121 and sm_120 alone, as the denoiser does. The loader takes klein's bf16
file or FLUX.2-dev's fp32 one rounded to bf16, through `vae_config.zig`, which Metal shares.

### Precision

fp32 throughout on the SDK's bf16 weights, as on Metal, against the same float64 ruler:

- Every convolution is `tf_iv_conv`, an implicit GEMM of fp32 FMAs (`__fmaf_rn`) on the FMA units: same-size 3x3s,
  the stride-2 downsamples padded right and below, and the 1x1s. Each output is one FMA chain over its taps and input
  channels in order, then the bias and the residual. There is no Winograd: with no tensor-unit workaround to pay for,
  the direct form keeps each output an fp32 dot product, with no `G g G^T` rounding or transform sums.
- `conv_in` (`2 x - 1`, three channels) is its own kernel, as on Metal.
- GroupNorm in two passes (the mean, then the squared deviations), then the affine and SiLU, `expf` and IEEE
  division.
- The mid block's attention runs on cuBLASLt in true fp32 (`CUBLAS_COMPUTE_32F`, fp32 operands, never TF32): q and k
  as torch's Linear, v as `W_v xn^T` so that `P V` is cuBLASLt's `X W^T` too, and the scores 4,096 query rows at a
  time. `tf_iv_softmax` takes the row's maximum, `expf` and the fp32 sum, and divides. The buffers fit inside the
  free picture tensors: no memory beyond the four 512 MB ones.
- `tf_iv_tokens` keeps a NaN logvar through the clamp, as `torch.clamp` does.

`idol_vae.cu` builds with the glue flags (`--fmad=false`, IEEE division and square root).

### Kernel contract

`tf-cuda-test idol-vae-contract` runs gates 1-4 over all eight `idol_vae.cu` kernels in 16 forms, each against
float64: `tf_iv_conv_in`, `tf_iv_conv` (3x3, 3x3 plus residual, stride-2 3x3, 1x1), `tf_iv_group_part` (sums,
squared deviations), `tf_iv_group_finish` (means, rstd), `tf_iv_norm` (plain, SiLU), `tf_iv_bias` (by column, by
column plus residual, by row), `tf_iv_softmax` and `tf_iv_tokens`. Gate 4's softmax rows are equal scores, so its
denominator is exact in fp32 and an fp16 sum stalls at 2,048.

Eleven controls each build one body of `idol_vae.cuh` with a planted flaw, and each must go red at its gate: half
inputs (`conv_in`, the conv's tile), symmetric padding in the stride-2 3x3, a half SiLU, half sums (the conv, both
GroupNorm passes, the softmax), a by-row bias read by column, swapped patch axes, and a NaN logvar lost to
`fminf`/`fmaxf`. The engine's kernels are the flaw-free instances. The full contract runs this set among its family
sets. Gate 5's records (`idol_vae_cuda_records.zig`) are nvcc 13.0's for sm_80, 86, 89, 120 and 121. No SUMMARY line
yet: it needs a GPU.

### Run a case

```sh
zig build idol-vae-cuda -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121   # -Dsm=120 for an RTX 50
zig-out/bin/tf-cuda-test idol-vae-contract
zig-out/bin/tf-idol-vae KLEIN/vae case/
```

The case comes from `tools/zig/idol_case.py vae`, as in `idol.md`. A pass is the line `against float64: ... pass`
(worst row and `1 - cosine` within twice the SDK's), five more encodes with identical bytes, and exit 0.

## Decoders

Idol's octree and Gaussian decoders (`idol.md`, [Decoders](idol.md#decoders)) on CUDA: `cuda_decoder.zig` beside
`decoder.zig`, the host steps shared (`decode.zig`, `octree.zig`, `gaussian.zig`: softmax, `systematic_sample`,
compaction, jitter, the activation, PLY and camera). `tf-idol-decode` on Metal and on CUDA is one driver
(`zig/tests/idol_decode_run.zig`), so a CUDA build reads the same case and runs the same gates.

**Not run on a GPU yet.** The decoders' host tests pass; their nvcc build for sm_120 and sm_121 and its gate-5 records
are pending, and none of their kernels has run on a device. `Decoder.load` admits sm_121 and sm_120 alone and refuses any other GPU by name.

### Precision

The Metal decoders' rounding points, in CUDA terms:

- Stem in fp32: `tf_idol_dec_embed` (`in_proj` as one FMA chain, plus `sinf` and `cosf` of the fp32 angle
  `fp32(fp32(x f) pi)`), then `input_layer` on cuBLASLt with `CUBLAS_COMPUTE_32F` and fp32 inputs, true fp32, and
  `tf_idol_dec_half` adds its bias and rounds once to fp16.
- Blocks in fp16 through the denoiser's CUDA kernels: cuBLASLt fp16 linears into fp32 sums, `tf_idol_bias` (bias,
  one rounding, GELU after it), `tf_idol_norm` and `tf_idol_gate_add`.
- `tf_idol_dec_kv`: `to_kv` of the fp16 latent as 16-deep fp32 FMA sums, the bias, one rounding.
  `tf_idol_dec_heads`: MultiHeadRMSNorm in fp32 in place (`F.normalize`, times gamma times 8), rounded to fp16.
- Head: `tf_idol_norm32`, cuBLASLt in true fp32 and `tf_idol_bias32`.
- Attention, fp32 scores, softmax and sums, scale 1/8:
  - Self attention is the denoiser's `tf_idol_attention` unchanged: to_qkv's packed rows are already its layout.
  - Cross attention to the latent's 8,192 rows is `tf_idol_dec_attention`, the same row arithmetic from query rows
    to other key rows (`idol_decoder_attention.cuh`).
  - Both round P once to nearest fp16 for the P x V mma and sum the softmax's denominator over the fp32 P. torch on
    CUDA does the same: its fp16 SDPA at head 64 runs the flash or memory-efficient kernel at any query count, and
    both round P to fp16 before P V. Metal needed P's hi and lo halves only because the M5's units take an fp32 P
    short (always low). Rounding to nearest has no such bias, so no split.
  - At 8 query rows or fewer (the octree's first two levels) `tf_idol_dec_few` runs fp32 throughout. torch on CUDA
    would round P there too. The cases come from the SDK on MPS, whose fp32 vector kernel takes those calls, and
    their gate at level 1 is the SDK's own distance to fp32 (0.00009), so this port keeps the cases' path. It is
    as close to fp32 as torch's CUDA path or closer.
- `idol_decoder.cu` builds with the glue flags (`--fmad=false`, IEEE division and square root).

The Metal contract's two nonfinite rules hold:

- `tf_idol_dec_heads` keeps a NaN norm, as `clamp_min` does (`fmaxf` would drop it).
- An infinite value stays infinite through the attention wherever its P is nonzero in fp16. There is no lo half
  here to meet it as 0. A P under fp16's least subnormal is 0, and 0 times infinity is NaN, as in torch's kernels.

### Kernel contract

`tf-cuda-test idol-decoder-contract` runs gates 1-4 over the six `idol_decoder.cu` kernels, the tiled and short
attention forms included. It also runs `tf_idol_attention` at the Gaussian decoder's self-attention shapes. The full
contract runs it among its family sets. Each kernel is its template's `BREAK = 0`. The 13 controls are the same
templates at other `BREAK` values (`contract_controls_idol_decoder.cuh`, in the controls image), and each must go red
at its gate:

- shapes: fp16 `pe`, fp16 `y` before the bias, fp16 `ctx w` before the bias, fp16 `x / n`, keys past Lk unmasked,
  the short form's P rounded to fp16;
- sums: `in_proj`, `to_kv`, the heads' squares and P V (both forms) summed in fp16;
- repeat: `tf_idol_dec_half` writing its input;
- nonfinite: `tf_idol_dec_heads` clamping by `fmaxf`.

Attention's references hold the kernel's own P: fp16 in the tiled form, fp32 in the short form. Its allowance is
each score's 64-deep fp32 sum, exp's approximation and the key sums' depth, P's fp16 rounding where an fp32 P could
fall on either side of a midpoint, and O's own rounding. Half the heads are built so O is P's fp16 remainder alone:
the tiled form must give 0 there, and the short form with P rounded to fp16 goes red. Gate-5 records
(nvcc 13.0, sm_80, 86, 89, 120 and 121) are pending the first nvcc build. No SUMMARY line yet: it needs a GPU.

### Run a case

```sh
zig build idol-decode-cuda
zig-out/bin/tf-idol-decode WEIGHTS decode262k/                 # the gates; writes .json, .ply, camera.json
zig-out/bin/tf-idol-decode WEIGHTS case/ time 262144 out.ply   # the native generator, cold then warm
```

The case comes from `tools/zig/idol_case.py decode`, as in `idol.md`. The time line reports `cuda_peak_bytes`, the
device bytes the decoders allocated at their peak, where Metal reports its allocated bytes.

## Background removal (BiRefNet)

BiRefNet's forward (Swin-L twice, the ASPP-deformable decoder) runs on CUDA in fp32, with its input and output
resizes. The host's preprocessing (`preprocess.zig`, `photo.zig`: decode, Pillow's resizes, erosion, crop and paste) is
the same code as on Metal, and `tf-idol-birefnet` on Metal and on CUDA reads the same case and runs the same gates
(`zig/tests/idol_birefnet_run.zig`).

**It compiles and passes host tests; it has never run on a GPU; GB10 qualification is pending.** `Net.load` admits
sm_121 and sm_120 and refuses any other GPU by name.

Precision, in CUDA's terms:

- Every conv and linear is `tf_br_conv`, an fp32 kernel with fp32 inputs. No TF32 and no cuBLASLt: cuBLASLt's fp32
  kernels do not document their summation order. In practice each output is one running sum per thread, or a split-K
  the heuristic picks, and one running sum over the squeeze's 51,840 terms drifts, as it did on Metal. So each output's
  sum runs in blocks of 256 terms. Each block is a sequential fp32 FMA chain from zero, and the blocks are added in
  order into the total.
- LayerNorm, BatchNorm (folded in float64 at load), window attention (scores, softmax and P V in fp32), the pooling and
  the gates use the same orders as on Metal.
- `idol_birefnet.cu` builds with the glue flags (`--fmad=false`, IEEE division and square root). GELU uses `erff` and the
  sigmoids use `expf`, each within 2 ulps.
- The deformable conv is `tf_br_deform`, an fp32 kernel with torchvision's `deform_conv2d` sampling: bilinear over the
  four neighbours, zero outside the picture, the modulation 2 sigmoid(m), and a weighted sum that is not fused.

`tf-cuda-test idol-birefnet-contract` runs gates 1-4 over every kernel in `idol_birefnet.cu`. Seven controls each add
an fp16 rounding point or sum, or drop a NaN, and the full contract runs this set among its family sets. The gate-5
records (`zig/src/core/contract/idol_birefnet_cuda_records.zig`) are nvcc 13.0's for sm_80, 86, 89, 120 and 121.

```sh
zig build idol-birefnet-cuda
zig-out/bin/tf-idol-birefnet WEIGHTS case/ photo.png
```

The case comes from `tools/zig/idol_birefnet_case.py`, as in `idol.md`.

JPEG photos need libjpeg-turbo 3.1.4.1 (`libturbojpeg.so.0` on Linux); other versions are refused. A distribution's
package is usually another version, so build it from the official release tarball into a private prefix and point
`TENSORFOLD_LIBTURBOJPEG` at it:

```sh
curl -LO https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz
echo 'ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022  libjpeg-turbo-3.1.4.1.tar.gz' | sha256sum -c
tar xf libjpeg-turbo-3.1.4.1.tar.gz
cmake -S libjpeg-turbo-3.1.4.1 -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/opt/libjpeg-turbo-3.1.4.1"
cmake --build build -j && cmake --install build
export TENSORFOLD_LIBTURBOJPEG="$HOME/opt/libjpeg-turbo-3.1.4.1/lib/libturbojpeg.so.0"
```

The build needs CMake and a C compiler; with NASM installed it also builds the SIMD paths. The prefix may install to
`lib64/` instead of `lib/` on some distributions.

## End to end

`tf-idol-generate` runs on CUDA as on Metal: one driver (`zig/tests/idol_generate_run.zig`) with a thin main for each
(`idol_generate.zig` on Metal, `idol_generate_cuda.zig` on CUDA). The arguments, flags, defaults (seed 42, 50 steps,
guidance 3, 262,144 Gaussians), outputs (`generate.ply`, `camera.json`, `start.f32`, `sample.f32`,
`tf-idol-generate.json`, and `preprocessed.png` for a photo) and JSON keys are the same. The one exception is the peak.
On CUDA it is `cuda_peak_allocated_gb`, the process's peak device allocation as `cuda.usage` counts it (the CUDA
context and cuBLASLt's own pools are not in it), and the summary line says `CUDA peak`. Metal keeps
`metal_peak_allocated_gb`. Each stage loads its CUDA model, runs it and frees it before the next: BiRefNet for a photo
(`photo_host.zig`'s preprocessing, shared with Metal), DINOv3, the VAE, the denoiser, then the decoders.

**It compiles and passes host tests; it has never run on a GPU; GB10 qualification is pending.** Every stage admits
sm_121 and sm_120 alone.

```sh
zig build idol-generate-cuda -Dnvcc=/usr/local/cuda-13.0/bin/nvcc -Dsm=121
zig-out/bin/tf-idol-generate SNAPSHOT VAE_DIR DINO_DIR picture.safetensors out/
zig-out/bin/tf-idol-generate SNAPSHOT VAE_DIR DINO_DIR photo.png out/ --mode synthetic
```

The tool is one module, `zig/src/idol_cuda_e2e.zig`, which the build also passes to the family as its `core`.
BiRefNet's host steps reach `core/` by path, as on Metal, so a separate core module beside them would hold the same
files twice.

## GB10 qualification

`tools/zig/idol_gb10_qualify.sh WEIGHTS NOTES [OUT]` qualifies every component and the end-to-end tool on a DGX Spark
in one run. It needs nvcc 13.0 with `cuobjdump`, Zig 0.17, GNU time and python3 with numpy. In order, it:

1. checks that `nvidia-smi` reports sm_121 (or sm_120). Without one, it records that the GPU steps did not run;
2. builds the fatbins for `SM` (121), the CUDA tools and `tf-cuda-test`; checks that `tf_idol_attention`'s SASS is
   identical with #1767's `idol_attention.cuh` at sm_80, 86, 89, 120 and 121; builds the contract images for every
   SM;
3. runs `idol-contract`, `idol-dino-contract`, `idol-vae-contract`, `idol-decoder-contract`,
   `idol-birefnet-contract`, the full `kernel-contract` and `kernel-contract-limits` over every SM;
4. runs each tool on its Mac case: the denoiser's case50 (and `blocks 0`), the decoders' decode262k and decode32k
   (and a timed 262,144-Gaussian run), the VAE's and DINOv3's cases, and BiRefNet's six. The JPEG case runs only
   when `TENSORFOLD_LIBTURBOJPEG` is set;
5. runs `tf-idol-generate` from the safetensors picture and from the opaque photo (seed 42, 50 steps, 262,144
   Gaussians), then once more with the SDK's noise (`--noise`). The SDK-noise run must reach a sample cosine of at
   least 0.998 against the SDK's sample and a sampled nearest-neighbour mean of at most 0.0100 against its PLY
   (Metal: 0.99835 and 0.00922, baseline 0.00896). Each sample is also compared with Metal's.

Each case runs in a directory of links to its inputs, so no tool writes into NOTES. Every timed run is under
`/usr/bin/time -v`, with nvidia-smi's memory and `MemAvailable` sampled every half second. GB10's memory is unified,
so nvidia-smi may report none of it. Each step's log goes to `OUT/logs/`, outputs and samples to `OUT/runs/`, and
results to `OUT/steps.tsv`. `OUT/summary.md` holds one page: each step's result and seconds, the contract lines, and
timings and peaks beside Metal's. The run continues past a failed step and exits non-zero if any step failed.
`--build-only` checks the GPU and builds; `--skip-build` reuses `zig-out/`; `--dry-run` prints the steps.
`SKIP_LIMITS=1` leaves out the all-SM build, the longest step, and `SKIP_SASS=1` the SASS check.

Copy these from the Mac:

- WEIGHTS: the speridlabs/idol snapshot (`config.json`, `denoiser.safetensors`, `decoder.safetensors`,
  `decoder_gs.safetensors`, `birefnet.safetensors`), with FLUX.2-klein-4B's `vae/` in `WEIGHTS/flux2-vae` and timm's
  DINOv3 ViT-H+/16 in `WEIGHTS/dino-timm-base` (or point `VAE_DIR` and `DINO_DIR` at them). DINOv3's weights are
  under Meta's DINOv3 License.
- NOTES, the case root, with these files under it:
  - `case50/`: `case.safetensors`, `exact.safetensors`, `reference.json`, `blocks0.safetensors`,
    `blocks0_f32.safetensors` and Metal's `tf-idol.json`;
  - `decode262k/` and `decode32k/`: `decode.safetensors`, `decode.json`, `reference.ply` and Metal's
    `tf-idol-decode.json`;
  - `vae/case/`: `vae.safetensors`, `vae.json` and Metal's `tf-idol-vae.json`;
  - `dino/case/`: `dino.safetensors`, `dino.json` and Metal's `tf-idol-dino.json`. Add `dino_blocks.safetensors`
    and `dino_blocks_f32.safetensors` (`idol_case.py dino` without `--no-blocks`) for the per-block gates;
  - `birefnet/cases/{avocado,webp,opaque,opaque-real,rgba,jpeg}/`: `birefnet.safetensors`, `birefnet.json`, Metal's
    `tf-idol-birefnet.json` and avocado's `birefnet_stages.safetensors`;
  - `birefnet/photos/`: `reference-avocado-input.png`, `food-avocado.webp`, `avocado-opaque.png`,
    `avocado-rgba.png` and `avocado-opaque.jpg`;
  - `e2e/sdk/`: the SDK's `sdk-record.safetensors` and `sdk.ply`. From Metal's runs: `sample.f32` in each of
    `e2e/sdknoise/`, `e2e/cold/` and `birefnet/e2e/ours-opaque-final/`.

The Metal numbers to compare (M5 Ultra, avocado, seed 42, 50 steps, 262,144 Gaussians) are 29.1 s and a 2.71 GB peak
footprint from the picture, and 30.07 s and 2.77 GB from the opaque photo. The SDK on MPS takes about 100.8 s and
12.88 GB.
