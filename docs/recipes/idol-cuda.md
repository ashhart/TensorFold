# Idol's denoiser on CUDA

Idol's denoiser (see `idol.md`) on the CUDA runtime, beside the Metal family and driven by the same case driver
(`zig/tests/idol_run.zig`): `tf-idol` on Metal and on CUDA reads the same case and runs the same gates. The FLUX.2 VAE
image encoder is ported too ([below](#vae-encoder)); DINOv3, the decoders and the PLY writer are not, and nothing
registers with the lane engine or the server.

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

