# Idol's denoiser on CUDA

Idol's denoiser (see `idol.md`) on the CUDA runtime, beside the Metal family and driven by the same case driver
(`zig/tests/idol_run.zig`): `tf-idol` on Metal and on CUDA reads the same case and runs the same gates. The DINOv3
image encoder runs on CUDA too ([DINOv3 encoder](#dinov3-encoder)); the VAE encoder, the decoders and the PLY writer
are not ported, and nothing registers with the lane engine or the server.

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
