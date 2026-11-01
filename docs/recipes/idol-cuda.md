# Idol's denoiser on CUDA

Idol's denoiser (see `idol.md`) on the CUDA runtime, beside the Metal family and driven by the same case driver
(`zig/tests/idol_run.zig`): `tf-idol` on Metal and on CUDA reads the same case and runs the same gates. Its decoders
follow ([Decoders](#decoders)). The image encoders are not ported, and nothing registers with the lane engine or the
server.

**Where it has run.** The kernel contract and the synthetic case have passed on a GB10 (sm_121) and on an RTX 5090
(sm_120).

- GB10, nvcc 13.0, on `05ab1990a` (GustavBlack's [run][gb10-current] of 15 Nov on #1767): `idol-contract` gives
  `SUMMARY idol: 13 kernels, 13 pass gates 1-4, 0 fail; 3 controls: 3 red as planted, 0 missed`, and `tf-idol` on the
  shared synthetic case (`idol-synthetic-case-2731c229`) has 100 of 100 branch-steps within the gate, all 23 joint
  blocks passing at `blocks 0` and a repeat with identical bytes; its 50 steps took 53.04 s. Its peak memory is not
  recorded yet.
- Earlier GB10 contract runs on `776a3335d` ([run][gb10]) and on `c3a6c0cb7`, after the controls moved to their own
  header ([run][gb10-rerun]), gave the same SUMMARY line. Both come before `36828402d`, which recompiled the norm
  kernels.
- RTX 5090, CUDA 13.3 (nvcc 13.3.73), on `f5b88dd98` ([run][rtx5090]): `idol-contract` gives the same SUMMARY line,
  and `tf-idol` on the shared synthetic case (`idol-synthetic-case-2731c229`) has 100 of 100 branch-steps within the
  gate and a repeat with identical bytes. Idol's CUDA sources are unchanged since `f5b88dd98`.

[gb10-current]: https://github.com/ashhart/TensorFold/pull/1767#issuecomment-6886995407
[gb10]: https://github.com/ashhart/TensorFold/pull/1767#issuecomment-6777718609
[gb10-rerun]: https://github.com/ashhart/TensorFold/pull/1767#issuecomment-6802382536
[rtx5090]: https://github.com/ashhart/TensorFold/pull/1767#issuecomment-6869833180

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
  The softmax denominator sums the fp32 P, and P is rounded to fp16 only for P·V; Metal's `id_attention` rounds P to
  fp16 first and sums the rounded values.
- `idol.cu` builds with the glue flags (`--fmad=false`, IEEE division and square root). `F.normalize`'s clamp keeps a
  NaN norm, as on Metal.

## Kernel contract

`tf-cuda-test idol-contract` runs gates 1-4 over every `idol.cu` kernel and the attention, with three controls that
each drop one fp16 rounding point; the full contract runs it among its family sets. Gate-5 records
(`zig/src/core/contract/idol_cuda_records.zig`) are nvcc 13.0's for sm_80, 86, 89, 120 and 121. On a GB10 and on
an RTX 5090 it gives the SUMMARY line above. On the 5090, nvcc 13.3's registers for four launches differ from the 13.0
records, all within their bounds.

## Run a case

```sh
zig build idol-cuda
zig-out/bin/tf-idol WEIGHTS case/
zig-out/bin/tf-idol WEIGHTS case/ blocks 0
```

The case comes from `tools/zig/idol_case.py`, as in `idol.md`.

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
