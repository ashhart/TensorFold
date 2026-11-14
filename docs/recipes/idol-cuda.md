# Idol's denoiser on CUDA

Idol's denoiser (see `idol.md`) on the CUDA runtime, beside the Metal family and driven by the same case driver
(`zig/tests/idol_run.zig`): `tf-idol` on Metal and on CUDA reads the same case and runs the same gates. The image
encoders, the decoders and the PLY writer are not ported, and nothing registers with the lane engine or the server.

**Where it has run.** The kernel contract has passed on a GB10 (sm_121) and on an RTX 5090 (sm_120); the synthetic
case has run on the RTX 5090. A case run on a GB10 is still to come.

- GB10, `idol-contract` on `776a3335d` ([run][gb10]):
  `SUMMARY idol: 13 kernels, 13 pass gates 1-4, 0 fail; 3 controls: 3 red as planted, 0 missed`.
  The rerun on `c3a6c0cb7`, after the controls moved to their own header, gave the same: 13 of 13 kernels pass gates
  1-4 and 3 of 3 controls are red as planted ([run][gb10-rerun]).
- RTX 5090, CUDA 13.3 (nvcc 13.3.73), on `f5b88dd98` ([run][rtx5090]): `idol-contract` gives the same SUMMARY line,
  and `tf-idol` on the shared synthetic case (`idol-synthetic-case-2731c229`) has 100 of 100 branch-steps within the
  gate and a repeat with identical bytes. Idol's CUDA sources are unchanged since `f5b88dd98`.

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
