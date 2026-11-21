# CUDA kernel contract records

The shared gates, registration rules and runner commands are in the [kernel contract](kernel-contract.md).

## CUDA records per nvcc release

nvcc allocates registers differently from one release to the next, so the CUDA records are kept per release.
`registry.cuda` holds what nvcc 13.0 compiles, the compiler of the release fatbins. `registry.cuda_sets` lists what
each later release compiles differently from the release before it: nvcc 13.2's set holds `tf_rms_rows`, `lane_gemv`
and its split-K form, prompt16's FP8 tile, qmmf's FP4 32-row cluster tile and its FP4 and MXFP8 16-row cluster
tiles, five 64-row fused-slice tiles of the MXFP8
and FP32-output paths, FP8G's 32-row cluster tile with FP32 output, the common EXL3 prompt products, Super's
decode-sized latent top-k and NVFP4 up experts, the Mamba scan and Super's NVFP4 prompt down kernels, three of
GLM-5.3-Flash's packed matmuls and three of its DFlash2 launches, and the training launches it compiles differently,
and nvcc 13.3 keeps 13.2's and changes four of GLM-5.3-Flash's KV page launches, its two prompt-indexer launches,
its wide decode select and the span scan `tf_glm_dsel_scan`. `kernel-contract` and
`kernel-contract-limits` read the release from each image's `.note.nv.tkinfo`, check the image against the newest set
at or below that release (13.0's when an image names none), and print which set they used. A release with no set of
its own is checked against the one below it, so its differences show as stale records: re-derive them with
`kernel-contract-limits` and add a set for that release rather than loosening the audit. Launch dimensions, dynamic
shared memory, the device's limits and the planted controls are checked the same way under every release.

`-Dnative-fp-decode` (off by default; nvcc 13.2 or later) builds prompt16, qmmf and fp8_experts for `sm_<n>a` from
sm_120 on and decodes their FP4/FP8 weight pairs with PTX ISA 9.2's `cvt.rn.bf16x2.e2m1x2`/`.e4m3x2` (qmmf: FP4 only).
Known finding: the hardware decode equals `fp4pair` on all 256 e2m1 bytes and `fp8pair` on every e4m3 pair without a
NaN code; the 1,020 pairs holding `0x7F` or `0xFF` decode to NaN, where the software gives ±480 (`tf-cuda-test
native-fp`). Those codes are NaN in e4m3, so only a broken checkpoint can tell the two apart. Default images keep the
software decode.

A restored 13.0 receipt can arrive after the later compiler sets. Such records mark `later_pending` so 13.0 and
13.1 use their base usage, while 13.2 and later keep their independently measured or pending selection.
A missing later receipt remains pending; a 13.0 measurement never substitutes for it.

Gemma 4 CUDA tower QKV norms and nine-term pooling register FP32 sums and BF16 outputs.
Their contract keeps the exact FP32 mirror and adds independent FP64 ranges at the BF16 nodes.
The sum fixtures use integers with exact FP32 partials below 2^24, including pooling cancellation near 2^20;
FP16 and BF16 simulations must fail in each of the three accumulator orders. Pooling declares its nine-term
reduction separately from the physical grid width, so the small stress grid remains a valid launch.
NaN and both infinity signs have explicit class controls. These registrations still require the device gates.
