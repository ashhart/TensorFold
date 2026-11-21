# CUDA kernel contract owners

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

## Shared M5 attention products

The shared `attention_nax` defaults to the relaxed tensor P x V: float P is truncated to ten fraction bits.
`TF_ATTENTION_PV_FP32=1` opts into full float P with FP32 FMA accumulation on M5; unset preserves the old arithmetic.
The cut registration's independent formula includes the operand cut, while the full registration uses the full P.
A third registration executes the real cut kernel against the full FP32 bound and must fail the shapes gate.
The product fixture writes float output so BF16 rounding cannot conceal that control.
Both modes still require exceptional-class, repeat, guard, shape, accumulator and separate launch checks.
See [the mode's scope and device checks](recipes/m5-attention-precision.md); other families' rounding is unchanged.

Flash Next's BF16 shared-down entries `tf_qmmf_shared16_half`, `tf_qmmf_shared16`, `tf_qmmf_shared32` and
`tf_qmmf_shared64` preserve the contiguous projection's FP32 result and store it directly in expert slot 10.
The baseline shared expert has all 640 intermediate channels and runs only on rank 0 at two ranks.
The contract checks all four geometries, nonfinite inputs, output bytes and untouched routed slots.
`moetile_records.zig` holds launch geometry and measured nvcc 13.0 resources on all five served SMs.
Later compiler receipts remain pending. Check each build with `tf-cuda-test kernel-contract-limits zig-out/fatbin`.
`tf-cuda-test moetile` compares the direct slot and the BF16 prompt tile plus scatter with the lane's contiguous
product and scatter at 1, 16, 17, 32, 64, 65, 1536 and 4218 rows, five repeats with guards.
`moetile-contract` runs the replacement gates and planted control.
`moetile-bench` rotates GPU event timings of the three at 64 to 4218 rows; no timing is a gate.

BF16 prompt K32 staging registers two unfused 128x128 entries, one for BF16 output and one for FP32 output.
They read the parent's K64 weight packs in two K32 halves and keep the K16 MMA order and output rounding.
Only unsplit prompt calls select K32; fused slices retain the parent K64 tile.
`qmmf-prompt` compares each new output directly with the parent's 128x128 K64 output, including forced small rows.
`qmmf-prompt-contract` keeps the four parent entries and adds both K32 replacements and their scaled controls.
Each new control launches K32 for its output type at small and production row counts.
`qmmf-prompt-bench` times the 64-row lane, prompt16, the parent K64 tile and K32 on each Flash Next BF16 shape.
Measured nvcc 13.0 resources cover both K32 entries on sm_120 and sm_121; later compiler receipts remain pending.
