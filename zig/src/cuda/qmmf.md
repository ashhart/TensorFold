# FP32 projection outputs

`cuda.qmmf.Lane.matmulF32` has the same inputs, packed weights, stride,
shape-driven split-K plan and scratch contract as `matmul`. Its output is
`[M, N]` FP32, requiring `M * N * 4` bytes. It preserves FP32 output for a
later residual or tensor-rank sum; the caller chooses when to round to BF16.

All five lane formats are supported: NVFP4, tensor-scaled FP8, MXFP8,
block-scaled FP8 and BF16. The change exposes the existing kernel template's
FP32 output path. It does not change weight packing or accumulation order.
Rebuild the qmmf fatbin together with the native runtime: older fatbins do
not contain the FP32 instances or reduction symbol.

`tf-cuda-test projection-f32` compares each format against an independent
FP64 calculation, checks conversion to the existing BF16 result, and varies
row count, activation stride, partial output tiles, fused reduction,
cluster reduction and explicit split reduction. The fixture is synthetic;
model correctness and multi-device collective behavior require separate
qualification.
