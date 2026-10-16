# Native MXFP8 projections

`cuda.mxfp8` packs row-major E4M3 code bytes and one UE8M0 scale byte per
32 input channels for `qlinear.Weight.mxfp8`. Both prompt and decode calls use
the lane matmul, with BF16 inputs, FP32 accumulation and BF16 output. Packing
permutes bytes; it does not requantize weights or expand them to BF16.

The logical shape is `[N, K]`, with positive dimensions and K divisible by 64.
`packCodes` writes `padded(N) * K` bytes; `packScales` writes
`padded(N) * K/32` bytes. Rows are padded to 128; padded codes are zero and
padded scales are one. `weight` returns the borrowed device view. The caller
owns its device buffers and the lane's split-K scratch.

UE8M0 bytes 0 through 254 mean `2^(byte - 127)`; byte 255 is NaN. In
particular, byte zero is a nonzero subnormal FP32 scale. See the
[CUDA datatype contract](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/struct____nv__fp8__e8m0.html).

`tf-cuda-test mxfp8` checks all 256 scale encodings through the production
matmul, then compares synthetic projections with an independent FP64
calculation. It also checks strided activations, partial output tiles,
row counts 1/7/17/33/256, prompt/decode dispatch, and cluster versus explicit
split reduction. Host tests cover scale permutation, padding, shared-memory
sizes, symbols and format refusals.

This primitive does not admit new checkpoint families or certify model
correctness, tensor parallelism or performance. Rebuild the qmmf fatbin with
the native runtime; previous fatbins lack the MXFP8 instantiations.
