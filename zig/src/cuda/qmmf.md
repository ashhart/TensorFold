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

# BF16 prompt tiles

On sm_12x, BF16 calls with at least 512 rows, N >= 128, K >= 512 and
128-column padded weights use a 128x128 CTA: WM=2, WN=4, 256 threads,
three K64 stages. Dynamic shared memory is 98,432 bytes, including the
stage barriers. Other formats, smaller grids and older devices use the
existing tiles. This threshold is a conservative screening choice; the
GB10 timing receipt must confirm that it pays.

Each stage copies two independent packed 64-column weight tiles with
`cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes`. Their
source stride remains `(K / 64) * 8192` bytes. One thread issues the bulk
copies and accounts for 16,384 bytes at that stage's barrier. Swizzled
activation rows use the existing coalesced 16-byte `cp.async` loads,
including zero fill past M and the caller's activation stride. Each warp
waits for both copy paths before it reads the stage. The CTA barrier
before the next prefetch also prevents reuse while another warp still
reads the old stage. Barrier parity changes each time a stage is reused.
The larger instances retain a 16-byte copy fallback in older SM images;
the launcher loads and selects them only on sm_12x.

## Byte contract

The BF16 accumulator starts at positive FP32 zero for each K slice.
Groups visit K in ascending 64-element steps. Each group executes the
same four `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
instructions, at K offsets 0, 16, 32 and 48, on the same operand fragments.
The hardware's internal K16 order is unchanged because the instruction
and fragment mapping are unchanged. Only independent output fragments
move between warps and CTAs; the stages change operand arrival time.

`Weight.sk` still overrides `splitK(N, K)`. It does not depend on M or
the CTA dimensions. A fused call resets the accumulator after exactly
`K / (64 * SK)` groups, retains slice zero, and adds each later slice in
ascending order with FP32 additions. This matches the old fused,
cluster and explicit reduction paths. The epilogue multiplies the final
FP32 sum by `Weight.scale` once, then either stores FP32 or applies one
round-to-nearest-even BF16 conversion. There is no new intermediate
rounding, split or reduction tree. Whole decoder layers retain their
shape-driven K slices; explicit pins let the comparison test exercise
the same slices on each tile geometry.

## Device checks and timing

Rebuild the qmmf image and runner together. For an existing complete
fatbin directory `FATBINS`, compile only the changed image, then embed it:

```sh
nvcc -fatbin -O3 -std=c++20 --expt-relaxed-constexpr \
  -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
  -gencode=arch=compute_121,code=sm_121 -Izig/kernels/cuda \
  zig/kernels/cuda/qmmf.cu -o "$FATBINS/qmmf.fatbin"
zig build compile-cuda-tests -Dfatbins="$FATBINS" -Doptimize=ReleaseFast
```

The build step emits the runner without installing it. To install the
runner with the same images, use `zig build -Dfatbins="$FATBINS"
-Doptimize=ReleaseFast`, then run:

```sh
zig-out/bin/tf-cuda-test qmmf-prompt
zig-out/bin/tf-cuda-test qmmf-prompt-bench
```

The device test compares all bytes against the unchanged lane kernels
for current whole-layer N/K = 16480/2560, 13952/2560, 2560/6144,
640/2560, 2560/640 and HC up 10240/320, both output types, SK=1 and the shape-driven split,
and M=1/16/64/65/127/128/129/511/512/1536/4218. It forces the new CTA
below the production threshold, checks production dispatch separately,
checks output guards and repeats the largest case five times. Small
SK=4 calls exercise both cluster and explicit reduction baselines.
Inputs use varied BF16 exponents and signs to expose a changed sum order.
A separate N=129 tail fixture plants NaNs, infinities, signed zeros and
subnormals in the inputs, and checks that inputs and weights stay unchanged.

The benchmark checks equality first, prints loaded register/shared-memory
usage and occupancy, then measures three A-B-A rounds with CUDA timing
events, three warmups and twenty calls per sample. It prints median
old/new times for all three shapes and a weighted 36/12/48-layer total.
Run on an idle GB10. These warm microbenchmarks do not prove the full
served mixer time or the complete kernel contract.

Each output tile now reuses operands over twice the rows and columns:
ideal input traffic per output halves, CTA count falls by about four,
and each resident CTA has eight warps instead of four. Bulk weight copies
also reduce load instructions. These are reasons to test a 1.6x speedup,
not a measured result. Register pressure, the three-stage shared-memory
allocation and hardware MMA throughput can limit the gain. Device byte
checks, timings and compile-limit receipts remain required before landing.
