# Kernel contract

Every GPU kernel, ours or a contributor's, registers with the kernel contract before it lands. One harness
(`zig/src/core/contract/`) runs five gates on Metal and on CUDA, and each gate has a planted control: a kernel built to
break that gate, which the gate must report as failing. A run where a control passes is a failed run, because then the
gate proves nothing.

A kernel's own unit test usually compares a few outputs within one tolerance. The gates ask more: what happens to NaN
and infinities, whether the bytes repeat, whether every tile edge is covered, whether the sums are really fp32, and
whether the launch fits every chip we serve.

## What a kernel registers

The test of a kernel fills a `contract.spec.Spec` and hands the harness a few hooks (`contract.gates.VTable`, or
`gates.hooks(T)` for a test type with `prepare`, `run`, `count` and `reference`).

- Its float64 reference: the kernel's own formula in float64. The input families in `zig/src/core/contract/` cover the
  common cases: `affine` (MLX's affine-quantized projections), `dense` (dots against dequantized weights, in K
  slices) and `rows` (RMS norm, softmax, sums, dots, maxima). Each builds inputs for every gate, computes the reference
  and the magnitude of the terms each output sums, and can replay the sums in another accumulator.
- Its served envelope: the shapes the engine calls it at, and for each dimension its tile, the step a valid size
  moves by and the least valid size.
- Its accumulator (`f32` for every kernel here) and the depth of its longest chain of roundings at a shape, plus any
  fp32 operations after the sums (`ops`) and intermediate roundings to the output type (`mids`).
- Whether it may write in place. If it replaces another kernel, the kernel whose bits it must equal.
- Its NaN rule: `propagate` (as float64 arithmetic on its formula does), or a stated rule the test checks itself
  (top-k, argmax, sampling and routing kernels state theirs).
- For gate 5, its launch in `zig/src/core/contract/registry.zig`: threads a group and threadgroup bytes on Metal, or
  symbol, image, threads a block, dynamic shared bytes and cluster width on CUDA, with the limits its compiles recorded.

## The gates

1. Non-finite. NaN, +Inf and -Inf go into the inputs at known places (a row each, one row holding both infinities).
   Every output's class (finite, NaN, +Inf, -Inf) must equal the float64 reference's, tolerance-free. A kernel with a
   stated rule checks the rule instead. Control: a row maximum by `fmax`, which drops a NaN.
2. Repeat and aliasing. Five runs on the same inputs give the same bytes, the inputs are unchanged afterwards, the
   guard bytes around every buffer are intact, and no output overlaps an input unless the spec says in place.
   Controls: a row sum by atomic adds (its order changes from run to run), and a row sum that writes into its input.
3. Shapes. Every served shape, then each tiled dimension at its least size, a tile minus a step, a tile, a tile plus a
   step and the largest served size. Each output must be within `|want| * outputs' half ulp + gamma(depth) * magnitude`
   of float64, or for a replacement kernel bit-equal to the kernel it replaces. Control: a row sum whose partial tail
   loses its last input; it passes at whole tiles and fails at the edges.
4. fp32 sums. The inputs are integers whose partial sums reach about 2^20 in any order and stay below 2^24, so fp32
   sums are exact whatever the kernel's order; the second half of each row mirrors the first, so the result is small.
   Before the kernel runs, the harness replays the sums on the CPU in fp16 and bf16 (one chain, contiguous halves, and
   32 lanes then their halves) and requires each to miss by at least 100 times the exact-sum bound: fp16 overflows,
   bf16 loses thousands. Then the kernel must meet the exact-sum bound, which enforces "no fp16 sums where fp32 is
   used". Where the output's own rounding hides a sum's error (a bf16 softmax of at most 4,096 terms in [0, 1]), the
   spec says so and the run prints the margins it could prove. Control: a dot product summed in half.
5. Limits. Every registered launch against each chip we serve, from its declaration and recorded compiles, so this
   part runs with no GPU in `zig build test`. Metal: threads a group and threadgroup bytes against M1 to M5; on M1
   and M2 a pipeline's thread limit falls as its registers rise (simd_qmm_bits' 8-bit tiles compile for 224), so
   every launch records its M1/M2 limit, unless its launcher sizes its groups to the pipeline or its source reserves
   the threads. A tensor-op kernel, which cannot be built for M1/M2 ahead of time, stays at 256 threads. CUDA:
   registers a thread (rounded per warp), static plus dynamic shared bytes, launch bounds and clusters against sm_80,
   sm_86, sm_89, sm_120 and sm_121. Tools re-derive the records: on a Mac, `tf-kernel-contract --offline DIR` compiles
   each pipeline for M1 and M2 with Xcode's Metal toolchain (native code built for T threads equal to the plain build
   means the plain pipeline runs T); on a CUDA build machine, `tf-cuda-test kernel-contract-limits DIR` reads registers,
   shared bytes and launch bounds from the fatbins' ELF images for every SM, counted as the driver counts them (sm_120
   and sm_121 images keep registers only in the `.nv.info` register records, and add the block's reserved 1 KiB to
   `.nv.shared`). On the GPU itself both runners check the loaded pipelines and the device's own limits. Controls:
   Metal, the row projection's pipeline launched at 1,024 threads where M1 and M2 compile it for 512, and 36 KiB of
   threadgroup memory; CUDA, a kernel launched past its `__launch_bounds__`, and one needing more registers than a block
   has.

## Running it

On a Mac, `zig build kernel-contract -Dcpu=apple_m1` runs every Metal kernel registered for this GPU, the controls
and gate 5. `zig build tf-kernel-contract` builds the program alone; `--only TEXT` runs the kernels whose name holds
TEXT, `--offline DIR` adds the M1/M2 compiles, and `--repro` prints the smallest input behind each known finding.

On a CUDA machine, build with `-Dnvcc` (the controls are compiled only then), and run `zig build kernel-contract
-Dnvcc=/usr/local/cuda/bin/nvcc`. For the limits of every SM, `zig build fatbins kernel-contract-fatbins -Dnvcc=...
-Dsm=80,86,89,120,121`, then `tf-cuda-test kernel-contract-limits zig-out/fatbin`. Its `RECORD` lines are the values
`registry.zig` holds.

Every run ends with a `SUMMARY` line: kernels passing gates 1 to 4, known findings, controls red, and limit problems.
The program exits 1 on any failed gate, any control that stayed green, or any limit problem.

## Adding a kernel

1. Write its test type in `zig/tests/contract/` (Metal) or `zig/tests/cuda/contract_*.zig` (CUDA): `prepare` builds
   the inputs for a shape and a gate's fill (usually through an input family), `run` launches once and waits, `count`
   and `reference` give the outputs and their float64 values, `simulate` replays the sums for gate 4.
2. Add it to the runner's list, with its served shapes and tiles.
3. Add its launches to `registry.zig`, and fill the records from `--offline` (Metal) or `kernel-contract-limits`
   (CUDA).
4. Post the `SUMMARY` lines from each backend you ran in the pull request, and say which chips you could not run.

A replacement kernel names the kernel it replaces and gives `replaced`, which runs that kernel on the same inputs.

## Known findings

A gate failure on a kernel already shipped is reported, not hidden. Until its fix lands, its spec can carry a
`known` entry naming the gate and what fails. That gate must keep failing (the run prints `KNOWN`); once a fix makes
it pass, the run fails until the entry is removed. Today: affine_mm's 4-bit form gives NaN for an infinite input where
MLX's formula keeps the infinity, because its codes enter the MMAs as q + 128 and its bias as b - 128 s.

## Coverage

Metal: softmax, RMS norm, bf16 top-k, the row projection (3, 4, 6 and 8 bits), affine_mm (4 and 8 bits), simd_qmm
(4 bits) and simd_qmm_bits (3, 6 and 8 bits), and the lane projection (6-bit groups of 32 on every chip; 4- and 8-bit
formats on the M5's tensor units, where its reg kernel must equal the cooperative one bit for bit).
CUDA: the RMS norm rows (16-byte form and the fp32-out tree), Nemotron's router logits and top-k, qmm_group's cluster
tile, lane_gemv and its split-K form (both bit-equal to it), and prompt16's FP4 tile.
