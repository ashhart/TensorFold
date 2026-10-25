# Kernel contract

Every GPU kernel, ours or a contributor's, registers with the kernel contract before it lands. One harness
(`zig/src/core/contract/`) runs five gates on Metal, CUDA and HIP, and each gate has a planted control: a kernel built
to break that gate, which the gate must report as failing. A run where a control passes is a failed run, because then
the gate proves nothing.

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
- For gate 5, its launch in `zig/src/core/contract/registry.zig`: threads a group and threadgroup bytes (the source's,
  and any its encoder sets) on Metal, or symbol, image, threads a block, dynamic shared bytes and cluster width on CUDA,
  with the limits its compiles recorded. HIP launches are in `zig/src/core/contract/hip_records.zig`: symbol, code
  object, threads a workgroup and dynamic LDS bytes.

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
   loses its last input; it passes at whole tiles and fails at the edges. A kernel with bf16 rounding points between
   its steps (MLX's op-by-op rows) checks each output against float64's value at those points, as a range over the
   spread its fp32 steps allow, and its controls each drop one rounding point, which that check must catch.
4. fp32 sums. The inputs are integers whose partial sums reach about 2^20 in any order and stay below 2^24, so fp32
   sums are exact whatever the kernel's order; the second half of each row mirrors the first, so the result is small.
   Before the kernel runs, the harness replays the sums on the CPU in fp16 and bf16 (one chain, contiguous halves, and
   32 lanes then their halves) and requires each to miss by at least 100 times the exact-sum bound: fp16 overflows,
   bf16 loses thousands. Then the kernel must meet the exact-sum bound, which enforces "no fp16 sums where fp32 is
   used". Where the output's own rounding hides a sum's error (a bf16 softmax of at most 4,096 terms in [0, 1]), the
   spec says so and the run prints the margins it could prove. Control: a dot product summed in half.
5. Limits. Every registered launch against each chip we serve, from its declaration and recorded compiles, so this part
   runs with no GPU in `zig build test`. Metal: threads a group and threadgroup bytes against M1 to M5; on M1 and M2 a
   pipeline's thread limit falls as its registers rise (simd_qmm_bits' 8-bit tiles compile for 224), so every launch
   records its M1/M2 limit, unless its launcher sizes its groups to the pipeline or its source reserves the threads. A
   tensor-op kernel, which cannot be built for M1/M2 ahead of time, stays at 256 threads. CUDA: registers a thread
   (rounded per warp), static plus dynamic shared bytes, launch bounds and clusters against sm_80, sm_86, sm_89, sm_120
   and sm_121. HIP: VGPRs a wave (a workgroup's waves spread over the WGP's four SIMDs, each rounded to the chip's
   allocation step), static plus dynamic LDS bytes, the kernel's `__launch_bounds__` and its wave size against gfx1030,
   gfx1100, gfx1151 and gfx1201, one chip a generation. Tools re-derive the records: on a Mac, `tf-kernel-contract
   --offline DIR` compiles each pipeline for M1 and M2 with Xcode's Metal toolchain (native code built for T threads
   equal to the plain build means the plain pipeline runs T); on a CUDA build machine, `tf-cuda-test
   kernel-contract-limits DIR` reads registers, shared bytes and launch bounds from the fatbins' ELF images for every
   SM, counted as the driver counts them (sm_120 and sm_121 images keep registers only in the `.nv.info` register
   records, and add the block's reserved 1 KiB to `.nv.shared`); with hipcc, `tf-hip-kernel-contract limits DIR` reads
   VGPRs, SGPRs, LDS, scratch, the bound and the wave size from the AMDGPU metadata note of each chip's code objects. On
   the GPU itself the runners check the loaded kernels and the device's own limits. Controls: Metal, the row
   projection's pipeline launched at 1,024 threads where M1 and M2 compile it for 512, and 36 KiB of threadgroup memory;
   CUDA, a kernel launched past its `__launch_bounds__`, and one needing more registers than a block has; HIP, a kernel
   launched past its `__launch_bounds__`, and 40 KiB of static LDS launched with 32 KiB more (hipcc fits a kernel's
   VGPRs to its bound, so a register overflow shows as a launch past the bound).

## Running it

On a Mac, `zig build kernel-contract -Dcpu=apple_m1` runs every Metal kernel registered for this GPU, the controls
and gate 5. `zig build tf-kernel-contract` builds the program alone; `--only TEXT` runs the kernels whose name holds
TEXT, `--offline DIR` adds the M1/M2 compiles, and `--repro` prints the smallest input behind each known finding.

On a CUDA machine, build with `-Dnvcc` (the controls are compiled only then), and run `zig build kernel-contract
-Dnvcc=/usr/local/cuda/bin/nvcc`. For the limits of every SM, `zig build fatbins kernel-contract-fatbins -Dnvcc=...
-Dsm=80,86,89,120,121`, then `tf-cuda-test kernel-contract-limits zig-out/fatbin`. Its `RECORD` lines are the values
`registry.zig` holds for that build's nvcc (see [CUDA records per nvcc release](#cuda-records-per-nvcc-release)).

On an AMD machine, run `zig build hip-kernel-contract -Dhip-arch=ARCH -Dhipcc=...` on a GPU of that chip (the
controls are built with the kernels). For the limits of every served chip, `zig build hip-kernel-contract-limits
-Dhipcc=...` compiles every kernel group and the controls for gfx1030, gfx1100, gfx1151 and gfx1201 and checks each
record against them, no GPU needed; `hip-kernel-contract-objects` installs those code objects into `zig-out/hsaco/`.
Its `RECORD` lines are the values `hip_records.zig` holds for that hipcc.

Every run ends with a `SUMMARY` line: kernels passing gates 1 to 4, known findings, controls red, and limit problems.
The program exits 1 on any failed gate, any control that stayed green, or any limit problem.

## Adding a kernel

1. Write its test type in `zig/tests/contract/` (Metal), `zig/tests/cuda/contract_*.zig` (CUDA) or
   `zig/tests/hip/contract_*.zig` (HIP): `prepare` builds
   the inputs for a shape and a gate's fill (usually through an input family), `run` launches once and waits, `count`
   and `reference` give the outputs and their float64 values, `simulate` replays the sums for gate 4.
2. Add it to the runner's list, with its served shapes and tiles.
3. Add its launches to `registry.zig`, and fill the records from `--offline` (Metal) or `kernel-contract-limits`
   (CUDA: nvcc 13.0's in `cuda`, and each later release's set where that release compiles it differently); a HIP
   kernel's go to `hip_records.zig`, filled from `hip-kernel-contract-limits`.
4. Post the `SUMMARY` lines from each backend you ran in the pull request, and say which chips you could not run.

A replacement kernel names the kernel it replaces and gives `replaced`, which runs that kernel on the same inputs.

## Known findings

A gate failure on a kernel already shipped is reported, not hidden. Until its fix lands, its spec can carry a
`known` entry naming the gate and what fails. That gate must keep failing (the run prints `KNOWN`); once a fix makes
it pass, the run fails until the entry is removed. Today: affine_mm's 4-bit form gives NaN for an infinite input where
MLX's formula keeps the infinity, because its codes enter the MMAs as q + 128 and its bias as b - 128 s. GLM-5.3-Flash's
`tf_glm_scores` scores a NaN dot as 0 where float64 keeps the NaN: its relu is `fmaxf(x, 0)`, as `tl.maximum` is in the
Python family's Triton glue it was ported from. Nemotron's Metal `tf_xup_w3` gives 0 for a NaN dot where float64 keeps
the NaN: its relu2 is `metal::max(bfloat(acc), 0.0f)`; the other up pipelines give its bytes on the same inputs.

HIP’s MLX stream tile keeps its finite dot2 sums unchanged. A non-finite result with an actual non-finite
operand is recomputed by full quantization groups using the factored float64 formula. The plain tile is
registered at all six supported bit widths and groups of 32, 64 and 128, with no known finding.

The HIP relu^2 replacement retains every non-finite plant and checks both unfused bytes and every
independent float64 output class. No inherited infinity discrepancy is accepted.
The paired launch retains every table plant and checks both members' separate-launch bytes and every
float64 output class, without an inherited exception.

## Coverage

Metal: softmax, RMS norm, bf16 top-k, the row projection (3, 4, 6 and 8 bits), affine_mm (4 and 8 bits), simd_qmm
(4 bits) and simd_qmm_bits (3, 6 and 8 bits), and the lane projection (6-bit groups of 32 on every chip; 4- and 8-bit
formats on the M5's tensor units, where its reg kernel must equal the cooperative one bit for bit), and tile_qmm's
prompt tiles and their decode twin (4, 6 and 8 bits; the twin bit-equal to the tiles). Gate 5 also holds the rest of
tile_qmm's pipelines and GLM-5.3-Flash's hc_expand_10.
LFM2.5-24B-A2B's shared rounds: the greedy pick (`lfm2_argmax`) and the conv and attention kernels that run every
stream's rows in one dispatch, each bit-equal to the per-stream kernels it replaces. Four controls edit the source: a tie
to the highest id, a nonfinite logit not flagged, and every stream reading or writing the first stream's state or cache.
Their launches are in `lfm2_records.zig` with the M1 and M2 limits from `--offline`; `lfm2_attn_segs` reserves its 128
threads in its source, as Metal cannot build it for M1 or M2 at another count.
Metal also checks Nemotron's routed experts on the engine's route tables, one row to 128: `tf_xup_w3` and `tf_xdown_w3`
against float64 at tf_load16's bf16 x sums (up's relu2 as an interval over its two bf16 roundings), and the rows2, rows4
and six geometry pipelines of each projection bit-equal to them (two controls: member rows read from one row, and
w3's x sums without their bf16 steps).
CUDA: the RMS norm rows (16-byte form and the fp32-out tree), the bf16 residual's norm tails (MLX's rounded rows and
the once-rounded ones, each checked on its residual and its output, with four controls each dropping a rounding point), Nemotron's router logits (and their 128-row tiles,
bit-equal to them) and top-k, the latent top-k's decode form (bit-equal to `tf_super_topk`), the latent combine and
shared add over bf16 share sums (bit-equal to `tf_super_combine` and `tf_super_moe_out` over the fp32 shares), the
Mamba scan of a verify window (its host twin's bytes), qmm_group's cluster tile, lane_gemv and its split-K form (both
bit-equal to it), the split-K form segmented over projections of one input (bit-equal to `decode()` on each), prompt16's FP4 tile with bf16 and fp32 out, qmmf's FP4 tiles with fp32 out (K slices met in a
cluster, fused, or by `reduce_kernel<true>`), the checkpoint math's FP8 128-row prompt tiles (both slice meetings,
bit-equal to the 64-row tiles) and NVFP4 experts: the decode-sized forms and the prompt tiles (up to down's NVFP4
rows, fp32 down and the down summing two width shares, each bit-equal to the kernels it replaces; sm_120 up), and the
reduce-scatter's `tf_rank_sum`, GLM-5.3-Flash's float32 norm weight forms `tf_glm_rmsnorm_4096_f32w` (its rows and
64-sums) and `tf_glm_hc_finish_f32w` (its normed rows, 64-sums, post and Sinkhorn comb), and the activation rows in
MLX's bf16 steps (GeGLU, and SwiGLU as DFlash drafters round it; twelve controls, each dropping one rounding point;
their 64-value sums bit-equal to the fp32 tree).
CUDA also checks the MLX affine token rows (4 bits, and 2 to 6 or 8 bits times a bf16 scale, each
output bit-equal to float64 at its rounding points; two controls, one dropping the rounding point, one folding the scale
into the group's scale and bias).
CUDA also checks qk_rows' attention inputs at head dims to 512 with a unit v norm and MLX's bf16
rows (three controls, each dropping one option).
CUDA also checks `window_attention`'s rows at head dims 256 and 512 (`window_rows.cu`: decode chunks and their merge,
prompt tiles with P as bf16 or in two halves, a row's last key, windows over a ring or none over a linear cache), each
output within float64's range over the spread its fp32 steps and tensor-core sums allow (`core/attention_range.zig`;
seven controls, each one planted fault).
CUDA also checks Dory's nGPT rows: the L2 norm rows with and without their bf16 add, and the normalized lerp residual
(`norms_l2.cuh`), each output within float64's range at its bf16 points (three controls, each dropping one point).
CUDA also checks Dory's paged caches (`window_pages.cu`): `tf_wrows_put` places rows in a shuffled arena bit for bit, and the
head dim 256 decode and prompt kernels read them through a page table, each output within float64's range (two controls:
swapped stream tables, a stale page id).
CUDA also checks Dory's attention inputs (`qk_l2.cuh`): RoPE first, each head's L2 norm, q times its scale vector, v as is, at
head dim 256 and positions past 199,800 (two controls, each dropping one rounding point).
CUDA also checks Dory's activation rows (`act_l2.cuh`): the scaled sigmoid output gate over all heads and the scaled SwiGLU,
each row then L2-normed, against float64's ranges at their bf16 points and expf's ulps (four controls, one point each).
CUDA also checks `dense_rows`' scaled epilogue (Dory's logits, fp32(bf16(x . w)) times a per-column scale), bit-equal to its
unscaled tile's outputs rounded and scaled on the host.
HIP: the RMS norm rows, a wave a row (fp32, fp16 and bf16 out) and a block a row (fp16 and bf16 out).

RADIO's fused bias, activation and residual (`tf_vit_bias_act`) replaces `tf_vit_bias` then
`tf_vit_pointwise`, byte for byte.
CUDA also checks GLM-5.3-Flash's KV page kernels on a page table out of order: the latent write and the pooled keys
(`tf_glm_lat_write`, `tf_glm_pool_keys`), the index scores and the selected tokens' rows (`tf_glm_scores`,
`tf_glm_dsa_tokens`), and the MLA chunks through their merge (`tf_glm_chunks_16` and `tf_glm_chunks_32`, dense and
sparse). Its prompt indexer (`tf_glm_index_scores`, `tf_glm_index_pools`) is bit-equal to `tf_glm_scores` and
`tf_glm_select_rows`, and its decode select at a 1,024-thread block a row (`tf_glm_select_rows_wide`) to
`tf_glm_select_rows`; both stated rules rank any NaN above +inf, as the GPU's `s + 0` makes every NaN one
positive NaN.

GLM's Metal window projections register narrow and wide rows at 2, 3, 4, 5, 6 and 8 bits, plus both gathered paths.
Independent FP64 linear cases accompany byte replacements against frozen pre-window-optimization kernels.
The gathered SwiGLU keeps its predecessor's clipped BF16 output and nonfinite rules; gate/up taps qualify its dots separately.
Controls dispatch real source mutations that couple rows or select the wrong expert; host tests validate their witnesses.
CUDA also checks the affine-3 experts' split decode pair, bit-equal to the pair kernel it replaces (two controls,
one dropping a last round of K groups short of the warps, one chaining the groups in reverse).
CUDA also checks Flash Next's NVFP4 experts on prompt tiles with the decode form's sums (SwiGLU gate-up and fp32
down, each bit-equal to the decode instance it replaces; the scaled-code prompt tile is the planted control).

## CUDA records per nvcc release

nvcc allocates registers differently from one release to the next, so the CUDA records are kept per release.
`registry.cuda` holds what nvcc 13.0 compiles, the compiler of the release fatbins. `registry.cuda_sets` lists what
each later release compiles differently from the release before it: nvcc 13.2's set holds `tf_rms_rows`, `lane_gemv`
and its split-K form, prompt16's FP8 tile, qmmf's FP4 32-row cluster tile, five 64-row fused-slice tiles of the MXFP8
and FP32-output paths, FP8G's 32-row cluster tile with FP32 output, the common EXL3 prompt products, Super's
decode-sized latent top-k and NVFP4 up experts, the Mamba scan and Super's NVFP4 prompt down kernels, three of
GLM-5.3-Flash's packed matmuls and three of its DFlash2 launches, and the training launches it compiles differently,
and nvcc 13.3 keeps 13.2's and changes `tf_nemo_topk` on sm_120 and sm_121, four of GLM-5.3-Flash's KV page
launches, its two prompt-indexer launches and its wide decode select. `kernel-contract` and
`kernel-contract-limits` read the release from each image's `.note.nv.tkinfo`, check the image against the newest set
at or below that release (13.0's when an image names none), and print which set they used. A release with no set of
its own is checked against the one below it, so its differences show as stale records: re-derive them with
`kernel-contract-limits` and add a set for that release rather than loosening the audit. Launch dimensions, dynamic
shared memory, the device's limits and the planted controls are checked the same way under every release.

A restored 13.0 receipt can arrive after the later compiler sets. Such records mark `later_pending` so 13.0 and
13.1 use their base usage, while 13.2 and later keep their independently measured or pending selection.
A missing later receipt remains pending; a 13.0 measurement never substitutes for it.

Gemma 4 CUDA tower QKV norms and nine-term pooling register FP32 sums and BF16 outputs.
Their contract keeps the exact FP32 mirror and adds independent FP64 ranges at the BF16 nodes.
The sum fixtures use integers with exact FP32 partials below 2^24, including pooling cancellation near 2^20;
FP16 and BF16 simulations must fail in each of the three accumulator orders. Pooling declares its nine-term
reduction separately from the physical grid width, so the small stress grid remains a valid launch.
NaN and both infinity signs have explicit class controls. These registrations still require the device gates.
