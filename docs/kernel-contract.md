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

`runAll` calls each kernel's optional `VTable.release` before moving to the next kernel, including after a failed
gate or planted control. `gates.hooks(T)` supplies `rig.reset()` for value or pointer rigs with a `reset` method.
Direct `one` callers remain responsible for cleanup.

- Its float64 reference: the kernel's own formula in float64. The input families in `zig/src/core/contract/` cover the
  common cases: `affine` (MLX's affine-quantized projections), `dense` (dots against dequantized weights, in K
  slices) and `rows` (RMS norm, softmax, sums, dots, maxima). Each builds inputs for every gate, computes the reference
  and the magnitude of the terms each output sums, and can replay the sums in another accumulator.
- Its served envelope: the shapes the engine calls it at, and for each dimension its tile, the step a valid size
  moves by and the least valid size.
- Its accumulator (`f32` where a kernel sums) and the depth of its longest chain of roundings at a shape, plus any
  fp32 operations after the sums (`ops`), intermediate roundings to the output type (`mids`), and the precision it
  stores at when that is narrower than the output's element (`store`: bf16 values in fp32 words).
- Whether it may write in place. If it replaces another kernel, the kernel whose bits it must equal.
- Its NaN rule: `propagate` (as float64 arithmetic on its formula does), or a stated rule the test checks itself
  (top-k, argmax, sampling and routing kernels state theirs).
- For gate 5, its launch in `zig/src/core/contract/registry.zig`: threads a group and threadgroup bytes (the source's,
  and any its encoder sets) on Metal, or symbol, image, threads a block, dynamic shared bytes and cluster width on CUDA,
  with the limits its compiles recorded. HIP launches are in `zig/src/core/contract/hip_records.zig`: symbol, code
  object, threads a workgroup and dynamic LDS bytes. XPU launches are in `zig/src/core/contract/xpu_records.zig`:
  symbol, image, work-group size, static SLM bytes, sub-group size and GRF mode, with the values ocloc's `.ze_info`
  recorded for the card class. The seam's records are in `metal_pair_records.zig`, which `registry.zig` concatenates.

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
-Dnvcc=/usr/local/cuda/bin/nvcc`. A family's suite alone runs as `zig build int4-contract -Dnvcc=...` (the GPTQ int4
experts and head, its `SUMMARY` line and each control) or as `tf-cuda-test int4-contract` on a built runner; its gate
registrations are in the filtered list too, so `tf-cuda-test kernel-contract int4` selects them without the suite's
skipped-slot check. For the limits of every SM, `zig build fatbins kernel-contract-fatbins -Dnvcc=...
-Dsm=80,86,89,120,121`, then `tf-cuda-test kernel-contract-limits zig-out/fatbin`. Its `RECORD` lines are the values
`registry.zig` holds for that build's nvcc (see [CUDA records per nvcc release](#cuda-records-per-nvcc-release)).

For the sm_110 image set, use `-Dsm=110` in the fatbin build and run the same limits command on that set.
The `DIR` must contain `norms.fatbin` and every registered image; the runner reads `norms.fatbin` first to select
the release or sm_110 audit mode. It reads sm_110 registers, static shared bytes and bounds from each cubin and
checks its device limits; the five release SMs retain their pinned `RECORD` entries.
On sm_110, some 48 KiB dynamic shared-memory kernels carry another 1 KiB of static shared memory.
Only their sm_110 audit uses the opt-in limit already raised by their loaders (`allowDynamicShared`).
For a mixed release and sm_110 set, each planted control must go red on each target separately.

On an AMD machine, run `zig build hip-kernel-contract -Dhip-arch=ARCH -Dhipcc=...` on a GPU of that chip (the
controls are built with the kernels). For the limits of every served chip, `zig build hip-kernel-contract-limits
-Dhipcc=...` compiles every kernel group and the controls for gfx1030, gfx1100, gfx1151 and gfx1201 and checks each
record against them, no GPU needed; `hip-kernel-contract-objects` installs those code objects into `zig-out/hsaco/`.
Its `RECORD` lines are the values `hip_records.zig` holds for that hipcc. The records hold ROCm 10.1's hipcc (HIP 7.16),
which ROCm 10.1 installs under `core/bin` of its root.

On an Intel GPU host, `zig build -Dxpu-contract=true xpu-kernel-contract-objects -Docloc=... -Dxpu-device=bmg` compiles
every registered XPU kernel and the controls to zebins, and `xpu-kernel-contract-limits` checks each record against the
BMG card's row, no GPU needed. Level Zero does not report a kernel's GRF mode, so the record states it, and the audit
fails a record whose mode is not the zebin's `grf_count`. `tf-xpu-kernel-contract limits DIR` runs that audit on a
directory another host built. The `RECORD` lines are the values `xpu_records.zig` holds for that ocloc.

On an Arc Pro B70, `zig build -Dxpu-contract=true tf-xpu-contract-run` builds `tf-xpu-contract-run` with the zebins
linked in, and `./tf-xpu-contract-run [--device N]` runs gates 1 to 4 over the registrations of the Qwen3.8-27B glue's
`add_rmsnorm`, `add_rmsnorm_dec`, `gated_norm`, `swiglu`, `argmax`, the top-k a sampled row draws from (`topk_part` and
`topk_merge`) and `embed`, the seven planted controls (the top-k's two: a merge breaking ties the wrong way, red at
`shapes`, and a flag read off the candidates only, red at `nonfinite`), the decode attention pair with its two planted controls, and the standalone merge with its two controls. Gate 5 checks the card's own properties against the BMG-G31 row and every loaded kernel's against its record. It loads the same
native zebins the audit reads, so the GRF mode the record states is the loaded image's. Its `SUMMARY` line names the PCI
id it ran on. `xpu-contract-run` runs it on this card. On a host without ocloc, `-Dxpu-images=DIR` links zebins that
another host made with the commands the receipt records.

The decode attention registration runs `attn_decode` followed by `attn_merge` as one operator: bf16 query rows,
bf16 K/V cache and a tree table to bf16 attention rows. Gate 3 requires the bytes of the upstream `1a5644892` pair,
whose decode grid is retained in `zig/kernels/xpu/contract_attn_upstream.cl`. It covers single rows and branching
trees of 2, 4, 8, 12 and 16 rows (heap-order parents; chains and other topologies are not fixtures), 32-key slice
edges, and 8,193 through 32,770 visible keys at selected root positions. The positions include 32,768 and 32,769
(zero-based). One control drops the last non-empty slice. The other launches the new kernel on the old
(slices, heads, rows) grid where the number of slices is at most the number of rows. Gate 4 uses uniform attention with cancelling integer values; it tests the value
and normalization sums, without separately proving the query/key dot accumulator. The default NaN rule compares
classes with the independent float64 formula. Int8 caches and multi-stream metadata are outside this registration.

`attn_decode` takes the exponentials of a key's six heads (`exp(m - mn)` and `exp(sc - mn)`) in one 16-lane vector
instead of twelve scalar ones; the arguments, the max and the update order are unchanged, and gate 3's bytes against the
upstream pair are the proof at every served shape. A planted control fills the scores into lanes 0 to 7 only, so every
`p` is `exp(-inf)`, and goes red at gate 3.

The `attn_merge` registration is a replacement: gate 3 requires the bytes of the merge it replaces, retained in
`zig/kernels/xpu/contract_merge_upstream.cl`, on the same partials (random maxima, sums and values, with other values
in the slices a row does not hold) at 1 to 16 tree rows and 33 to 131,073 visible keys, including 32-key slice edges.
Gate 4 uses unit weights and cancelling integer values, and gate 1 plants NaN and infinities in the partials, an
infinity of each sign in a row's last slice among them. Its two planted controls go red at gate 3: the real kernel on
the old one-subgroup-a-head grid, and a copy that weighs the lanes past the last slice (it adds the last slice again).
The tree table is the only row source; the multi-stream row positions are outside it.

On any host, with no device, `zig build cpu-contract` runs the CPU kernels (Qwen3.5-2B's affine projection and RMS
norm on the host's cores) through gates 1 to 4 and their eight controls; a CPU kernel has no launch, so gate 5 has
nothing to check. `zig build test-kernel-contract`, CI's no-device step, runs the same gates as a test, and `zig build
test` the kernels' own test that the bytes are the same at 1 to 16 threads and at column splits every 8 columns to 256 and every 256 beyond (each alignment of the four-tile pass), on x86_64 and arm64.

`--only` filters generic Metal launch labels too; both global launch-limit controls still run.

Every run ends with a `SUMMARY` line: kernels passing gates 1 to 4, known findings, controls red, and limit problems.
The program exits 1 on any failed gate, any control that stayed green, or any limit problem.

## Adding a kernel

1. Write its test type in `zig/tests/contract/` (Metal), `zig/tests/cuda/contract_*.zig` (CUDA),
   `zig/tests/hip/contract_*.zig` (HIP) or `zig/tests/cpu/contract_*.zig` (CPU): `prepare` builds
   the inputs for a shape and a gate's fill (usually through an input family), `run` launches once and waits, `count`
   and `reference` give the outputs and their float64 values, `simulate` replays the sums for gate 4.
2. Add it to the runner's list, with its served shapes and tiles.
3. Add its launches to `registry.zig`, or to a records file it concatenates as the seam's are, and fill them from
   `--offline` (Metal) or `kernel-contract-limits`
   (CUDA: nvcc 13.0's in `cuda`, and each later release's set where that release compiles it differently); a HIP
   kernel's go to `hip_records.zig`, filled from `hip-kernel-contract-limits`, and an XPU kernel's to
   `xpu_records.zig`, filled from `xpu-kernel-contract-limits`. Its test type goes in `zig/tests/xpu/contract_*.zig`
   and, where it needs a planted control, `zig/kernels/xpu/contract_gate_controls.cl`.
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
HIP's prompt attention tile (`tf_fa_paged_*`, `tf_fa_wide_*`) writes 0 for a row whose every score is NaN where
float64 gives NaN, as its running max is `fmaxf` and a row at max -inf takes p = 0; and an infinite value at a key
the row sees under fp32's least weight gives NaN where float64 gives inf, as 0 x inf enters its P x V.
Qwen3.8 Flash Next's `tf_sattn_nax` and `tf_attn256_nax` score masked keys -FLT_MAX, the value each row's running
max starts from, so a row whose every score is -Inf weighs its masked keys 1 and gives a finite average where float64
gives NaN. `tf_sattn_nax`'s padding lanes also read key 0 at p = 0, so a non-finite value in key 0's row reaches every
row whose key count is not a multiple of 32 (0 x Inf is NaN where float64 gives the infinity).
MiniMax-M3.1-Flash's `mm_prompt_attention`, Spark-X2.6-4B's `sp_prompt_attention`, Gemma 4's `gemma_prompt_attention`
(31B and 12B, sliding and global), Nemotron's `tf_attention_nax` and Qwen3.8 Flash Next's `tf_attn256_nax` give NaN
for a row whose tile or block reads a non-finite value at a key masked for that row, and Gemma 4's
`gemma_decode_attention` for a decode row whose 32-key step reads one: P = 0 times Inf inside the P x V mma, where
float64 leaves the masked key out. These were seen on the M5, where their cases run. Each kernel has an entry of its own
that plants such a value and carries the finding, so its other entries keep a clean gate 1.
Hy4-flash's `tf_hy4_attend_chunks` and `tf_hy4_attend_scored` give NaN for a row and head whose 512-key chunk starts
with a -Inf score, where float64 leaves that key out: the running max is still -Inf there, so the first rescale is
exp(-Inf - -Inf). Its `tf_hy4_attend_nax` gives NaN for a row whose 32-key tile reads a non-finite value at a key it
masks (0 x Inf inside the P x V mma), and 0 for a row whose every score is NaN, where float64 gives NaN: Metal's `max`
drops the NaN, so the running max stays -Inf and the kernel's guard weighs every key 0. Each has an entry of its own.
Hy4-flash's `tf_hy4_route_select_rank` ranks a NaN key 0, since every comparison with it is false, so it races the
row's best expert for the first pick; the core select it stands beside never picks a NaN key.
Hy4-flash's `tf_hy4_group_scores` scores a NaN dot as 0 where float64 keeps the NaN: its relu is `max(dot, 0.0f)`.
Hy4-flash's SwiGLU clip, in `tf_affine_experts_up_4` and `_8` and in affine_mm's `am_swiglu`, gives a finite output for
a NaN gate or up sum where float64 keeps the NaN: `min(gate, 10)` and `clamp(up, -10, 10)` drop it. Its 4-bit tensor
gathers carry affine_mm's +128 code offset finding above.

HIP’s MLX stream tile keeps its finite dot2 sums unchanged. A non-finite result with an actual non-finite
operand is recomputed by full quantization groups using the factored float64 formula. The plain tile is
registered at all six supported bit widths and groups of 32, 64 and 128, with no known finding.

HIP's routed item-rows stream tile (`Routing.item_rows` on plain fp32-output `affine_dot2_stream`, 4-bit, 4 rows,
enabled on RDNA3.5 for single-round `K <= 1024` plans), whose rows past an item's count leave the lane fold, replaces
`item_rows = 0` byte for byte when the largest item holds 3 or 4 rows. Its controls run over items one row short and go
red at gate 3.

HIP's shared-row decode score (`plan_rows.hip`: q fp32, d = 256, spans of 16,384 keys or more, launches of more than
one row) replaces `tf_plan_score256_*` byte for byte with two launches on its grid, a block a plan row. First
`tf_plan_lone_score256_*` scores the rows of one-row slots as `tf_plan_score256_*` does, at its occupancy, and its other
blocks leave. Then `tf_plan_rows_score256_*`, counted from each multi-row slot's `first`, has the block of every second
row score that row and the next of the same slot from one read of the keys; the other blocks leave, so a chunk never
crosses a slot. Its registrations run both head groups (16 over 2, 24 over 4) and both cache kinds as one operator, on
launches of 1 to 64 rows: slots of 3, 1, 4 and 2 rows one position apart (at the span's end, mid-span, across a block's
first key, at the span's start), and 5 and 7 slots of one row each, over spans of 16,384 to 262,144 keys with NaN and
infinite keys that some rows of a chunk see and others do not. Its controls go red at gate 3: one leads the chunks over
the whole launch (every slot's `first` 0 and its count every row), so a chunk crosses a slot, and one launches the
chunks alone, so no launch scores the rows of one-row slots. A row scored by both launches gets the same bits from
each, so no control can show it; each kernel's own exit test keeps the two sets apart.

HIP's shared-row decode values (`plan_rows.hip`, under the same conditions with at most 8 query heads a block) replace
`tf_plan_apply_*` byte for byte with two launches on its grid. `tf_plan_lone_apply_*` weighs the rows of one-row slots
as `tf_plan_apply_*` does, and `tf_plan_rows_apply_*` has the block of every fourth row of a multi-row slot, counted
from its `first`, weigh up to four rows of that slot from one read of each 128-key tile's values. Each row sums only
its own keys in key order; a key past them is left out, not weighed 0, so a NaN or infinite value there cannot reach
the row. Its registrations take fp32 scores and stats as inputs and run both head groups and both cache kinds on the
score's launches (slots of 3, 1, 4, 2 and 6 rows, and 5 and 7 one-row slots), with NaN and infinite values at the key
only a slot's last row weighs, at the key past all its rows and at scattered keys, and NaN and infinite scores and a
NaN max at gate 1. Gate 4 weighs every key 1, 1/2 or 1/4 and sums integer values. Its controls go red at gate 3: one
leads the chunks over the whole launch, and one launches the chunks alone.

The HIP relu^2 replacement retains every non-finite plant and checks both unfused bytes and every
independent float64 output class. No inherited infinity discrepancy is accepted.
The paired launch retains every table plant and checks both members' separate-launch bytes and every
float64 output class, without an inherited exception.
Limits do not count scratch, so `spills_on_gfx1100` in `hip_records_relu2_widths.zig` names any relu^2 width record
that spills on gfx1100 (none under the records' hipcc), and its test fails when a record spills that the list does not
name or a named one stops spilling.

## Isolated masked-value panels

`metal masked-V` entries isolate one hidden +Inf or NaN in otherwise finite Q/K/V. Every output class and every
finite output is checked against binary64 over the actual visible set, including a visible exceptional-value control.
The fixtures cover causal tiles, sliding rings, indexed/padded lists, page mappings and shared-context masks.
Latent MLA treats its shared K/V storage as both key and value; a visible poisoned latent can also make its score NaN.
The tensor kernels' dropped visible NaN scores and zero residual probability parts times visible Inf are separate
known findings from hidden-value leakage. No production kernel is repaired by these registrations.

The source-only focused runner builds without an offline Metal compiler:

```sh
zig build --build-file build.masked-values.zig tf-masked-values test-masked-values-host -Dcpu=apple_m1
./zig-out/bin/tf-masked-values
```

It runs the same entries registered in the main Metal runner. Its equal-score/unit-value panels do not discriminate
accumulator precision, so gate 4 states that limitation instead of claiming a stress proof. The two device controls
plant a hidden NaN leak and erase a visible infinity; each must fail gate 1. A known finding must still reproduce.
LFM3 entries register only on a tree that actually contains its source. DeepSeek CSA2 and H3 are operator checks,
not new native serving paths. Flash Next is excluded. This focused run does not replace the complete backend gates.

## Coverage

The kernel-by-kernel inventory is in [Kernel contract coverage](kernel-contract-coverage.md).

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
its wide decode select, the span scan `tf_glm_dsel_scan` and 12 of the GPTQ int4 launches' 21 instantiations.
`kernel-contract` and `kernel-contract-limits` read the release from each image's `.note.nv.tkinfo`, check the image
against the newest set at or below that release (13.0's when an image names none), and print which set they used.
A release with no set of its own is checked against the one below it, so its differences show as stale records:
re-derive them with `kernel-contract-limits` and add a set for that release rather than loosening the audit.
Launch dimensions, dynamic shared memory, the device's limits and the planted controls are checked the same way under
every release.

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
