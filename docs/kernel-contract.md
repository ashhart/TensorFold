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

`tf-cuda-test kernel-contract` alone is the full serial run in one process. Two opt-in modes change that, each apart
from the other, and they compose:

- Filter: `tf-cuda-test kernel-contract WORD...` runs the kernels, controls and launches whose name holds a WORD. It
  skips the standalone suites and the offline limits audit, and marks its `SUMMARY` `(filtered)`.
- Parallel: `tf-cuda-test kernel-contract --jobs N` (N from 1 to 16) starts N child processes of the same binary,
  each a shard `K/N`: every N-th kernel, control, standalone suite and launch record, on the visible GPUs in turn.
  The shares are equal in count; a kernel's cost is not known when they are cut, so one shard can finish before
  another. It prints the children's logs and one merged `SUMMARY`, and exits 1 if any child crashed or printed no
  `TALLY` line, or the merged counts show a problem. N never rises by itself.
  A process is planned for 28 GiB of host memory for a full run and 8 GiB when WORDs filter it, and 7 GiB of GPU
  memory. The run is refused when N processes do not fit the host's available memory (`MemAvailable` in
  `/proc/meminfo`, so Linux only), or when `ceil(N / GPUs)` processes do not fit the smallest GPU's total memory
  (not its free memory); several processes share a GPU only when they fit. On an integrated GPU, which shares the
  host's memory, N times both plans must fit the host's available memory and no GPU total is checked. GPUs that
  differ in compute capability, multiprocessor count or name are refused, since registration reads those to pick
  kernel configurations and their shards would not list the same kernels.
  A shard stops when the process that started it does (`PR_SET_PDEATHSIG`), SIGINT and SIGTERM sent to that process
  are forwarded to its shards, and once one shard fails to finish the others are stopped.
  `--shard K/N [--device D]` runs one shard alone and prints its `TALLY` line (the machine-readable counts the
  parent merges) in place of a `SUMMARY`; `--device` goes with `--shard` only, and so does `--parent PID`, which
  the `--jobs` process passes so that a shard refuses to run once that process is gone.
  With words the filter applies first and each shard takes its share of what is left.
  `kernel-contract --help` prints these options.
  The standalone suites of a full run are the entries of `list` in `contract_back_suites.zig` and of `served` in
  `contract_after.zig`; the serial run and every shard read that one table, so a new suite is listed in one of them
  and a suite called from anywhere else runs in neither.

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

Metal: softmax, RMS norm, bf16 top-k, the row projection (3, 4, 6 and 8 bits), affine_mm (4 and 8 bits), simd_qmm (4
bits) and simd_qmm_bits (3, 6 and 8 bits), and the lane projection (6-bit groups of 32 on every chip; 4- and 8-bit
formats on the M5's tensor units, where its reg kernel must equal the cooperative one bit for bit), and tile_qmm's
prompt tiles and their decode twin (4, 6 and 8 bits; the twin bit-equal to the tiles). Gate 5 also holds the rest of
tile_qmm's pipelines and GLM-5.3-Flash's hc_expand_10.
On the M5's tensor units, Metal also checks what nax.h's `mma_16x32` keeps of an fp32 operand
(`metal_mma_operand.zig`): the left, the right and both (attention's fp32 P, `gemm_nax`'s f32 kernels), each output a
single product bit-equal to float64 on operands with their low 13 mantissa bits cleared. That is 10 bits kept, rounded
toward zero, fp32's exponents and subnormals kept, NaN and the infinities passed. Four controls state the operand as
fp32 or as rounded to nearest at 10 bits, on each side, and must go red at `shapes`. The cut was measured on the tensor
units only, so the cases run only there; what M1-M4 keep of an fp32 operand is not measured.
`core/attention_range.zig` names that cut (`operand_cut`, 2^-10) as a form's `p_cut`: one-sided, since each p only
shrinks, so it moves an output down by up to 2^-10 of its positive values' share and up by its negative values'. On the
M5, Nemotron's `tf_attention_nax` states its fp32 P with it, and MiniMax-M3.1-Flash's `mm_prompt_attention` does on
every chip (before the M5 that only widens its range); each bf16 output must lie between its range's ends rounded to
bf16. Nemotron's adds a case with P spread over about
[2^-6, 1] and values in [0, 1), where the cut's bias adds up; its control, the same range without the cut, goes red at
`shapes`. Before the M5, `tf_attention_nax` keeps attention_ref's bound.
Qwen3.8 Flash Next's prompt attention on the M5 does too: `tf_sattn_nax` (the indexer's dense and selected rows) and
`tf_attn256_nax` (the causal chunk without the indexer), head dim 256, 24 heads over 2. Each bf16 output,
bf16(bf16(o) * bsig(gate)), must lie between its range's ends rounded to bf16 and through the gate's rounded sigmoid.
Each adds a P-spread case whose cutless range is a planted control, and a source edit gate 3 must catch: `tf_sattn_nax`
counting its padding lane (key 0 again), `tf_attn256_nax` seeing one key past the row. Before the M5 neither is
registered.
On the M5, Nemotron's SSD steps (`prefill_mixers.zig`'s Mamba mixer) run `gemm_nax`'s five f32 kernels with both
operands fp32, so each product enters cut, smaller by up to 1 - (1 - 2^-10)^2. Their entries
(`metal_nemotron_ssd_gemm.zig`, y = sur x dtx and the state dd^T x bh^T a head, at the step's served row counts) state
it as a one-sided range: float64's value less that share of the positive terms and plus that of the negative, beside
the fp32 sums' `mma_step` roundings for every k16 step. Gate 4 runs 11-bit integers times 5-bit ones, which the cut
keeps. Two cases take operands in (0, 1], where the bias adds up; their controls state the operands as fp32 and go red
at `shapes`, as does a source edit dropping the ragged K tail's last column. Before the M5 they are not registered;
their launches are in `nemotron_ssd_records.zig`.
Spark-X2.6-4B's `sp_prompt_attention` registers both pipelines the engine builds (sliding layers over a 512-key window
in a 2,560-slot ring, and full layers), its fp32 P stated with the cut, each with a P spread case
and its cutless control, and one source edit each (a key past the window, a key past the row); before the M5 they are
not registered.
On the M5, Gemma 4's `gemma_prompt_attention` (the 31B's and 12B's sliding and global layers) and
`gemma_decode_attention` with `gemma_attention_merge` (their global layers; the 12B's assistant launches the 12B's)
state their fp32 P with the cut too, each form's counts taken from the kernel's own steps, at the sliding ring, image
rows and decode chunks the engine serves. A P-spread case on the 31B's global prompt and decode has a cutless control
each, red at `shapes`, and a score edit (2 log2e) in each kernel is a control. Their launches are in
`gemma_attention_records.zig`.
LFM2.5-24B-A2B's shared rounds: the greedy pick (`lfm2_argmax`) and the conv and attention kernels that run every
stream's rows in one dispatch, each bit-equal to the per-stream kernels it replaces. Four controls edit the source: a tie
to the highest id, a nonfinite logit not flagged, and every stream reading or writing the first stream's state or cache.
Their launches are in `lfm2_records.zig` with the M1 and M2 limits from `--offline`; `lfm2_attn_segs` reserves its 128
threads in its source, as Metal cannot build it for M1 or M2 at another count.
Metal also checks Nemotron's routed experts on the engine's route tables, one row to 128: `tf_xup_w3` and `tf_xdown_w3`
against float64 at tf_load16's bf16 x sums (up's relu2 as an interval over its two bf16 roundings), and the rows2, rows4
and six geometry pipelines of each projection bit-equal to them (two controls: member rows read from one row, and
w3's x sums without their bf16 steps).
Metal also checks DeepSeek V4.1's routed experts at hidden 5120, width 2304, rows 1 to 16: the gate/up (`tf_dsv41_xp_up`:
fp8 act_quant of the input, bf16 projections, clamped SwiGLU, route weight, bf16, act_quant of the result) against an
oracle in the kernel's fp32 order whose silu may move by exp's error, and the down (`tf_dsv41_xp_down_one`/`_rows`, a
DeepSeek-only copy of Kimi K3's chains with the full E8M0 scale range) against its bytes (also over the release's real
FP4 expert bytes), each with four controls. The copied plan and combine (`tf_dsv41_route_plan`, `tf_dsv41_xp_combine`)
are checked end to end through the GPU's own plan, and every launch is recorded for gate 5.
Metal also checks DeepSeek V4.1's FP8 rows tile (`fp8_tile.metal`, `tf_dsv41_fp8_tile_*`), which runs decode launches
of 6 to 16 rows on simdgroup MMA: one registration replaces `tf_dsv41_fp8_w*` run one row a launch and must be
bit-equal to it at every shape of its shapes gate (the seven release projections and every row count from 1 to 16),
beside f32, bf16, forced-form and release-fixture registrations against float64. A probe holds the tile byte-equal to
the row kernel one row a launch on every dense FP8 shape the decode forward runs, f32 and bf16, rows 1 to 16 from row 0
and 1 to 15 from row 1 at every form, the 16/17 seam and a three-launch batch, with NaN and Inf planted. Four controls
go red: non-finite inputs flushed, group and slice roots added in a chain, group roots rounded to fp16, and writes past
its output.
Metal also checks DeepSeek V4.1's one-row EXL3 projection (`core/exl3_projection.metal`, `tf_exl3_matmul_row`), which
takes a decode row instead of `tf_exl3_matmul_hoisted`: five registrations (mcg 2 and 8 bits, mul1 2.5 and 4 bits,
3inst 6 bits; bf16 and f32 out) must equal that kernel's bytes, all rows in one launch, at the eight dense decode
shapes (`wq_a`, `wkv`, `wq_b`, a `wo_a` group, `wo_b`, the shared expert's w1/w3 and w2, `ix_wq_b`) and the N and K
edges, each row launched alone. Its control drops the TwoSum error term and goes red at `shapes`.
Its 2 to 16 row kernels (`core/exl3_rows.metal`, `tf_exl3_matmul_rows{rp}x{q}`: q lanes share a column's decode and
each keeps rp rows) are held to the same five registrations' bytes at 2 to 16 rows, all rows in one launch, with every
count routed to them; their control drops the rows body's TwoSum error term and goes red at `shapes`. Two registrations
of `tf_exl3_swiglu` (w1 mul1 2.5 and w3 mcg 2 bits; w1 3inst 6 and w3 mul1 1.5 bits with biases) must equal the bytes
of w1 and w3 through `tf_exl3_matmul_hoisted` then `tf_dsv41_f_swiglu`, y then its act_quant round trip, at the shared
expert's 1, 2, 4 and 16 rows; their control drops the up clamp.
The decode gate/up and down for an expert's slot of 2 to 16 rows (`tf_dsv41_xp_upc_rows`, `tf_dsv41_xp_down_rows`)
decode its weights once for every row of a pass, each row in the one-row kernel's order: both run with every row on
the same six experts as well, and a probe holds a launch's act, out and combine output bit-equal to each row launched
alone at 1 to 16 rows on the same, disjoint and random experts, with NaN and Inf planted. Rows that read their pass's
first row are a control in the registrations (red at shapes) and in the probe.
DeepSeek V4.1's routed experts over EXL3 trellises (`deepseek_v41/experts_exl3.metal`, host `experts_exl3.zig`) run
main's route plan, `tf_exl3_rotate_in` for each (token, expert) pair, `tf_dsv41_x3_up` (gate and up) or
`tf_dsv41_x3_down` over each slot's pairs, in which a lane owns a column and adds a pair's products in
`tf_exl3_matmul_row`'s order, then `tf_dsv41_x3_act` (gate and up rotated out to bf16, main's clamped SwiGLU times the
route weight, rounded to bf16), `tf_exl3_rotate_out` into main's per-pair layout and main's combine. Four registrations
(up and down; mul1 at 2 to 5 bits by expert, and mul1, mcg and 3inst by expert; f32 and bf16 combine) must equal the
serial reference (`experts_exl3_ref.zig`: the one-row reference's dot per pair, flush-to-zero fp32 rotations and the
device's silu read as a table over every bf16 gate) byte for byte, slot sums included, at 1 to 16 rows and at the
5120 x 2304 release shape. A lane runs the whole chain at 1, 2, 3, 4, 8 and 16 rows, holding each pair's sums equal
to `tf_exl3_matmul_row` launched alone and to routed `tf_exl3_matmul_hoisted`. Three controls go red at shapes: the
up's and the down's TwoSum error term dropped, and the act without the route weight. A descriptor table shorter than
the call's experts is refused (`error.ShortExpertTable`) before any launch, its outputs left as they were.
The decode MoE's `tf_dsv41_route_gate_plan` (the router's matvec, route and plan in one launch: the last of its groups
to finish reads the logits as atomics, ranks the picks and plans) registers twice: its matvec in the matvec suite (exact
bytes at 384 experts, 1 to 16 rows, four controls under `TF_GCTL`), and its route and plan in the route suite on given
logits (k 0) with the route's four controls.
The row-split seam's four kernels (`pair_rows.metal`, launched by `pair_rows.zig`) move bf16 frames, never sums:
`pair_stage` copies this rank's half of a frame row into the slot the relay sends, and `pair_gather` writes both halves
of every row, its own natively and the peer's from the relayed frame. `pair_argmax` flags a row whose logits are not
finite (its id word carries bit 31, its value word a NaN) and takes the lower id on a tie; `pair_pick` turns a flagged
rank's word into the id sentinel, so both Macs refuse a non-finite round where one Mac does. Six controls edit the
source: a stage reading the peer's half, a gather writing its own half twice, and an argmax or pick tie to the higher
id must go red at `shapes`, and an argmax or pick that does not flag at `nonfinite`.
The Llama family's own kernels (`metal_llama*.zig`): its glue (embedding, the norms at HF's rounding points, the
residual add, RoPE with bf16 cos and sin, the SiLU gate, the decode attention's partials and their merge, the argmax
draw by its stated rule), its row projections (1 to 32 rows) and, on the M5's tensor units, its four GEMM forms and the
prompt attention at both served geometries; gate 5 reads their pipelines on this GPU. Twelve controls edit the
source, one a kernel group: an embedding reading its own row, a norm's mean over the threads, a subtracting add, an
add-norm dropping its residual, a SiLU with gate and up swapped, a rope at the window's first position, a decode that
never rescales, a merge dropping each block's scale, a row projection dropping its tail, a GEMM reading its weights
untransposed and a prompt attention seeing the next key go red at `shapes`, and an argmax never picking a NaN at
`nonfinite`. The INT4 release's row projections and dequantization have two more: every group taking its row's first
zero point, and a word's nibbles read in reverse, both red at `shapes`.
Ripple-65B's own kernels (`metal_ricursive*.zig`): its glue, each output inside float64's range at the release's bf16
rounding points over the spread its fp32 steps allow (the scaled embedding; the add-norm in the forms the forward runs,
the value residual's lerp, the sublayer's add and both; q and k's norms with RoPE and a ring slot on sliding layers,
without RoPE on global ones; the output gate's fp32 sigmoid; the clamped SwiGLU), the decode attention's partials over a
1,024-key window in a 3,072-slot ring and over a global layer's every key, and on the M5's tensor units the prompt
attention for both layer kinds; the NVFP4 MLP (E2M1 nibbles low first, an unsigned E4M3 scale a block of 16, the
tensor's fp32 global): its row kernels, bf16 of the global times the fp32 sum, the prompt's exact dequantization and the
GEMM taking the global before its one rounding. Llama's row projections, merge, argmax and GEMM forms register again at
Ripple's shapes (k to 40,960, 80 query heads, 100,352 logits), and Nemotron's `tf_sample_full`, Ripple's sampled draw,
is held to a stated rule: each draw equals lanes' `gpu_full` twin. Fifteen controls edit the source, each red at
`shapes`: an embedding reading its own row, an add-norm not writing its residual back, a lerp dropping its blend, a prep
rotating a global layer or writing one ring slot late, a gate rounding its sigmoid's exp to bf16, a clamp missing its
lower bound, a decode seeing the key before its window, a prompt attention seeing the 64 keys before it, one whose keys
wrap 64 slots early, nibbles read high first, a scale's exponent bias off by one, the global applied after the sum
rounds, a dequantization leaving out a block's scale, and a GEMM never taking the global. Their launches are in
`ricursive_records.zig`.
CUDA: the RMS norm rows (16-byte form and the fp32-out tree), the bf16 residual's norm tails (MLX's rounded rows and
the once-rounded ones, each checked on its residual and its output, with four controls each dropping a rounding point), Nemotron's router logits (and their 128-row tiles,
bit-equal to them) and top-k, the latent top-k's decode form (bit-equal to `tf_super_topk`), the latent combine and
shared add over bf16 share sums (bit-equal to `tf_super_combine` and `tf_super_moe_out` over the fp32 shares), the
Mamba scan of a verify window (its host twin's bytes), qmm_group's cluster tile, lane_gemv and its split-K form (both
bit-equal to it), the split-K form segmented over projections of one input (bit-equal to `decode()` on each), prompt16's FP4 tile with bf16 and fp32 out and its gate/up SwiGLU pair (bit-equal to two lean products then `tf_q35_mlp_act`; controls: gate and up swapped, the gate's scale one fp32 ulp up), qmmf's FP4 tiles with fp32 out (K slices met in a
cluster, fused, or by `reduce_kernel<true>`), the checkpoint math's FP8 128-row prompt tiles (both slice meetings,
bit-equal to the 64-row tiles) and NVFP4 experts: the decode-sized forms and the prompt tiles (up to down's NVFP4
rows, fp32 down and the down summing two width shares, each bit-equal to the kernels it replaces; sm_120 up), and the
reduce-scatter's `tf_rank_sum`, GLM-5.3-Flash's float32 norm weight forms `tf_glm_rmsnorm_4096_f32w` (its rows and
64-sums) and `tf_glm_hc_finish_f32w` (its normed rows, 64-sums, post and Sinkhorn comb), full GLM-5.3's bf16-weight
norms `tf_glm_rmsnorm_6144` and `tf_glm_rmsnorm_2048` (rows and 64-sums against float64; two controls, each width
launched with stride 0 so every row reads row 0), and the activation rows in
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
CUDA also checks the Qwen3.5 family's pending kept rows (`contract_qwen_pending.zig`): each `tf_q35_gdn_slots_*` scan
replaying them, written back or not, and `tf_q35k_apply_pend`, bit-equal to `tf_q35k_apply` and then the scan at the
27B's widths (two controls: the last pending row dropped, the last two swapped).
CUDA also checks Dory's paged caches (`window_pages.cu`): `tf_wrows_put` places rows in a shuffled arena bit for bit, and the
head dim 256 decode and prompt kernels read them through a page table, each output within float64's range (two controls:
swapped stream tables, a stale page id).
CUDA also checks Dory's attention inputs (`qk_l2.cuh`): RoPE first, each head's L2 norm, q times its scale vector, v as is, at
head dim 256 and positions past 199,800 (two controls, each dropping one rounding point).
CUDA also checks Dory's activation rows (`act_l2.cuh`): the scaled sigmoid output gate over all heads and the scaled SwiGLU,
each row then L2-normed, against float64's ranges at their bf16 points and expf's ulps (four controls, one point each).
CUDA also checks `dense_rows`' scaled epilogue (Dory's logits, fp32(bf16(x . w)) times a per-column scale), bit-equal to its
unscaled tile's outputs rounded and scaled on the host.
CUDA also checks EXL3's replacements of `linear_body` (`exl3_linear.cu`), each byte-equal to it at every row count it
takes: the mid-M kernels (3, 4, 5 and 6 bits) and `linear_wc` (4 and 6 bits) at 1-128 rows (`Linear.min_rows` 1 runs
them below 17 rows), the projection groups against each member's own launch, and `tf_exl3_rot_group` against
`tf_exl3_rot_in` (1-16 rows). Their served shapes reach every form `plan()` gives them (row passes, warps, split or
unsplit K, `linear_wc`'s tiles); a host test sweeps K and N to 16384 through `exl3.pick` and `groupable` and fails on a
form no registration runs. A fixed plan keeps the linear. Gate 4 proves an fp16 sum would overflow on a constant-weight
stress input (all-ones states: one weight at every width). Six controls: another K split (3, 4 and 5 bits; `linear_wc`),
swapped scales, a swapped suh.
HIP: the RMS norm rows, a wave a row (fp32, fp16 and bf16 out) and a block a row (fp16 and bf16 out).
HIP also checks the decode tail (`tf_tail`): the residual add, of a row of the activation kind or of nine slots' fp32
rows times their weights summed in slot order, and the RMS norm after it, against float64 at Qwen3.6-35B-A3B's and
Qwen3.8-27B's widths, decode rows and windows, and the widest row the launch takes.
HIP also checks the 64-row prompt attention tile (`prefill.hip`): `tf_fa_paged_bf16` and `tf_fa_paged_f16` over a
permuted page table at the 27B's, the 35B-A3B's and Nemotron's head counts, each output within attention_ref's float64
bound, and `tf_fa_wide_bf16` and `tf_fa_wide_f16`, which no model calls, bit-equal to them on flat keys. A build runs
its chip's cache type (fp16 on gfx1030, bf16 on the rest); gate 5 holds all four.
HIP's unquantized embedding gather (`tf_embed_dense`) reads a BF16 table byte for byte and an fp32 table bit-equal to
the gather before its table kind; two controls gather the next row and read a BF16 table as fp16.
HIP also checks Clef's joint schema head (`joint_schema.hip`, `contract_js_*.zig`): the affine layer norm, the row
softmax over FP32 scores to 16,384 keys, GELU, BF16 widening and rounding, the span sums over piece rows and over table
rows by id (FP32 in prompt row order) and their means, step 5's option summary, and step 7's logits (prior, cosine and
the scorer's input rows) and their gate, each within float64's bound at the head's BF16 points. Two controls: a layer
norm summed in bf16 and a softmax that skips non-finite scores.
CUDA checks the same head (`joint_schema.cu`, `zig/tests/cuda/contract_js_*.zig`) with the same cases, plus the
lexical sums read from the engine's packed BF16 lm_head (`tf_js_lex_sums`) and the head's residual add and broadcast.
Every kernel has a planted control (`contract_js_controls.cuh`), each naming the gate that catches it.

HIP also checks the router over the rows a checkpoint already holds in bf16 (`tf_router_bf16`, and its fp16-activation
form), bit-equal to the fp32-rows router it replaces at 1 to 16 rows, with both kinds' non-finite plants through gate 1
and the exact sums of gate 4.

Metal also checks Idol (`metal_idol*.zig`, each part's controls in [its recipe](recipes/idol.md)): the denoiser's fp16
projections and GELU form, split-K fp32 rows, LayerNorms, RePo's angles, heads, elementwise steps and four attention
forms at idol-sdk's fp16 points, and `id_gemm32`'s true fp32 sums, the NAX fp32 path a gate-4 control (19 controls); the
decoders' point embedding, fp16 stem cast, `to_kv`, RMS-normed heads and attention, P V as fp16 P hi + lo, which a
control without the lo half must fail on heads whose O is that remainder alone (15); the FLUX.2 VAE encoder's
convolutions (direct, implicit GEMM, Winograd), GroupNorm, 512-wide attention and condition rows in fp32 against float64
(12); DINOv3's bf16 projections, LayerNorms, RoPE heads, LayerScale, SwiGLU and attention at transformers' bf16 points
(11); and BiRefNet's fp32 implicit GEMM convolutions in nine served forms, 3x3, LayerNorms, shifted-window attention,
align_corners resize, patch and tile copies, modulated deformable columns, pool, gate and head against float64 (21).

HIP also checks the DeltaNet prompt recurrence at half a value head a block (`tf_gdn_prompt_128`, its y and state),
bit-equal to the plain window scan `gated_delta_tile` on the same tokens.
Its prompt prework in one launch (`tf_gdn_prompt_prep`: conv, q and k head norms, v, gate, beta and the new conv state)
is bit-equal to the chain it replaces (`tf_conv_prefill`, `tf_copy_cols`, `tf_rms`, `tf_gdn_gate_prefill`).
Four controls: the recurrence one row short of the prompt and with one key head, and the prework norming q with k's
weights and writing gate and beta to each other's outputs.

HIP also checks Flash Next's indexer block score kernel (`tf_fn_idx_scores`, `contract_idx_scores.zig`, record in
`hip_records_sparse.zig`): staging each tile's rows of `q` in LDS when `heads * dims <= 512`, checked against the
fp64 relu-summed head dots with one control leaving out each row's last head (`shapes`).

HIP also checks EXL3's projections for decode rows (`exl3/exl3.hip`, CUDA's `exl3_block/linear.cu`): `tf_exl3_rot_in`
(bf16 rows times suh, rotated by 128 and rounded to fp16) and `tf_exl3_linear` at 2, 3, 4, 5, 6 and 8 bits (mul1) on 4
and 8 waves, each against float64 at deepseek_v41's indexer wk and Engram wkv (and its head at 8 bits) and the plan's
8-wave edge. The linear's gate 4 is exact: all-ones trellis states give one weight (-0.651 at every width), so 6144
terms of 4096 times it stay integers under 2^24 where fp16 overflows. Two controls: rot_in reading the next block's suh,
and the linear reading its input rows one row late.

HIP also checks DeepSeek-V4.1's indexer top-512 (`deepseek_v41/topk512.hip`, Metal's `topk512.metal` arithmetic:
`contract_dsv41_topk512.zig`, records in `hip_records_dsv41.zig`) at Metal's served shapes, rows and rule, each row against
the CPU's stable sort. Four controls, as on Metal: NaN ranked above +inf (`nonfinite`), ties going to the higher index
and the last visible score dropped (`shapes`), and a write into its input (`repeat`).

HIP also checks Inkling 2's attention at Mini's 24 query and 8 key heads (`inkling2/attention.hip`,
`contract_inkling2*.zig`, records in `hip_records_inkling2.zig`; `tf-hip-kernel-contract image inkling2`). It does not
match #4125's CUDA bits; it keeps CUDA's rounding points and is held to the same float64 reference
(`core.contract.inkling2_attn_ref`, with HIP's scalar fp32 chains): `tf_ink2_prep` (sliding and global, tau at n_floor
8,192), `tf_ink2_rel_table`, and the chunked attention and its merge on sliding layers (`tf_ink2_attn`) and global ones
(`tf_ink2_attn_fold`) to 1M keys on a ring that wraps. Every P is rounded to bf16 before its sum and P V, the sink enters
last as a raw logit and the merge takes the raw gate's sigmoid, rounding once. The split launch (a block a 1,024-key
sub-chunk) and the prompt launch (`tf_ink2_attn_prompt`: 8 pairs a block share one LDS stage while the next tile loads
into registers, 37,120 bytes or 36.25 KiB) are bit-equal to the fold, and three streams in one launch are bit-equal to
each stream alone, at its row and cache offsets (prep, sliding, fold and split). The float64 bound allows P's bf16
rounding either way, so two exact cases pin it (sliding and fold): rows below position 64 with q 0, no sink, a gate
whose sigmoid is 1 and integer values, and biases drawn so each bf16(exp2f) is certain, where every sum is exact in
fp32 and each output must equal bf16(fp32(sum P v / sum P)) bit for bit. Eight controls go red at `shapes`: a bias one
distance late, no sink, one q a bf16 step up (prompt), a window one key long, the sink and gate given after their
sigmoid (as #4387's glue did), tau off, a table read one column late, and a prep writing one stream's keys into
another's slots. `attend()` refuses a ring that is not whole 64-key tiles or is shorter than the window plus the
call's rows less one, a call with no streams (only one stream takes the prompt kernel), and partial buffers holding
fewer than rows x heads x stride partials. It does not refuse a global position past the ring, since the contract's
4,096-slot ring over 1M keys wraps on purpose: the engine glue keeps a global layer's positions inside its ring.

On gfx12, HIP also checks the WMMA dense tile (`dense_wmma12_gemm`, both block geometries, fp32 and BF16 out). The
small block meets its depth-k bound. The big block (64-deep K steps over swizzled LDS rows) replaces the 32-deep block
it was before #2245's second lever, byte for byte: gate 3 runs a frozen copy of that block (`tf_ref_wmma12_k32`, in the
controls' code object, with its own gate-5 record) on the same inputs, and the kernel tests hold both geometries to
float64. Three controls go red: the small block summing each 16-deep step in BF16 (`tf_ctl_wmma12_bf16_sum`) at gate 4,
and at gate 3 against the frozen block, the big block loading its operands without the stores' swizzle
(`tf_ctl_wmma12_unswizzled`) and the big block without its K loop's barriers (`tf_ctl_wmma12_unordered`). The last is a
race, so gate 3 launches it up to eight times a shape and counts it red if any launch differs.

On gfx12, HIP also checks the WMMA product with silu x up in its epilogue (`dense_wmma12_gemm_silu`, a 2-wave 16 x 64
block and the 128 x 128 block, BF16 out), over a gate/up weight interleaved in 16-row groups. Each block replaces the
plain WMMA product on that weight, its split into gate and up columns (`tf_copy_cols`) and `tf_silu_mul`, byte for
byte, at Qwen3-8B's served rows and each block's edges; gate 1 holds the unfused bytes and every float64 class, and
gate 4 negates the stress rows' x so that every gate and up sum is positive and silu keeps its error. One control goes
red at gate 3: the pair epilogue applying silu to the fp32 gate sum instead of its BF16 rounding
(`tf_ctl_silu_unrounded`), once the served kernel shows the unfused bytes on the same inputs.

RADIO's fused bias, activation and residual (`tf_vit_bias_act`) replaces `tf_vit_bias` then
`tf_vit_pointwise`, byte for byte.
CUDA also checks GLM-5.3-Flash's KV page kernels on a page table out of order: the latent write and the pooled keys
(`tf_glm_lat_write`, `tf_glm_pool_keys`), the index scores and the selected tokens' rows (`tf_glm_scores`,
`tf_glm_dsa_tokens`), and the MLA chunks through their merge (`tf_glm_chunks_16` and `tf_glm_chunks_32`, dense and
sparse). Its prompt indexer (`tf_glm_index_scores`, `tf_glm_index_pools`) is bit-equal to `tf_glm_scores` and
`tf_glm_select_rows`, its decode windows' scores (`tf_glm_index_scores` at the decode tiling) to `tf_glm_scores`,
and its decode select at a 1,024-thread block a row (`tf_glm_select_rows_wide`) and in spans over many blocks
(`tf_glm_dsel_scan`, then `tf_glm_dsel_merge` a block a row) to `tf_glm_select_rows`; both stated
rules rank any NaN above +inf, as the GPU's `s + 0` makes every NaN one positive NaN.
Full GLM-5.3's indexer values run on the same kernels (`contract_glm_full.zig`): `tf_glm_scores` at pools of 1 with
64 roped query dims against float64 (allowing the roped queries' bf16 rounding), `tf_glm_index_scores` bit-equal to
it, `tf_glm_dsa_tokens` at pools of 1, and `tf_glm_index_write`'s LayerNormed keys roped on 64 dims, with four
controls: queries or keys left unroped, keys read as pools of 4, and token lists one position late.
CUDA also checks `TF_GLM_KV_HOST`'s gather (`tf_glm_kvh_*` in `glm_ops.cu`, launched by `cuda_kv_host.zig`): the
marks (a slot for each distinct pick; past its slots, the call flagged and no row left marked), the picked rows into
their slots from host pages or the layer's row cache, the token lists' slots, the marks cleared, the dense rows through
pages out of order, and the layer's cache kept, each exact against the host's replay (nine controls, one a kernel: a
stale mark, a dropped row or swapped slots).

Ornith-1.5-122B's routed block on Metal registers the experts' decode rows with the shared expert as an extra matrix
(`q27_rows_extra`, 4 and 8 bits), the router's softmax and top-k with renormalized bf16 weights (`q27_route`: picks
equal to the host's, each weight within float64's range at its bf16 points), the tensor units' expert sort and gather
(bit for bit), the experts' SwiGLU, and the weighted combine with the gated shared expert (in place and from its slot).
The same kernels register at Qwen3.6-35B-A3B's served widths (experts of 512 over 2,048).
qwen27's attention registers at the 27B's 24 query heads over 4 and the 35B-A3B's 16 over 2: a round's rows through the
engine's own plan and launcher against float64 attention, prompt rows through the prompt kernel on every chip that
builds it (`attention_prompt_sg` before the M5), and attention_io's packing, append and keep checked bit for bit (keep
a second time on the caches the first left). A plan check shows the decode shapes plan prefix_1 to _6 for chains and
_2 to _6 for trees, and a tree over two tail chunks. Its known finding: before the M5 a masked key's zero probability
times an infinite value gives NaN in P V. Its controls: a row seeing one key past its own, the one-Mac merge dropping
a round's later tail chunks, the prompt kernel losing its bounded last step, and an append one row late. prefix_5 and
_6 have no launch record yet: past 256 threads they need a reserved thread count before M1 and M2 can admit them.
GLM's Metal window projections register narrow and wide rows at 2, 3, 4, 5, 6 and 8 bits, plus both gathered paths.
Independent FP64 linear cases accompany byte replacements against frozen pre-window-optimization kernels.
The gathered SwiGLU keeps its predecessor's clipped BF16 output and nonfinite rules; gate/up taps qualify its dots separately.
Controls dispatch real source mutations that couple rows or select the wrong expert; host tests validate their witnesses.
GLM-5.3-Flash's other served Metal launches (`metal_glm_launch.zig`, names `metal glm launch ...`) each run the loader's
own library text beside the kernel they replace, on the same inputs, and must give its bytes in every written buffer,
untouched guards and poisoned padding included. The decode select's merge (rows within 3,072 blocks) and its mark and
merge (past 8,192 blocks, to rows past 65,536 blocks where the merge falls back to the whole row) keep
`glm_index_select`'s key lists over ties, signed zeros, NaNs, infinities and a floor that overshoots. The windowed KDA
step (`glm_kda_window`, 64 heads, and `glm_kda_window_tp`, 32) keeps `kda_rows`' outputs, next state and next conv
window at 2 to 16 rows. A window's dense attention as `attendRows` launches it (`glm_latent_scores_rows`,
`glm_softmax_rows`, `glm_latent_values_rows`) keeps main's one-row launches' scores, probabilities and outputs row by
row, 1 to 2,048 keys across the 256-key join and the 1,024-key split, against a frozen copy of main's `glm_attn.metal`.
The hyper-connection boundary as `core/hc.zig` launches it (the expand's parts on six groups a slice to 4 rows) keeps
main's expand and split (a frozen `hc_boundary.metal`) in the streams, partial sums, normed row, post and comb, as raw
words, for separate streams, in place and at a first read. The decode experts at `moe.zig`'s outputs a simdgroup keep
the four-output gate/up and down kernels' activations and outputs. On an Ultra's GPU the KDA step's two groups a head
with the norm after (`glm_kda_parts`, `glm_kda_norm`) keep `kda_rows`' bytes at 2 to 16 rows. The boundary's two-group
split runs in the same boundary entries, and the attention entry runs the prefetched score and value bodies with a
control that multiplies the prefetched block. The one-row router and indexer-gate logits (`glm_route_logits_one`, fp32
and bf16 out) also meet float64 at every gate, each row its own one-row launch, and give `tf_route_logits`' bits. The
core route select (`tf_route_select`, its picks by simd max and min and its groups' offsets by prefix sums) keeps main's
picks, weights, held groups, pick lists and counts at GLM's shape (every expert held, or either half) and Step-5's, over
ties, NaNs, signed zeros and -inf biases. `glm_mla_norms` (three roles, and four with the query norm) keeps main's
`glm_rms`, `glm_layer_norm` and `glm_scale` launches; the attention entry's scores kernel scales the queries by 1/16 as
it loads them, against main's `glm_scale` first. The boundary that combines a MoE block's picks (`tf_hc_expand_mix_moe`,
and its parts to 2 rows) keeps main's combine then expand, and a one-row MoE layer whose gate/up groups replay the
select keeps main's select then the one-row gathers (its picks held to the 16 experts given weights by -inf biases
elsewhere). A prompt chunk's routed experts and deep dense products in column groups (`glm_gather_glu_32`/`_64`,
`glm_gather_scatter_32`/`_64`, `glm_mm_grouped`) keep `core/affine_mm.metal`'s one-launch gathers and `tf_affine_mm` at
both gather heights, with a partial last column group in the dense entry. Gate 1 plants NaN and both infinities and
requires the predecessor's bytes, NaN payloads included; these
entries compare and keep sums rather than requalify them, so gate 4 does not apply. Each has a control, a planted source
fault red at `shapes`, and its launches are in `glm_launch_records.zig`.
GLM's Metal DFlash2 drafter also registers its ring-buffer block attention (`glm_draft_attend`, `glm_draft_attend_merge`), top-k selector (`glm_draft_topk`), candidate-subset gather head (`glm dflash2 head`, q4/q8 affine gather and row paths), same-buffer GPU accept (`glm_draft_accept`) and GPU follow (`glm_draft_follow`), with planted controls for ring overrun, masked-key leakage, stream-0 ring aliasing, candidate-score tie/drop faults, wrong accepted token, and oversized sliding window.
CUDA also checks LFM2.5-24B-A2B's kernels: the router's fp32 logits (`tf_lfm2m_route`) and its top-4 rule, Liquid's
3-tap short conv with its per-row state snapshots and the state commit (`liquid_conv.cuh`), each against the host's
replay of its fp32 and bf16 steps (three controls: ties to the higher expert, bx unrounded, every stream committing into
the first stream's slot). Their paged attention at head dim 64 runs over shuffled 512-key pages with several streams
a launch: the decode rows' chunks and merge (`tf_paged_attn_chunk_segs_64`, `tf_paged_attn_merge_segs_64`) and the
prompt rows (`pattn_kernel<64, 8, 4, 8>`), head by head within `attention_ref`'s float64 bound (two controls: the
decode reading its values from the neighbouring lane's dims, the prompt from the neighbouring 8 dims).
CUDA also checks the affine-3 experts' split decode pair, bit-equal to the pair kernel it replaces (two controls,
one dropping a last round of K groups short of the warps, one chaining the groups in reverse).
CUDA also checks Flash Next's NVFP4 experts on prompt tiles with the decode form's sums (SwiGLU gate-up and fp32
down, each bit-equal to the decode instance it replaces; the scaled-code prompt tile is the planted control).
CUDA also checks Flash Next's GPTQ int4 experts and head (`fn_int4.cu`, `zig/tests/cuda/contract_int4.zig`) at the
checkpoint's widths: gate/up with SwiGLU, down to fp32 and bf16, the head's dense rows and the two prompt forms
bit-equal to the plain ones, each output against float64 at the weights `qweight` and the fp16 scales store. A gate/up
output's allowance is its epilogue's own worst case (`int4_ref.zig`; the arithmetic is in
[int4 contract allowances](int4-contract-allowances.md)): the two fp32 sums through silu's slope and the up value,
then bf(g), bf(u), bf(silu) and the store, each at its own value's size, and the silu 0 that expf's fp32 overflow
leaves where float64 keeps a denormal-sized value. Gate 4's bf16 proof of that epilogue stops at 1 / (4.0393 u) =
63.4x, because the halves of a mirrored row cancel and a bf16 sum misses by want itself, so the registration states
60. The epilogue's scalar model (`tf_ctl_int4_swiglu` at fault 0) passes the same gates. Five controls: a group sum
rounded to bf16 before its scale (gate 4, the down), each group's sums rounded to bf16 (gate 4, the gate/up), a head
whose words are packed unshuffled (gate 3), and on the gate/up silu of the up sum times the gate sum (gate 3) and
gate, up and silu cut toward zero (gate 3).
A plan whose shared expert's slot is skipped must leave those rows untouched. The 21 launches' usage is in
`int4_records.zig`: nvcc 13.0's from each SM's cubin, the release fatbins' compiler, and nvcc 13.3's where it differs
(the `cuda_133` set).
CUDA also checks the bf16 experts (`experts_bf16.cu`, a checkpoint's own bf16 weights: Nemotron 3.5 Transcribe's routed
and shared experts): relu^2 up, fp32 down and bf16 down, in the decode and prompt forms, against float64 (two controls:
the fp32 down launched one 32-input group short, and a down summing in bf16). nvcc 13.2's prompt forms and 13.3's
decode forms on sm_120 and sm_121 take other registers than 13.0's and have their own records.
CUDA also checks the MXFP4 experts (`experts_mxfp4.cu`, E2M1 codes and E8M0 scales read as stored, each weight decoded
to bf16 in registers, then the bf16 experts' chain): SiTU's up (an expert's gate and up matrices, each sum rounded to
bf16, then ERNIE 5.5's SiTU at beta 4 and linear beta 25), fp32 down and bf16 down, in the decode and prompt forms,
against float64 on ERNIE 5.5's routed shapes (four controls: the fp32 down launched one 32-input group short, each
group's scale read from the next, the up's gate and up matrices swapped, and SiTU's up unclamped). The up's bound adds
SiTU's slopes times how far each sum's bf16 can move within the sum's error. Its records are nvcc 13.0's.
CUDA also checks GLM's NVFP4_NZ5 block 32 experts (`tf_nvfp4_nz5_*`), each bit-equal to the block 16 instance it
replaces on the same values: a planted +5 is a 4 there under its scale two e4m3 steps up (x1.25), and the E2M1
entry reading NVFP4_NZ5's codes, each +5 as -0, is the planted control. Gate 4's stress fill holds +5s among its 4s
and 6s, two on equal inputs mirrored by a -4 and a -6 (NVFP4_NZ5 has no -5), with the float64 reference reading
0b1000 as 5; the same E2M1 entry on that fill is a second control, red at gate 4.
CUDA also checks Nemotron's pocket (`nemotron_ops.cu`, the learned neurons the living-weights learner adds beside the
shared expert): relu^2 up into act rows and down added into the shared expert's slot, a block row a row and tiled, the
down to fp32 and bf16, against float64, the neurons past the count NaN so a read of one shows (two controls: the up's
relu by `fmaxf`, and the down launched one neuron short).

Metal also checks DeepSeek V4.1's CSA2 attention (`csa2_attn.metal`): decode rows (ring or linear windows) and prompt rows
(16-key tiles at absolute positions), fp32 and bf16 out, against float64's softmax over the window, the selected rows and
the sink with the inverse RoPE, and a row's bytes equal across chunk splits and between the two launches (four controls,
each one planted fault, and one on the row bytes). Its slot kernels (`*_segs`: the window key, the ratio-2 pool with
the inputs it keeps, the FP4 round trip, the key packing and the decode attention) run the rows of several slots in one
launch over a table of the slots' state, each bit-equal to its per-slot kernel launched a slot at a time at 1 to 4 slots
and 1 to 17 rows, NaN and infinities included; each but the prefetch attention has a control that reads or writes the
first slot's state for every row. The indexer's decode scores (`tf_dsv41_ix_scores_dec_segs`) and candidate blocks
(`tf_cand_segs`) do the same over each slot's keys, position and key count, bit-equal to the per-slot launches at
ratios 1 and 2, with and without a candidate mask, and a slot with no keys left alone; the top-512 runs unchanged over
score rows of one stride. Their controls score the first slot's keys and count the first slot's keys for every row.
The slot table entry `TfDsv41Seg` lives in one header (`segs.h`) the five sources start with; a kernel reads back its
size and field offsets against segs.zig's `Seg`. After the gates, every per-slot kernel these slot kernels share a body
with is checked byte-equal to the base's body, kept only in the tests (`dsv41_base`), on the same random and non-finite
inputs at one stream.

CUDA also checks the grouped EXL3 experts' `tf_exl3_plan_finish`: its bf16 and fp32 SwiGLU and its fp32 and bf16 down
epilogues bit-equal to main's kernel (kept in `contract_controls.cu`), and its SwiGLU-OAI (`act_mode` 2,
MiniMax-M3.1-Flash's) against float64 at its rounding points (two controls: the up without its + 1, alpha 1).
Its twin `tf_exl3_plan_finish_after` clamps silu's output instead, `min(silu(g), L) * clip(u, -L, L)` (Ling-3.1-flash's
order): its fp32 SwiGLU at limits 10 and 2 and without one, and its fp32 down, against float64 at Ling's widths. A
control runs `tf_exl3_plan_finish` against the after order at limit 2, where silu's term differs by up to L - silu(L)
(0.24; 4.5e-4 at limit 10).

CUDA also checks the grouped EXL3 experts (`tf_exl3_grouped_0`, `_1` and `_2`, one a codebook), which on sm_120 and
sm_121 load a warp's next k tile while the current one decodes: each bit-equal to main's loop (kept in
`contract_controls.cu`) at one, two and three k tiles a warp and at GLM-5.3-Flash's, MiMo-V2.6-Flash's and the Qwen3.5
EXL3 pack's widths (two controls: the next load reading tile i + 2, the last tile decoding the words before it).
Older SMs build main's loop, and their records are main's.

CUDA also checks MOSS-Image's glue (`moss_image.cu`: the norm, plain and modulated; q/k norm and rotary into packed
rows; situ; the gated add), each output within float64's range at its torch bf16 points (five controls, each dropping
one point), and `vision_kernel` at head dim 128 against `contract_attn.zig`'s float64 reference (`moss-contract`).

CUDA also checks Nemotron 3.5 Transcribe's audio encoder glue (`nemo_audio.cu`: log-mel frames, their per-feature
normalisation, q/k norm and rotary in both stacks' forms, the speaker head's taps, ReLU and predictions, and the fusion
of the two branches), each output within float64's range at its bf16 points, and its head-64 attention against
`contract_attn.zig`'s float64 reference (`audio-contract`; eight controls built from `nemo_audio.cuh` with one thing
dropped each).

CUDA also checks GLM-5.3-Flash's image tower on that attention: `vision_kernel` at head dim 64 against the same
float64 reference (`contract_vit_full.zig`, the MOSS-Image test's type at any head dim), and `glm_vis_qk`, which
norms and rotates q and k in place in the rows it reads, bit-equal to `glm_vis_qkv` at segments of one row (one
control: the rotation before a head's bf16 rounding).

CUDA also checks ERNIE 5.5's MultiMax (`ernie5_5/head.cu`, `tf_ernie55_multimax`), which rewrites the head's fp32
logits in place as x + t0 relu(r0 - x) + t1 relu(x - r1) + t2 relu(r2 - x)^2 + t3 relu(x - r3)^2, each step rounded
once in the vendor's order with nothing fused. Each output must be the host's replay of those fp32 steps
(`ernie5_5/multimax.zig`), at a decode row and a 16-row verify window over the 201,216-logit vocabulary and over a
TP=2 rank's half; gate 4 runs whole logits with whole ranges and power-of-two weights, where every step is exact
(`ernie5_5-head`; two controls: each term's multiply and add fused into an fma, and the last term's square dropped).

CUDA also checks ERNIE 5.5's DSA key lists (`ernie5_5/index.cu`, `tf_ernie55_window_merge`): each row's indexer
picks (`topk_rows`' out, `picked` of them held to [0, k] and ending at the first -1), then its sliding window's
min(seq, 128) positions up to seq - 1, then -1, as vLLM-ERNIE's `merge_swa_topk` lays them out for the sparse MLA,
with the count of entries a row. The lists must be the host's (`ernie5_5/window.zig`) to the byte, and every row's
entries, rows told a negative count or one past k included, are rows below seq that ascend to seq - 1 with its picks
below the window, from a decode row at the first position and deep in context, across seq = 128 and seq = k + 128,
to prompt chunks and the 1M context's last rows (`ernie5_5-index`; two controls: every row taking all k picks, and
the window's oldest position dropped).

CUDA also checks ERNIE 5.5's mHC pre side (`ernie5_5/mhc.cu`, `tf_ernie55_mhc_pre`): a row's 24 mixes of its 4 bf16
streams with fn's float32 rows (a thread's 64 terms in one fma chain, then the warps' butterflies), times ERNIE's mix
RMS 1 / (sqrt(mean) + eps); pre, post (2 sigmoid) and comb (each row's softmax plus hc_eps, then 20 rounds of
Sinkhorn's column and row sums); then the collapse sum_s pre_s x_s, its RMSNorm from the unrounded sums' squares, and
the norm's weight times the bf16 collapse, rounded to bf16 once. Each output is held to its float64 bound alone, as
GLM's float32-weight finish is: the normed rows, post and comb (`ernie5_5/mhc.zig`), from rows near unit RMS, offset
by 8 and below 1e-6, where the mix RMS's eps outweighs sqrt(mean); gate 4 runs pre at 1/2 on one stream, where every
fp32 sum is exact (`ernie5_5-mhc`; two controls: the plain mix RMS 1 / sqrt(mean + eps), and comb's softmax over a
column).

CPU: Qwen3.5-2B's affine projection (4 bits, groups of 64, `cpu_kernels.zig`) at the widths the 2B calls it, one row
to a 128-row prompt chunk, registered once for each chain count of its projection classes (16; 32 for the delta rule's
a and b; 8 for the head), and its RMS norm over 2048-wide rows; each registration's runs take 1 to 8 threads in turn,
so gate 2's five runs are five thread counts (the 16-output a and b projections have two tiles, so all five of theirs
run on two). Eight controls: a projection zeroing a non-finite activation, writing its
activations, skipping the rows past its last block of eight, and summing each group in half; a norm taking a
non-finite input as zero, writing its output over its input, scaling by the next element's weight, and summing its
squares in bf16.

CUDA also checks the mla image (`mla.cu`, MLA over a 512-wide latent with 64 roped columns and 128-wide heads, on
GLM's chunk bodies): the head-wise gate, the rope (plain, and with YaRN's attention factor at 900,000 positions), the
128-wide absorb_q and expand_v, full GLM-5.3's 192-wide absorb_q (`tf_mla_absorb_q_192`, 21 and 22 heads), the
256-wide expand_v (`tf_mla_expand_v_256`), and the chunk tiles (`tf_mla_chunks_r16`, 576-wide q and keys over a linear
cache, through `tf_glm_merge`; the split form with `tf_mla_merge_part`; and through the sink merge,
`tf_mla_merge_sink`, which counts each head's fp32 sink once as a zero-value key, at four chunks of 32 heads, one of 64
and two of 32), each against float64 (eleven controls, real launches each given one planted fault: the gate reading the
next head's column, the plain rope one position late, the YaRN rope without its factor, both absorb_q forms and both
expand_v forms reading the next head's weights, the chunks the next head's queries, the split chunks reading a rank's
page z as chunk z, the per-rank merge one chunk short, and the sink merge with every sink left out). The sink merge's
body is Hy4-flash's (`tf_hy4_attend_merge`, which now instantiates it from `common/mla.cuh` in its own image), and
`tf-cuda-test mla-sink` checks the two give the same bytes at Hy4-flash's shapes.

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
