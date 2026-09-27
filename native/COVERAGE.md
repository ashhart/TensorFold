# Native Metal coverage

This fork ports the inference hosts to Zig and embeds the original author's Metal
kernels. MLX-C supplies arrays, scheduling, safetensors, and general operations.
This matrix distinguishes exercised behavior from physical-device validation.

## Early speculation and adaptive-depth milestone

- Metal sampling now queues batched MTP work behind target verification before its
  host read. The accepted MTP prefix, hidden state and first draft are reused in the
  next round. Late speculation and fixed-depth switches retain diagnostic baselines.
- All **150 serial/MTP comparisons** pass, including **84 exact scheduling comparisons**
  of proposal hashes, round counts and accepted drafts across early/late and queued/host
  modes. Twelve adaptive runs also match serial output, using measured native costs.
- All **414 MTP prefix/cache/continuation checks** pass with real checkpoints: Nemotron
  tensor/SIMD and Flash, widths 1/3/16, at 0/31/2,044/2,051/9,999/10,007 cached positions.
  Each checks every batched hidden/logit row and every retained prefix, including zero.
  Fixed real target states isolate the MTP computation; these are not full-target 10K
  generation runs. The separate target long-context checks cover that integration.
- **12,288 adaptive-depth choices/acceptance updates** match the original Python policy.
  **360 scheduling fixtures** call Python's actual `speculate`/`settle` methods with an
  identity head to isolate sampling positions and retained-row selection.
- These checks exposed two bugs: MTP draft noise was keyed one position too far ahead,
  and Flash retained a speculative pooled cache after rolling back below the sparse
  threshold. Both are fixed. Flash's 64-stream-row MTP projection stays row-exact by
  splitting it into <=16-row calls.
- Full-target Flash rollback also passes all **48 accepted-prefix comparisons** at
  2,044/2,051/2,063 tokens, including the newly added window that crosses the sparse
  threshold and retains rows below it. Every cache array and continuation is exact.
- After early-MTP integration, full-target Python comparisons pass again at 10,007
  Nemotron tokens and 2,051 Flash tokens: respectively **917,504** and **744,960**
  final-block logits, plus all sixteen continuation IDs in serial and early-MTP modes,
  match exactly. Context copies are disabled and the MTP budget is fifteen.
- All **566 host allocation failures** pass, including scheduler prepare/propose/
  speculate/settle cleanup and invalid-budget/commit recovery. MLX active memory returns
  to zero. All **121 shared CPU/Metal fixtures** and **17 safety-enabled host tests** pass.

The [runtime audit](RUNTIME_AUDIT.md) still lists GPU proposal handoff, serial pipelining,
attention buffer reuse, and arithmetic configuration differences. Final engine benchmarks
and the root README comparison remain pending.

## Reduced-vocabulary and queued-MTP milestone

- Native MTP now uses the original reduced vocabulary by default: 32,768 Nemotron IDs
  and 79,592 padded Flash IDs. `test-draft-vocab` independently verifies every mapped
  ID and selected packed weight/scale/bias row against Python; packed uint32 data is
  compared directly, without a lossy float conversion.
- Metal MTP can queue a whole dependent proposal chain before reading tokens. All
  **90 serial/MTP comparisons** pass across full/reduced heads, queued/host modes,
  budgets 1/3/15, greedy/Metal/CPU sampling and Nemotron tensor/SIMD plus Flash.
  All **36 queued/host pairs** also match every proposal window's SHA-256, round count
  and accepted-draft count. Context copies are disabled throughout this matrix.
- Added 32 independent CPU sampler fixtures, including mapped token IDs and ties.
  All **121 shared CPU/Metal fixtures** and **16 safety-enabled host tests** pass.
- Reduced-head construction and cleanup add 98 host allocation failure points;
  **425 total failures** pass with all bytes released and zero retained MLX memory.
- The [runtime audit](RUNTIME_AUDIT.md) identifies remaining scheduling work: early MTP
  speculation/reuse, adaptive depth, serial pipelining and attention buffer reuse.
  Whole-chain queuing alone does not establish parity with Python's full pipeline.
  Final original-engine performance comparisons remain pending.

Reproduce with `zig build test-draft-vocab`, `zig build test-mtp-runtime`,
`zig build test-metal -Dmetal-tensors=true`, and `zig build test-allocation-failures`
using `.zig-toolchain/zig`. The baseline and earlier milestones below retain their
original counts to distinguish historical runs from the expanded checks.

## Model paths

| Path | Native implementation | Executed checks on M5 Max |
| --- | --- | --- |
| Qwen3.8-27B target | Embedding, quantized projections, norms, GDN, attention, gated MLP, head | 993,280 Python logits exact; sampled and greedy completions |
| Dense tree verification | Parent/depth metadata, recurrent replay, KV gather and partial commit | 513-token prefix, branched rows, continuation and all 64 caches exact |
| Dense SIMD | simdgroup quantized matmul and row attention | Forced SIMD tree/cache check; 128-token GPU-sampled DFlash2 equals serial; physical M1–M4 untested |
| DFlash2 | Five layers, hidden taps, dynamic convolution, selector codebooks, candidate tree | 128 sampled tokens match Python serial; GPU radix top-k preserves output |
| Context copy | Suffix matching and target-verified proposals | Host tests and generation path |
| Nemotron target | Mamba convolution/SSM, grouped norm, NoPE attention, top-6 routing, ReLU² experts, shared experts | 524,288 Python logits exact; verified rows and all 52 caches exact |
| Nemotron MTP | Embedding/hidden fusion, attention, MoE, cloned draft cache | 32 sampled tokens match Python serial with both CPU and GPU samplers |
| Nemotron SIMD | Row quantized projections and expert kernels; MLX SDPA | Forced SIMD verified rows, rollback and all 52 caches exact |
| Nemotron long context | 128-wide tensor attention at 10K visible keys; SDPA otherwise | Full-model Python logits and 16-token serial/MTP continuations exact at 9,999 and 10,007 tokens, both backends |
| Flash Next target | Four residual streams, hyper-connections, GDN, sparse attention, top-10 experts/shared gate | Full-model Python logits and serial/MTP continuations exact at 2,051 and 2,063 tokens; all 48 caches checked separately |
| Flash Next PLE | N-gram hashing/EOS reset, positional packed-row reads, gate, dilated convolution | Shipped hash constants checked; all 128 shard boundaries checked against MLX; rollback history/conv cache exact |
| Flash Next MTP | Embedding and stream fusion, HC attention/MoE, separate output mixer | 16 sampled tokens match Python serial |
| Flash sparse selection | Pool four keys, select top 512 blocks, include causal tail, merge attention | Prefixes 2,044/2,051/2,063; cold and populated pool, eight rows and rollback at every nonzero row |

## Shared Metal utilities

`test-metal` generates fixtures using the original Python wrappers and independently
loads/dispatches them through Zig/MLX-C. Tests fail on any bit mismatch.

| Utility | Cases |
| --- | --- |
| GPU sampling | BF16/f32 logits; greedy and sampled; seeds 0/1234/5678/UINT64_MAX; mapped and full token IDs; top-k 0/20/2048; top-p .8/.95/1; ties and dominant logits; positions through 262,144 |
| Radix top-k | Vocabularies 31/4,097; k 1/16/31/64; deterministic ties; BF16 conversion of f32 input |
| Tensor attention | D128/D256; 1/3/8 queries; 513/9,999/10,007 keys; cache views with unused trailing capacity; every batch row equals serial |
| Sparse attention | Original fused Python attention with deterministic synthetic weights at actual Flash dimensions; pooled keys and projected outputs match; rejected rows do not affect continuation |
| PLE normalization | 36 cases: 1/3/16 rows, zero/tiny/unit/large inputs, three seeds; exactly matches the original square/mean arithmetic and detects the former fused substitution |

All 89 fixtures pass on M5 Max. The fixtures use synthetic inputs to reach expensive context branches cheaply.
The separate full-checkpoint tests establish model integration. Kernel export checks
ensure embedded sources remain verbatim; source export alone is not a numerical test.

## Reproduce

```sh
bash scripts/fetch-zig.sh
.zig-toolchain/zig build test -Doptimize=safe
.venv/bin/python tools/export_native_kernels.py --check
.zig-toolchain/zig build test-metal -Doptimize=safe -Dmetal-tensors=true
.zig-toolchain/zig build test-variants -Doptimize=safe
.zig-toolchain/zig build test-allocation-failures -Doptimize=safe
.zig-toolchain/zig build test-nemotron-simd-reference -Doptimize=safe
.zig-toolchain/zig build test-models -Doptimize=safe
.zig-toolchain/zig build test-cache-stress -Doptimize=safe
.zig-toolchain/zig build test-drafts -Doptimize=safe
.zig-toolchain/zig build test-checkpoint-files -Doptimize=safe
.zig-toolchain/zig build test-model-schemas test-schema-failures -Doptimize=safe
.zig-toolchain/zig build test-long-context -Doptimize=safe
.zig-toolchain/zig build test-long-cache -Doptimize=safe
.zig-toolchain/zig build test-ple -Doptimize=safe
```

The suite needs the MLX prefix described in README and Python development dependencies
in `.venv`. Omit `-Dmetal-tensors=true` on GPUs without tensor units. Model paths default
to `build/models`; override with `-Dmodel-root=/absolute/path`. All fixtures and weights
stay in ignored `build/` directories. `test-models` deliberately serializes model loads.
The cache and draft suites also serialize their model loads; run these targets separately,
since simultaneous suites compete for unified memory and GPU execution time.
Use `-Dcache-family=0|1|2` or `-Ddraft-family=0|1|2` to select Qwen, Nemotron or Flash.

## Expanded acceptance and cache checks

The shared acceptance policy is used by both DFlash trees and MTP chains. Host tests
cover all 32 chain acceptance lengths against 33 output budgets, accepted/rejected/bonus
EOS, and 1,000 deterministic random trees. The fifteen host tests pass with safety
checks, including four supported config fixtures and 129 invalid recipe mutations,
safetensors parser allocation failures and all 128 PLE shard lookup boundaries.

`test-cache-stress` has passed on all five model/backend combinations below. Each
accepted prefix is compared with serial execution: logits, every cache array and its
metadata, and the next token's logits after restoring the original cache. It also drops
entire speculative passes, tests invalid commits, inserts EOS IDs into the PLE history,
and repeatedly restores snapshots and resets the model.

| Model/backend | Accepted-prefix checks | Prefix lengths | Post-warmup reset cycles | Active MLX memory, baseline = maximum |
| --- | ---: | --- | ---: | ---: |
| Qwen tensor | 384 | 0, 1, 15, 16, 17, 63, 64, 127, 128, 511, 512, 513 | 128 | 15,133,588,480 bytes |
| Qwen forced SIMD | 384 | Same | 128 | 16,734,960,640 bytes |
| Nemotron tensor | 96 | 0, 1, 15, 16, 17, 33 | 128 | 18,816,686,464 bytes |
| Nemotron forced SIMD | 96 | Same | 128 | 17,778,989,440 bytes |
| Flash, positional PLE reads | 96 | Same | 128 | 79,022,784,536 bytes |

Qwen additionally checks every path in randomized trees with 2, 7, 15, 16, 17, 31 and
32 rows, on each backend. Its chain windows reach the 128-row prefill limit; family
windows reach 16 rows. Memory figures measure MLX active allocations after synchronization,
not process RSS or peak unified-memory use.

These tests exposed a borrowed recurrent-state handle in Qwen's saved verification pass:
restoring the live cache could invalidate the handle needed for a later partial commit.
The pass now retains its replay base. Flash's saved pooled-key handle received the same
lifetime fix. Dense commit also validates position and ancestor paths before changing
the cache, and releases temporary replay arrays at the end of each commit.

`test-drafts` passes all 44 serial/drafted comparisons: 17-token greedy, 32-token CPU
sampling, 32-token Metal sampling and a two-token CPU output budget. MTP budgets are
1, 3 and 15. Use `-Ddraft-scenario=0|1|2|3` to select a scenario. The comparison
requires actual decode rounds so an immediate EOS cannot masquerade as draft coverage.
Completion evidence for this larger matrix is tracked in [WORK_PLAN.md](WORK_PLAN.md).

## Checkpoint and PLE row reads

All safetensors loads validate dtype names, bounded dimensions, checked byte-size
arithmetic and contiguous non-overlapping offsets against the actual file size before
calling MLX. Shard names from indexes must be simple safetensors filenames. Duplicate
tensor names across loaded files are rejected. Native readers additionally validate
the dtype/shape of all 384 PLE packed weight/scale/bias tensors and their total row count.

`test-checkpoint-files` verifies missing files, short headers, oversized declarations,
bad payload lengths, wrong row buffer sizes, out-of-range rows, and files truncated
after opening. Parser and file-opening checks inject failure at every Zig allocation.
The dense loader now cleans up array/linear ownership if map insertion fails.
All four fixed checkpoint recipes now validate required tensor names, shapes and dtypes
before model transformations and inference kernels: 1,847 Qwen, 81 DFlash2, 763 Nemotron
(including MTP), and 3,414 Flash tensors. `test-model-schemas` independently verifies
all 6,105 against actual headers and index references. `test-schema-failures` passes
38 native CLI rejection cases: missing/truncated files, missing tensors/MTP, bad
dtype/rank/shape, missing or invalid index entries, unsafe paths and wrong shard references.
The fifteen safety-enabled host tests also pass. Native ownership-boundary allocation
failure checks are described below; MLX's internal C++ allocator is not instrumented.

`test-ple` independently loads each shard with MLX and compares its first, adjacent,
middle, penultimate and final rows against native positional reads: 640 exact rows.
The production lookup bounds scratch storage to 256 packed rows (25,600 bytes), followed
by GPU dequantization. Full-model active MLX memory dropped by 32,002,375,680 bytes in
the repeated cache check. The previously intermittent maximum-budget Flash timeout
did not recur in the complete Flash draft matrix after this change.

## Limits

Physical validation is on one M5 Max with 128 GiB, with both normal and forced SIMD
dispatch. Older Apple GPUs still require execution on those devices before claiming
hardware qualification. Flash Next fits the tested 2,063-token context through positional
PLE reads; this does not establish its maximum feasible context length on this machine.

This is functional coverage of the three upstream Metal inference recipes. Native
dispatch selects one implementation of each operation; historical or optional fused
optimization variants are not all separate native execution modes. CUDA, HTTP serving,
chat templates, vision, and disk prefix-cache persistence are outside this Metal port.

## Full-model Python comparisons at attention thresholds

`test-long-context` has passed 20 native serial/drafted comparisons against the original
Python arithmetic. Every logit in the final prefill block is compared exactly, followed
by all 16 generated tokens (Metal sampling, seed 5678, temperature .7, top-k 12, top-p .8).
Context copies are disabled in drafted runs, so DFlash2/MTP heads execute.

| Model/backend | Prompt lengths | Compared logits per final block | Native peak MLX bytes at longer prompt, drafting |
| --- | --- | --- | ---: |
| Qwen tensor + DFlash2 | 9,999 / 10,007 | 3,724,800 / 5,711,360 | 28,813,688,832 |
| Qwen forced SIMD + DFlash2 | 9,999 / 10,007 | 3,724,800 / 5,711,360 | 27,498,844,068 |
| Nemotron tensor + MTP | 9,999 / 10,007 | 1,966,080 / 917,504 | 21,533,663,066 |
| Nemotron forced SIMD + MTP | 9,999 / 10,007 | 1,966,080 / 917,504 | 20,460,568,482 |
| Flash + MTP | 2,051 / 2,063 | 744,960 / 3,724,800 | 84,910,995,412 |

These tests use repeating token IDs to bound the Python PLE oracle's resident shard set.
Python keeps PLE tables sharded instead of concatenating another 32 GB copy. All original
normalization, convolution, attention and projection arithmetic is retained. The oracle
uses the serving runtime's row-invariant vocabulary projection, not the raw model's
batch-dependent output matmul. These are correctness checks, not final engine benchmarks.

The Flash comparison exposed a real normalization error after token 432: native PLE used
the MTP RMS kernel, whose fp32 reduction differs from PLE's separate square/mean operations.
Small BF16 differences entered recurrent state and eventually changed most logits. Native
PLE now follows the original operation order; the strict long tests pass without tolerance.
`tools/native_flash_trace.py` and `--trace-dir` compare 242 layer/projection intermediates;
`--gdn-layer N` / `--trace-gdn N` trace one recurrent block across the entire prefill.
Use `-Dlong-family=0|1|2` and optionally `-Dlong-tokens=N` to isolate a long-context case.
Use `-Dlong-backend=0|1` for tensor/SIMD. Qwen's SIMD oracle composes the original
unstacked lane host, `simd_qmm` projections and `row_attention`, matching native's
configuration. The original stacked `row_forward` changes shape-dependent reduction
splits and is numerically different; it is covered separately by variant replay.
The 17-token short code-prompt Nemotron SIMD completion also matches Python exactly,
reproducibly checked by `test-nemotron-simd-reference`.

`test-long-cache` also passes all five model/backend combinations: Qwen tensor and
forced SIMD each check 144 accepted prefixes at 9,999/10,007 tokens (16/128-row
windows); Nemotron tensor and forced SIMD each check 32; Flash checks 32 at
2,051/2,063 tokens. All 384 checks compare every cache array, verified logits and
continuation after partial acceptance, complete rejection and rollback. These are
separate from the short-context reset/memory checks above.

## Optional implementation variants

`test-variants` passes **1,104 native launches across 54 embedded variants**, with
every output bit matching Python. It runs upstream assertions before recording
inputs, templates, launch geometry and expected arrays; Zig replays the launches
using its embedded catalog, validates source hashes, and never executes fixture source.
The generator fails if any required variant is absent. Its Python tests pass 89 cases;
one group-128/width-1,856 combination is skipped because that quantization is undefined.

| Family | Executed variants and boundaries |
| --- | --- |
| Dense fused/row paths | Three lane-fuse variants; both historical GDN steps; row-forward norms, pre/post-GDN, MLP, tree, partial sums, fused gate/up, six quantized projection epilogues; groups 32/64/128 |
| SIMD projection | Scalar and simdgroup matrix, 1–128 rows, five original projection shapes, dependency inputs and the upstream scaling-prologue example |
| Nemotron | Row projections and both expert projections with original fp32 error-bound assertions; routing ties, selection bias, saturated sigmoid, zero probability, 0/1/2 shared slots; three residual/norm variants |
| Flash | Quantized row projections, tiled embedding, stacked SwiGLU; top-10 ties/random/dominant routing at 32/512 experts; grouped/ungrouped gate-up and down, shared and unshared outputs, 1/3/16 rows |
| Expert-down bounds | Independently exact dot products at widths 32/64/128/480/512/544/768/1,024; rejected zero, half-group and oversized widths |
| Additional attention and PLE | Direct/non-direct attention at D128/D256, 513/10,007 keys and 1/3/8 queries; original eight-group PLE lookup at every group boundary, 1/3/16 rows; fp32/BF16 router outputs at 256/512 threads |

The narrow expert-down cases exposed out-of-bounds reads in all three original down
kernels: inactive SIMD lanes still loaded 16 inputs and packed weights below width 512.
Python and embedded Metal now guard those lanes and validate group-32 widths through
1,024. The supported checkpoint width keeps the same arithmetic.

These fixtures intentionally use contiguous inputs. Separate attention tests cover
strided caches. Unwritten output regions (branched-tree terminal state and unused
expert-group slots) are initialized to zero on both sides. SIMD's Python constexpr
dimensions become equal-valued native template arguments; the kernel body is unchanged.
The custom-prologue example is finite diagnostic coverage, not support for arbitrary
runtime shader source. These additional embedded variants are diagnostic entry points;
production inference still selects the implementations documented in the model matrix.

After these changes, Flash again passes all 96 short accepted-prefix checks and 128
post-warmup reset cycles. Active MLX memory remains exactly 79,023,013,912 bytes across
the measured cycles. All 89 shared Metal fixtures and 15 safety-enabled host tests pass.
All twelve Flash serial/MTP comparisons also pass again: greedy, CPU and Metal sampling,
two-token output budgets, and MTP budgets 1/3/15.

The complete [86-kernel inventory](KERNEL_INVENTORY.md) distinguishes production
integration sites and diagnostic replay. `tools/native_kernel_inventory.py --check`
rejects stale fixtures and kernels without either integration or diagnostic coverage.
Static integration sites alone do not prove execution: the model/cache/draft and shared
utility suites above establish that separately.

## Allocation and MLX error recovery

`test-allocation-failures` passes **327 injected host-allocation failures** across
temporary scopes, weight-map insertion/replacement, owning dense-weight insertion,
linear preparation, indexed/unindexed checkpoint reads, and kernel dispatch/cache hits.
Both tensor and SIMD branches run, with in-place resizing enabled and disabled. The
latter forces allocate/copy growth so failures after a scope's first allocation are
also tested. Every allocation is freed; synchronized MLX active memory returns to zero.

The diagnostic also rejects null array handles and invalid kernel arity, exercises a
real MLX reshape error, verifies subsequent operations still succeed, and repeats that
error/recovery cycle sixteen times without retained memory. Kernel/config construction
now rejects null handles. Family ownership is initialized explicitly at runtime, and
the diagnostic substitutes a failing Zig allocator only within its single-threaded scope.
This checks the native ownership boundary; it does not inject faults into every private
allocation in MLX, the Metal driver, or the operating system. All five real-model cache
checks pass again after these ownership changes.
