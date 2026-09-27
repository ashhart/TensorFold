# Remaining native inference coverage and performance work

Goal: close every identified gap executable on this M5 Max, push completed milestones
to `feat/zig`, and finish with reproducible end-to-end Python/Zig measurements in the
root README immediately after the fork attribution. Hardware qualification on other
Macs remains explicitly separate.

Completion requires execution evidence, not just source export or a passing smoke test.

- [x] Automated serial/draft regression for DFlash2 and both MTP families, CPU/Metal
  sampling, greedy/sampled settings, small/maximum budgets, tensor/forced-SIMD paths.
- [x] Randomized tree verification, every accepted prefix, maximum windows, rejection,
  EOS/token-budget boundaries, repeated reset/cache fork/restore cycles.
- [x] Full-model Python/native long-context comparisons across sparse and 10K dispatch
  thresholds; memory measurements establish what fits rather than assuming a limit.
- [x] Adversarial routing, sampling, PLE hash/history/shard boundaries and quantization
  fixtures, including direct checks of each additional implementation variant.
- [x] Loader/config/shape/dtype failure checks, missing/truncated shards and missing MTP
  heads; allocation failure cleanup and sustained memory-growth checks.
- [x] Inventory and native execution coverage for remaining upstream fused projection,
  GDN and row-forward optimization variants; distinguish selectable production paths
  and diagnostic variants in the final inventory.
- [x] Audit original runtime optimizations (reduced draft vocabulary, queued MTP,
  stacked SIMD projections), documenting production choices and any remaining behavior
  that needs a native implementation or an independent execution check.
- [x] Early MTP speculation/accepted-state reuse and adaptive draft depth, with independent
  original-policy fixtures and real-model batch/serial cache checks.
- [ ] Finish pipelined serial decode, GPU proposal handoff and attention buffer reuse.
  Nemotron GPU proposal handoff is implemented and checked; preserve the bounded PLE
  memory strategy when evaluating Flash paths.
- [ ] Final matched end-to-end benchmarks against the original Python engines: same
  checkpoints, prompts, seeds, sampling and token counts, serial and drafting, repeated
  runs, cold process/load/prefill/decode/total breakdown and token parity. Document any
  unavoidable runtime or prefill differences and regressions honestly.
- [ ] README comparison table directly below the fork notice, detailed machine/method
  and raw measurements checked in, final verification and remote push.

Prior evidence: native/COVERAGE.md describes the baseline through commit 8d7f2d3.
New milestone evidence and any discovered failures will be recorded below.

## Acceptance/cache milestone

- Ten safety-enabled host tests pass, including 1,056 acceptance/budget combinations,
  explicit EOS cases and 1,000 random trees.
- All five full-model/backend cache suites pass: 1,056 accepted-prefix comparisons in
  total, all cache arrays and continuation logits exact. Qwen also checks every path of
  seven random trees per backend. No active MLX memory growth across 128 measured
  forward/partial-commit/reset cycles for each combination (640 cycles total).
- Fixed invalid borrowed array handles in saved Qwen recurrent replay and Flash pooled
  attention records. Qwen commit now rejects invalid position/path arguments atomically.
- Draft suite now rejects immediate-EOS runs without decode rounds. Nemotron's original
  synthetic prompt hit this case; replaced it with the code prompt before counting its
  matrix results. The corrected Nemotron matrix passes all 18 comparisons. Qwen's six
  comparisons passed before the handle fix; the final binary will be checked again.
- Flash passes serial versus MTP budgets 1 and 3 with greedy sampling. The maximum
  budget (15) also timed out when rerun alone. This is an unresolved execution failure,
  not a passing case or an established hardware limit; stage diagnostics are in place.
- Four checked-in checkpoint config fixtures exercise 43 adversarial mutations with
  three invalid replacements each. Validation now rejects unsupported activation,
  cache dtype, bias, routing normalization, PLE and MTP configuration choices instead
  of silently running the hard-coded recipe. Tensor/shard failure coverage remains open.

## Bounded PLE and checkpoint reader milestone

- Flash reads the selected packed PLE rows directly from safetensors instead of
  materializing whole embedding shards. All 640 boundary/adjacent/middle rows across
  128 shards match independent MLX load/dequantization exactly.
- Flash's nine serial/MTP matrix comparisons now pass, including budget 15 with Metal
  sampling. A separate greedy maximum-budget reproduction matched the original output
  hash and completed in 1.386 s instead of the prior 85.354 s; diagnostic only, not the
  final performance comparison. All 44 comparisons passed after integration: eight
  Qwen, 24 Nemotron and 12 Flash, spanning greedy, 32-token CPU/Metal sampling and the
  two-token budget boundary. The previous 33-case full suite passed, then all 11 added
  32-token CPU cases passed with `-Ddraft-scenario=2`.
- Flash's 96 accepted-prefix checks and 128 measured reset cycles pass. Active MLX
  memory stays at 79,022,784,536 bytes, versus 111,025,160,216 before bounded PLE reads.
- Thirteen host tests pass. The safetensors parser rejects malformed shapes/dtypes,
  overlapping/gapped/truncated payloads and arithmetic overflow. File tests exercise
  missing/truncated files, header length limits, invalid row reads and allocation
  failure cleanup. All production safetensors loads now run this preflight validation.
- Remaining loader work: validate every model-specific tensor shape and required tensor,
  exercise missing MTP heads, and broaden allocation-failure cleanup beyond the new reader.

## Long-context and tensor metadata milestone

- Sixteen full-model Python/native comparisons pass: Qwen at 9,999/10,007 tokens,
  Nemotron at both lengths with tensor and forced SIMD, and Flash at 2,051/2,063 tokens;
  each checks serial and neural drafting with context copies disabled. Every final-block
  logit and all 16 continuation tokens match exactly. Memory figures are in COVERAGE.md.
- Corrected the Flash oracle to call the original serving runtime's row-invariant head.
  A remaining real mismatch was traced to PLE normalization: native had reused the MTP
  fused RMS kernel, which changes fp32 reduction rounding. Following PLE's original
  square/mean operations fixes long-context parity. Layer and per-block trace tools remain
  available to reproduce and localize future discrepancies.
- All 6,105 required tensor names/shapes/dtypes are validated against real checkpoints
  and by production loaders. Thirty-eight CLI metadata rejection cases and fifteen
  safety-enabled host tests pass. Broader allocation cleanup and MLX failure injection
  remain open; metadata failures do not establish MLX-internal OOM cleanup coverage.
- Additional focused PLE normalization fixtures detect the old implementation at 15
  BF16 elements. Running them also exposed that the header reader rejected MLX's absent
  metadata represented as null; the parser now accepts that optional representation.
  All 89 Metal fixtures pass, including all 36 new normalization cases, and all 59
  embedded kernel sources remain identical to upstream Python.
- Remaining backend/oracle checks include Qwen long-context Python SIMD parity and the
  Nemotron short code-prompt SIMD reference. Final performance measurements are pending.

## Long-cache and optional-kernel milestone

- All 384 long-context accepted-prefix checks pass: 144 each for Qwen tensor/forced
  SIMD, 32 each for Nemotron tensor/forced SIMD, and 32 for Flash. Every cache array,
  verified row and continuation matches serial after partial commits and full rejection.
- Native replay now exercises 1,041 original Python launches across 44 optional and
  production kernel variants. The embedded catalog has 84 kernels; source checks pass.
  The Python fixture suite passes 89 tests and skips one undefined quantization shape.
  Coverage includes fused/row dense paths, scalar/matrix SIMD with dependency/custom
  prologue examples, Nemotron expert/routing/norm variants, and Flash grouped experts,
  projections, embeddings and routing ties. A required-variant inventory prevents
  silently losing a diagnostic path. Details and limitations are in COVERAGE.md.
- Fixed out-of-bounds reads below width 512 in all three original Flash expert-down
  shaders and their native copies. Independent all-ones dot-product checks cover eight
  widths from 32 to 1,024, shared/unshared outputs, and invalid-width rejection.
  Checkpoint-width arithmetic is preserved. Diagnostic outputs intentionally left
  unwritten by original kernels are zero-initialized before comparison.
- Fifteen safety-enabled host tests pass after the native dispatch refactor to support
  variants with six outputs. All 89 shared Metal fixtures pass. Flash's 96 short-cache
  checks and 128 reset cycles pass again with no active-memory growth (79,023,013,912 bytes).
  All twelve Flash serial/MTP comparisons pass again after the normalization and shader
  changes, including CPU/Metal sampling, greedy and the two-token budget boundary.
  Remaining work includes backend oracles, allocation cleanup,
  the final kernel/production-path inventory audit, and end-to-end engine benchmarks.
- The current static inventory leaves `lane_attention_partial` and
  `lane_attention_partial_128` without direct native fixture execution (production uses
  the direct variants). Flash's original eight-group `ple_lookup` specialization also
  needs diagnostic coverage; production uses the separately verified bounded row reader.
  Audit runtime optimizations such as reduced draft vocabulary and queued MTP proposals
  separately from operation coverage before declaring parity with all Python modes.

## Remaining kernel, SIMD oracle and ownership milestone

- All remaining catalog variants execute: 1,104 launches across 54 variants pass exactly.
  Added both non-direct attention widths, eight-group PLE lookup at group boundaries,
  and fp32/BF16 router output specializations. All 86 embedded sources match Python.
  KERNEL_INVENTORY.md lists every kernel, its native integration sites and diagnostic
  counts; the generator rejects stale fixtures and uncovered catalog entries.
- The four Qwen SIMD long comparisons pass at 9,999/10,007 tokens, serial and DFlash2:
  every final-block logit and all sixteen continuation IDs match. Total long-context
  comparisons are now twenty. The SIMD oracle uses original unstacked projections and
  row attention; Python's stacked row_forward is a different rounding configuration,
  already exercised by variant replay. Short Nemotron SIMD code-prompt parity also
  passes as a dedicated reproducible target.
- A new GPU-backed ownership diagnostic passes 327 injected host allocation failures,
  tensor/SIMD and resize/fallback growth. It covers scopes, both weight stores, linear
  preparation, actual indexed/unindexed checkpoint files and kernel dispatch. All bytes
  are released and active MLX memory returns to zero. Null handles, invalid arity and
  real MLX API errors recover cleanly, including sixteen repeated error cycles.
  MLX/driver internal allocation sites are not instrumented by this native-boundary test.
- All fifteen host tests and all five real-model cache checks pass after the ownership
  changes. Runtime optimization audit and final engine benchmarks remain outstanding.

## Reduced-vocabulary and queued-MTP milestone

- Embedded the two original ID lists directly through the build graph. Native MTP
  selects their packed head rows by default; full-vocabulary and per-token host-read
  switches preserve both configurations. CPU/Metal sampling use the original token
  IDs for noise and tie ordering. Target verification always uses the full vocabulary.
- All 90 real-checkpoint serial/MTP comparisons pass, including 36 queued/host pairs
  with exact proposal hashes and acceptance/round counts. Budgets 1/3/15 cover greedy,
  Metal and CPU sampling, Nemotron tensor/SIMD and Flash. Context copies are disabled.
- Original Python row-selection fixtures verify all 32,768 Nemotron and 79,592 padded
  Flash IDs and packed weight/scale/bias rows. All 121 shared CPU/Metal fixtures and
  sixteen safety-enabled host tests pass. Allocation cleanup now checks 425 failure
  points, including reduced-head construction; retained active MLX memory is zero.
- RUNTIME_AUDIT.md records the remaining Python scheduling features separately from
  complete kernel execution coverage. Early speculation/state reuse, adaptive depth,
  serial pipelining and attention-buffer reuse still need native work or an explicitly
  justified memory constraint. Final benchmarks and the README comparison remain open.

## Early speculation and adaptive-depth milestone

- Native target samples feed batched MTP before the host reads verification results;
  only the accepted prefix and its last state/draw survive for the next chain. The
  original adaptive-depth policy uses native window/step calibration and measured
  acceptance/cost updates. Both late and fixed-depth diagnostic paths remain available.
- All 150 real-model serial/MTP comparisons pass, including 84 exact early/late and
  queued/host proposal-stream comparisons. Twelve adaptive runs preserve serial output.
  All 414 real-checkpoint MTP accepted-prefix/cache/continuation checks pass through 10K.
- Original Python policy methods supply 12,288 passing depth decisions and 360 passing
  sampling-position/retained-row fixtures. Corrected draft noise from P+2 to P+1 for
  the first proposal after pending token P. Target verification had protected final
  output, but the wrong key could lower draft acceptance.
- Fixed Flash sparse rollback retaining pooled keys below the pooling threshold. Its
  MTP hidden projection now splits 64 stream rows into row-exact calls. The original
  Python MTP matmul choices differ; RUNTIME_AUDIT.md records those configurations.
- All 121 shared CPU/Metal fixtures, seventeen host tests and 566 allocation-failure
  points pass; retained MLX memory is zero after ownership/error tests.
- Flash's full-target long-cache suite now includes the 2,044-token crossing window.
  All 48 accepted-prefix/cache/continuation comparisons pass at 2,044/2,051/2,063 tokens.
- Full-target Python parity passes again after integration: Nemotron at 10,007 tokens
  (917,504 final-block logits) and Flash at 2,051 (744,960 logits). Every logit and all
  sixteen continuation IDs match in serial and early-MTP modes, with fifteen drafts
  and context copies disabled.
- GPU proposal handoff, pipelined serial decode and attention buffer reuse remain,
  followed by original-engine benchmarks and the README comparison table.

## Nemotron GPU handoff milestone

- Queued Metal proposals now feed target embedding without a host token read. Incoming
  proposals, target draws and early MTP draws share one read. A host override remains.
- All 48 focused serial/MTP comparisons and 36 exact scheduling comparisons pass:
  tensor/SIMD, greedy/Metal sampling, reduced head, depths 1/3/15, early/late speculation.
- All 192 short lazy-GPU-input accepted-prefix/cache/continuation comparisons pass,
  including windows containing both EOS IDs. Another 256 reset cycles show no active
  MLX memory growth. Logical EOS truncation has explicit host coverage at every position.
- All 64 long GPU-input cache/continuation comparisons pass on tensor/SIMD at
  9,999/10,007 tokens. All eighteen safety-enabled host tests pass.
- Flash's maximum-depth shared-driver regression passes seven serial/MTP and four
  exact scheduling comparisons, including full/reduced heads and adaptive depth.
- The final Nemotron maximum-depth matrix passes another 22 serial/MTP and sixteen
  scheduling comparisons with full/reduced heads on both backends, including adaptive
  runs. This partially overlaps the focused matrix; counts are separate execution runs.
- Flash GPU handoff, serial pipelining and buffer reuse remain, followed by final
  original-engine performance comparisons and the README table.
