# Remaining native inference coverage and performance work

Goal: close every identified gap executable on this M5 Max, push completed milestones
to `feat/zig`, and finish with reproducible end-to-end Python/Zig measurements in the
root README immediately after the fork attribution. Hardware qualification on other
Macs remains explicitly separate.

Completion requires execution evidence, not just source export or a passing smoke test.

- [ ] Automated serial/draft regression for DFlash2 and both MTP families, CPU/Metal
  sampling, greedy/sampled settings, small/maximum budgets, tensor/forced-SIMD paths.
- [x] Randomized tree verification, every accepted prefix, maximum windows, rejection,
  EOS/token-budget boundaries, repeated reset/cache fork/restore cycles.
- [ ] Full-model Python/native long-context comparisons across sparse and 10K dispatch
  thresholds; memory measurements establish what fits rather than assuming a limit.
- [ ] Adversarial routing, sampling, PLE hash/history/shard boundaries and quantization
  fixtures, including direct checks of each additional implementation variant.
- [ ] Loader/config/shape/dtype failure checks, missing/truncated shards and missing MTP
  heads; allocation failure cleanup and sustained memory-growth checks.
- [ ] Inventory and native execution coverage for remaining upstream fused projection,
  GDN and row-forward optimization variants; distinguish selectable production paths
  and diagnostic variants in the final inventory.
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
