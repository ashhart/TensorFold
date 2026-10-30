# Independent template accumulators

For [#1633](https://github.com/ashhart/TensorFold/issues/1633), interleaved large string concatenations no longer repeatedly copy the growing prompt. Saved strings also keep their bytes and request provenance when an accumulator grows or branches.

Base: `f350cd3ea253a042e205746928271f9d8f71a82d`. Candidate implementation and regression tests: `78b0a0399545c680d2af887980e8f2ccfd2a24f8`. The memory and correctness measurements below use synthetic host inputs on Apple M5 Pro, 48 GiB RAM, macOS 27.0, Zig 0.17.0, Python 3.11.16 and Jinja2 3.1.6. Those measurements involve no checkpoint, inference backend, rank group or GPU workload.

## Memory and correctness

The public renderer accumulates 1,024-byte messages plus literal separators while interleaving a separate 4,096-byte concatenation. The allocator tracks peak live render-scratch allocation, excluding the compiled template's standing allocation, input and returned result. It refuses allocations above 240 MiB. These are allocator bytes, not process RSS or GPU memory.

| Messages | Base peak bytes | Candidate peak bytes | Change |
| --- | ---: | ---: | ---: |
| 128 | 17,146,558 | 3,556,664 | −79.3% |
| 256 | 74,847,780 | 5,647,106 | −92.5% |
| 512 | 225,995,904 | 11,949,772 | −94.7% |

The [base scaling test](baseline-scaling.txt) fails its linear memory bound after checking every rendered byte and request/literal provenance offset. The [expanded base run](baseline-regressions.txt) has three failures: interleaved allocation growth, corruption of saved string bytes, and the bound for two growing chains with a grouped long right operand. Its other 29 tests pass. The final [candidate template suite](template-tests.txt) passes all 35 tests, including both doubling checks, saved aliases, branching, self-concatenation, separate text/span relocations, and identical 4,096-byte request/literal fragments with distinct provenance.

An independent Jinja2 oracle covers nine families at four sizes, including macros that mutate the same namespace while an outer expression holds its earlier left operand. The [base](baseline-parity.json) matches 29/36 cases; the [candidate](parity-summary.json) matches all 36 across 2,530,055 output bytes in each of two runs. The candidate runs have identical corpus and result bytes. A separate [ReleaseFast run](releasefast-parity.json) also matches all 36 cases and produces byte-identical results to Debug. Seven baseline mismatches are saved-branch, self-concatenation and macro-mutation cases. This fixes existing corruption; it does not claim equality to corrupted baseline output.

## Host rendering speed

A separate [ReleaseFast benchmark](host-speed/README.md) alternates five ABBA/BAAB blocks and validates output bytes and provenance on every render. The median of ten per-run medians falls from 0.949 ms to 0.196 ms at 128 interleaved messages (−79.3%) and from 3.597 ms to 0.474 ms at 256 messages (−86.8%). The short-chat control is effectively unchanged: 8.35 µs versus 8.38 µs. The base exceeds the 240 MiB scratch cap at 512 messages in ReleaseFast, so that cell has no paired speed comparison. These are host-render times, not model throughput.

## CUDA model gate

The [Qwen chat model receipt](model/README.md) records an alternating baseline/PR/baseline/PR run on one DGX Spark GB10 with identical GPU assets, an 8,192-token context and two lanes. All 292 responses match exactly across drafted/plain, solo/concurrent, repeated and cross-build checks; all reported cached-token counts are zero. Decode changes are +0.01% to +0.80%, and fresh-server first-request TTFT is 575.50 ms versus 575.82 ms (+0.06%). These ordinary-template measurements establish no large observed regression, not a meaningful model-speed gain. The targeted host-render improvement is measured separately above.

A subsequent [fixed decode-equivalence test](decode-equivalence/README.md) uses six matched fresh-server pairs and 256-token replies. All twelve per-cell 95% intervals lie wholly inside the prespecified ±2% tolerance; their endpoints range from −1.680% to +1.073%. All 480 responses match exactly across builds and concurrency. This establishes practical equivalence for the tested checkpoint/settings, rather than inferring it from small point estimates.

## Exact reported DeepSeek template

A later [exact-template follow-up](deepseek-template/README.md) passes all 168 processes and 840 full-output comparisons, including 2,001 seeded filler messages. Both revisions fit the 240 MiB cap, so this shape does not reproduce the reported allocation failure. At 2,001 messages without tools, candidate render scratch is 15.2% higher and the descriptive single-ABBA render time is 10.3% longer. The original tool schemas and private tool-heavy conversation were unavailable. The synthetic string-accumulation gains above remain separate evidence; the PR stays draft without claiming that this follow-up proves resolution of the production incident.

## Design choices required for exactness

Only concatenation results carry an arena-owned builder. An append can reuse that builder only when both the text and provenance slices identify its current complete value. Older snapshots and derived strings start a new builder, so evaluating the right operand cannot silently change the already-evaluated left operand.

Text and provenance arrays grow geometrically by allocating and copying, without freeing or remapping their earlier arrays. Arena ownership alone is insufficient: `Allocator.free` can poison an old slice even if the arena retains its physical storage. Appends preserve existing text and span entries, including self-appends. Ordinary buffers do not attach their stack addresses to strings, and public render results contain only caller-owned text and spans.

## Reproduction

From the repository root, with `zig` at 0.17.0 and Python with Jinja2 3.1.6:

```sh
zig test --dep sys -Mroot=zig/src/core/template/template.zig -Msys=zig/src/platform/sys.zig -lc
python3 docs/receipts/template-accumulation/verify_parity.py --zig zig --out /tmp/template-parity
zig build test-server-deepseek-chat-host -j2 --summary all
zig build test-cuda-host test-server-cpu test-kernel-contract test-golden fuzz-cortheon -j2 --test-timeout 2m --summary all
zig fmt --check zig/src/core/template/{eval,provenance,template,accumulation_test}.zig
python3 tools/zig/lean_check.py
```

The standalone scaling test can run against the base evaluator without the new retained-buffer tests. `baseline-regressions.txt` records that failing stage before the three additional relocation/provenance tests were added. The portable parity verifier writes its synthetic corpus, results, versions and source hashes; it returns nonzero on any byte mismatch.

## Integrated checks and limits

- [Host checks](host-checks.txt): 299/299 build steps succeeded, 1,519 tests passed and three skipped in the final aggregate run. Other test groups were cached; these are executed counts, not a sum across attempts. All five requested targets succeeded.
- Server CPU tests: 868 passed and three Linux-only voice-tool tests skipped on macOS, including request-field provenance and 178 committed DeepSeek prompt goldens. A separate focused DeepSeek run passed 20/20 tests. Those goldens primarily exercise the native fallback renderer; the independent Jinja corpus exercises the changed evaluator.
- Server wire goldens: 322/335 equal with 13 known differences and zero unexpected differences; security goldens 55/55 equal. Cortheon fuzz: 16/16 passed.
- Formatting and `git diff --check` passed. Lean: 426 findings on base and branch, zero new; inherited findings are not called clean.
- The initial sandboxed aggregate run could not bind loopback test servers or query macOS memory pressure. Two waiting server-test controllers were stopped, then the successful aggregate ran with those accesses available. That interrupted attempt is not a pass.
- Plain `zig build -j2 --summary all` was attempted but failed because the installed Xcode lacks the Metal compiler component. `xcrun -sdk macosx metal -v` confirms the same error outside the sandbox. The default build and Metal-dependent tools are not qualified by the host checks.
- Full macOS `zig build test` was not run because it includes GPU device checks. The bounded CUDA Qwen gate above covers model token parity, served decode and fresh-server TTFT. Resumed-cache equality, the reported multi-rank million-token request, other checkpoints and Metal model/device gates were not run. No GPU kernel changed, so a new device-contract `SUMMARY` is not applicable.

## Coordination

ATC check was complete and free before claiming #1633. ATC overlap reported `pullFilesLimit`: its fallback covers at most 300 files and open PR #1446 contained 301. An independent paginated audit at `2026-10-30T07:44:45Z` covered all 96 open PRs and 4,618 changed-file entries, with a second inventory confirming every head and base. None touched the four changed source/test files. ATC also reported conflicts involving the separate sampled-heads PR #1609; those are not changes on this branch. No touching PR required a merge-tree check.

Before this receipt update, ATC check was complete with advice `yours` and PR #1652 as the only linked implementation. ATC overlap still reported `pullFilesLimit`; its global coverage is not claimed complete. A fresh independent paginated audit covered 100 open PRs and 4,691 changed-file entries, checking all 30 branch and receipt paths and excluding this PR. A second inventory confirmed stable PR heads and bases, and the local target list stayed unchanged. No other PR touched these paths, so no touching PR required a merge-tree check.

The decode-equivalence receipt update refreshed the independent audit at `2026-10-30T10:29:01Z`: 101 open PRs, 4,709 changed-file entries and all 49 source/receipt paths, with stable second-snapshot heads and bases. No other PR touched these paths. Official ATC overlap still reported `pullFilesLimit`; its global coverage is not claimed complete.
