# Compact-handle CUDA qualification

The candidate-only drafted/plain gate passes: **18 complete cold responses, nine exact pairs and 4,608 token IDs**, plus six generation/history tokenization snapshots. **All 12 fixed decode cells meet the ±2% practical-equivalence gate.** TTFT is descriptive.

This compares original base `f350cd3ea253a042e205746928271f9d8f71a82d` with compact-handle implementation `8d00b490f5ddf45721546e9b9816a083a46e5fa2` on DGX Spark GB10, using pinned Qwen3.8-Flash-Next-FP4-Local assets. Earlier receipts retain their original revision labels.

## Build and correctness gates

The current candidate completed 218/218 CUDA build steps and a separate 112/112-step build using the retained baseline kernels. The baseline installation was reused unchanged. All 103 GPU assets match between the measured installations. Four independently compiled CUDA fatbins differed from the baseline, so the candidate qualification build embeds the baseline assets. [Environment](environment.json) and window captures pin source, executables, checkpoint metadata, image, driver and helpers.

The actual Qwen template matches Jinja2 3.1.6 bytes for six cases on each revision, covering generation on/off; this test does not check provenance. [Host checks](../value-layout-followup/README.md) record 37 focused tests in each of three modes, 299/299 build steps, 1,099 passed tests and three platform skips. Lean remains 426 on base and branch, zero new.

P0 covers prose, code as chat and a 34-message conversation: greedy seed 1234 and temperature-one seeds 1234/1235 (top-k 20, top-p 0.95), each drafted/plain with 256 reply tokens. All nine pairs match complete token arrays and independently recomputed token/output hashes; all cache counts are zero. The first cold request counts once within the 18. Restoration succeeded without cleanup errors.

## Fixed decode method

[protocol.json](protocol.json) freezes six paired blocks **AB, BA, BA, AB, AB, BA**, with fresh servers. The unchanged decode client covers three fixtures × temperatures zero/one × one/two streams: 12 cells, two repetitions, 256 reply tokens. Each run has 36 measured responses, one excluded warmup and three excluded solo references. Full decode totals: 480 responses, comprising 432 measured and 48 excluded.

Aggregate decode is `streams × 255 / (latest last chunk − earliest first chunk)`. Each cell uses six paired log ratios of candidate/base per-run medians, with a two-sided 95% t interval (`df=5`, critical value `2.570581835636305`), then exponentiates endpoints. All 12 intervals must lie wholly within `[0.98, 1.02]`. Requests/repetitions are not independent samples. No interim intervals, replacement pairs or result-selected extra runs are permitted.

Validation checks complete cold streams, hashes, timestamps, tokenizations and repeat/solo/concurrent/build equality, including every candidate decode response against P0 plain. Full qualification requires 30 window identity captures and successful restoration. Settings: context 8,192, two lanes, prompt cache zero, 9 GiB reserve and 6 GiB guard.

## Decode results

Throughput columns are arithmetic means of six per-run medians. The paired geometric change and confidence interval come unchanged from the frozen comparator and determine the gate.

| Fixture | Temperature | Streams | Base mean tok/s | Candidate mean tok/s | Paired geometric change | 95% paired interval | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 34-message conversation | 0 | 1 | 61.16 | 61.02 | -0.233% | [-0.559%, +0.095%] | equivalent |
| 34-message conversation | 0 | 2 | 85.04 | 84.94 | -0.111% | [-0.584%, +0.364%] | equivalent |
| 34-message conversation | 1 | 1 | 54.68 | 54.61 | -0.131% | [-0.376%, +0.115%] | equivalent |
| 34-message conversation | 1 | 2 | 54.78 | 54.79 | +0.012% | [-0.203%, +0.228%] | equivalent |
| Prose | 0 | 1 | 56.17 | 56.00 | -0.309% | [-1.105%, +0.494%] | equivalent |
| Prose | 0 | 2 | 102.77 | 103.14 | +0.360% | [-0.589%, +1.317%] | equivalent |
| Prose | 1 | 1 | 47.72 | 47.76 | +0.073% | [-0.520%, +0.669%] | equivalent |
| Prose | 1 | 2 | 62.90 | 62.87 | -0.050% | [-0.567%, +0.470%] | equivalent |
| Code as chat | 0 | 1 | 63.23 | 63.00 | -0.358% | [-1.175%, +0.466%] | equivalent |
| Code as chat | 0 | 2 | 111.08 | 110.87 | -0.181% | [-0.821%, +0.464%] | equivalent |
| Code as chat | 1 | 1 | 62.34 | 62.25 | -0.142% | [-0.659%, +0.377%] | equivalent |
| Code as chat | 1 | 2 | 81.51 | 81.40 | -0.141% | [-0.713%, +0.434%] | equivalent |

## TTFT: descriptive results

Measured cells show arithmetic means of six per-run median request TTFTs. The last row separately averages six first-cold excluded warmups per side. TTFT has no equivalence gate.

| Fixture | Temperature | Streams | Base mean ms | Candidate mean ms | Change ms | Change % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 34-message conversation | 0 | 1 | 511.793 | 511.248 | -0.545 | -0.11% |
| 34-message conversation | 0 | 2 | 797.017 | 794.532 | -2.485 | -0.31% |
| 34-message conversation | 1 | 1 | 514.206 | 510.457 | -3.749 | -0.73% |
| 34-message conversation | 1 | 2 | 801.953 | 799.136 | -2.817 | -0.35% |
| Prose | 0 | 1 | 118.997 | 121.030 | +2.033 | +1.71% |
| Prose | 0 | 2 | 180.663 | 179.668 | -0.995 | -0.55% |
| Prose | 1 | 1 | 122.837 | 123.505 | +0.668 | +0.54% |
| Prose | 1 | 2 | 181.404 | 182.307 | +0.903 | +0.50% |
| Code as chat | 0 | 1 | 113.333 | 111.833 | -1.500 | -1.32% |
| Code as chat | 0 | 2 | 169.230 | 167.928 | -1.301 | -0.77% |
| Code as chat | 1 | 1 | 112.179 | 112.602 | +0.423 | +0.38% |
| Code as chat | 1 | 2 | 166.586 | 168.498 | +1.911 | +1.15% |
| First-cold excluded warmup | 0 | 1 | 572.219 | 596.913 | +24.693 | +4.32% |

## Aborted client attempt and limits

An earlier P0 attempt stopped after one valid drafted response: zero completed pairs and zero decode observations. Its validator required lists while the shared streaming client emits tuples. The fix accepts both with unchanged shape/type/arithmetic checks. An actual shared-client synthetic SSE regression fails the original predicate and passes the fix and JSON roundtrip. Production was restored; the aborted response is retained and excluded entirely.

The completed corrected experiment contains **498 responses** (18 parity plus 480 decode), or **499 lifetime responses** including the isolated aborted request. The decode sample contains 122,880 token IDs; the corrected experiment contains 127,488. The aborted response is excluded.

Qwen uses namespace counters without the growing accumulator motivating this change; these checkpoint-specific gates do not measure accumulator speedup. Intervals assume independent, approximately normal block log ratios; six blocks give limited precision. Wide intervals are inconclusive; outside-band results do not automatically prove regression. Cached resume, long-prompt/last-release speed and Metal device gates remain unrun. The local Metal build remains blocked by its missing Xcode compiler component; this receipt verifies the Linux CUDA build. The reporter's missing schemas, private history and patched multi-rank image remain outside this qualification.

## Evidence and reproduction

The corrected protocol and helpers were frozen at `e635dacd9`; the current build identity manifest was committed at `3a5ab15a6`. [Summary](summary.json), [parity receipt](parity/P0.json), [decode receipts](decode/) and [window metadata](windows/) retain the complete fixed sample. Published run receipts, outcomes and identity captures preserve every parsed value. Window settings omit private paths and container identifiers; the publication manifest records retained and omitted keys. All 30 identity paths resolve to two canonical phase-specific captures, preserving the intentional parity/decode client difference. The environment manifest retains the exact controller-pinned bytes.

[Independent verification](independent-verification.json) reconstructed all response hashes, per-run rates and confidence intervals directly from raw timings, including a 50-digit Decimal calculation. All twelve statuses agree. A separate [presentation audit](presentation-verification.json) checked every displayed mean and percentage. These reports fingerprint the original raw JSON and private presentation draft; [publication.json](publication.json) maps raw to published byte hashes, with parsed values preserved. [Final restoration](final-restoration.json) confirms production healthy, no owned test workload and the unchanged 4 GiB kernel reserve.

```sh
python3 docs/receipts/template-accumulation/compact-cuda/compare.py \
  docs/receipts/template-accumulation/compact-cuda/decode \
  --parity docs/receipts/template-accumulation/compact-cuda/parity/P0.json \
  --identity docs/receipts/template-accumulation/compact-cuda/environment.json \
  --windows docs/receipts/template-accumulation/compact-cuda/windows \
  --output /tmp/compact-cuda-summary.json
```

## Coordination

ATC check was complete with advice `yours` and PR #1652 as the only linked implementation. Official overlap remains incomplete with `pullFilesLimit`. A separate paginated audit at `2026-10-30T15:14:52.694814+00:00` covered all 110 open PRs, 4,998 changed-file entries and all 153 branch paths. A second inventory confirmed stable PR heads, bases and local targets. No other PR touched these paths, so no touching PR required a merge-tree check.
