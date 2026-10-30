# Compact template builder handles

The revision restores the original 40-byte `Str` and 48-byte `Value` on the tested 64-bit target. It replaces the per-string builder pointer with a one-based `u32` handle in existing padding and a renderer-owned registry of buffer pointers. A handle is reused only when it is in bounds and both text and provenance slices still match the builder's current snapshot. Macro calls share the renderer; public outputs carry no handle. The retained-array logic protecting saved aliases is unchanged.

This removes the diagnosed value-size regression while retaining independent accumulators. The original [issue #1633](https://github.com/ashhart/TensorFold/issues/1633) production incident remains unreproduced; these are bounded synthetic host tests of the supplied template.

## Pinned method

- Original base: `f350cd3ea253a042e205746928271f9d8f71a82d`.
- Earlier pointer-based PR: `d33e5e769d7a4bb801cc39e0a32400488cb89777`.
- Revised implementation: `8d00b490f5ddf45721546e9b9816a083a46e5fa2`.
- Apple M5 Pro, 48 GiB RAM, macOS 27.0, Zig 0.17.0. Frozen fixtures use Python 3.11.16 and Jinja2 3.1.6.

The [runner](run.py) was committed at `2f6f3382f` before measurement. The [protocol](protocol.json) records the complete 13-case allocation corpus plus 21-case compatibility corpus, three revisions and Debug, ReleaseFast and ReleaseSafe: 306 fresh processes, one render each. It verifies the original template and corpus checksums and reuses the unchanged allocation diagnostic harness with every probe disabled. No instrumentation patches or timing observations are used here. The supplied template SHA-256 is `7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e`.

Each render has the same fixed 240 MiB combined compiled-template/render-scratch cap. Inputs, expected output and returned output are outside it. Scratch must be released on success or error. A cap refusal is an incomplete render, never a correctness pass; no larger-cap retries or replacements were run. These are live allocator requests, not RSS or physical memory.

## Results

All 306 processes ran. **297 completed renders matched all expected Jinja bytes across 176,407,221 output bytes**, with identical canonical provenance spans across revisions and modes wherever both completed. The eight compatibility cases with independent request-byte coverage supplied 72 exact provenance checks. All 102 revised and all 102 earlier-PR renders completed. The nine cap refusals were the three previously recorded large-argument baseline cases in all three modes.

The revised 2,001-message filler scratch is **145,623,620 bytes, exactly the original baseline**, versus 182,966,960 bytes for the earlier PR in ReleaseSafe. The measured `Value` size is again 48 bytes. Using the previously independently verified 2,003,001 copied list elements gives a calculated 96,144,048-byte payload, removing the extra 16,024,008 bytes. That payload is a calculation from the unchanged workload and measured layout, not a new instrumented counter observation.

ReleaseSafe is shown because it matches the CUDA server's template module. MiB means 1,048,576 bytes. Scratch excludes 891,842 bytes of standing compilation allocation for the exact template and 33,032 bytes for the separate synthetic control.

| Workload | Base MiB | Earlier PR MiB | Revised MiB | Revised vs base |
| --- | ---: | ---: | ---: | ---: |
| 2,001 filler messages, no tools | 138.88 | 174.49 | 138.88 | +0.00% |
| 157 large schemas, small dialogue | 5.27 | 6.35 | 5.45 | +3.34% |
| 8 cycles, 64-byte object arguments | 0.98 | 1.05 | 0.98 | -0.01% |
| 8 cycles, 8 KiB object arguments | 4.38 | 2.00 | 2.58 | -41.12% |
| 8 cycles, 8 KiB JSON-string arguments | 3.25 | 2.36 | 1.84 | -43.40% |
| 64 cycles, 64-byte object arguments | 6.82 | 7.62 | 6.82 | +0.00% |
| 64 cycles, 8 KiB object arguments | 117.87 | 17.53 | 15.73 | -86.66% |
| 64 cycles, 8 KiB JSON-string arguments | 144.73 | 16.22 | 14.46 | -90.01% |
| 128 cycles, 64-byte object arguments | 23.87 | 20.07 | 24.12 | +1.05% |
| 128 cycles, 8 KiB object arguments | Cap refusal | 45.82 | 39.89 | Not comparable |
| 128 cycles, 8 KiB JSON-string arguments | Cap refusal | 33.01 | 36.33 | Not comparable |
| 128 cycles, 8 KiB objects + large catalogue | Cap refusal | 51.92 | 47.93 | Not comparable |
| Separate synthetic interleaved control | 0.83 | 0.34 | 0.29 | -65.02% |

Argument sizes refer to the complete HF-serialized object JSON (64 or 8,192 UTF-8 bytes); payload text is 22 or 8,150 bytes. JSON-string arguments contain that same object. The catalogue contains 157 synthetic schemas totalling 245,760 compact UTF-8 bytes; it is not the reporter's missing schema set or a token-count claim.

The large-tool benefit remains: 64 cycles with 8 KiB object arguments use 16,491,748 bytes versus 123,597,630 on the base (-86.66%). Some other shapes still have higher peaks than base or the earlier PR. In particular, the large-catalogue short dialogue is +3.34% versus base and 128 small-argument cycles are +1.05%. The 128-cycle JSON-string case rises 10.04% versus the earlier PR (34,618,268 to 38,094,060 bytes), while the original base still refuses it under the cap. Registry storage, retained buffer growth and arena allocation boundaries remain distinct from `Value` payload size; no complete causal attribution of those peak differences is claimed. The change does not eliminate the template's quadratic list concatenations.

[Separate fixed host timing comparison](../value-layout-timing/README.md) reports uninstrumented ReleaseSafe renders. No GPU or served-speed conclusion follows from these host measurements.

## Evidence and checks

[Metadata](metadata.json), [all process records](results.json), [all three-way comparisons](summary.json), and [independent verification](verification.json) retain source/binary identities, output/provenance identities and every refused allocation's live/request totals. Public JSON preserves parsed measurements; raw logs remain local because compiler and stack diagnostics include local paths.

[Checks](checks.json): 37 focused template tests pass in each of Debug, ReleaseSafe and ReleaseFast. The integrated host run completes 299/299 build steps, with 1,099 tests passed and three Linux-only voice-tool tests skipped on macOS; other groups are cached. Wire goldens have 322/335 equality, 13 known differences and zero unexpected differences. Formatting and diff checks pass; lean remains 426 findings on base and branch, zero new.

Plain `zig build -j2` was attempted and fails because the Xcode Metal compiler component is unavailable. No GPU kernel changed. No new Spark/model run was performed; earlier model/decode receipts qualify their explicitly named earlier implementation and are not relabelled as measurements of this revision. The original huge multi-rank request, patched image, missing schema set and private tool-heavy history remain unavailable. The PR stays draft pending outstanding qualification and original reproduction evidence.

## Reproduce

Use the unchanged corpus generators and external template documented in the earlier [allocation diagnostic](../allocation-diagnostic/README.md) and [compatibility receipt](../deepseek-template/README.md), then run:

```sh
python3 docs/receipts/template-accumulation/value-layout-followup/run.py \
  --template /tmp/deepseek-template.jinja \
  --allocation-corpus /tmp/allocation-corpus.json \
  --compatibility-corpus /tmp/compatibility-corpus.json \
  --candidate 8d00b490f5ddf45721546e9b9816a083a46e5fa2 \
  --output /tmp/value-layout-results --zig zig
```

The independent paginated overlap audit covers 112 open PRs, 5,003 file entries and all 82 publication paths. A second snapshot confirms stable heads, bases and inventory, with no other PR touching these paths; no touching PR requires a merge-tree check. ATC's `pullFilesLimit` remains explicit rather than claiming complete global coverage from ATC alone.
