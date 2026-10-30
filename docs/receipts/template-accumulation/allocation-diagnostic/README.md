# Allocation diagnostic for the reported DeepSeek template

This follows the [exact-template compatibility test](../deepseek-template/README.md) for [issue #1633](https://github.com/ashhart/TensorFold/issues/1633). That earlier fixed test did not reproduce the reported 66 GiB mapping and found higher candidate scratch on the 2,001-message filler case. This diagnostic separates evaluator string-join traffic from list-concatenation traffic and exercises repeated tool calls with larger arguments. It does not change production source.

## Frozen method

Base `f350cd3ea253a042e205746928271f9d8f71a82d`; candidate `9dac03d6be8871d2d4c9404e53e9d44c07d7cd6a`. Production implementation is unchanged from the earlier receipts. Apple M5 Pro, 48 GiB RAM, macOS 27.0, Zig 0.17.0, Python 3.11.16, Jinja2 3.1.6. Debug, ReleaseFast and ReleaseSafe are included. ReleaseSafe matches the CUDA server's template-module compilation setting; these remain standalone host renderer tests. No timing comparison is made from instrumented runs.

The supplied 11,410-byte template is unchanged, with SHA-256 `7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e`. [fixtures.py](fixtures.py) freezes 12 exact-template cases: the earlier 2,001-message filler control; a small dialogue with 157 synthetic schemas totalling 245,760 serialized UTF-8 bytes; two-tool-call cycles at 8, 64 and 128 cycles with 64-byte serialized object arguments, 8,192-byte serialized object arguments, or JSON-string arguments containing the same serialized object; and the 128-cycle large-object case with the large catalogue. A thirteenth, separately labelled synthetic template interleaves independent large string accumulations as a positive diagnostic control. Argument sizes refer to the complete HF-serialized JSON object (64 or 8,192 UTF-8 bytes); its payload text is 22 or 8,150 bytes. The string variant contains that same serialized object. The catalogue is a byte-size proxy, not a claim of 60,000 tokens or equality to the reporter's missing schemas.

Full expected bytes come from the immutable Jinja2 sandbox with Hugging Face JSON serialization conventions. Each revision/mode runs one unchanged control process and one instrumented process per case: 156 planned processes in total. Every completed render must equal the full Jinja output and have ordered, nonempty, disjoint, in-bounds provenance spans. Canonical span hashes compare all completed versions. This diagnostic compares provenance structure and cross-variant identity; it does not add an independent byte-coverage oracle. The earlier exact-template receipt retains its separate eight-case provenance oracle.

Both variants use the same 240 MiB cap on combined compiled-template and render-scratch live allocations. Input JSON, expected output and returned output are outside the cap. Scratch is freed on success and failure. Refusals remain explicit, with no larger-cap retry; unfinished cases have partial counters and cannot claim full-output correctness.

## What the counters measure

[instrument.py](instrument.py) inserts probes into disposable archives of the pinned source, asserting unique source anchors. It wraps only the allocator used by evaluator string joins and the payload allocator for list/tuple `+`. [diagnostics.zig](diagnostics.zig) uses fixed global counters; logging happens after rendering, and the probe adds no render allocation. The unchanged control has zero probe counters. Each control/instrumented pair must agree on completion status, output/provenance identities and measured upstream peak.

- Allocation requests, successes, failures and successful requested bytes describe logical traffic through those two sites. Resize/remap growth is recorded separately. These are not physical memory, arena-slab sizes or a complete account of rendering allocation.
- List payload copies count both operands' elements after concatenation succeeds. They exclude the list wrapper, slices, dictionaries, frames, request conversion and other allocation sites. Payload bytes equal completed copied elements times the measured size of `Value`.
- String prefix bytes count explicit left-hand-side seeding of a fresh join buffer. They exclude relocation copies inside buffer growth. Attempted and completed operations are distinguished, particularly on cap refusal. The string allocator category also includes join-local builder and provenance storage.
- Upstream peak is tracked separately at the compiled template's child allocator, including all render-scratch arena allocations. It must not be added to logical payload traffic or interpreted as RSS.

## Results

**Large tool arguments expose repeated string-prefix copying on the baseline, while the plain filler exposes a separate value-size cost in the candidate.** All 156 planned processes ran: 138 completed with full Jinja byte equality across 112,912,140 output bytes and 18 refused an allocation at the fixed cap. Every refusal belongs to the baseline in the three 128-cycle large-argument cases, across all three modes and both control/instrumented builds. The candidate completed every case. A refusal is not a completed correctness check.

All 78 control/instrumented pairs agree on completion status, compiled standing allocation and observed upstream scratch peak. The 69 completed pairs also agree on canonical provenance hashes; the nine refused pairs have no final output to compare. The full completed-results set agrees on spans across revisions and modes wherever both exist. This establishes no observed allocation effect from these probes in these runs, not a general claim that instrumentation has no effect.

ReleaseSafe is shown below because it matches the CUDA template module's compilation mode. The values are measured live scratch requested from the child allocator, excluding 891,842 bytes of standing compilation allocation for the exact template. The separate synthetic template's standing allocation is 33,032 bytes. MiB means 1,048,576 bytes.

| Workload | Base scratch MiB | PR scratch MiB | Change |
| --- | ---: | ---: | ---: |
| 2,001 filler messages, no tools | 138.88 | 174.49 | +25.64% |
| 157 large schemas, small dialogue | 5.27 | 6.35 | +20.46% |
| 8 cycles, 64-byte serialized object arguments | 0.98 | 1.05 | +7.32% |
| 8 cycles, 8 KiB serialized object arguments | 4.38 | 2.00 | -54.29% |
| 8 cycles, 8 KiB JSON-string arguments | 3.25 | 2.36 | -27.40% |
| 64 cycles, 64-byte serialized object arguments | 6.82 | 7.62 | +11.79% |
| 64 cycles, 8 KiB serialized object arguments | 117.87 | 17.53 | -85.13% |
| 64 cycles, 8 KiB JSON-string arguments | 144.73 | 16.22 | -88.80% |
| 128 cycles, 64-byte serialized object arguments | 23.87 | 20.07 | -15.90% |
| 128 cycles, 8 KiB serialized object arguments | Cap refusal | 45.82 | Not comparable |
| 128 cycles, 8 KiB JSON-string arguments | Cap refusal | 33.01 | Not comparable |
| 128 cycles, 8 KiB serialized object arguments + large catalogue | Cap refusal | 51.92 | Not comparable |
| Separate synthetic interleaved control | 0.83 | 0.34 | -59.25% |

A cap refusal means the next arena allocation would exceed the combined 240 MiB limit; it does not mean 240 MiB was consumed. For example, the ReleaseFast 128-cycle object case has 163,644,410 live bytes and refuses a 90,948,990-byte allocation because the resulting 254,593,400 bytes would exceed 251,658,240. No larger-cap retries were run. Partial counters are preserved without treating them as complete workload totals.

### String interference is input-dependent

For 64 cycles with 8 KiB serialized object arguments, both revisions execute 5,895 string joins. The candidate reuses a builder 127 more times. Completed explicit left-prefix seeding falls from 71,478,600 to 1,202,504 bytes, and ReleaseSafe scratch falls from 123,597,630 to 18,384,054 bytes. These are different measurements: prefix seeding excludes buffer relocation and is not live memory.

This demonstrates the mechanism inside the exact supplied template with controlled large tool arguments. The small-argument and large-catalogue-only cases do not demonstrate the same interference. It does not reproduce the original private conversation or the reported 66 GiB mapping, and no served speedup is claimed.

### Larger values increase list-copy traffic

Measured `Str` size increases from 40 to 48 bytes, and `Value` from 48 to 56 bytes. The 2,001-message filler performs exactly 2,001 list concatenations, copying 2,003,001 `Value` elements on both revisions. That payload rises from 96,144,048 to 112,168,056 bytes, an exact 16,024,008-byte increase. The list-count formula and element size independently reconcile the measured payload allocator totals.

The filler has identical string-join counts, including 5,993 reuses, and only 19,995 explicit prefix-seeding bytes on each revision. It does not exercise the interference that the tool cases expose. Its ReleaseSafe scratch rises from 145,623,620 to 182,966,960 bytes; the proven extra list payload does not fully attribute this peak difference. Larger values also affect other structures, and arena growth and buffer strategies affect peaks. The earlier ReleaseFast timings remain a separate, unchanged receipt.

The PR remains draft. This evidence supports the string-copy fix for some request shapes while establishing a regression that needs attention before landing. The original issue remains open pending the reporter's complete synthetic request and exact patched build details.

## Evidence

[results.json](results.json) retains all 156 process records, full and partial logical counters, measured layouts and output/span identities. [control-comparisons.json](control-comparisons.json) retains all 78 observer comparisons. [metadata.json](metadata.json) defines the counter order and pins the executable and source identities. [fixtures-summary.json](fixtures-summary.json) records every case's sizes and list-copy expectations. [verification.json](verification.json) independently reconciles the fixture, logs, counters, cap requests and arithmetic. Public JSON is compacted without changing parsed measurement values, with original artifact hashes retained.

## Served-path limits

`HfText.load` compiles with the process allocator. `renderFn` converts request data/options, then calls `renderProvenanceGlobals`, whose fresh scratch arena uses the compiled template's child allocator and is released on success or error. Output text and spans are copied onto the caller's allocator. A request can invoke multiple prompt/history/probe renders; request-side JSON, output and token arrays have different lifetimes, and tokenizer memo storage is separate.

A nonempty compiled checkpoint template is selected before the native DeepSeek fallback; the `native_deepseek` flag alone does not bypass Jinja. The base and candidate share this routing. The reporter's evaluator stack is consistent with that compiled-template path. These tests do not reproduce HTTP concurrency, the original patched image, tokenizer/checkpoint globals, process allocator retention, physical footprint or GPU inference. Logical counters cannot identify the source of the original 66 GiB mapping without the missing reproduction evidence.

## Reproduction

From the repository root with the pinned tools and the separately downloaded template from the earlier receipt:

```sh
python3 docs/receipts/template-accumulation/allocation-diagnostic/fixtures.py \
  --template /tmp/deepseek-template.jinja --output /tmp/allocation-corpus.json \
  --summary /tmp/allocation-fixtures.json
python3 docs/receipts/template-accumulation/allocation-diagnostic/run.py \
  --template /tmp/deepseek-template.jinja --corpus /tmp/allocation-corpus.json \
  --output /tmp/allocation-results --zig zig
```

The runner refuses a different template, corpus or Zig version. Its metadata records source revisions and all diagnostic source, fixture and executable hashes. The complete generated corpus and original logs remain local; parsed measurements and independent verification are retained in this receipt.

## Publication checks

Formatting and diff checks pass. Lean remains 426 findings on the base and 426 on the branch, with zero new findings. Independent method review and reconstruction of all fixtures, raw results and counter arithmetic pass. No production source changed in this follow-up, and no Spark or GPU operation was performed.

The independent paginated overlap audit covers 108 open PRs, 4,905 file entries and all 68 branch/publication paths. A second snapshot confirms unchanged heads, bases and PR inventory. No other PR touches these paths, so no touching PR requires a merge-tree check. ATC's own `pullFilesLimit` remains explicit; its global coverage is not claimed complete.
