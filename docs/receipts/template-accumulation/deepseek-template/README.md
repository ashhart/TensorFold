# Exact DeepSeek template follow-up

The reporter supplied the [exact template on issue #1633](https://github.com/ashhart/TensorFold/issues/1633#issuecomment-6521146049). Its SHA-256 is `7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e`: 11,410 bytes and 222 lines. The reporter identifies `deepseek-ai/DeepSeek-V4.1-Flash-1017` revision `1bae0e1088993b9b82de08f59de5be5a34e42170` as the base checkpoint. This test uses the supplied template, without downloading model weights or running inference. The template is downloaded separately and is not vendored in this contribution.

## Protocol

Base `f350cd3ea253a042e205746928271f9d8f71a82d`; candidate `f1c7c6d3a6632daed5f26c3d2d240f8d56926e1b`. Candidate production source is unchanged from the earlier measured `a8dd856068836faaf35ff46da1d63f1e103b0e7d`. Apple M5 Pro, 48 GiB RAM, macOS 27.0, Zig 0.17.0, Python 3.11.16 and Jinja2 3.1.6.

[fixtures.py](fixtures.py) reconstructs the issue's public seed-424242 filler generator: one system message followed by 2,000 alternating user/assistant messages, each containing 234 sampled NATO words. Prefixes contain 9, 129, 257, 513, 1,001 and 2,001 messages. The original 157 tool schemas were not supplied; a separately labelled case uses 157 synthetic schemas. Additional cases exercise thinking, generation headers, object/string tool arguments, reversed tool-result arrival, mid-conversation systems, reminders, structured JSON and literal/request marker collisions. These fixtures do not contain the private original conversation and do not establish its token count.

The unchanged template renders in Jinja2's immutable sandbox with Hugging Face's `tojson` convention: `json.dumps`, Unicode preserved, insertion-order keys and standard separators, without HTML escaping. Every successful native render must equal the full oracle bytes. Eight cases additionally supply independently constructed request-byte coverage; all cases check span structure and compare canonical span hashes across repetitions and successful builds/modes. Structural checks alone are not a complete independent provenance oracle for the remaining cases.

[run.py](run.py) compiles both pinned revisions in Debug and ReleaseFast, then runs a fixed baseline/candidate/candidate/baseline block per case and mode. Each entry is a fresh process with two warmups and three timed renders. Timing covers rendering, including scratch allocation and release; it excludes compilation, fixture parsing, oracle generation and validation. Reported ReleaseFast times are medians of the two process medians per side. They are descriptive host-render timings, not served TTFT, decode rates or confidence intervals.

The allocator refuses combined compiled-template and live render-scratch allocations above 240 MiB. Input JSON, expected text and returned output are outside that cap. The standing compile allocation and peak incremental scratch are recorded separately; these are requested live allocator bytes, not physical process footprint or GPU memory. Refusals are retained as failures to complete within the fixed cap, with no retry at a larger cap.

## Results

**Correctness passes; the reported allocation failure is not reproduced, and this exact-template workload shows no speedup.** Both revisions complete all 21 cases in both modes and both process repetitions: 168/168 processes, 840 complete byte comparisons, 320 independent request-byte coverage checks and 361,968,320 compared output bytes. All canonical span hashes agree across modes, revisions and repetitions. There are no cap refusals, leaks, panics, mismatches or timeouts.

The ReleaseFast table reports descriptive medians of two process medians per side and maximum observed render-scratch bytes per cell. Positive time/memory changes are worse. This single ABBA block does not establish statistical performance equivalence or a general regression estimate.

| Messages and tools | Base ms | PR ms | Time change | Base scratch bytes | PR scratch bytes | Scratch change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9, no tools | 0.056 | 0.054 | -3.59% | 190,410 | 213,008 | +11.87% |
| 129, no tools | 0.338 | 0.335 | -1.04% | 3,062,150 | 3,975,608 | +29.83% |
| 257, no tools | 0.745 | 0.745 | -0.01% | 8,953,478 | 7,282,810 | -18.66% |
| 513, no tools | 1.689 | 1.776 | +5.17% | 17,867,916 | 19,535,832 | +9.33% |
| 1,001, no tools | 4.736 | 4.897 | +3.39% | 38,300,330 | 49,615,532 | +29.54% |
| 2,001, no tools | 12.461 | 13.740 | +10.27% | 152,630,868 | 175,842,300 | +15.21% |
| 2,001 + 157 synthetic tools | 13.332 | 14.083 | +5.63% | 150,622,056 | 173,306,018 | +15.06% |

Standing compiled-template allocation is 891,842 bytes on both revisions. Maximum combined tracked live allocation across all runs is 176,734,142 bytes, below the fixed 251,658,240-byte cap. At 2,001 messages without tools, scratch is 145.56 MiB versus 167.70 MiB. This is not a replication of the reporter's 66 GiB mapping.

[results.json](results.json) preserves every parsed process result and all five render samples, including 336 warmups and 504 timed renders. [metadata.json](metadata.json) pins template, corpus, source revisions, runner, harness and executable identities. [fixtures-summary.json](fixtures-summary.json) defines the corpus; [verification.json](verification.json) records independent reconstruction of the issue generator, Jinja output, eight provenance oracles, raw logs and all table calculations. JSON is compacted for publication without changing parsed values; original artifact hashes are retained.

The scaled filler shape has one persistent growing string. Its message-body joins use local buffers and do not displace the baseline's global string accumulator. Code inspection therefore explains why the baseline already handles this shape efficiently. Repeated message-list concatenation remains quadratic; adding a builder pointer to string values can increase the storage copied in those lists. That is a plausible contributor to the higher measured scratch, not an isolated causal measurement. Arena growth and retained arrays also affect peaks.

The earlier synthetic interleaved-string and saved-alias regressions remain valid, but this new evidence does not establish that the PR resolves the reporter's production incident. The PR remains draft and no longer requests automatic closure of #1633. The missing original tool schemas and tool-heavy message structure matter to reproducing that incident. The reporter also ran a locally patched image, whose exact code and request globals are unavailable. The tested baseline template directory matches the reporter’s cited upstream `3f0a715`, which does not identify that patched running image.

## Checks and coordination

Independent review accepted the bounded allocator, failure classification, byte oracle, provenance scope and timing method. Eight mocked negative controls reject cleanup, leak, panic, mismatch and invalid-exit cases instead of calling them expected cap refusals. Zig formatting and diff checks pass; lean remains 426 base / 426 branch, zero new. Production source is unchanged, so the earlier broad host and model receipts were not rerun.

ATC ownership was complete and `yours`. Official overlap remains incomplete with `pullFilesLimit`. A separate fully paginated audit at `2026-10-30T12:01:09Z` covered 106 open PRs, 4,886 file entries and all 57 publication paths. A changed PR head was refreshed; the final second snapshot was stable. No other PR touched these paths, so no overlapping PR required a merge-tree check.

## Reproduction

With the pinned Zig and Python/Jinja2 versions, from the repository root:

```sh
curl -fL https://github.com/user-attachments/files/34178715/chat_template.jinja.txt -o /tmp/deepseek-template.jinja
python3 docs/receipts/template-accumulation/deepseek-template/fixtures.py \
  --template /tmp/deepseek-template.jinja --output /tmp/deepseek-corpus.json \
  --summary /tmp/deepseek-fixtures.json
python3 docs/receipts/template-accumulation/deepseek-template/run.py \
  --template /tmp/deepseek-template.jinja --corpus /tmp/deepseek-corpus.json \
  --output /tmp/deepseek-results --zig zig
```

The full corpus and original process logs stay local; committed summaries retain all parsed measurements, expected-output identities and harness identities. The fixture generator and runner both validate the template checksum.

## Limits

The exact template also repeatedly rebuilds lists (`ns.msgs`, merged parts and tool-text lists). This string-builder change does not optimize list concatenation and therefore does not promise linear total rendering memory for every template. No DeepSeek checkpoint inference, four-rank deployment, original private request, original tool schemas or 765k-token prefill was tested here. The earlier Qwen decode-equivalence receipt remains a separate gate. No Spark operation or production change was required.
