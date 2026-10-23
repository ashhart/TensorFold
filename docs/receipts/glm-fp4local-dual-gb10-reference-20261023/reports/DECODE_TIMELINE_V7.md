# Dual GB10 decode investigation — 2026-10-23

## Contract and scope

The user authorized controlled 8K, 32K and 128K decode timeline/routing tests,
followed by measured precision-preserving optimization, including service
stops and restarts. Only the isolated TensorFold FP4-Local pair on port 8890
is in scope. The vLLM/EXL3 services, OliveTin and original weights stay unchanged.

Baseline: immutable `local-reference/tensorfold-fp4local:exp-v5-20261022`, binary SHA-256
`4c75b2b6182560bdbee9eef8267218bfaa975541464761a7ad0476c8fec2e0ad`,
image `sha256:77a6204988f5012919aca93198a267f5cc8faafb0f49ee9848d5479a13d0fb20`.
Source base `f8d254c5e0f9ab91eb325a2e7c28d2e5cb07c94d` plus pinned loader and
local changes recorded in `provenance-v5.json`. The v6 source supplies optional
detailed diagnostic intervals; its three experimental optimizations stay OFF.

Hardware: two NVIDIA GB10s, SM121a, CUDA 13.0, Zig 0.17.0, TP2 with existing
RoCE transport. Original NVIDIA FP4-Local view and existing DFlash2 revision
are unchanged. Context 524288, four admitted streams, FP8 latent KV, existing
A4 prefill / A16 verify, no extra Dense FP8 and no changed split-K.

The recipe's Python 0.6.6 coverage statements are stale for this native tree.
Use the actual native code and local receipts, not Python-only tuning flags.

## Measurements

- Fixed public text fixtures with exact rendered token arrays near 8192,
  32768 and 131072; record actual counts, all request bodies, fixed seed,
  thinking enabled, temperature 1 and top_p 0.95, fixed output cap.
- Baseline ordinary serving without instrumentation, then opt-in route and
  stage probes; profiler timings are diagnostic and not headline throughput.
- Short bounded Nsight Systems windows on BOTH ranks after prefill, with no
  CPU sampling/context-switch tracing or driver/system permission changes.
- Distinguish GPU kernel busy time, gaps, waits, host spans and overlapped
  events. Never sum nested intervals or infer CPU time as wall minus event.
- Route histograms exclude the shared-expert sentinel. Record true layer,
  verify width, context and rank. Opt-in copies must not alter computations.
- Watch RAM, swap activity, GPU clocks and temperatures. No swapoff, cache
  dropping, automatic context reduction or clock changes to improve scores.

## Acceptance and rollback

No optimization is selected before measured bottlenecks identify its target.
Candidate mathematics must retain weight/activation precision, scale application,
rounding and accumulation order. Kernel replacements require byte-equality
against immutable baseline assets, guards/repeat checks and focused contracts.
Served fixed-seed token hashes must match at all three context sizes; cover
draft/plain, concurrency, cache, cancellation/retry and affected API features.
Keep cold-prefill checks separate and compare uninstrumented repetitions with
the same inputs. Small gains require alternating repeats; measurement noise
is not a performance improvement. No percentage speedup is promised in advance.

Restore v5 if a candidate has incorrect output, unstable behavior, insufficient
capacity or no repeatable served gain. This is a local dual-GB10 experiment,
not complete upstream all-device qualification or a broad quality benchmark.

## Progress

- Initial state confirmed: v5 running; approximately 17 GiB MemAvailable on A.
- v5 baseline completed: 12 requests, six exact fixtures, two repetitions each.
  Per-fixture token hashes and speculative counters match between repetitions;
  decode throughput spread is 0.002–0.908%. See
  `results/decode-v7-baseline-summary.json` and its retained JSONL inputs.
- Baseline prompt caching was observed, not forcibly flushed. Cached token
  counts are recorded. Do not use these prefill times as a cold benchmark.
- Diagnostic runs use a zero-byte prompt cache and separate event/routing
  probes from Nsight tracing. The full request has host-round and GPU-stage
  records; Nsight records bounded steady-state windows, not the entire prompt
  or every output token. These two kinds of evidence must not be conflated.
- v7 diagnostic image was built and used only for profiling. The default v5
  controller, image and weights are unchanged.
- Route host tests and two round-trace unit tests passed on node-b. The first
  build found a Zig 0.17 reflection API mismatch, corrected before rebuilding.
- Full events run passed all six fixed-seed hashes, with 706 forwards/rank,
  59,304 matched layer-route records and no discarded/aborted rounds. The
  controlled-width 8K-to-128K MLA increase is approximately 1.7–1.8 ms;
  routed experts and KDA projections dominate instead.
- First 8K Nsight window: 2,082.5 ms span, 1,977.1 ms traced GPU activity union
  (94.9%), 53,224 kernel launches, all correlated. It is not evidence of
  saturated DRAM bandwidth, but does not support large host launch starvation
  as the main cause. All twelve task/context/rank windows are now complete.

## Next candidate contract: low-eight-row expert specialization

Implement a default-OFF, independent grouped expert kernel specialized for
the observed 1–8-row decode windows. Keep the original kernel and common MMA,
block-scale FMA and epilogue helpers unchanged. Each item advances by eight
members, with a compile-time false high-half predicate; counts above eight
must still be covered by further iterations. Never change router ordering,
weight precision, activation precision, split-K, global scale or GLM clipping.

Expected mechanism is reduced live registers/predicates, not half the tensor
core arithmetic or elimination of already-shared weight reads. Dispatch only
on the explicit flag for decode rows<=8; prompt and larger batches retain the
old path. Keep WARPS=4. Qualify function resources, byte-exact output/guards/
input preservation on real dimensions and awkward item counts before any
served use. Promote only if uninstrumented full-model repetitions improve;
otherwise restore v5 and retain the negative result.

## Completed findings (2026-10-23)

### Unprofiled baseline, not the user's previous benchmark

The six synthetic fixtures use exactly 8,192 / 32,768 / 131,072 rendered prompt
tokens and 256 generated tokens, temperature 1, top_p .95, seed 119, thinking
enabled at low effort. The output count includes reasoning. Each cell has two
repetitions. No all-context quality or completed-HTML claim is made.

| Fixture | 8K token/s | 32K token/s | 128K token/s |
|---|---:|---:|---:|
| Chinese summarization prompt | 33.95 | 30.18 | 30.31 |
| HTML prompt | 44.37 | 47.66 | 29.28 |

Hashes and speculative counters match between repeats, with a maximum 0.91%
throughput spread. The six event-instrumented and six Nsight requests also
match the corresponding immutable-v5 output hashes. Prefill cache state is
recorded, not controlled in the unprofiled baseline; these are decode results.

### Where decode time goes

Full-request stage/routing instrumentation covers 706 target forwards per rank
and 59,304 layer-route records across both ranks, with no unmatched metadata,
dropped rounds or aborted rounds. Host-round records are aggregated by width
and context (46 records), not a per-round CPU trace. Nsight supplies twelve
approximately two-second representative decode windows, **not full-request
Nsight captures**. See `results/events-v7-summary.{json,txt}` and
`results/nsys-v7/ALL_CONTEXTS_AUDIT.md` for reproduction and attribution limits.

- Routed expert kernels: 33.56–43.54% of the Nsight window.
- Source-matched KDA input/output projections: 17.16–19.66%.
- Named communication kernels: 4.45–5.78%, potentially including peer waits.
- Traced GPU activity: 94.51–95.53%. This is not achieved occupancy or DRAM
  bandwidth utilization.
- All gaps: 5.057% (duration-weighted). The next kernel was already queued for
  4.811% of the window; launch-late gaps account for only 0.129%. A large host
  submission bottleneck is not supported. Even deleting every gap would have
  only a roughly 5% ideal benefit, and most gaps may not be removable.

At a controlled two-row target verification width, rank-0 GPU stage totals
are 50.832 / 51.795 / 53.853 ms at 8K / 32K / 128K. MLA rises from 6.894 to
8.596 ms, while routed experts remain near 18.6–19.0 ms and the full KDA stage
near 14.0–14.1 ms. Context-dependent MLA growth is real but not the dominant
explanation for the large token/s variation in these fixtures.

For example, the HTML fixture's accepted/drafted ratio changes from 81.2% at
32K to 52.8% at 128K; generated tokens per round fall from 3.28 to 1.82. But
the 128K output consists entirely of reasoning within the 256-token cap, while
the 32K result contains only 22 reasoning tokens. Do not attribute this change
solely to context length or claim general acceptance degradation at 128K.

Draft calls take about 5.9–6.2 ms out of roughly 64 ms mean full rounds. This
means halving the drafter alone would save only around 5% end-to-end in these
samples. `draw_nested_ms` includes waiting for previously queued target GPU
work; it is not an independent 20–40 ms sampling bottleneck.

### Expert reuse and the low8 experiment

Two-row verification selects about 13.2 distinct routed experts per layer
from 16 token/expert pairs; wider windows already share repeated experts via
the grouped planner. These repeated pairs are not a new unimplemented reuse
opportunity. No host/device copy of complete router decisions is added to the
normal serving path; diagnostic probes remain off.

The new default-off low8 kernel retains the original K loop, block FMA order,
weight/input precision, global scales, clipping and epilogues. It compiled in
an independent v8 image. Focused dispatch host tests: 4 passed. Full build:
1,133 passed, 4 skipped. Local GPU checks on node-b passed finite bit equality,
zero/extreme cases, nonfinite classifications, shuffled members, counts above
8/16, output guards and unchanged inputs/weights. This is local operator
qualification, not a broad model quality guarantee or all-GPU certification.

Gate/up registers fell from 128 to 96 (with 8 bytes local storage/thread), and
down from 96 to 71. Theoretical occupancy increased, but no stable benefit
appeared in five interleaved repetitions of ten route plans per width:

| Verify rows | Baseline/candidate speed ratio, 64 MiB eviction | Without eviction |
|---|---:|---:|
| 2 | 0.9895 | 1.0115 |
| 3 | 0.9978 | 0.9979 |
| 4 | 1.0028 | 1.0049 |
| 8 | 0.9984 | 0.9993 |

Greater than 1 means the candidate is faster. Timing includes gate/up + down
only, not full-model serving. The 64 MiB write exceeds this GPU's 24 MiB L2,
but is an eviction approximation, not a DRAM counter or proof of cold cache.
Both suites passed all correctness checks. Since there is no convincing
operator benefit, **v8 is not promoted and was not loaded for full-model
serving**; avoid spending another full load on an unpromising candidate.

The unflushed two-row 1.0115 ratio is not a reliable gain: the baseline timing
shifted during the repetitions, and the median of paired repetition ratios is
0.9874. Local allocation alone does not establish an actual register spill.
Combined up+down timing also cannot rule out a small single-stage benefit
being cancelled by a regression in the other stage; separate-stage events and
an A/A control are the smallest useful follow-up if revisiting this candidate.

Nominal minimum weight bytes divided by measured operator time yield about
203–218 GB/s. Together with the lack of improvement after reducing registers,
this supports a weight-traffic limitation, but does not prove DRAM saturation.
The next useful kernel investigation would measure DRAM/L2 traffic and issue
stalls before changing pipeline depth or cache policy again. No driver
profiling permissions or security settings were modified in this round.

### What remains worth doing

1. Optimize the actual KDA projection arithmetic/data access rather than the
   already-tested output-copy fusion. Preserve its MXFP8-to-BF16 MMA and exact
   accumulation/rounding boundaries unless the user authorizes a precision change.
2. Improve DFlash's useful tokens per target forward, not only drafter speed.
   Confidence/window decisions require task-diverse controlled tests; the
   existing .7 confidence setting is unchanged. Reducing rejected verification
   work could matter more than another tiny kernel launch saving.
3. Consider deeper expert data-layout or memory-pipeline changes only after
   hardware-counter evidence identifies avoidable traffic/stalls. Reusing the
   already-grouped experts or merely increasing theoretical occupancy is not
   sufficient.

An actual 20% expert-stage reduction would imply only roughly 7–10% whole-run
speedup; a 20% KDA-projection reduction roughly 3–4%. These are Amdahl estimates,
not achieved results or promises. No weight, split-K, precision, sampling or
context/concurrency settings were changed.

## Preserved artifacts and recovery

- v7 diagnostics: `provenance-v7.json`, `results/source-v7-20261023.tar.gz`,
  `tfctl-v7`; image exists on both nodes.
- v8 rejected candidate: `provenance-v8.json`,
  `results/source-v8-low8-20261023.tar.gz`, `tfctl-v8`; image exists on node-b
  only. It is not connected to normal startup or OliveTin.
- Exact baseline/candidate embedded fatbins are preserved in `results/`.
- GPU checks: `results/low8-gpu-b{,-unflushed}.jsonl`.
- Normal startup still delegates to immutable v5: on node-a run
  `/srv/tensorfold-experiment/tfctl start original 8890`.
- Restoration completed: both containers use the pinned v5 image, original
  profile, 8890, 512K, four streams, DFlash on, temperature 1 and top_p .95.
  Startup took 378.7 seconds. See `results/restore-v5.log`.
- Post-restoration 32K code fixture matches the original full token hash and
  measured 47.40 token/s versus the prior 47.66 mean. The smoke suite completed
  eight receipts, with both draft/plain hash comparisons equal and successful
  tool/Responses requests. Receipts: `results/decode-v7-restored-v5.jsonl` and
  `results/restored-v5-smoke-v8.jsonl`.
- Final MemAvailable: A 17.23 GiB, B 17.80 GiB. In six two-second idle samples,
  A had zero page-ins/outs; B had 10 page-ins and zero page-outs. Resident swap
  was 118.75 / 179.79 MiB; residency is not ongoing swapping. No full-capacity
  four-stream 512K test was performed in this profiling round.
- Temporary v7 diagnostic monitors were stopped. No sampling/profiling
  overhead, low8 switch or v6 experimental optimization is enabled in v5.
