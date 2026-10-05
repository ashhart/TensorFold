# Sleep/wake latency and retained-cache performance

Bounded concurrent checkpoint hashing reduced median wake-to-first-token latency
from 30.53 to 16.58 seconds for Qwen 27B with DFlash2, and from 7.16 to 6.02 seconds
for FrogNano 4B with DFlash. Both complete SHA256 verification passes remain.
These measurements use the same 8K prompt and 64-token reply in each trial.

The optimized Qwen wake still takes longer than an ordinary fresh-process start
at this prompt length. Retaining a prefix improves the first request after readiness;
it does not guarantee a faster result when reload time is included.

[Individual measurements, profiles and methodology](model-sleep-performance-results.json)
include excluded priming runs and all measured repetitions.

## Method

- Baseline: `84b9bd415687376cde93cf1000df7859c4f91004`, based on 0.6.5.
  The optimized runtime differs only in `CheckpointIdentity` file hashing.
  The results record source fingerprints and pinned target/draft revisions.
- Single-device CUDA, unified memory, compute capability 12.1, PyTorch
  2.13.0+cu130 and CUDA 13.0. Drafts enabled, one stream, checkpoint precision,
  BF16 prefill, 16,384-token served context.
- Prompt: `[1 + i % 128 for i in range(8192)]`. Generate 64 tokens with
  seed 1234, temperature 1, top-k 20, top-p 0.95 and EOS stopping disabled.
  This synthetic exact-prefix replay isolates retained-state reuse; it is not
  a natural conversation workload or a production cache-hit-rate estimate.
- For each trial, launch a fresh process, initialize the runtime and frontend,
  and generate the uncached reply. Then enable disk-prefix sleep, sleep, wake,
  and replay the request. The cold baseline initializes no sleep adapter before
  its first request, so it does not pay this feature's checkpoint checks.
- One excluded priming trial and three measured trials per model per revision;
  alternate model order across repetitions. The baseline campaign precedes the
  optimized campaign. Compare per-run quantities using their medians.
- Filesystem and compiled-kernel disk caches are warm. Fresh-process startup
  includes process launch, imports, weight load and frontend initialization.
  Wake includes both checkpoint checks, snapshot verification and runtime load.
  Resumed request timing includes the first lazy disk-prefix restoration.
- No other CUDA compute process was present before each trial. All benchmark
  processes exited and released their runtime after testing.

## Bottleneck and change

A separate Qwen phase profile attributed 22.91 of 29.59 seconds of wake time to
two sequential checkpoint checks: 77.4% of the transition. The target and draft
manifest contains 17 files totaling 24.02 GiB.

Two CPU-only hash passes, with the second reversing strategy order, produced
identical complete-file hashes:

| Hash strategy | Median time for one manifest |
| --- | ---: |
| One reader, `read` | 11.50 s |
| One reader, `readinto` | 11.44 s |
| Two concurrent readers | 6.22 s |
| Four concurrent readers | 4.52 s |
| Eight concurrent readers | 4.49 s |

Four readers reduce a check by about 61%; eight add little in this workload.
The implementation therefore hashes up to four files concurrently, retaining
1 MiB reads, full SHA256 digests, target/draft path keys, and verification before
and after loading. A read failure still prevents a successful transition.

The optimized phase profile measured 15.68 seconds total: 9.02 seconds hashing,
6.34 seconds loading and 0.32 seconds verifying the prefix snapshot. These are
single diagnostic profiles, separate from the repeated comparisons below.
Larger gains for Qwen reflect its multi-shard target: concurrency is across files,
so a single large shard remains serial within its reader.

## Transition and total response time

| Median metric | FrogNano before | FrogNano after | Qwen before | Qwen after |
| --- | ---: | ---: | ---: | ---: |
| Sleep | 2.55 s | 2.09 s | 13.10 s | 6.07 s |
| Wake to ready | 6.87 s | 5.73 s | 30.02 s | 16.06 s |
| Wake through first token | 7.16 s | 6.02 s | 30.53 s | 16.58 s |
| Fresh-process start through first token | 6.21 s | 6.34 s | 11.92 s | 11.58 s |
| Output tokens/s including wake through completed reply | 8.22 | 9.62 | 1.86 | 3.14 |
| Output tokens/s including fresh-process start through completed reply | 9.23 | 9.05 | 4.04 | 4.13 |

Wake through first token improves by 15.8% for FrogNano and 45.7% for Qwen.
Total output throughput including wake improves by 17.0% and 68.4%, respectively.
FrogNano's optimized wake is modestly faster than its paired startup baseline;
Qwen's remains slower. Three repetitions do not establish statistical bounds,
especially for small differences. The cold-baseline variation is reported rather
than attributed to a code path that did not change.

## What the retained prefix saves

All six measured optimized restored requests reuse 8,191 of 8,192 prompt tokens
(99.9878%). All 64 output tokens match their uncached counterparts exactly.
Each trial reaches zero CUDA allocated and reserved bytes while asleep, loads
one prefix lazily after wake, and reports no load or memory-admission failures.
The CUDA allocator figures exclude driver/library memory.

| Optimized runtime, median request metric | FrogNano fresh | FrogNano restored | Qwen fresh | Qwen restored |
| --- | ---: | ---: | ---: | ---: |
| First token after request begins | 1.512 s | 0.291 s | 3.138 s | 0.516 s |
| Complete 64-token reply | 2.245 s | 0.924 s | 7.054 s | 4.330 s |
| Request output tokens/s | 28.50 | 69.27 | 9.07 | 14.78 |

Request output TPS is generated output tokens divided by request elapsed time;
it excludes startup/wake. Total output TPS adds startup/wake time to that
denominator. Neither counts cached input tokens as generated work. These are
first-request measurements, not steady-state decode or aggregate serving throughput.
Fresh and resumed requests also differ in process warmup; decode differences
cannot be attributed solely to KV reuse. A normally restarted server can build
its own warm cache after the first request.

## Checks and remaining experiments

Focused lifecycle, storage, HTTP and adapter checks: 125 passed, one module
skipped in combined collection. Both real CPU tensor snapshot modules also pass
separately (51 cases, overlapping the combined run). All 25 ownership/reload
tests pass, including complete multi-block hashes, same-name target/draft files,
read failure before release, and a checkpoint modified during reload with its
size and timestamp preserved. Scoped lint introduces no new findings; existing
findings remain. The broader suite limitations in the previous receipts still apply.

The remaining costs suggest these next experiments:

1. Measure longer histories and distinct retained conversations to find the
   workload where saved prefill outweighs verification and reload. Include
   changed suffixes; exact replay alone cannot characterize normal cache reuse.
2. Time an actual A-to-B-to-A switch, including both outgoing sleeps and incoming
   wakes, with output-token throughput across the whole workload. The results
   above time one runtime's startup or wake, not a complete swap.
3. Profile loader allocation, staging and graph initialization inside the
   remaining 6.34-second load. Checkpoint validation is still the largest cost,
   but integrating validation with loading would need to preserve rejection of
   changed content and failed-wake cleanup before replacing either check.
4. Test cold storage and competing I/O. These warm-cache results do not establish
   that concurrent readers help every disk. Lazy prefix restore is synchronous;
   its queueing cost under concurrent resumed requests remains unmeasured.

This pass changes checkpoint hashing only. It adds no new sleep backend,
retention policy or restart-persistence guarantee.
