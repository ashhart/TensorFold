# Warm resume versus cold start

The primary comparison is **warm resume versus cold start on the same current
runtime**, including the whole path to the first token and the completed reply.
The measurements below use runtime `938d085`, drafts, an 8K prompt and a 64-token reply.

- **Warm resume:** wake the sleeping runtime, reload weights, restore the retained
  conversation prefix lazily, and generate the reply.
- **Cold start:** launch a fresh process, load the same model and draft, process
  the full prompt without retained KV state, and generate the reply.

Both use warm filesystem and compiled-kernel disk caches. Here, cold means a
fresh runtime with no conversation cache; cold-storage startup is unmeasured.
The resume timer starts at wake, excluding the earlier sleep operation. Full
model-switch timing must also account for outgoing sleep and the other model.

## Primary comparison

| Median metric | FrogNano cold | FrogNano warm | Qwen 27B cold | Qwen 27B warm |
| --- | ---: | ---: | ---: | ---: |
| Start/resume through first token | 6.34 s | 6.02 s | 11.58 s | 16.58 s |
| Start/resume through completed 64-token reply | 7.07 s | 6.66 s | 15.50 s | 20.39 s |
| Total output tokens/s, including start/resume | 9.05 | 9.62 | 4.13 | 3.14 |
| Reused prompt tokens | 0 | 8191 | 0 | 8191 |

FrogNano warm resume is about 5% quicker to the first token and 6% quicker to the
completed reply in this sample. Qwen warm resume takes about 43% longer to the
first token and 32% longer to complete the reply; its total output TPS is 24%
lower. These are medians of three repetitions, which do not establish statistical
bounds on small differences.

For Qwen, the current gap is **5.00 seconds to the first token and 4.89 seconds to
reply completion**, computed from the measured medians. Improvements should be
judged by reducing these warm-versus-cold gaps. The earlier 46% wake improvement
is useful diagnostic history, not a warm-versus-cold advantage.

All six measured warm requests reused 8,191/8,192 prompt tokens and reproduced
all 64 output tokens exactly. A cache hit alone does not establish a net speedup:
verification, reload and lazy restoration all count toward the result.

[Individual measurements, profiles and methodology](model-sleep-performance-results.json)
include excluded priming runs, all measured repetitions and the computed comparison.

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

## Investigation: checkpoint hashing

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

## Secondary comparison: progress before and after the hashing change

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

## Diagnostic request metrics after readiness

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

The next optimization target is lower first-token and completed-reply time for
warm resume against its paired cold start. Keep exact output, full checkpoint
integrity checks, cache accounting and asleep memory reclamation as guardrails.
Repeat the cold control for each candidate and report the complete elapsed time;
a faster internal phase is insufficient if total warm latency regresses.

The remaining costs suggest these next experiments:

1. Profile the current warm/cold readiness difference, splitting checkpoint
   validation, target loading, draft loading and other initialization. The warm
   phase profile attributes 9.02 seconds to hashing and 6.34 seconds to loading;
   verify the corresponding cold stages before changing shared loading code.
2. Measure longer histories and distinct retained conversations to find the
   workload where saved prefill outweighs verification and reload. Include
   changed suffixes; exact replay alone cannot characterize normal cache reuse.
3. Time an actual A-to-B-to-A switch, including both outgoing sleeps and incoming
   wakes, with output-token throughput across the whole workload. The results
   above time one runtime's startup or wake, not a complete swap.
   Count all generated output and the entire switch interval when comparing
   aggregate throughput. Include an already-resident cached-request control to
   separate cache reuse from process and reload costs.
4. Test cold storage and competing I/O. These warm-cache results do not establish
   that concurrent readers help every disk. Lazy prefix restore is synchronous;
   its queueing cost under concurrent resumed requests remains unmeasured.

This pass changes checkpoint hashing only. It adds no new sleep backend,
retention policy or restart-persistence guarantee.
