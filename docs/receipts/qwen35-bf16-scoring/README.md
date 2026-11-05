# Dense BF16 Qwen3.5 prompt-score qualification

The [20261105 layer diagnostic](layer-diagnostic.md) records the requested 32-ID follow-up after rebasing onto current main. The first excess is at layer 0; the stricter score failures remain. Its native/reference producers are separate from the historical serving and full-CUDA producer below. [layers.json](layers.json) and [asset.json](asset.json) bind the new evidence.

This receipt records qualification of the experimental dense BF16 text format and native nonstreamed
`/v1/completions` prompt scores for the tested single-GPU configuration.
It records functional, numerical and regression checks separately from timing.
BF16 prefill and concurrency remain slow; performance remains unqualified.

The scope follows [issue #1136's agreed gate](https://github.com/ashhart/TensorFold/issues/1136#issuecomment-6553918935):
served scores, same-weight FP64 prompt/decode head and reduction evidence, red
planted controls, and unchanged existing formats. That agreement explicitly
allows the independent-engine distance comparison to fail. The requested direct
HTTP-versus-Transformers comparison is reported in both directions, with its
counterexamples. A broader hidden-state bound never replaces that comparison.
Maintainer acceptance and merge authority remain separate from this receipt.

## Measured results

| Check | Result |
| --- | --- |
| Served API matrix | 32/32 passed |
| Same-weight FP64 | 6 prompt rows, 12 decode rows, 24 projection captures passed; no argmax differences |
| Actual HTTP versus Transformers | 4511 scored positions collected; strict distances fail in both directions |
| Existing formats | MLX4/ModelOpt: 20 full reply-hash matches each; scoring refusal unchanged in four alternating processes each |
| Cache and cancellation | Fresh/retained/edited/restarted full reply hashes match; all observed-cancellation checks pass |
| Complete CUDA backend | Exit 0; 559 kernels pass, 367 controls red, zero gate/limit failures, six unchanged known findings |
| Five-SM compiled-resource audit | Exit 0; 1493 records, zero stale records, zero applicable launch failures |

Worst prompt/decode FP64 head errors are 2.455665e-6/2.588786e-6;
logprob errors are 5.418979e-6/6.144033e-6. The HTTP comparison has
1719 positions outside the FP32-reference distance and 1911 outside the
BF16-reference distance. Worst native-to-FP32/BF16 absolute logprob distances
are 5.546444/5.767185. These failures remain visible under the agreed gate.

| Public BF16 workload | Native | SGLang 0.5.19 |
| --- | ---: | ---: |
| Code greedy decode, median tokens/s | 23.68 | 27.70 |
| Chat greedy decode, median tokens/s | 29.32 | 27.58 |
| Code warm TTFT, median seconds | 0.352 | 0.076 |
| Chat warm TTFT, median seconds | 0.767 | 0.079 |
| Code C4 greedy aggregate decode, tokens/s | 8.9 | 109.6 |
| Chat C4 greedy aggregate decode, tokens/s | 9.0 | 109.5 |
| Code C4 whole-request throughput, tokens/s | 8.910 | 108.572 |
| Chat C4 whole-request throughput, tokens/s | 8.973 | 108.313 |

Three measured repetitions per short cell are descriptive evidence; they do
not establish a production percentile SLO. Native full reply hashes match solo
and plain execution in all 60 measured concurrency requests. Six greedy
cross-engine content/reasoning fingerprints match; cross-engine full-token-ID
equality is unverified. Sampled cells, SSE chunk statistics and all observations
remain in the linked asset and summary. No performance improvement is claimed.

| Fresh prompt IDs | Native TTFT, seconds | SGLang TTFT, seconds |
| ---: | ---: | ---: |
| 2048 | 49.515 | 0.447 |
| 8192 | 199.777 | 1.901 |
| 32768 | 801.786 | 9.211 |
| 65536 | 1601.814 | 24.608 |

Each cold row is one admitted first request with eight reply tokens after a
fresh server becomes ready. Startup and OS-cache scope are stated below.
SGLang's generic cache qualifier remains false because its HTTP cache counter
is absent; independent scheduler chunks total exactly N new tokens and zero
cached tokens in each row. No missing field is replaced with zero.

MLX4 baseline/candidate median-of-process-median decode changes are
+0.112% code/+0.137% chat; ModelOpt changes are -0.040%/-0.065%.
These small differences do not establish statistical noninferiority. The
selected EXL3 export is refused with `UnexpectedTensor` in all four processes
(F16 embeddings where BF16 is expected); no EXL3 inference is qualified.

## Exact source and configuration

[summary.json](summary.json) binds every result to its actual source and binary.
[source-provenance.json](source-provenance.json) retains historical identities;
subsequent documentation commits do not relabel earlier executions.
The GPU evidence binds parent #2008 at `789846adb` and pinned main `0d19cc19`.
The submitted stack also includes the parent's `30e9b872` receipt/test follow-up:
its four resource tables match the qualified cubins and its two added host
checks pass. Numerical gates and inference behavior are unchanged. Parent
history stays in the stack until the parent lands.

One NVIDIA RTX PRO 6000 Blackwell Max-Q workstation GPU, sm_120, rank 1,
96 GB class, 300 W power limit, Linux x86_64 and driver 615.78.08.
The host has an AMD Ryzen 9 9950X3D (16 cores/32 logical CPUs) and 128 GB-class
physical RAM. Serving uses four CPU cores and a 32 GiB
container cap, which does not establish fit on a physical 32 GiB machine.
Build/contract/reference jobs have separate budgets. The complete CUDA runner
uses available host CPU/RAM; it is not capped to the serving quota.
ReleaseFast, Zig 0.17.0, CUDA compiler 13.0.88; native version 1.2.2.
The clients run on the test host over HTTP loopback, without authentication.
Only a loopback host port is exposed. One model process owns the GPU lock at a time.

[models.json](models.json) pins the public checkpoints. The new format uses
`Qwen/Qwen3.5-27B` at `fc05daec18b0a78c049392ed2e771dde82bdf654`.
All 11 measured shard SHA-256s and byte counts match the pinned public
Hugging Face LFS manifest; the verification record is included in the asset.
Weights, tokenizer and drafter assets are unchanged. BF16 projections and
activations retain FP32 sums and the stored FP32 GDN scales.
All qualification inputs use the public checkpoint and synthetic public fixtures.
The baseline uses pinned-main source and the same prebuilt CUDA assets.

## Numerical and API evidence

The API matrix covers [public ID arrays](fixtures.json) of lengths
1, 32, 129, 257 and 4097, including the 4096-row prompt seam. It checks
first-row null, score/rank/top-k shape, full drafted/plain reply tokens,
concurrent/solo score bits, repeatability, refusals and recovery.
This exactness matrix is not a service-latency percentile benchmark.

The independent FP64 check computes head dots and full-vocabulary logsumexp
on identical native normalized BF16 rows and unchanged checkpoint weights.
It separates prompt and decode, worst absolute error and worst bound ratio,
selected/top tokens, ranks and argmax margins. Synthetic same-weight captures
cover the named projection classes. This is not a whole-model FP64 forward
or layer-by-layer activation equality with another engine.

The FP32 GDN-scale norm is registered in the complete CUDA runner at real
128-wide served dimensions, with non-BF16 scale values, nonfinite scales,
repeat/guard checks, FP64 bounds and a wrong-weight-dtype control.
Its integer stress inputs overflow FP16 sums while remaining exactly summable
in FP32. BF16 output rounding can hide BF16 sum error; that stated limitation
is not an unreported skip or evidence of FP32 accumulation by itself.
The scoring reduction's separate stress fixture distinguishes both reduced
accumulators; its omitted-tail and FP16-head controls remain red.

## Actual served comparison with Transformers

The CPU reference uses the same pinned checkpoint and IDs, eager attention,
BF16 and promoted-FP32 decoder arms, and a full-vocabulary FP32 scoring head.
Both arms cover every scored position, including all 4096 scored positions
of the 4097-ID fixture. Native scores come from actual HTTP responses with
cache reuse disabled, `draft: false`, top 20 entries and one reply token.
Position 0 is null; selected IDs, positions and zero cached tokens are checked.

The loader explicitly restores and byte-verifies all 96 stored FP32
`A_log`/GDN-norm parameters after Transformers auto-dtype loading; BF16
projections stay BF16. The earlier auto-dtype reference silently downcast
those parameters and is superseded for same-stored-value fidelity.
Its original failed counterexamples and checksums remain in the historical asset.
The reference uses CPU, Python 3.12.3, torch 2.15.0+cu132,
Transformers 5.22.1, accelerate 1.15.0, safetensors 0.8.0, numpy 2.5.4,
16 threads and approximately 53.9 GB peak RSS. Both arms took approximately
445 seconds. This larger CPU reference budget is not the serving RAM quota.
Teacher-forced continuation IDs position the auxiliary historical native
captures; those captures do not qualify the new served binary. Prompt rows are
the causal prefix of each forward, which includes five continuation IDs. A
separate prompt-only and library-equivalence repeat is verified for fixtures
through 257 IDs; the 4097-ID forward is not repeated for those auxiliary
checks because of CPU cost. Both BF16 and FP32 scored arms still cover every
prompt position at 4097 IDs.

The current summary reports every fixture's worst and mean distances, rank
agreement, and counts outside each measured BF16/FP32 direction. No tolerance
is widened. Collection success means complete, well-formed evidence, not
numeric equivalence or a passing cross-engine envelope.

## Served measurements and regressions

Both engines use the same physical GPU, public BF16 checkpoint, quotas and
rendered prompt IDs: code and no-thinking chat. Ordinary decode requests
128 tokens, one warmup per prompt excluded and three measured repetitions.
Concurrency measures C1/C4, greedy and sampled temperature 0.7/top-k20/top-p0.95,
three repetitions per cell and four 16-token warmups excluded.
Concurrency uses seed 1234; sampled stream i uses 1234+i. Decode repetitions
use seeds 1234/1235/1236, which are inert at temperature 0. `ignore_eos` fixes
output length.
Native drafting remains enabled; solo and `draft:false` checks verify full
reply hashes before speed is interpreted. No external drafter is loaded.
Decode rate excludes the first output token; end-to-end throughput includes
the full client makespan and prefill. ITL and round gaps describe observed SSE
chunks, which may contain several tokens, rather than individual GPU kernels.

The standard server is SGLang 0.5.19 (torch 2.13.0+cu130, Transformers 5.12.1)
in the pinned image listed in the summary,
BF16 weights/KV, FP32 SSM, Triton attention, radix disabled, `no_buffer`,
chunked prefill 4096 and static-memory fraction 0.85. The attached startup policy
disables TF32 and BF16/FP16 reduced-precision partials. Its toolchain differs
from native; these are whole-server measurements. SGLang does not return the
native full-token hash, so cross-engine token-ID equality is unverified;
streamed content/reasoning fingerprints are compared separately.

MLX4 and ModelOpt use four alternating baseline/candidate processes,
three decode repetitions per prompt and eight concurrency cells for each
first baseline/candidate pair. Greedy and sampled full reply hashes and
scoring-admission behavior are compared. Small rate differences are descriptive,
with all observations retained; no statistical noninferiority claim is made.
The selected EXL3 export's actual admission outcome is recorded, without
claiming inference support after a refusal.

Retained, edited, fresh and post-restart chat checks use the same complete
4122-ID prompts and verify full reply hashes. Observed cancellation checks
require an actually active request and verify the following request's state.
Cold 2k/8k/32k/64k measurements use exact archived arrays, fresh server processes,
cache0, C1, context N+32 and eight reply tokens. The 64k read timeout is 3600s;
others 1800s. All admitted attempts are retained.
Eight-token cold replies may have too few SSE chunks for a decode rate;
the observed TTFT remains measured and missing rates remain null.
TTFT begins after server readiness. Weight-load time is separate and the OS
file cache is not flushed. Compiler/kernel disk caches are also preserved,
including the mounted SGLang cache. Startup warmups/graph setup precede
readiness; “cold” here describes prompt/KV state.
SGLang's missing HTTP cache counter stays null and its generic cold qualifier
stays false/exit1; independent scheduler cached/new-token counts are separate
evidence, never a rewritten HTTP zero.

Memory runs are untimed. Sampled device/pinned/RSS counters, process high-water
marks and cgroup peaks have distinct scopes. Large-context admission with a
4097-ID request is not full 64k peak memory. Separate post-timing counters and
Linux high-water marks through the actual cold requests have their own scope. Sampling is not an allocator peak
when the runtime exposes no peak counter. Quotas and admission are not proof
of fit on a smaller physical machine.

| Memory scope, GiB | Process peak RSS | Native device allocator peak | Pinned sampled maximum |
| --- | ---: | ---: | ---: |
| Cache0/context8192/C4, 4097-ID requests | 5.984 | 54.025 | 0.060 |
| Cache2/context8192/C4, retained exactness | 6.000 | 54.025 | 0.453 |
| Cache0/context65568/C1, actual 65536-ID request | 5.992 | 55.302 | 0.060 at post-request read |

Native post-cold lifetime counters include model load and that request. The
last row's pinned value is one post-request observation, not a sampled maximum
through the timed workload. Physical process VmHWM and cgroup charge peaks
are also retained for every process where available. SGLang's largest single
process VmHWM is 6.815 GiB in the short run and 6.821 GiB through the 64k run;
these are not sums of its workers. Its short-run cgroup charge peak reached
about 32 GiB, and its 64k charge peak was 6.068 GiB. Cgroup charge includes
different mappings/accounting and peaks at different times from process RSS;
it is not interchangeable with summed physical RSS. SGLang device/pinned
allocator peaks were not collected and remain unmeasured.

## Checks, evidence and unrun coverage

Full API/FP64 per-capture reports live in the linked asset; the source tree
keeps aggregate summaries, worst row/token identities and bound formulas.
The summary records plain/native build commands, complete unprivileged host
checks, fuzz/golden coverage, executed versus cached/skipped counts, zero-new
lean comparison. Exact published-head CI is reported in the
[PR checks](https://github.com/ashhart/TensorFold/pull/2020/checks) and its final
description; those runs do not relabel the archived device producer. CUDA evidence includes the
complete backend SUMMARY, exit status, existing known findings, every planted
control, five-SM compiled-resource audit and actual sm_120 live checks.
A five-SM cubin instruction-byte proof covers the canonical-load cleanup.
Diagnostic norm-only checks supplement the complete runner.

The complete device command exits **0** with:

```text
SUMMARY 559 kernels: 559 pass gates 1-4, 0 fail, 6 known findings; 367 controls: 367 red as planted, 0 missed; 0 limit problems; gates 4023784 ms, limits 760889 ms
```

All six known findings match the previous declarations: the BF16/FP8 Qwen
paged and contiguous decode merges skip a NaN chunk, and the two GLM index
score paths suppress a NaN through their existing ReLU rule. No new known
entry is added. Complete per-case output, side suites and actual live checks
are retained in the linked log. This proves the tested sm_120 backend scope.
The initial missing-auxiliary-image setup failure and corrected runner hash
remain separate; the served native binary is unchanged.

[asset.json](asset.json) links immutable, checksum-pinned public assets and
archive member hashes. Full per-request outputs and complete logs stay outside
the source tree. Private infrastructure identifiers, personal contacts, user
paths, credentials and generated prose are excluded. Generic container paths,
public model/software/hardware identities, numeric evidence and token hashes
remain. Failed/superseded attempts retain their real source identities.
See [commands.md](commands.md) and the [recipe](../../recipes/qwen35-bf16-scoring.md)
for reproduction.

Other physical GPUs/backends, multiple ranks, FP8 KV, optional grammar/tool
paths, vision, int8 prompts and learning are not qualified for the new format.
GB10 execution remains with the maintainer as offered. Compiling another SM
proves its resource record, not device execution.
