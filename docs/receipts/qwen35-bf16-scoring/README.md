# Dense BF16 Qwen3.5 prompt-score qualification

Draft qualification for #1136 and PR #2020. Native correctness, API, reference,
retention and observed-cancellation checks passed for the configurations below.
The complete CUDA contract exits 1 with 342 launch-limit problems. BF16 prefill
and four-stream throughput are substantially slower than the standard server.
This receipt does not establish a performance-qualified serving format.

## Source and environment

[manifest.json](manifest.json) identifies the source, stacked parent, native
binary and artifact hashes. Production inference/kernel trees are unchanged
from the build at `327ef639`; the final strict contract source is `5332d97e`.
Later changes add documentation, reports and finite qualification clients.

One NVIDIA RTX PRO 6000 Blackwell Max-Q workstation GPU, sm_120, rank 1,
96 GB class and 300 W limit; Linux x86_64, driver 615.78.08, Zig 0.17.0,
nvcc 13.3.73, ReleaseFast, native version 1.2.1. This is a main-line candidate,
not a measurement of release 1.1.1. The physical host is 128 GB class.
Serving has a four-CPU quota and 32 GiB container cap. Both engines publish only
a loopback host port; the container listener uses its private bridge interface.
One model process owns the GPU lock at a time.

[models.json](models.json) pins every public checkpoint/tokenizer. The BF16
checkpoint is `Qwen/Qwen3.5-27B` at
`fc05daec18b0a78c049392ed2e771dde82bdf654`. Projection weights and activations
remain BF16; sums and the checkpoint's stored FP32 GDN scales remain FP32.
No private checkpoint, request fixture, new weights or drafter is distributed.
Vision, int8 prompts and learning are refused for the new format. Multiple ranks,
other physical chips/backends, FP8 KV and optional grammar/tool generation are
unqualified here. Compiled five-SM images are not execution receipts for those chips.

## Native API and independent reference

[api.json](api.json) records 32 passed checks, one warmup excluded: drafted/plain,
scored/unscored, repeat, concurrent/solo score bits, top-k admission, refused
requests and recovery. [fixtures.json](fixtures.json) contains exact public
arrays of lengths 1, 32, 129, 257 and 4097. The last crosses the prompt-chunk seam.
The receipt retains full reply IDs, full token hashes and FP32 score-bit/rank hashes.
This is an exactness matrix, not a fixed-prompt service-latency distribution.

[fp64.json](fp64.json) passes six prompt rows, 12 decode rows and 24 named
projection captures. Worst absolute head error: `2.588785719126463e-6`;
worst absolute logprob error: `6.144032695942769e-6`; zero argmax differences.
Absolute-error and bound-ratio coordinates are separate, with selected/top/rank
checks and every capture's derived bounds.

This computes FP64 head dots and full-vocabulary logsumexp on identical native
BF16 normalized rows. It is not a full-model FP64 forward. Projection captures
use deterministic synthetic BF16 rows against actual checkpoint matrices;
they do not establish layer-by-layer whole-model fidelity. No tolerance was widened.
See the [recipe](../../recipes/qwen35-bf16-scoring.md) for capture/build commands.

## Served measurements

Both engines use the same pinned public BF16 checkpoint, physical GPU, CPU/RAM
limits and exact rendered prompt IDs: 14 code tokens and 31 no-thinking chat
tokens. Tokenization receipts are [native](rendered-prompts.json) and
[standard](standard/rendered-prompts.json). Both request 128 reply tokens with
`ignore_eos`, seed 1234, greedy temperature 0 or sampled 0.7/top-k 20/top-p 0.95.
Sampled stream i uses seed 1234+i. No external drafter is loaded; native drafting
stays enabled. Native concurrent runs also check each reply against solo and
`draft: false` references. The standard server provides no full token-ID hash,
so its cross-engine exact-token equality is unverified. All six greedy streamed
content/reasoning fingerprints match across engines.

Single-stream decode uses `bench_openai.py`, three repetitions per prompt,
one 128-token warmup per prompt excluded. Concurrency uses `bench_concurrent.py`,
three repetitions per cell, levels 1/4, four 16-token warmups excluded: 60 measured
requests, plus ten solo and ten plain references for native exactness. Every
measured native reply has 128 tokens, zero cached tokens and a full token hash;
there are zero mismatches, failed requests or unmeasured streams. Standard
measured requests also have zero failures/unmeasured streams. Each four-stream
cell is homogeneous; it does not represent a mixed production workload.

| Greedy case | Native decode tok/s | SGLang decode tok/s | Native TTFT ms | SGLang TTFT ms | Native C4 aggregate tok/s | SGLang C4 aggregate tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Code | 23.83 | 27.62 | 346.64 | 75.83 | 8.9 | 109.5 |
| Chat | 29.38 | 27.51 | 760.50 | 79.73 | 9.0 | 109.4 |

Native sampled C1 code/chat: 20.2/29.3 tok/s; sampled C4 aggregate: 8.7/9.0.
Raw reports retain TTFT, decode, end-to-end throughput, makespan, ITL and round-gap
figures: [native decode](native-decode.json), [native concurrency](native-concurrent.json),
[standard decode](standard/sglang-decode.json), [standard concurrency](standard/sglang-concurrent.json).
Decode excludes the first token; C4 aggregate uses the common batch decode span.
End-to-end throughput is a separate field. These are descriptive medians over
three repetitions, not a statistical significance claim. ITL p95 is not request
latency p95. Generated samples are omitted; interval clock origins are normalized
per repetition. Numeric rates and fingerprints are retained.

The [standard configuration](standard/configuration.json) uses SGLang 0.5.19,
Python 3.12.3, PyTorch 2.13.0+cu130/CUDA 13.0, Triton attention, BF16 weights/KV,
FP32 SSM state, radix retention disabled and `no_buffer`. The attached
[precision policy](sglang-precision/sitecustomize.py) disables optional BF16/FP16
reduced-precision partial reductions and TF32; the installed image was checked
with all three flags false. These flags differ from stock PyTorch defaults.
The engines and toolchains differ; these are whole-server measurements, not an
isolated-kernel speedup or a numerical oracle.

## Loaded-model cold prefill

Each cell starts a fresh server process, waits for readiness, uses cache-off
settings, one stream, context N+32 and one measured request with eight output
tokens. Startup/model loading is excluded. SGLang's startup/health warmups are
retained in the scheduler evidence; they are not measured requests. Exact
arrays and request bodies are under [cold/](cold/summary.json).

| Input tokens | Native TTFT s | SGLang TTFT s | Native HTTP cached tokens | SGLang scheduler cached tokens |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 49.569 | 0.443 | 0 | 0 |
| 8,192 | 199.272 | 1.907 | 0 | 0 |
| 32,768 | 799.221 | 9.260 | 0 | 0 |
| 65,536 | 1,601.291 | 24.579 | 0 | 0 |

The standard server omits the HTTP cached-token field. Its original generic
client reports remain `qualified: false` and exit 1; they have not been changed
to pass. [Scheduler evidence](cold/sglang-scheduler-cache.json) independently
records zero cached tokens in every measured prefill batch, whose new-token
counts sum to the exact input length. Leading startup batches of 80/1/1 tokens
are retained separately. This is out-of-band cache evidence, not an HTTP-counter
qualification. The 64k HTTP read timeout is 3600 seconds for both engines;
other cells use 1800 seconds. This changes the transport wait, not a speed gate.

## Retention, restart and cancellation

The public [chat fixtures](cache/chat-fixtures.json) render to 4122 tokens each
and share the first 4096 exactly. With a 2 GiB retained-state budget, observed
cached counts are original 0/4096/4096 and edited 4096/4096/4096. All replies match
the cache-off references by full token hash:
[fresh](cache/fresh.json), [retained](cache/retained.json).
The cache client's six negative/positive evidence tests pass on Python 3.11/3.14
and on the Linux qualification host. Missing/short hashes, early completion,
unknown counters and an edited first request without reuse cannot pass.

[Restart checks](cache/lifecycle-restart.json) reproduce original/edited replies
with cache off after process restart. Scoring recomputes the full prompt and
returns the same score bits with zero reused tokens after retained history.

[Observed cancellation](cache/observed-cancellation.json) passes seven checks.
Before socket reset, native status reports 4097 tokens still in prefill and zero
generated tokens. Before stream close, it reports one generated token of 128.
The native cancellation-acknowledgement counter increments by one in each case;
the source increments it only after the engine finishes with `cancelled`.
Each subsequent scoring request reproduces the original full reply IDs and FP32
score hash. Earlier unobserved disconnect-recovery checks remain separately scoped.

Failed fixture/harness attempts are retained under [attempts/](attempts/README.json).
Raw completions declare no history and therefore did not retain a snapshot;
the initial reuse assertion was wrong. The actual chat test satisfies reuse.
The first observed-cancellation client used `/v1/stats` and received HTTP404;
the corrected existing route is `/stats`. Neither failure is hidden as a pass.

## Existing-format regressions

[All alternating observations](existing-formats/summary.json) use process order
parent/candidate/parent/candidate, three repetitions of 128 tokens per prompt.
Each arm retains its own median and TTFT. The first parent/candidate pair also
runs all eight concurrency cells with one repetition, greedy/sampled, C1/C4,
solo and plain references. All 20 measured full token hashes match across builds
for each format; every native exactness comparison passes. Prompt-score requests
remain refused with HTTP400 as before.

| Checkpoint format | Code parent → candidate tok/s | Chat parent → candidate tok/s |
| --- | ---: | ---: |
| Qwen3.5 MLX 4-bit | 83.56 → 83.60 | 80.30 → 80.25 |
| Qwen3.8 NVIDIA ModelOpt NVFP4 | 63.33 → 63.36 | 63.35 → 63.40 |

Values are geometric means of each arm's two process medians. Observed changes
range from -0.064% to +0.084%; some concurrent cells are slightly slower, and all
raw observations are retained. No statistical noninferiority claim is made.
The ModelOpt checkpoint is Qwen3.8, not a relabeled Qwen3.5 export.

The public [EXL3 export](existing-formats/exl3-admission.json) is refused by both
builds: its embedding tensor is F16 while the existing native path requires BF16.
Both parent/candidate attempts reproduce `UnexpectedTensor`. No stored weights
were converted. EXL3 inference and performance remain unqualified for this export.

## Memory

[Memory receipts](memory.json) separate Linux process RSS, container memory,
CUDA-owned buffers and sampled pinned storage. VmHWM includes startup and is
about 32.08 GiB on this 128 GB-class host; it is not a physical-footprint counter
including GPU memory. The 32 GiB cgroup budget does not prove a physical 32 GiB
machine can run this workload. Counter scopes differ.

| Separate untimed case | CUDA-owned peak GiB | Observed pinned maximum GiB |
| --- | ---: | ---: |
| 8192 context, C4, cache off | 54.487 | 0.060 |
| 8192 context, C4, retained chat | 54.487 | 0.453 |
| 65568 context, C1 admission plus 4097-token requests | 55.764 | 0.060 |

CUDA-owned peak excludes driver-private allocations. Pinned values are sampled
maxima, not exact allocator peaks. The last row is not a measured device peak
during the full 64k prefill. Completed 64k request RSS/cgroup evidence is separate.
No physical 32 GiB-host sizing or energy-cost claim follows from these figures.

## Complete CUDA contract and host checks

[cuda-contract.txt](cuda-contract.txt) retains the complete run;
[cuda-contract.json](cuda-contract.json) identifies its binary and verdict:

```text
SUMMARY 550 kernels: 550 pass gates 1-4, 0 fail, 6 known findings; 365 controls: 365 red as planted, 0 missed; 342 limit problems; gates 4013421 ms, limits 726300 ms
exit: 1
```

All four new function records pass on the live sm_120 device. Six existing NaN
findings reproduce: Qwen BF16/FP8 paged and contiguous attention, and GLM
scores/index scores. No known entry or tolerance was added to conceal a failure.
The 342 limit problems comprise 188 registry problems and 154 live mismatches.
This is a failed full contract, not a clean qualification.

[limits-comparison.json](limits-comparison.json) retains separate parent/candidate
five-SM asset audits: both exit 1 with 637 problems and identical 639-line
failure/stale-record multisets, including the two final failure lines. All 20
new function/SM records are current. Parent full/live-only execution was not run;
this comparison is not a matching parent live SUMMARY. Existing declarations
are unchanged apart from appending the four new records.

[Host/CI evidence](host-and-ci.json): full unprivileged host execution passed
6870 tests with 46 skips; the final strict-fixture rerun passes with cached
results distinguished. Plain build passes, lean has 424 inherited/zero new
findings, and all six hosted no-GPU jobs plus format passed on `284339b0`.
Publication-head CI is tracked separately on the PR.
[Overlap evidence](overlap.json) retains newly reported conflicts inherited
from the declared parent; ATC coverage is incomplete. Landing order needs
maintainer coordination. The pinned main merge-tree check is clean.

## Reproduce the finite clients

Use Python 3.11+, the native build in the recipe, loopback port 8091 and model
alias `qwen35-bf16`. Start a fresh context-8192/parallel-4 server with cache off,
run `fresh`, then restart with `--prompt-cache-gib 2` and run `retained`:

```sh
python3 docs/receipts/qwen35-bf16-scoring/reproduce-cache.py http://127.0.0.1:8091 qwen35-bf16 docs/receipts/qwen35-bf16-scoring/cache/chat-fixtures.json fresh --output fresh.json
python3 docs/receipts/qwen35-bf16-scoring/reproduce-cache.py http://127.0.0.1:8091 qwen35-bf16 docs/receipts/qwen35-bf16-scoring/cache/chat-fixtures.json retained --reference fresh.json --output retained.json
python3 docs/receipts/qwen35-bf16-scoring/test_reproduce_cache.py
```

For cold prefill, start context N+32/parallel-1/cache-0 before each call. The
finite adapter uses the existing stream, timing and receipt implementations:

```sh
python3 docs/receipts/qwen35-bf16-scoring/reproduce-cold.py . docs/receipts/qwen35-bf16-scoring/cold/ids-65536.json cold.json
```

For observed cancellation, use a fresh context-8192/parallel-4/cache-0 native
server with `--dashboard`, which exposes the existing `/stats` diagnostics:

```sh
python3 docs/receipts/qwen35-bf16-scoring/reproduce-disconnect.py http://127.0.0.1:8091 qwen35-bf16 docs/receipts/qwen35-bf16-scoring/fixtures.json --output cancellation.json
```

All these Python programs are bounded development clients. Native inference
continues through the shared lane scheduler. Launch-limit catalog resolution,
performance acceptance and the explicitly unqualified scopes remain outstanding.
