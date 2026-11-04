# Dense BF16 Qwen3.5 prompt-score qualification

Draft qualification for #1136 and PR #2020. Historical native API, score-head
reference, retention and observed-cancellation checks passed as scoped below.
The Transformers BF16-distance comparison remains failed.
The complete CUDA contract exits 1 with 342 launch-limit problems. BF16 prefill
and four-stream throughput are substantially slower than the standard server.
This receipt does not establish a performance-qualified serving format.

## Source and environment

[manifest.json](manifest.json) identifies the source, stacked parent, native
binary and artifact hashes. These measurements bind the original build at
`327ef639` and strict contract source at `5332d97e`.
[Source provenance](source-provenance.json) distinguishes these actual tested
identities from later rebased ancestry. The historical GPU results do not
qualify a new native build from the rebased head.

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

## Dumps kept out of the tree

The per-request dumps and the complete CUDA contract log are kept out of the
tree; [asset.json](asset.json) links their immutable public evidence assets and
pins every file hash, byte count and archive member hash. The checkpoint and
fixtures are public. Personal/infrastructure metadata was reviewed before upload.

| file | sha256 | bytes |
| --- | --- | ---: |
| `qwen35-bf16-scoring-dumps-20261103.tar.gz` | `d8f5763c8130ac5326ab5eb166999b3265152a897c731f4ed70768b5f64a29b7` | 376,782 |
| `qwen35-transformers-prompt-scores-20261103.json` | `423d2d653a5e0c900ceca2e4f4d6fcbfe982f6c7a2e8a22f9a9ccb290f48a194` | 5,367,360 |
| `qwen35-transformers-run-20261103.log` | `6c826b51cc1725422d83cfb9a501a5264739eae1d3da5092490d50d61c67ea73` | 4,128 |

The tarball unpacks to 3,258,590 bytes and holds `cuda-contract.txt`, the
`cold/ids-*.json` and `cold/request-*.json` replay bodies, the six per-request
concurrency dumps and `attempts/`. The recipe sections below keep every command
that produced them; the summary receipts stay here.

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

## Independent transformers forward

**Qualification remains failed.** The original experiment completed, but its
measured BF16-distance envelope does not pass.
[reference-verdict.json](reference-verdict.json) reports three rows outside the
FP32-reference envelope and one outside the BF16-reference envelope, even with
the existing native-head rounding allowance. The broader hidden-distance bound
is a numerical diagnostic and does not replace this acceptance comparison.
These are captured native rows, not a direct HTTP prompt-score comparison.
The 4097-token fixture has no FP32 arm. Maintainer acceptance remains outstanding.


[tools/zig/qwen35_transformers_reference.py](../../../tools/zig/qwen35_transformers_reference.py)
runs the pinned checkpoint on CPU in two arms with the same token ids: the
stored-BF16 model, and the same model with every decoder layer's stored
parameters and buffers promoted to FP32 for its forward. The full payload and run
log are kept out of the tree with the dumps (their sha256 are in
[asset.json](asset.json)); [transformers-comparison.json](transformers-comparison.json)
is the tracked summary and holds the tables below. The script reads the pinned
checkpoint, the tracked [fixtures.json](fixtures.json) and [api.json](api.json)
receipts, an FP64 reference record and the finite native capture tree; the
summary withholds the host name and every absolute runner path.

Conditions: CPU only (`cuda_available: false`; no GPU was used), python 3.12.3,
torch 2.15.0+cu132, transformers 5.22.1, accelerate 1.15.0, safetensors 0.8.0,
numpy 2.5.4, 16 threads, peak RSS 53.0 GB, wall 224 s. Captures were recorded with
torch 2.13.0a0+9186a08b2c.nv26.07 and CUDA 13.3. Checkpoint `Qwen/Qwen3.5-27B` at
`fc05daec18b0a78c049392ed2e771dde82bdf654`, 11 BF16 safetensors shards (55,563,022,432 weight bytes, 1,199 weight-map entries),
config sha256 `f8d190c5b89c1521220f935d2567a587d6e291ed69066a45a106560b05a2174c`,
eager attention, 27,356,728,560 parameters.

Method: each scored entry is `logit - logsumexp` over the full 248,320-token
vocabulary in float32, position 0 is null as served, no KV cache is reused, and
the first five served greedy continuation tokens are teacher-forced per fixture.
The BF16 arm's hidden states are bitwise equal to a plain library `model.forward`
(max abs diff 0.0), and extending a prompt by the five served continuation tokens
is bitwise inert (0.0 over 257 rows). Promotion is exact, so the two arms differ
only in activation and accumulation precision; the FP32 arm is not a trained FP32 checkpoint and no
FP32-partial-sum arithmetic is claimed (see #1468). The 4097-id fixture has no
FP32 arm.

**A. transformers BF16 vs transformers FP32, same ids** (fixture 0 has 1 id and 0
scored rows)

| fixture | ids | rows | max abs dlogprob | mean abs dlogprob | max abs dlogit | argmax agree | rank agree |
|---|---|---|---|---|---|---|---|
| 1 | 32 | 31 | 0.0577 | 0.00704 | 0.2688 | 31/31 | 28/31 |
| 2 | 129 | 128 | 0.0280 | 0.00112 | 0.1239 | 128/128 | 127/128 |
| 3 | 257 | 256 | **4.035** | 0.0978 | 9.029 | 249/256 | 233/256 |
| 4 | 4097 | 4096 | bf16 only | - | - | - | - |
| all | | 415 | 4.035 | 0.0612 | 9.029 | 408/415 | 388/415 |

The distance is not uniformly small: the worst logprob row is fixture 3,
position 216, token 6761 (BF16 -0.605836 vs FP32 -4.641201, argmax disagrees),
and 16-position block maxima rise from 0.025 at the prompt start to 4.04 over
positions 208-223. This is transformers' own BF16 distance from FP32, not a
serving error.

**B. served captures vs both arms.** 15 captures, 18 rows: the last two scored
prompt rows `[L-2, L-1]` then decode rows `[L+1, L+4]`, verified by exact row
pick before comparison; no nearest-row search. `bound` is
`native_vs_fp64_box + tf_bf16_vs_fp64_box + |dh·W_t| + max_v|dh·W_v| +
rounding_slack`, where `dh` is the captured-vs-reference BF16 hidden difference
added through the output projection `|dh·W_t|` and the worst-case rows of `dh·W_v`
here. The bound holds on 18/18 rows, the max ratio is 0.169451 and the max bound
is 15.8293. The receipt-style form that omits `dh` (`native_box + D + slack`)
fails 1/18 (row 13: `|native - tf_bf16|` = 0.003021 against a 0.001006 bound) and
is kept only as a diagnostic.

| # | cap | phase | fx | pos | tok | native | tf-bf16 | tf-fp32 | abs(n-16) | bound | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | prompt | 1 | 30 | 321 | -0.003733 | -0.003702 | -0.003771 | 0.000031 | 0.1861 | 0.0002 |
| 1 | 0 | prompt | 1 | 31 | 7890 | -0.003687 | -0.003553 | -0.003576 | 0.000134 | 0.7516 | 0.0002 |
| 2 | 1 | decode | 1 | 33 | 271 | -0.367115 | -0.363819 | -0.371754 | 0.003296 | 0.1433 | 0.0230 |
| 3 | 2 | decode | 1 | 34 | 248068 | -0.291004 | -0.273081 | -0.293707 | 0.017923 | 0.2104 | 0.0852 |
| 4 | 3 | decode | 1 | 35 | 271 | -0.564259 | -0.561222 | -0.568481 | 0.003036 | 0.2216 | 0.0137 |
| 5 | 4 | decode | 1 | 36 | 248069 | -0.000017 | -0.000019 | -0.000017 | 0.000002 | 0.3554 | 0.0000 |
| 6 | 5 | prompt | 2 | 127 | 220 | -0.000072 | -0.000074 | -0.000074 | 0.000002 | 0.1599 | 0.0000 |
| 7 | 5 | prompt | 2 | 128 | 15 | -0.000443 | -0.000509 | -0.000500 | 0.000067 | 1.9843 | 0.0000 |
| 8 | 6 | decode | 2 | 130 | 220 | -0.000240 | -0.000227 | -0.000206 | 0.000013 | 1.0871 | 0.0000 |
| 9 | 7 | decode | 2 | 131 | 16 | -0.000046 | -0.000044 | -0.000051 | 0.000002 | 0.3920 | 0.0000 |
| 10 | 8 | decode | 2 | 132 | 198 | -0.000561 | -0.000582 | -0.000559 | 0.000021 | 0.1732 | 0.0001 |
| 11 | 9 | decode | 2 | 133 | 262 | -0.000084 | -0.000086 | -0.000082 | 0.000002 | 0.1481 | 0.0000 |
| 12 | 10 | prompt | 3 | 255 | 814 | -6.881475 | -6.051020 | -4.500031 | **0.830456** | 4.9008 | 0.1695 |
| 13 | 10 | prompt | 3 | 256 | 20139 | -0.003700 | -0.000679 | -0.001114 | 0.003021 | 14.9119 | 0.0002 |
| 14 | 11 | decode | 3 | 258 | 5979 | -0.000961 | -0.000662 | -0.000826 | 0.000299 | 5.2837 | 0.0001 |
| 15 | 12 | decode | 3 | 259 | 44896 | -0.000786 | -0.000896 | -0.003065 | 0.000111 | 11.3577 | 0.0000 |
| 16 | 13 | decode | 3 | 260 | 5533 | -0.008688 | -0.006145 | -0.000908 | 0.002542 | 14.7874 | 0.0002 |
| 17 | 14 | decode | 3 | 261 | 264 | -0.006615 | -0.007076 | -0.002056 | 0.000462 | 15.8293 | 0.0000 |

Max `|native - tf_bf16|` is 0.830456 (row 12, prompt, fixture 3, position 255,
token 814); the other 17 rows are at most 0.0179.

Limits: the FP32 arm is float32 arithmetic over promoted stored BF16 values; no
whole-model FP32 copy is materialised. The 4097-id fixture has no FP32 arm and no
capture resolves to it. Prompts spanning more than one 4096-row chunk are not
exercised. The payload's `host` field is redacted; all other environment
identities are as recorded.

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
figures: [native decode](native-decode.json) and [standard
decode](standard/sglang-decode.json); the two per-request concurrency dumps are kept out of
the tree like the other per-request dumps (checksums in [asset.json](asset.json)).
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

Failed fixture/harness attempts are kept out of the tree under `attempts/`
(checksums in [asset.json](asset.json)).
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

[cuda-contract.json](cuda-contract.json) identifies its binary and verdict; the
complete run log `cuda-contract.txt` is kept out of the tree (checksum in
[asset.json](asset.json)); its verdict lines are:

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
findings, and all six hosted no-GPU jobs plus format passed on `75843aed`.
Historical hosted passes retain their actual original commit IDs. Both workflows
on rebased head `54f94594` were cancelled. Publication-head CI is tracked on the PR.
[Overlap evidence](overlap.json) retains newly reported conflicts inherited
from the declared parent; ATC coverage is incomplete. Landing order needs
maintainer coordination. The pinned main merge-tree check is clean.

## Follow-up validation

[followup-checks.json](followup-checks.json) records seven verdict/privacy tests
on Python 3.11/3.14 and Linux Python 3.12, unchanged non-CLI executable ASTs,
CPU head-byte/FP64-profile equality, and zero new lean problems (473 → 430).
The forward tool now exits nonzero when the measured envelope fails, and its
JSON output removes host names, local input paths and raw command-line paths.
`complete` means the experiment completed; `qualification.status` is separate.
The standalone verdict client can check the recorded result without weights:

```sh
python3 tools/zig/qwen35_reference_verdict.py docs/receipts/qwen35-bf16-scoring/transformers-comparison.json
python3 tools/zig/test_qwen35_reference_verdict.py
```

The first command intentionally exits 1 for this failed comparison.

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
finite adapter uses the existing stream, timing and receipt implementations;
with the kept-out dumps tarball at hand, extract a replay body first:

```sh
tar -xzf qwen35-bf16-scoring-dumps-20261103.tar.gz cold/ids-65536.json
python3 docs/receipts/qwen35-bf16-scoring/reproduce-cold.py . cold/ids-65536.json cold.json
```

For observed cancellation, use a fresh context-8192/parallel-4/cache-0 native
server with `--dashboard`, which exposes the existing `/stats` diagnostics:

```sh
python3 docs/receipts/qwen35-bf16-scoring/reproduce-disconnect.py http://127.0.0.1:8091 qwen35-bf16 docs/receipts/qwen35-bf16-scoring/fixtures.json --output cancellation.json
```

All these Python programs are bounded development clients. Native inference
continues through the shared lane scheduler. Launch-limit catalog resolution,
performance acceptance and the explicitly unqualified scopes remain outstanding.
