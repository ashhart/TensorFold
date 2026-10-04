# CUDA sleep/wake qualification on 0.6.5

The Qwen and Nemotron-H reference adapters are integrated with TensorFold 0.6.5
(`609ca41`). The qualified integration commit is `ca765ac20319`.
The [0.6.4 results](model-sleep-reference-validation.md) remain historical evidence;
this receipt and its [machine-readable results](model-sleep-065-results.json)
cover the new release. The CUDA runtime and qualification files match the exported
source; the control telemetry fix is verified by host HTTP tests.

## Integration changes

- Both root and `/v1` lifecycle controls keep their dedicated sleep credential when
  API authentication is enabled. Neither credential grants the other's authority.
- Authenticated `/health` stays minimal. The telemetry client obtains lifecycle
  state from protected metrics, so sleeping, draining and waking are not shown as ready.
- Nemotron's new MTP temperature scaling is pinned across wake. Adaptive costs are
  remeasured and the throughput estimate restarts; these affect proposal depth,
  while target sampling and the preserved prefix remain unchanged.
- The existing compact prefix format covers the new drafter. Scratch buffers,
  CUDA events and graphs are rebuilt after reload.

## Runtime qualification

Checkpoints and revisions are unchanged from the
[reference checkpoint table](model-sleep-reference-validation.md#checkpoints-and-settings):
`nvidia/Qwen3.8-27B-NVFP4` with `z-lab/Qwen3.8-27B-DFlash2`, and
`Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit`.
The environment uses single-device CUDA, unified memory, compute capability 12.1,
PyTorch 2.13.0+cu130 and CUDA 13.0.

All four configurations pass two complete cycles. Serial, drafted, warm and restored
outputs match exactly. Each cycle collects the old engine, preserves the frontend,
and reaches zero CUDA allocated and reserved bytes while asleep. Wake leaves prefix
tensors on disk until a matching request loads them, with no load failures or
memory-admission misses.

Direct prompts repeat token IDs 1–128 with a per-stream offset. Each reply has 32
tokens, with seed 1234 plus that offset, temperature 1, top-k 20, top-p 0.95 and EOS
stopping disabled. These probes test state restoration, not model quality.

| Configuration | Context / prompt | Prefixes | Snapshot MiB | Cached tokens per request | Median sleep / wake |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nemotron + adaptive MTP | 4096 / 2048 | 1 | 78.24 | 2047 | 8.54 / 25.04 s |
| Nemotron without MTP | 4096 / 2048 | 1 | 76.24 | 2047 | 8.60 / 21.22 s |
| Qwen + DFlash, one stream | 4096 / 2048 | 1 | 314.75 | 2047 | 12.38 / 30.19 s |
| Qwen + DFlash, two streams | 16384 / 8192 | 2 | 1397.53 | 8191 | 14.00 / 32.82 s |

Sleep timings include checkpoint hashing and transactional snapshot writing. Wake
includes checkpoint reload, graph construction, calibration where used, and snapshot
validation. Filesystem caches were warm. Allocator counters exclude the driver
context and some library allocations.

| Request | Cold first token | Prefix in memory | After wake, cycle 1 | After wake, cycle 2 |
| --- | ---: | ---: | ---: | ---: |
| Nemotron + adaptive MTP | 0.270 s | 0.021 s | 0.074 s | 0.071 s |
| Nemotron without MTP | 0.270 s | 0.016 s | 0.065 s | 0.070 s |
| Qwen + DFlash, one stream | 0.735 s | 0.095 s | 0.291 s | 0.278 s |
| Qwen + DFlash, two streams, request 1 | 2.795 s | 0.134 s | 0.944 s | 0.946 s |
| Qwen + DFlash, two streams, request 2 | 6.485 s | 0.133 s | 0.943 s | 0.946 s |

First-token timings include lazy disk restoration and any queueing. Serial reference
requests warm kernels before these measurements. The small number of repetitions
is not a performance guarantee.

## HTTP and conversation continuity

Both families pass with API authentication enabled and disabled. The authenticated
runs use a separate inference key and sleep secret, reject cross-use of credentials,
exercise root and `/v1` controls, keep health minimal and protect metrics.
Nemotron uses a 4096-token served context for authenticated qualification; Qwen uses
16384 tokens and two stream slots. The unauthenticated comparison runs use 16384
for both families.

Every HTTP run verifies streamed-request draining, 503 admission, 409 transition
conflicts, Origin and Level 1 refusal, discovery and stored-response reads while
asleep, and continuation of the same `previous_response_id` after wake. Output hashes
and history lengths remain unchanged. Nemotron reuses 58 of 59 continuation tokens;
Qwen reuses 55 of 56. Wake performs no eager prefix loads. Normal shutdown removes
the authenticated runs' snapshot files.

The drain probe uses the serial reference engine when cache preservation is required,
so an unrelated drafted prompt does not evict Nemotron's retained conversation before
sleep. Drafted replies are separately compared with serial replies on both sides of wake.

## Comparison with unmodified 0.6.5

Each family runs release → enabled before sleep → enabled after wake → release,
using the same checkpoint and a 16384-token context. Qwen has two stream slots and
DFlash; Nemotron uses serial request execution and adaptive MTP. A seeded 64-token
completion has identical serial/drafted output hashes across all four phases.

Decode uses `tools/bench_openai.py`, greedy sampling, 64 reply tokens and three
measured repetitions after warm-up. Cold prefill uses three distinct public Python
standard-library prompts at each of 2K and 8K tokens, a 1024-token warm-up and two
reply tokens. Actual prompt lengths differ by at most one token.

| Reference and metric | Release before | Enabled before sleep | After wake | Release after |
| --- | ---: | ---: | ---: | ---: |
| Nemotron, Fibonacci decode (tokens/s) | 138.40 | 139.17 | 138.80 | 138.46 |
| Nemotron, GPU explanation decode (tokens/s) | 151.46 | 151.94 | 151.48 | 150.85 |
| Nemotron, 2K cold first token (s) | 0.275 | 0.278 | 0.279 | 0.276 |
| Nemotron, 8K cold first token (s) | 1.127 | 1.139 | 1.139 | 1.134 |
| Qwen, Fibonacci decode (tokens/s) | 50.14 | 50.15 | 50.13 | 50.33 |
| Qwen, GPU explanation decode (tokens/s) | 40.75 | 40.81 | 40.69 | 40.95 |
| Qwen, 2K cold first token (s) | 0.790 | 0.904 | 0.786 | 0.917 |
| Qwen, 8K cold first token (s) | 2.895 | 2.909 | 2.883 | 2.908 |

These are per-phase medians. The alternating order reduces drift but does not establish
a statistical bound on small differences. Individual repetitions are retained in JSON.
Decode medians vary by less than 1% across phases for both families. Qwen 2K cold
prefill varies from 0.786 to 0.917 seconds, including a slower unmodified-release
phase. A fresh baseline/enabled repeat measured 0.786/0.794 seconds, with equal
output hashes. The larger slowdown did not repeat; the original samples are retained.

## Host validation and limits

The integration suite passed **365 tests, six skipped**. Focused telemetry and HTTP
checks passed **115 tests**; real CPU tensor codec, storage and lifecycle checks passed
**74 tests**; broader CLI, Responses, telemetry and capacity checks passed **166 tests,
287 skipped**. Counts overlap and must not be added.
The compact Nemotron GPU test passes with and without MTP, restoring into a fresh
runtime with different cost calibration and comparing identical and changed-suffix
prompts against serial generation.

The broad host run was stopped after 5m13s: **1,948 passed, 419 skipped, 15 failed,
and six subtests passed**. Thirteen failures require unavailable TUI or Metal support;
the other two GLM checks reproduce on unmodified `609ca41`. A separate release-focused
run passed 53 tests, skipped two, and had 15 Metal setup errors. The full suite is not
claimed green or complete. The sleep-specific changes add no lint diagnostics;
existing repository lint failures remain, including diagnostics imported from upstream.

Use the commands in the [0.6.4 reproduction section](model-sleep-reference-validation.md#reproduction-and-limits)
from the 0.6.5 integration. For authenticated HTTP, supply the separate inference key
through `--api-key-env VARIABLE` as described in the [usage guide](../model-sleep.md).

Qualification is limited to these pinned single-device CUDA checkpoints and retained
text prefixes in one process. Level 1, Metal sleep, tensor parallelism, image state
and restart persistence remain unsupported. Existing eviction rules apply: Nemotron's
two retained boundaries are not two independent conversation slots. Longer contexts,
cold-storage reloads, disk contention, aggregate throughput and GPU allocation-failure
injection remain unmeasured. Lazy restoration uses synchronous I/O.
