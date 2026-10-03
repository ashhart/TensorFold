# Model sleep/wake (prototype)

Level 2 sleep releases model weights and device caches while keeping the HTTP server
and its conversation history alive. Wake reloads the existing local checkpoint with
the original served context, precision, drafter and concurrency settings. It does not
write another copy of the weights to disk.

The reference adapters support single-device CUDA dense Qwen (`qwen3_5`), including
its serial and concurrent engines, and Nemotron-H (`nemotron_h`), with its serial
request execution and optional integrated MTP head. Sleep is opt-in. Level 1 (GPU to
CPU), other families, Metal and tensor parallelism are not implemented. The
[0.6.4 reference validation](research/model-sleep-reference-validation.md) covers
the pinned Qwen NVFP4 and Nemotron affine checkpoints, drafting, unified-memory
reclamation, disk prefixes and HTTP conversation continuity. Other checkpoints and
formats need their own qualification. Earlier receipts cover
[synthetic runtimes](research/model-sleep-validation.md),
[Qwen weights](research/model-sleep-cuda-validation.md) and
[Qwen prefix preservation](research/model-sleep-cache-validation.md).

## Start and control

Set a secret in the server's environment, then start the supported model with sleep
enabled. The default environment variable is `TENSORFOLD_SLEEP_TOKEN`; change its name
with `--sleep-token-env`. `--sleep-timeout` limits how long sleep waits for accepted
requests to finish (default 120 seconds).

```bash
export TENSORFOLD_SLEEP_TOKEN="$(openssl rand -hex 32)"
tensorfold serve /path/to/qwen-checkpoint --backend cuda \
  --drafter /path/to/dflash2-checkpoint --enable-sleep-mode
```

For Nemotron-H, the checkpoint includes its MTP head; no separate `--drafter` is needed:

```bash
tensorfold serve /path/to/nemotron-checkpoint --backend cuda \
  --enable-sleep-mode --sleep-cache-dir /path/to/cache-storage
```

Use `--no-drafts` to disable drafting for either family. The checkpoint files must remain
available and unchanged throughout the server's lifetime. The prototype verifies
their full SHA-256 content before releasing the runtime and when reloading it;
verification reads the entire checkpoint and contributes to transition latency.

The following calls require the same secret in the caller's environment:

```bash
curl -fsS -X POST 'http://127.0.0.1:8080/sleep?level=2' \
  -H "Authorization: Bearer $TENSORFOLD_SLEEP_TOKEN"
curl -fsS 'http://127.0.0.1:8080/is_sleeping' \
  -H "Authorization: Bearer $TENSORFOLD_SLEEP_TOKEN"
curl -fsS -X POST 'http://127.0.0.1:8080/wake_up' \
  -H "Authorization: Bearer $TENSORFOLD_SLEEP_TOKEN"
```

These routes also accept a `/v1` prefix. They return 404 when sleep is disabled,
401 for an invalid secret, and 403 for a browser `Origin` header. The secret protects
these lifecycle routes; it does not add authentication to the existing inference API.

Responses include `state`, `ready`, `is_sleeping`, `level`, `active_requests`,
`last_error`, and CUDA `memory.allocated_bytes` / `memory.reserved_bytes`.
`GET /health` keeps returning 200 for a live sleeping server; use `ready` for inference
readiness. Prometheus exposes `tensorfold:model_ready` and
`tensorfold:model_lifecycle_state{state="sleeping"}`; control telemetry displays the
lifecycle state.

## Requests and conversations

Sleep closes admission and drains all accepted requests, including preparation and
open streams. New inference and token-helper POSTs receive 503 while draining,
sleeping or waking. Health, model discovery, metrics and stored-response reads remain
available. Wake must complete before sending the next generation request.

The same app, tokenizer, templates, served model IDs, counters and Responses store
survive. A client can continue a stored `previous_response_id` after wake, within the
store's existing retention limits. Chat Completions clients continue by sending their
message history as usual. This is same-process preservation, not restart persistence.

By default, sleep discards runtime prefix caches and the next turn prefills its
history again. Add `--sleep-cache-dir /path/to/cache-storage` to preserve reusable
text prefixes on disk. This option requires `--enable-sleep-mode`.

```bash
tensorfold serve /path/to/qwen-checkpoint --backend cuda \
  --drafter /path/to/dflash2-checkpoint --enable-sleep-mode \
  --sleep-cache-dir /path/to/cache-storage
```

Each snapshot includes attention K/V, convolution history, recurrent state, positions,
and compatible drafter context. Only committed rows are saved, with their original
bits. Qwen preserves its gated-delta state and optional DFlash context; Nemotron-H
preserves Mamba state, its retained hidden row, and optional MTP attention state.
Sleep writes and validates the complete selected set before releasing any
runtime state. A failed write, including a full disk, refuses sleep and keeps the
original runtime awake. Snapshot identity binds the files to the current process,
checkpoint content, runtime, device capability and precision/context settings.

Wake validates the snapshots but leaves their tensors on disk. A matching request
loads the longest valid strict prefix under memory admission; unrelated prefixes
stay on disk. Reply tokens and the new turn may still need prefill. A prefix that
cannot fit, or whose file becomes unusable after wake, is a cache miss. Corruption
detected during wake leaves the model sleeping for retry.

The snapshot count follows the engine's existing prefix-cache capacity. Every live
retained prefix is saved; older disk-only entries fill any remaining slots. Prefixes
evicted before preservation are unavailable, and image state is not saved.
Nemotron clears its live retained prefixes when an unrelated drafted prompt starts;
sleep cannot preserve a conversation prefix already displaced by that request. Control
responses expose `cache.saved_prefixes`, `snapshot_bytes`, `omitted_prefixes`,
`unavailable_prefixes`, `loaded_prefixes`, `load_failures` and `memory_misses`.
The unavailable count covers saved files found unusable during a later save or
load; it cannot count conversations evicted before sleep. An old disk-only prefix
that is already missing or corrupt is omitted from the next generation. A failure
writing the new generation still refuses sleep and preserves the live runtime.

Files live in a private process directory under the configured location. They contain
conversation tokens and model state. A new generation replaces the old one only after
it succeeds, so saving can temporarily require space for both generations. Tensor
staging uses chunks of at most 8 MiB. Normal shutdown removes this process's files;
an abrupt exit can leave its directory for manual removal. These files do not provide
conversation or cache restoration after a process restart.

Repeated completed sleep/wake calls are idempotent. Concurrent transitions return 409.
A drain timeout or checkpoint preflight refusal leaves the original runtime awake.
A failed reload returns 503 and remains sleeping for retry after partial allocations
are cleaned up. An unrecoverable release or cleanup error closes admission, sends an
error and stops the service; restart the process to recover.

## Qualification

Run the tool with a CUDA-enabled TensorFold environment from the repository root:

```bash
PYTHONPATH=src python tools/qualify_sleep.py --synthetic --output sleep-serial.json
PYTHONPATH=src python tools/qualify_sleep.py --synthetic --parallel 2 --output sleep-concurrent.json
PYTHONPATH=src python tools/qualify_sleep.py --synthetic --synthetic-draft \
  --parallel 2 --seed 1234 --output sleep-drafted.json
PYTHONPATH=src python tools/qualify_sleep.py --model /path/to/qwen-checkpoint \
  --draft /path/to/dflash2-checkpoint --parallel 2 --cycles 3 --output sleep-model.json
PYTHONPATH=src python tools/qualify_sleep.py --model /path/to/qwen-checkpoint \
  --draft /path/to/dflash2-checkpoint --parallel 2 --preserve-cache --seed 1234 \
  --cycles 3 --output sleep-cache.json
PYTHONPATH=src python tools/qualify_sleep.py --model /path/to/nemotron-checkpoint \
  --preserve-cache --seed 1234 --cycles 3 --output sleep-nemotron.json
```

The tool constructs a real CUDA engine, compares serial and concurrent/drafted token
IDs, measures allocated/reserved memory and Linux process RSS, checks old engine
collection, then compares the same prompts after each wake. It records checkpoint
hashes and timing in JSON. Synthetic mode exercises a random two-layer affine model;
`--synthetic-draft` uses 64 layers of width 2048 and a matching random DFlash2 model.
Neither establishes pretrained-model quality or full-model memory behavior.
CUDA allocator counters exclude the driver context and some library allocations.
The tool detects the family from the checkpoint configuration. Nemotron-H uses its
integrated MTP by default; add `--no-drafts` to qualify it without MTP. Its CUDA
engine serializes requests, so the direct qualification tool requires `--parallel 1`.

The [CUDA family guide](recipes/adding-a-cuda-family.md#sleepwake-reference-implementations)
describes the shared lifecycle and each reference adapter's extension points.

With a sleep-enabled server running and its bearer secret in the client's environment,
exercise HTTP stream draining and stored conversation continuation with:

```bash
python tools/qualify_sleep_http.py http://127.0.0.1:8080 MODEL --output sleep-http.json
```

For a server started with `--sleep-cache-dir`, add `--require-cache` to check that
stored-response continuations reuse the same cached-token count after wake.

The [reference results](research/model-sleep-reference-validation.md) cover both pinned
families on 0.6.4. The [earlier Qwen results](research/model-sleep-cuda-validation.md)
retain the original comparisons against 0.6.3.
For other configurations, run the benchmarks from
[Contributing](../CONTRIBUTING.md#the-receipt) against the last release in the same
environment. Longer prompts, aggregate concurrent throughput,
cold-storage reload latency, image input and other checkpoint formats
remain to be measured.
