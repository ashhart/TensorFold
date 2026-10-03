# Model sleep/wake (prototype)

Level 2 sleep releases model weights and device caches while keeping the HTTP server
and its conversation history alive. Wake reloads the existing local checkpoint with
the original served context, precision, drafter and concurrency settings. It does not
write another copy of the weights to disk.

The first adapter supports single-device CUDA dense Qwen (`qwen3_5`), including its
existing serial and concurrent engines. It is opt-in. Level 1 (GPU to CPU), other
families, Metal and tensor parallelism are not implemented. Qualification includes
[synthetic runtime checks](research/model-sleep-validation.md) and
[full-checkpoint NVFP4 checks](research/model-sleep-cuda-validation.md) with a drafter
and unified-memory reclamation. Other checkpoint formats still need hardware qualification.

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

Use `--no-drafts` when serving without a drafter. The checkpoint files must remain
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

This milestone releases KV, recurrent and drafter prefix caches. The next turn
prefills its history again. Preserving those reusable caches across sleep/wake is a
separate milestone described in the [research](research/model-offloading.md).

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
```

The tool constructs a real CUDA engine, compares serial and concurrent/drafted token
IDs, measures allocated/reserved memory and Linux process RSS, checks old engine
collection, then compares the same prompts after each wake. It records checkpoint
hashes and timing in JSON. Synthetic mode exercises a random two-layer affine model;
`--synthetic-draft` uses 64 layers of width 2048 and a matching random DFlash2 model.
Neither establishes pretrained-model quality or Spark behavior.
CUDA allocator counters exclude the driver context and some library allocations.

With a sleep-enabled server running and its bearer secret in the client's environment,
exercise HTTP stream draining and stored conversation continuation with:

```bash
python tools/qualify_sleep_http.py http://127.0.0.1:8080 MODEL --output sleep-http.json
```

For release qualification also run the HTTP exactness, cold-prefill and decode
benchmarks from [Contributing](../CONTRIBUTING.md#the-receipt) against the last release
on the same hardware. Long prompts, drafted execution, image input, NVFP4/EXL3 packs
and unified-memory reclamation need their own measurements.
