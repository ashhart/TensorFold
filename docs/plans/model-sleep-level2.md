# Level 2 weights sleep/wake prototype

Implement the first step of [the reviewed research](../research/model-offloading.md)
against TensorFold 0.6.3. Keep the same HTTP app, tokenizer, templates, model IDs,
metrics and Responses store alive while releasing and reconstructing the runtime.
The first adapter is single-device CUDA dense Qwen (`qwen3_5`), with existing serial,
drafted and concurrent paths. Other families, MLX, Level 1, TP and strict conversation
cache preservation remain subsequent adapters/milestones; refuse unsupported opt-ins.

Local implementation and focused verification are complete. See the
[validation receipt](../research/model-sleep-validation.md) for actual CUDA results,
host-suite limitations and links to full-checkpoint qualification, and the
[usage guide](../model-sleep.md) for the prototype API.

## Lifecycle and HTTP contract

- Opt in with `--enable-sleep-mode`; require a secret from `--sleep-token-env`
  (default `TENSORFOLD_SLEEP_TOKEN`). Never print its value. Existing serving behavior
  stays unchanged when disabled.
- `POST /sleep?level=2`, `POST /wake_up`, `GET /is_sleeping` require a bearer secret.
  Reject browser Origin requests. Level 1 is unsupported in this prototype.
- A shared lifecycle controller owns `awake`, `draining`, `sleeping`, `waking`, `error`.
  Request admission covers the entire POST, including preparation, image encoding,
  grouped decisions, Anthropic/Responses translation and token helpers. Internal
  same-thread reentry belongs to the already-admitted request.
- Sleep atomically closes admission and waits for accepted requests, including their
  worker completion. New POSTs receive structured 503; existing streams finish.
  `--sleep-timeout` bounds draining. Timeout before teardown returns to awake.
- Serialize state changes; completed repeats are idempotent and conflicting operations
  return 409. Health, model discovery, metrics and stored-response reads stay available.
  Control telemetry reports sleeping/waking separately from generation readiness.
- On restore failure, release partial runtime allocations and remain sleeping for retry.
  If teardown or failed-restore cleanup cannot recover, keep admission closed, report
  error and stop the service nonzero; recovery requires a fresh process.

## Runtime ownership

- Resolve target/draft checkpoints to local paths and retain a verifiable content
  identity and scalar reload options. Detect changed/missing backing files on wake.
- Preserve served context as explicit, precision, draft settings and stream settings;
  fail rather than shrink the context or choose different math.
- Stop/join the dense-Qwen worker after requests drain, synchronize CUDA, detach both
  app engine and vision, release all target/draft/cache/graph ownership and cached
  device-pointer tables, collect references, then empty freed CUDA allocations.
- Remove CLI locals retaining the original weights/engine during the serving loop.
  Loader callbacks must not capture those objects.
- Reconstruct through the existing family loader, verify runtime settings, then publish
  the engine and vision together before reopening admission. Keep CPU-only frontend
  objects and the app-keyed Responses store unchanged. CUDA prefixes are recomputed.
- Constructors failing after allocations must not retain tensors through saved exception
  tracebacks. Clear failure frames before measuring or cleaning partial loads.

## Implementation units

1. Shared lifecycle controller and focused thread/race/failure tests; this defines the
   stable interface used by the HTTP and runtime adapter units.
2. Authenticated HTTP integration, health/control reporting and route/stream/Responses
   regression tests. Own the HTTP modules, health consumers and related tests.
3. Dense-Qwen CUDA adapter, CLI opt-in/capability checks, reload identity and ownership
   tests. Own runtime loader/adapter files, CLI options and related tests.
4. Integrated verification, usage documentation and a repeatable hardware qualification
   tool. Inspect the complete diff; run meaningful regression checks once integrated.

## Verification and completion

Tests first where practical: rejected requests must not reach preparation; admission
must remain closed throughout drain/release/load; a timeout leaves the original runtime
usable; failed wake cleans allocations and permits retry; fatal cleanup never reopens
admission; repeated cycles release weak references. Exercise real HTTP routes on both
handler implementations, request-body framing, authorization, stream completion and
stored `previous_response_id` continuation across replacing the runtime.

Use host fakes for lifecycle races and injected failures, without claiming they prove
GPU reclamation. On available local CUDA, measure allocated/reserved and process memory
before sleep, asleep and after wake; compare pre/post token IDs and repeated cycles.
For a real checkpoint, qualify drafted versus serial and concurrent versus solo, and
record cold-prefill/decode before and after on the same hardware. Run the repository
host suite and directly affected CUDA checks where dependencies permit.

The local implementation is complete when supported opt-ins work end to end, regression
tests pass, the full diff is reviewed, and the evidence distinguishes host checks, actual
GPU measurements and pending Spark/Metal qualification. Do not claim hardware or model
combinations as qualified without results. Keep TP independent of later KV preservation.
