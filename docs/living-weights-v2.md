# Living Weights v2: guarded learning controller (experimental)

This change adds a dependency-free Python orchestration/evaluation layer for the **existing native Living Weights pocket learner**. It does not replace or generalize the native model weight trainer and does **not** make dense, MoE, hybrid or unsupported TensorFold models trainable. The boundary is deliberately explicit: the orchestration/quality contract is architecture-independent, whereas the model's actual parameter update and safe persistence remain the responsibility of `--slide`.

## Why this exists

Native TensorFold can write learned facts into safetensors, persist them across restarts, and delete selected facts. Its documented failure modes include paraphrase misses, near-subject leakage, interference and shrinking accuracy as fact count grows. The controller adds source attestation, positive/negative/anchor probing, regression checks against earlier committed facts, local write serialization, conflict rejection, durable receipts, and conservative native delete on a failed post-training gate. These are **additional checks**, not a substitute for the native learner's own held-out checks.

## How to run

Use an already running, authenticated TensorFold server, started with `--slide` and a copied supported checkpoint. Never teach a Hugging Face cache snapshot directly. Native hardware/model support is still constrained as documented in `docs/living-weights.md`.

```sh
PYTHONPATH=tools python3 -m living_weights_v2 plan docs/living-weights-v2-manifest.example.json
PYTHONPATH=tools python3 -m living_weights_v2 teach docs/living-weights-v2-manifest.example.json \
  --base-url http://127.0.0.1:8080 --api-key-file "$HOME/models/slide-key"
PYTHONPATH=tools python3 -m living_weights_v2 evaluate docs/living-weights-v2-manifest.example.json \
  --api-key-file "$HOME/models/slide-key"
PYTHONPATH=tools python3 -m living_weights_v2 status
python3 -m pytest -q tools/tests -k living_weights_v2
```

The example is fictional. Edit it for a real, user-approved lesson and correct probes. `asserted_by_user` means approval of the supplied claim; the tool does not automatically verify a document, URL, or third-party claim. The manifest hash is a provenance fingerprint, not a content authenticity signature.

The read-only `evaluate` command uses standard `/v1/models` and `/v1/chat/completions`, and can probe any endpoint implementing both routes. It has no weight-training capability and scores conservative affirmative answer presence rather than semantic correctness.

The controller expects one atomic statement per `facts[]` item and sends it separately to the current native `/v1/slide/learn` endpoint. Long documents must be decomposed into individually approved claims first. A native learner may extract more than one fact from a statement; the controller then refuses to certify ownership, stops and marks recovery required.

## Transaction and safety model

1. Check the server model list, native fact graph, already committed facts, and local unresolved writes.
2. Reject a contradictory subject/relation, repeated known fact, and graph entries left `deleting`.
3. Probe positive questions, hard negatives, unrelated anchors on the current model.
4. If all positive answers are already correct, report `already_answers` without writing any weights. This is *not* evidence that the model learned the fact.
5. Write an `inflight` intent to a local SQLite WAL ledger and call the native SSE learning route.
6. Require a completed stream, exactly one native `learned.recalled=true`, an explicit `saved` event, and attribution of exactly one new graph fact.
7. Probe every positive, negative, anchor, and previously committed positive question. Also verify the previously committed facts were healthy before beginning the write. All must pass. Conservatively compare stable replies on negatives and anchors exactly after case/whitespace normalization. This can reject harmless stylistic drift.
8. If post-gates fail, ask native batch deletion to remove the attributed fact. Report `rolled_back` only after every `forgot.deleted=true`, graph state `deleted`, and the pre-edit probe outputs are restored. If attribution, graph reads or deletion fail, mark `needs_recovery` and refuse more writes for that model.

`committed` certifies **only the observed test questions**, not generalized semantic learning. It is not a formal proof that no unrelated output changed. The native delete is not byte-exact inverse training. It may leave interference from later refits or 4-bit group rounding; check remaining facts and performance on real GPU hardware after recovery.

The process lock serializes writers using the same local ledger on POSIX hosts. It cannot prevent learning from other clients, other machines, or other ledger files. The native server's own one-at-a-time learning gate remains authoritative. Concurrent external writers can make the graph attribution ambiguous and force manual recovery.

An unhandled crash after recording `inflight` deliberately blocks further work. Inspect the native graph, journal and checkpoint rather than blindly repeating the fact. There is intentionally no `--force`, automatic ledger reset, or unfenced retry mechanism.

## Architecture portability contract

The controller is independent of model tensor shapes, expert layouts, kernel backends and quantization because it interacts through three narrow native capabilities: deterministic chat evaluation, fact-graph inspection, and transactional *best-effort* learning/deletion SSE APIs. Models without all three **must refuse write attempts**. It is not an API for generic gradient training.

To support dense and new MoE/hybrid families natively, follow a separate implementation program:

- Add per-family trainability descriptors (`supports_gradients`, legal writable tensors, quantizer, architecture revision, maximum update budget, known rollback semantics), rather than detecting support from model names.
- Introduce staged checkpoint updates with a native transaction ID, base-checkpoint digest and compare-and-swap commit; publish on-disk manifests that survive power loss.
- Add a general objective over token log-likelihood, paraphrase invariance, locality loss and retained-task rehearsal; keep a non-training verifier to avoid a model grading only itself.
- Implement real backend VJPs/gradients for dense MLP, shared experts, MoE routing and state-space components separately, qualified on Metal/CUDA/HIP with numerical checks. Never silently train a subset and claim complete family support.
- Test F16/BF16/FP32 plus actual deployment quantizers, including post-roundtrip regression tests. Persist native safetensors only after a staging verifier passes.
- Measure dense factual recall, cross-subject disambiguation, multi-hop reasoning, procedural skill transfer, deletions and out-of-domain regression from clean copies at 1, 10, 100 and 1,000 facts. Include seed variance, thinking modes, restart, backend transfer, latency, RAM/VRAM and write amplification.

These remaining items are not implemented in this PR. Until then, the new controller is an explicit, bounded quality/safety improvement over existing native `--slide` only.

## Operational constraints

- Python 3.10+ and a local POSIX lock. No third-party Python packages.
- Loopback IP is required by default; `--allow-remote` explicitly enables authenticated remote HTTP(S). Use TLS if remote.
- HTTP redirects are refused, including redirects on the same origin, so the key stays with the configured server.
- The native `/v1/slide/graph` JSON must expose a `facts` array with positive integer IDs and states. Unknown schema fails closed.
- Graph comparisons ignore native recheck counters/verdicts and computed links; other changes to existing facts require recovery.
- One controller instance per ledger/checkpoint; avoid direct concurrent `/v1/slide/learn` requests.
- Prompts are tested at temperature 0 with a fixed token cap; reasoning-on and adversarial role/system-message tests are future native integrations.
- Model capability remains native-authoritative. A `501` is an unsupported model, not something the controller attempts to bypass.
- SQLite receipts can contain personal fact text and model answers. Keep the ledger private, rotate it under a managed retention policy, and do not commit it to source control.
- No cross-device numerical or performance qualification was performed in this environment.

## Recommended follow-on PR sequencing

1. Native staged-edit transaction and recovery protocol, with deterministic commit/abort and crash fault injection.
2. Native architecture descriptors and trainability refusal matrix for dense, MoE, state-space and hybrid models.
3. General differentiable objective and retain/replay stream; numerical gradient verification for every backend.
4. Progressive long-document/curriculum learning and capacity planning, with measured 100/1,000-fact performance.
5. GUI learning observability, per-fact lineage/provenance and inspectable model-level evaluation receipts.
