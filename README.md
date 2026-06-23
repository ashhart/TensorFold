# TensorFold Runtime

TensorFold Runtime is an exact-first local inference runtime for hosting standard transformer model files with less resident memory pressure.

The current build covers the V0 foundation and the first V1 MLX streaming path from `plan-origin.md`:

- Parse `.safetensors` metadata without loading tensor data.
- Build a layer-aware manifest with tensor offsets, dtypes, shapes, and sizes.
- Lazily expose tensor bytes through `mmap`.
- Track tensor access timing and cache residency decisions.
- Build a conservative memory-budget streaming plan.
- Provide a CLI for inspecting real model shards.
- Stream GPT-OSS layers into an MLX model shell without making all model weights resident.
- Phase-load embeddings and output heads so the pinned set can stay tiny.
- Select retained layers under a resident-weight budget.

It does not change model weights, quantize them, or modify the transformer architecture. The goal is to create the memory virtualization layer first, then add layer streaming, KV-cache virtualization, and hot/cold tensor intelligence on top.

## Quick Start

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m tensorfold doctor
PYTHONPATH=src python3 -m tensorfold demo create /tmp/tensorfold-demo --force
PYTHONPATH=src python3 -m tensorfold inspect /tmp/tensorfold-demo/toy.safetensors
PYTHONPATH=src python3 -m tensorfold pack /tmp/tensorfold-demo/toy-moe --out /tmp/tensorfold-demo/toy-packs
```

## Serve (OpenAI-compatible)

Host a model behind an OpenAI-style endpoint with low-resident streaming:

```bash
PYTHONPATH=src python3 -m tensorfold serve /path/to/qwen3_5_moe-model --port 8421
```

This exposes `GET /v1/models` and `POST /v1/chat/completions` (JSON and SSE streaming), with greedy or temperature sampling, EOS-aware stopping, and the model's chat template applied. Both `qwen3_5_moe` and `gpt_oss` checkpoints are supported. For Qwen thinking templates, reasoning is disabled by default (`--enable-thinking` re-enables it); for GPT-OSS harmony output, the analysis channel is parsed out of `content` into `reasoning_content` and never streamed, and `--reasoning-effort low|medium|high` (default low) controls how much the model deliberates. The server keeps a KV/SSM cache checkpoint at the end of rendered conversation history, so a follow-up turn that extends the same conversation only prefills the new suffix — `usage.prompt_tokens_details.cached_tokens` reports the reuse. Requests are served one at a time; all MLX work (including model construction) stays on one dedicated inference thread because MLX streams are thread-bound. `--exact-mode` selects the exactness contract (`target-verified` default, or `exact-strict` for the bitwise mode); the active mode is echoed on each chat response as `exact_mode` (see the speculative decoding section).

### Speculative decoding

Greedy requests use speculative decoding by default: a drafter proposes several tokens, and one streamed 40-layer pass verifies the whole block. Because the streaming cost per pass is nearly flat in token count, every token a draft gets right is a token that did not cost its own pass.

There are two exactness modes, selected with `--exact-mode`:

- **`target-verified` (default)**: every emitted token came from a target-model verification path. This is *not* a bitwise guarantee — multi-token verify passes evaluate under chunked numerics, so a position that is a near-tie under single-token decoding can resolve differently. Those near-tie differences vs the single-token baseline are documented and counted in the `speculative` telemetry (`near_tie_events`), and a `--draft-margin` rejects near-tie agreements the chunked numerics could otherwise steer. The outputs remain valid greedy decodes.
- **`exact-strict`**: the honest bitwise mode (emitted tokens are bitwise-identical to single-token greedy). On GPT-OSS this runs speculation on a temporal sliding cache (proven bitwise exact, forensics `max_abs=0.0`); the requested `--sliding-cache` is overridden to `temporal`. On Qwen and other architectures it runs speculation off, because SSM/chunk exactness there is unproven and only single-token greedy is bitwise exact.

Two drafters:

- `prompt-lookup` (default, free, no extra model): a token-level LZ matcher with an incremental n-gram index over the whole context (prompt, tool/file content, generated text). It proposes the continuation of the longest recent suffix match, shrinks its proposals after repeated rejection, and reports `matches_attempted`/`matches_found`/`average_match_length`. Excellent for code, edits, quoted text, and structured output — exactly the agentic/opencode workload where the model reproduces spans of its input.
- `model` (`--draft model --draft-model PATH`): a small fully-resident same-tokenizer MLX model drafts tokens. Helps general prose where there is no literal repetition, at the cost of the draft model's resident memory. The draft tokenizer must match the target (verified at load).

Speculative state is transactional: the cache is snapshotted before each verify block and either committed or rolled back (`CacheTransaction`) — no speculative path may leave durable generation state mutated without the verifier committing it. The standing correctness contract is enforceable on the real model:

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py verify-speculative /path/to/model
```

Run it after any change to drafting, caching, or rollback. The exactness contract has two tiers, stated precisely:

- **Hard invariant**: a garbage drafter (nothing ever accepted) must reproduce baseline greedy token-for-token. This isolates snapshot/rollback correctness — any violation is a state-management bug.
- **Drafted runs**: experimental. Multi-token verify/refeed passes evaluate under chunked numerics, and the bonus/correction token after such a pass is not margin-gated; on GPT-OSS the `copy-edit` suite case diverged from single-step greedy at normal margins, while `--draft-margin 999` restored exactness by accepting no drafts. Treat prompt-lookup speculation as a measured workload-specific accelerator, not a blanket correctness path.

The opencode-style workload suite (rewrite, patch, complete, summarise-to-JSON, copy-edit, test generation) measures speedup and accepted-tokens-per-pass per workload:

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py speculative-suite /path/to/model
```

Qwen3.6-35B results (96 max tokens, ~2GB profile, LZ prompt-lookup drafter; every workload measured byte-exact against single-token greedy in these runs — a measured observation under `target-verified`, not a bitwise guarantee on this hybrid-SSM model):

| Workload | off | on | Speedup | Accepted/pass | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| rewrite-file | 30.1s | 13.1s | 2.30x | 7.25 | 94% |
| apply-patch | 28.3s | 14.6s | 1.94x | 4.50 | 89% |
| copy-edit | 32.1s | 17.0s | 1.89x | 4.89 | 90% |
| complete-function | 18.2s | 13.2s | 1.37x | 1.76 | 100% |
| generate-tests | 36.3s | 29.6s | 1.23x | 1.52 | 58% |
| summarise-json | 18.4s | 19.4s | 0.95x | 1.00 | 15% |

Mean 1.61x; the adaptive proposal shrink caps the worst (novel-text) case near break-even. The server reports per-request telemetry in the chat response under `speculative` and in its logs: `draft_policy`, `accepted_tokens_per_pass`, `acceptance_rate`, drafter match stats, `zero_accept_rounds`, `disabled_rounds`, `near_tie_events`, and `estimated_regression_avoided_passes` (gated rounds the adaptive draft gate skipped to keep speculation from becoming a regression).

Sidecar drafting status: Qwen3.5-0.8B shares the 35B's tokenizer (verified, 248k vocab) and runs as `--draft model`, but measured only ~1.1 accepted tokens/pass at 14-16% acceptance against this finetuned target — a net slowdown, below the ~2 tokens/pass usefulness bar. A distilled-from-target sidecar is the open path for prose speedups; prompt-lookup already gives prose a modest lift (1.2x on the story benchmark) and code its 2x+.

Measured on the 35B at the ~2GB profile, served code-echo (87 tokens, warm), greedy:

| Mode | Wall | tok/s | Output |
| --- | ---: | ---: | --- |
| `--draft off` | 28.9s | 3.0 | reference |
| `--draft prompt-lookup` | 13.9s | 6.3 | byte-identical |

Speedup tracks how repetitive the output is: code/echo/edit workloads see 2x+; freeform prose with the prompt-lookup drafter falls back toward 1x (no literal n-gram to match) and is where a draft model earns its keep. Partial draft acceptance rolls the hybrid KV/SSM cache back to a pre-verify snapshot and re-feeds only the accepted prefix, so correctness holds on the Qwen linear/full-attention mix.

Point any OpenAI-compatible client at `http://127.0.0.1:8421/v1`. For opencode, add a custom provider to `opencode.json` (see opencode's provider docs for the exact schema in your version):

```json
{
  "provider": {
    "tensorfold": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "TensorFold",
      "options": { "baseURL": "http://127.0.0.1:8421/v1" },
      "models": { "Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-8bit": {} }
    }
  }
}
```

Expectations: decode runs at streamed-hosting speed (~3 tok/s on the 35B at the ~2GB profile), so short chat turns feel fine and long generations are slow; the resident budget is set with `--resident-budget` (default 2GiB). Currently `model_type=qwen3_5_moe` only. The history-checkpoint prefill split can flip greedy near-ties versus the single-pass benchmark path on this hybrid-SSM architecture; outputs remain valid greedy decodes.

## Benchmarks

### Verified Internal Benchmark Artifacts

These are internal benchmark artifacts from TensorFold development runs. They
are included to make the current performance envelope visible; each row states
the scope and exactness mode so the number is not accidentally over-claimed.

| Model/profile | Hardware | Speed | Memory | Scope |
| --- | --- | ---: | --- | --- |
| Qwen3.6-35B-A3B MLX 4-bit, guarded hard-quarter profile | 24GB Mac mini | 22.122 tok/s | 2.568GB RSS, 1.060GB resident weight peak | Exact guarded generation, 56 generated tokens |
| Nemotron-3-Ultra-550B-A55B MLX 4-bit, deferred fixed hot-set + cold buddy substitution | M3 Ultra, 256GB unified memory | ~16.5-16.7 tok/s | fixed hot-set path, target peak below 190GB RSS | Fresh near-exact generation, coherent held-out prompts |
| Nemotron-3-Ultra-550B-A55B MLX 4-bit, speculative block-verify ceiling | M3 Ultra, 256GB unified memory | ~22.8 tok/s | fixed hot-set path, target peak below 190GB RSS | Verifier path ceiling with oracle draft tape; not a fresh real-drafter benchmark |

The Nemotron Ultra ~22 tok/s result is useful because it shows the verifier path
can run there on M3 Ultra hardware. It should not be quoted as a general
Nemotron throughput claim until a non-oracle drafter reproduces it on held-out
prompts. The honest citable fast Nemotron row today is the fresh near-exact
~16.5-16.7 tok/s path.

Discover cached Hugging Face models:

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py discover
```

Inspect model metadata without loading tensor payloads:

```bash
PYTHONPATH=src python3 -m tensorfold inspect /path/to/model-dir --json
```

Run a normal MLX generation benchmark:

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py mlx /path/to/model-dir --max-tokens 8
```

Load only one planned TensorFold layer into MLX arrays:

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py mlx-selective /path/to/model-dir --layer 0
```

Run a deduped pinned-plus-layer streaming cycle into MLX arrays:

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py mlx-stream /path/to/model-dir --layers 2 --evaluate
```

Stream weights into an instantiated MLX model shell:

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py mlx-model-stream /path/to/model-dir --layers 1 --evaluate
```

Run a GPT-OSS streaming forward/next-token prototype:

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py mlx-forward-gpt-oss /path/to/gpt-oss-model --prompt Hello
```

Generate multiple GPT-OSS tokens with streamed layers and KV cache:

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py mlx-generate-gpt-oss /path/to/gpt-oss-model --prompt Hello --max-tokens 2
```

Generate many GPT-OSS requests in one streamed pass per layer:

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py \
  mlx-generate-gpt-oss-batch /path/to/gpt-oss-model \
  --prompt "Write a small story about a robot who learns to paint." \
  --batch-size 64 \
  --max-tokens 32 \
  --loader-backend native \
  --pin-policy phase \
  --resident-budget 2GiB \
  --defer-cache-clear \
  --final-cache-clear
```

Generate Qwen3.5-MoE tokens with streamed layers and KV cache:

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py mlx-generate-qwen3-moe /path/to/qwen3_5_moe-model --prompt Hello --max-tokens 2
```

Run the current low-resident optimized profile:

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py \
  mlx-generate-gpt-oss /path/to/gpt-oss-model \
  --prompt Hello \
  --max-tokens 2 \
  --loader-backend native \
  --pin-policy phase \
  --resident-budget 2GB \
  --defer-cache-clear \
  --final-cache-clear
```

GPT-OSS 20B now streams selectively like the Qwen path: each ~875 MB layer is ~846 MB of 32-expert bank, so TensorFold streams the ~28 MB base (attention, norms, router) and loads only the top-4 expert rows the router selects. Fresh single-request story measurements on the local GPT-OSS 20B MLX model:

| Mode | Time | tok/s | MLX/Metal peak | Output |
| --- | ---: | ---: | ---: | --- |
| Raw full-resident MLX, batch 1, generation only | 1.60s / 128 tok | 80.1 | 22.25 GB | reference |
| TensorFold selective streaming, hot set 0 | 47.3s / 128 tok | 2.7 | 1.81 GB | token-exact |

Historical progression: the original full-layer streaming path moved ~21 GB per forward pass (~0.5 tok/s); selective experts cut that to ~3.2 GB (~2.3 tok/s); persistent shard handles (mmap/header state now lives for the loader's lifetime instead of being rebuilt per load call) reached ~5.4 tok/s on the earlier 24-token warm profile. The fresh 128-token story table above is the current comparison point.

Measured GPT-OSS expert locality (128-token generation): the top-8 of 32 experts per layer serve 81% of selections (top-12: 92%, top-16: 97%) with 53% consecutive-token overlap. Zero-copy Metal mapping is deprioritized: with persistent handles the copy path already runs near memory bandwidth.

That locality is now exploited by **hot expert residency** (`--expert-hot-set N`): each layer keeps up to N expert rows resident, lazily grown from rows that misses load anyway (zero warm-up IO), then refreshes the resident set toward the most frequently selected rows using only rows that were already loaded for misses. The current memory/speed dial on a 128-token story is:

| `--expert-hot-set` | tok/s | MLX/Metal peak | resident row hit |
| ---: | ---: | ---: | ---: |
| 0 (pure streaming) | 2.4-2.7 | 1.81 GB | - |
| 8 | 8.9 | 7.34 GB | 75% |
| 12 | 12.0 | 9.83 GB | 86% |
| 16 | 7.8 | 12.08 GB | 91% |

For reference, full-resident native MLX is 80 tok/s at batch 1 and ~22.25 GB. Hot-set 12 is the current single-request sweet spot; hot-set 16 raises hit rate but falls off a memory/allocator cliff. Telemetry (`expert_prefetch` in the result JSON) reports row hit rate, resident expert bytes, overflow passes, and refreshes.

Throughput mode is the stronger low-memory result: batching shares each streamed layer base and the per-layer union of selected experts across many requests, so total tokens/sec rises while resident weights stay small. Same story prompt, 32 generated tokens per request:

| Mode | Batch | Total tok/s | Per-request tok/s | MLX/Metal peak | Resident weight peak | Exact vs raw batch |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TensorFold pure streaming, hot set 0 | 32 | 73.6 | 2.30 | 2.10 GB | 1.77 GB | not checked |
| TensorFold pure streaming, hot set 0 | 64 | 138.6 | 2.17 | 2.39 GB | 1.77 GB | yes |
| TensorFold hot set 4 | 32 | 101.2 | 3.16 | 5.01 GB | 4.79 GB | not checked |
| TensorFold hot set 8 | 32 | 128.8 | 4.02 | 7.55 GB | 7.33 GB | not checked |
| TensorFold hot set 12 | 32 | 144.5 | 4.52 | 10.03 GB | 9.81 GB | not checked |
| Raw full-resident MLX | 64 | 1475-1511 | ~23 | 23.4 GB | all weights | reference |

The honest read: TensorFold already clears 80 aggregate tok/s under a tiny resident-weight budget by batching, and it does so token-exact against raw MLX in the pure streaming batch-64 check. It is not faster than raw full-resident MLX when the machine can afford ~23 GB; raw MLX batch throughput is an order of magnitude higher. The remaining gap is weight movement and per-pass orchestration, not GPU math.

Current Qwen3.5-MoE 35B smoke result on the local cache, using the raw prompt `Hello` and 2 generated tokens:

| Mode | Output | Time | Resident weight peak | MLX/Metal peak |
| --- | --- | ---: | ---: | ---: |
| Normal raw `mlx_lm.generate` | `,\n\n` | 7.16s | all weights resident | 36.872 GB |
| TensorFold phase streaming | `,\n\n` | 5.88s including final cache clear | 0.896 GB | 0.965 GB |
| TensorFold phase streaming, `--resident-budget 2GB --final-cache-clear` | `,\n\n` | 6.64s including final cache clear | 1.791 GB | 1.861 GB |

Generate the same prompt for multiple requests at once (throughput mode):

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py \
  mlx-generate-qwen3-moe-batch /path/to/qwen3_5_moe-model \
  --prompt "Write a small story about a robot who learns to paint." \
  --batch-size 8 \
  --max-tokens 32 \
  --loader-backend native \
  --pin-policy phase \
  --retain-layers 31 \
  --defer-cache-clear \
  --final-cache-clear
```

Benchmark the full-resident equivalent with identical scheduling (`mlx-batch-raw`):

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py \
  mlx-batch-raw /path/to/qwen3_5_moe-model \
  --prompt "Write a small story about a robot who learns to paint." \
  --batch-size 16 \
  --max-tokens 32
```

Current Qwen3.5-MoE 35B story throughput (32 tokens per request, warm page cache; TensorFold uses retain 31 base layers and phase pinning; raw rows are generation-only with the ~37GB weight load reported separately, about 2-5s warm):

| Mode | Wall time | Total tok/s | MLX/Metal peak |
| --- | ---: | ---: | ---: |
| Raw full-resident, batch 1 (generation only) | 0.42s | 76.2 | 37.733 GB |
| Raw full-resident, batch 16 (generation only) | 1.16s | 443.3 | 37.733 GB |
| TensorFold batch 1 | 9.46s | 3.4 | 1.971 GB |
| TensorFold batch 4 | 9.76s | 13.1 | 2.097 GB |
| TensorFold batch 8 | 11.08s | 23.1 | 2.273 GB |
| TensorFold batch 16 | 10.45s | 49.0 | 2.641 GB |

Batched decoding shares each streamed layer base and the per-layer union of selected experts across all requests, so wall time stays nearly flat while generated tokens scale with batch size. The honest comparison: when the model fits in memory, steady-state full-resident MLX is still roughly 9x faster at batch 16 and far faster at batch 1 — story-length decode is dominated by TensorFold's per-step streaming and orchestration overhead, not by model math (0.42s of compute per 32 tokens). TensorFold's current wins are cold-start one-shot latency (streaming ~1GB instead of loading ~37GB), throughput per resident GB, and running on machines where 37GB simply does not fit.

Outputs are greedy per request; batched Metal kernels can flip near-tie argmax choices versus single-request decoding (raw full-resident MLX shows the same behavior), so a batch row usually matches the single-request output for a long prefix but bitwise equality across batch sizes is not guaranteed.

Predictive expert prefetch (`--expert-prefetch previous`) predicts each layer's experts from the previous token, prefetches those rows, then synchronously merges any missing rows so the math stays exact. Expert table reuse (`--expert-reuse --expert-reuse-cap N`) instead keeps each layer's assembled expert table resident across tokens, loading only uncovered rows and rebuilding when the table outgrows the cap; oversized prompt-pass unions are never cached. Both are exact and fully instrumented under `expert_prefetch` in the result JSON. Both currently measure as net slowdowns on the story workload (~43% consecutive-token row overlap is not enough: prefetch wastes ~57% of prefetched rows, reuse hits only ~27% of rows while paying per-layer concat and giving up retained-base budget), so both are off by default. The instrumentation is the point: it bounds what expert locality can buy on this model.

The per-token floor for exact streaming is the host copy: ~1GB of expert rows per token must move through `numpy` into MLX arrays (~3ms per layer warm), plus streamed base layers. Levers that attack the floor rather than shuffle it: speculative decoding (one streamed pass verifies several tokens), request batching (above), and an upstream zero-copy load path.

`resident weight peak` is TensorFold's logical sum of model tensors kept resident by the hosting policy. `MLX/Metal peak` is MLX allocator peak memory. Process RSS can be higher because Python, MLX, mmap, and allocator address space are not all returned to the OS immediately; it is recorded by the benchmark but is not the resident-weight budget.

This is still a prototype. The current optimized path beats normal MLX on this cold short-generation smoke test by avoiding full-model residency. Dense longer generations still have the bandwidth wall described in `plan-origin.md`: without quantization, sparsity, or hot/cold tensor intelligence, every generated token must touch every dense layer.

Benchmark an already-running oMLX OpenAI-compatible server:

```bash
PYTHONPATH=src python3 benchmarks/bench_cached_models.py omlx --model model-id --max-tokens 8
```

## MVP Architecture

```text
.safetensors shards
   -> TensorFold manifest
   -> layer/tensor index
   -> mmap-backed runtime
   -> future inference backend adapter
```

The first useful product claim is deliberately modest:

> Inspect and access standard model tensors lazily, with enough structure to support memory budgeting, layer streaming, and prefetch scheduling.

That keeps quality intact while giving us a concrete base for the larger idea.
