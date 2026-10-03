# Adding a CUDA family

A family serves CUDA when it exports `cuda_engine`. Keep implementation files under the family's `cuda/`
package and import PyTorch inside backend code, so family discovery also works on an MLX installation.

## Package and engine

```python
MODEL_TYPES = ("mymodel",)
TITLE = "My model"
MODELS = ("example/checkpoint",)

def cuda_engine(model_dir, *, drafter="", tp=1, rank=0, master="", master_port=29551,
                no_drafts=False, mtp_drafts=None, **options):
    from .cuda.engine import MyEngine
    return MyEngine(model_dir, ...)
```

Validate supported weights, context and rank settings before allocating model state.
An optional `CUDA_APP` subclasses `tensorfold.cuda.server.App` for family-specific request handling.

The engine exposes `eos`, `generate(prompt, max_tokens, sampling, on_tokens)` and, for a follower rank,
`follow()`. `generate` receives token IDs and keyed sampling settings, reports newly committed tokens
through the callback, honors its stop result where supported, and returns statistics. A `generate` that also
takes `stop_eos` is passed `stop_eos=False` for an `ignore_eos` request and decodes past end tokens; without it,
an end token ends every reply, unless the family's `CUDA_APP` hands `ignore_eos` to the engine and sets
`reads_ignore_eos`, as GLM's does. Expose cache capacity
so the server can reject oversized prompt-plus-reply requests before streaming.

Use the dense Qwen engine as a starting point. Both ranks must agree on settings, request headers,
prefix identity and forward order. Do not advertise shared requests merely because the CLI accepts
`--parallel`; the HTTP scheduler and engine both need that support.

## Exactness

A row must get exactly the same bits alone and in every supported verify window. Check projections,
attention, routing, recurrent state, accepted-path commits and subsequent decode. Compare eager calls
with CUDA graphs. For two ranks, compare with that same two-rank engine's serial reference.

Keep split-K and attention partitions independent of window width. Use stable routing ties and fixed
reduction order. Where ranks combine fp32 partials, gather and add in rank order. Library matmuls whose
algorithm changes with row count cannot define an unchecked exact verify path.

Use a separate fp32 forward for quality checks and set `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` for that
reference. Exactness against serial and quality against a trusted model are separate checks.

## Memory and packaging

Budget checkpoint storage, cache capacity, workspaces, captured graphs and file-backed tables on every
rank. On unified-memory systems, host allocations and GPU allocations compete for physical memory.
Reject an explicit context that cannot fit and report the estimated fitting capacity.

Declare non-Python kernel sources and data files as package data. Verify the wheel contains them and the
extension compiler is available in the supported container. Keep PyTorch and Triton versions tied to the
qualified toolchain instead of replacing the container's packages implicitly.

## Tests and measurements

Put GPU tests in `tests/cuda/`, with imports and bad-setting checks that can run without weights.
Cover each window width, sparse-attention transitions, partial keeps, resumed versus fresh prompts and
rank synchronization. Add concurrent versus solo checks before exposing concurrent execution.

Use the [public fixture command](README.md#measurements) for server measurements. Keep model/runtime pins,
launch commands and output hashes with results. Time dependent chains and full requests; do not select a
kernel solely from independent microbenchmarks.

## Sleep/wake reference implementations

Dense Qwen and Nemotron-H provide two single-device CUDA Level 2 reference adapters.
Both use the same HTTP lifecycle, checkpoint identity, transactional prefix store,
bounded safetensors transport and lazy disk restore. Their runtime ownership and
state layouts remain in the family package.

| Reference | Runtime hooks | Prefix codec | State preserved |
| --- | --- | --- | --- |
| Dense Qwen | [`qwen3_5/cuda/sleep.py`](../../src/tensorfold/families/qwen3_5/cuda/sleep.py) | [`prefix_snapshot.py`](../../src/tensorfold/families/qwen3_5/cuda/prefix_snapshot.py) | Attention, gated-delta/convolution state, optional DFlash context |
| Nemotron-H | [`nemotron_h/cuda/sleep.py`](../../src/tensorfold/families/nemotron_h/cuda/sleep.py) | [`prefix_snapshot.py`](../../src/tensorfold/families/nemotron_h/cuda/prefix_snapshot.py) | Attention, Mamba state and pending window buffers, retained hidden row, optional MTP context |

Export `CUDA_SLEEP_LEVELS = (2,)` and a lazy `cuda_sleep()` function returning the
family's hooks module. The shared adapter consumes this contract:

| Hook | Responsibility |
| --- | --- |
| `FAMILY` | Stable family identifier included in snapshot identity |
| `settings(engine)` | Tensor-free values describing the served context, math and drafting settings; compare them after reload |
| `cache(engine)` | A `PrefixCache` view with retained `(token_ids, state, draft_state)` entries and the existing capacity |
| `prefix_codec()` | Return the family's explicit snapshot codec |
| `attach(engine, store)` | Connect lazy disk lookup to the new runtime, including memory admission |
| `close(engine)` | Stop any runtime workers before the shared adapter drops the engine and reclaims allocations |

Hooks must not retain the previous engine, weights, callbacks bound to them, or any
device tensors. The reload factory receives pinned local paths and scalar options;
it must recreate the same effective context and settings. On failed reload, all
temporary allocations and file-reader staging must be releasable before retry.
Include serial twins, captured graphs, draft heads and closures when checking ownership.
The shared adapter pins `capacity_plan["context_window"]` when available, separately
from the served HTTP limit: rounded cache capacity can exceed the checkpoint's
native context and must not become the explicit context passed to the loader.

The codec exposes `save_prefix`, `verify_prefix` and `load_prefix`; the two reference
modules show their signatures. Delegate file I/O and integrity checks to
[`cuda/prefix_snapshot.py`](../../src/tensorfold/cuda/prefix_snapshot.py), then validate
the family's complete state layout before allocating restored tensors. Store only
committed KV rows. Qwen creates independent growing `State` objects; Nemotron restores
compact saved rows into a fresh engine's fixed-capacity buffers. Neither serializes
weight objects, graphs or arbitrary Python objects.

Keep the family's retention rules explicit. Nemotron clears live retained prefixes
when an unrelated drafted prompt starts; its two retained boundaries are not two
independent conversation slots. Sleep preserves the state still retained at drain.

Qualify repeated cycles with exact serial/drafted output, cached-token reuse, zero
allocated and reserved bytes while asleep, failed-save rollback, failed-load retry,
memory-denied cache misses and HTTP conversation continuation. Use
`tools/qualify_sleep.py` and `tools/qualify_sleep_http.py --require-cache` as the
reference checks. The [0.6.4 validation](../research/model-sleep-reference-validation.md)
records both full-model implementations. See [model sleep/wake](../model-sleep.md)
for supported operation and snapshot lifetime. Adding a family is separate from adding Level 1, multi-rank
coordination, or restart persistence.
