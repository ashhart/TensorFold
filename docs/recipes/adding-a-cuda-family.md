# Adding a CUDA family

This guide covers the Python engine, frozen at 0.6.5. New families go on the Zig engine: see
[adding a Zig family](adding-a-zig-family.md).

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
