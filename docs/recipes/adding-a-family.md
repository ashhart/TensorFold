# Adding an MLX family

This guide covers the Python engine, frozen at 0.6.5. New families go on the Zig engine: see
[adding a Zig family](adding-a-zig-family.md).

Create `src/tensorfold/families/<name>/`. The CLI discovers packages by `MODEL_TYPES`, matching
`config.json` or its `text_config`. Keep backend imports inside the loader so discovery does not load a model.

## Package interface

```python
MODEL_TYPES = ("mymodel",)
TITLE = "My model"
LANES = True
MODELS = ("example/checkpoint",)
KERNEL_PACKAGE = "tensorfold.kernels.mymodel.v1"
KERNEL_VERSION = "v1"

def load(model_dir, **options):
    ...  # return model, tokenizer
```

Optional declarations include `DRAFTER`, `MLX_ENV`, `check`, `engine_settings`, `kernel_version` and `setup`.
`check` must reject unsupported configuration before downloading weights, then validate the weight index
when available. List shared kernel modules in `KERNEL_DEPENDENCIES` so snapshot keys include them.

`weight_bytes(model_dir, ple_on_ssd)` can provide a conservative resident-weight estimate for MLX startup
admission. Without it, the CLI counts every safetensors file's size. Read headers only, use the loader's
placement decision, and exclude only tensor bytes kept on the host (file mappings, or the SSD reads of
`--ple-on-ssd`). Runtime cache and workspace admission still applies after loading; this hook does not change
the memory budget.

## Model interface

`load` returns a model with `lane_family = True` and a tokenizer. The complete model protocol is in
`src/tensorfold/engine/lane_family.py`.

| Member | Contract |
| --- | --- |
| `make_cache()` | New state for one stream |
| `hidden(inputs, cache)` | Consecutive hidden rows, advancing the cache |
| `head(hidden)` | Target logits |
| `keep_rows(cache, rows, keep)` | Restore precisely the accepted prefix |
| `exact_width`, `window_costs` | Checked window width and measured cost by width |
| `prefill(inputs, cache)` | Optional prompt path, called on the engine's planned chunks |
| `hidden_rows`, `keep_rows_streams` | Shared forward and independent stream commits |
| `batch_rows`, `max_streams`, `shared_costs` | Shared-forward limits and costs |
| `speculate`, `settle`, `draft_streams` | Optional draft-head operations |
| `adopt_cache(cache)` | Restore family-specific cache classes from snapshots |
| `release_rounds()` | Drop the last forward's rollback buffers when no stream is live, including after startup probes |

Start with `exact_width = 1`. Use one sampler consistently for serial and drafted calls; the host and GPU
implementations can differ at near-ties. Prompt chunks must follow the engine's plan both fresh and resumed.
The CLI finds message markers in the tokenizer's chat template. The plan uses assistant-message starts
and the second message's start as resume points, with chunk starts at least 256 and at most 2,048 tokens
apart. Without usable markers it uses 2,048-token chunks from position zero. A cache lookup must not
choose a different prefill shape for an otherwise identical prompt.

## Verification

Check each projection and attention row alone and in a window using exact equality. Compare recurrent
state after partial keeps, then continue decoding. Check shared forwards against streams run separately,
including unequal stream lengths. Disable an unsupported wider path when a load-time check fails.

Row-dependent matmul dispatch, shape-dependent reductions, attention masks, routing ties and different
rounding points can break exactness. Fix arithmetic order per row. Pass per-step values in buffers instead
of creating a new kernel specialization for each token. Evaluate derived weights on the loading thread.

Compare model quality against a trusted forward on a named public fixture. Use teacher-forced NLL and
top-token agreement, and report the reference's own numerical variation. Tests on small synthetic models
must be supplemented by the real model dimensions.

## Tests and measurements

Use `tests/lane_fakes.py` and `tests/test_family_streams.py` for model-free engine cases. Add family tests
for forward correctness, row equality, shared streams, rollback, snapshots and context/memory admission.
Exercise thinking on and off, tools, later instructions and templates that rewrite earlier turns.

Measure whole requests with the [public benchmark fixtures](README.md#measurements). Record revisions,
commands, runtime and output hashes. Measure complete dependent chains before promoting a kernel change.
