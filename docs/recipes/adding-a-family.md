# Adding a model family

A family is a package in `src/tensorfold/families/<name>/`. Nothing registers it: the CLI scans the folder,
reads each package's `MODEL_TYPES`, and picks the package whose list holds the checkpoint's `model_type`
(from `config.json`, or `text_config.model_type`). Start from the family closest to yours and copy its layout.

## The package

```python
# src/tensorfold/families/mymodel/__init__.py
MODEL_TYPES = ("mymodel",)        # config.json model_type values this package serves
TITLE = "My Model"                # shown by `tensorfold models`
LANES = False                     # False: the serial engine; True: the lane engine (mlx_lm-style batch caches)

def load(model_dir, **options):   # -> (model, tokenizer); ignore options you don't know
    ...

# optional
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200"}   # set before MLX starts, unless the environment sets them
def engine_settings(model): ...                # keyword arguments for the engine, e.g. {"max_rows": 16}
def kernel_version(model): ...                 # a name for the kernels (prefix snapshots are keyed by it)
def setup(app, model, **options): ...          # extras on the server app, e.g. a draft model
```

Without `kernel_version`, snapshots are keyed by a hash of the package's source files, so editing a kernel
never reuses a snapshot computed by the old one.

## The model object (serial engine)

The serial engine (`engine/family_engine.py`) needs three things:

| Member | What it returns |
| --- | --- |
| `make_cache()` | one cache object per layer (mlx_lm's cache classes work) |
| `hidden(inputs, cache)` | the last hidden states `[1, L, D]` for token ids `[1, L]`, advancing the cache |
| `head(hidden)` | logits `[..., V]` |

That is enough for correct decoding: one token a round, sampled with the exact keyed rule. Get this right first.
It is the reference every faster path must reproduce bit for bit, and prompt caching and snapshots work with it
unchanged. The rest is speed:

| Member | Effect |
| --- | --- |
| `multi_row_exact = True` | the model promises that a forward of several consecutive rows gives each row the bits of a one-row forward. The engine then verifies drafts in one pass. Check it at load time (see below) and set it only when the check passes. |
| `keep_rows(cache, rows, keep)` | roll the caches back to the first `keep` rows of the last `rows`-row call (KV trim, recurrent states of the kept row) |
| `gpu_tokens = True` | `hidden` accepts the token as an unread GPU array. The engine then samples on the GPU (`engine/gpu_sampling.py`, same rule) and queues the next step before reading the token, so Python overlaps the GPU. |
| `gpu_sampling = True` | draw tokens with `gpu_sampling` in synchronous rounds too (saves a large host-side top-k) |
| `mtp`, `draft(...)`, `drafts`, `last_streams`, `absorb_draft_context(...)` | a draft head. Each round verifies the pending token and up to `drafts` drafts, then asks `draft` for the next ones from the kept rows. See `families/qwen4_exp/runtime.py`. |
| `adopt_cache(cache)` | convert a cache read from a snapshot into the model's own cache classes |

Copy windows come for free once `multi_row_exact` and `keep_rows` exist. When the context already holds the
text being written (file edits, repeated code, tool arguments) and at least 8 tokens before the cursor match
it, the engine verifies up to 7 copied tokens a round.

## Proving row invariance

Drafted decoding is exact only when a row's arithmetic does not depend on how many rows share the call. Check it
on the real weights at load time, the way `rows_match_serial` does in the Flash Next and Nemotron packages:

1. prefill a short prompt;
2. from two copies of that cache, run k single-row steps on one and one k-row step on the other, for k = 2, 3,
   4 (and up to your largest window);
3. compare the logits with `mx.array_equal`. Any difference means no drafts on this machine.

Things that break row invariance:

- MLX's quantized matmul picks a different kernel by row count, and some of them sum in a different order. On
  an M3 Ultra with MLX 0.32.0, 2 to 4 rows did not match one row (on an M5 Max with MLX 0.31.2 they did).
  `families/qwen4_exp/kernels.py` has a matvec (`qmv_rows`) that gives every row MLX's one-row bits: a
  simdgroup per row, weights read once for all rows.
- Reductions whose split depends on the row count (a threadgroup per k rows, a split-K chosen by shape).
- Attention kernels chosen by query count, and masks that differ between one-row and multi-row calls. Give
  each row its own key range, or write one kernel that treats every row the same way (`attention_rows` in the
  Flash Next kernels).
- Templating a Metal kernel on a per-step value (row count, key count). Each new value compiles a new kernel,
  and different specialisations can round differently. Pass such values in a small buffer.
- Fast-math reassociation: any edit to a kernel's source can change its bits. After every kernel change,
  regenerate the serial reference output and compare again.

## Checking quality

Row invariance says drafted equals serial. It says nothing about whether serial is right. Compare your decode
path with a reference forward (mlx_lm's model, or a straightforward MLX implementation of the architecture)
by teacher-forced negative log-likelihood over a few thousand tokens of real text, at short and long context.
The two should agree to within noise (a few thousandths of a nat). Also check that they choose the same argmax
on at least 98% of positions. A kernel that is fast but shifts the NLL is not done.

## Measuring

1. The bandwidth floor: bytes of weights read per token (all dense weights, plus the experts a token uses)
   divided by the machine's memory bandwidth. That is the best one-row step can do.
2. The kernel count per token. Each dependent kernel costs a few microseconds even when it does nothing (we
   measured about 5.5 us of GPU time on an M3 Ultra, 2.5 to 3 us per small op on an M5 Max), so 1,000 small
   kernels cost 3 to 6 ms.
3. The cost of each extra verify row, split by component: stub one component at a time and time 1, 2, 4 and
   8 rows. Deeper drafting pays only while an extra row costs less than the tokens it is expected to add.

Time kernels in dependent chains (each call's input is the previous call's output), never as many independent
calls of one shape: independent calls fill the GPU whatever the kernel's parallelism and favour the wrong
variant. Confirm every kernel change with a whole-model timing, and ideally a live A/B on real requests.

## Tests to write

- the family's forward on a tiny random config on the CPU (whole prompt equals token by token);
- row invariance on the real weights (at load time, or a test that is skipped when the weights are absent);
- drafted equals serial for your engine path: the fakes in `tests/test_thinking_budget.py` and
  `tests/lane_fakes.py` show how to drive the engines without a model.
