# Qwen3.8 Flash Next on CUDA

The native CUDA build serves Flash Next (`qwen4_exp`) on one NVIDIA GPU from an MLX affine 4-bit, group-32 checkpoint
with its MTP head. It replays the Python 0.6.1 engine's Triton kernels from a captured set beside its own CUDA
kernels, so it gives that engine's teacher-path tokens. @AdrianBinDC started the port in
[#476](https://github.com/ashhart/TensorFold/pull/476); it is tracked in
[#472](https://github.com/ashhart/TensorFold/issues/472). It has been run on a GB10 (sm_121) only.

## Run

The engine needs a captured kernel set (see [The kernel set](#the-kernel-set)); without `TENSORFOLD_CUDA_KERNELS` it
refuses to load.

```bash
TENSORFOLD_CUDA_KERNELS=/path/to/flashnext-kernels tensorfold-native serve /path/to/checkpoint --temperature 0
```

- `--context N` takes up to the model's 262,144 tokens; the default is the whole window. Past 2,048 keys each attention
  layer reads 512 blocks of 4 keys a row, chosen by the model's block indexer, as the Python engine does.
- `--parallel N` admits N streams. Each stream's caches are its own: about 110 MiB of DeltaNet state and the n-gram
  tail, plus 28 KiB a token of keys, values and indexer keys, and with drafts its MTP head's cache.
- `--no-drafts` serves without the MTP head; `"draft": false` on a request does the same for that request.
- `TF_FLASHNEXT_DRAFTS` (1 to 6, default 6) and `TF_FLASHNEXT_CONFIDENCE` (default 0.60) set the head's chain.
- The engine uploads 73.3 GiB of the checkpoint: the n-gram tables stay mapped from the shards, and the vision tower is
  not loaded.
- It refuses another quantization or other shapes (config.json is checked at load, and the log names the field),
  `--segments` and `--slide`, and a folder whose shared experts carry Living Weights neurons.

## What the core runs and what the family runs

The family plugs into the core's pieces and adds no scheduler:

- The core's round loop and lane host drive every token. Drafted windows, shared rounds of streams without drafts, and
  prompts a chunk at a time between other streams' rounds (`prefill_step`) all go through them.
- The core's prompt cache keeps and resumes a stream's state at the marks it asks for. The family copies the state
  (`cuda_snap.zig`), with the MTP head's cache rows for a drafting stream; the head reads one token past a row, so
  states keep with a lookahead of 1, as Nemotron's do.
- The routed experts over a window's or a chunk's rows run on the core's plan (`cuda/grouped.zig`) and its rows kernel
  (`experts_rows.cu`, `cuda.experts.RowsExperts`), which gives every row the decode kernel's sums.
- Captured Triton launches go through `cuda.aot.Set`, so a launch with no exact variant takes the general one.

The family supplies its config check, weight loading and packing, the layer graph on one batched pass
(`cuda_batch.zig`), and kernels for what is its own: the Gated DeltaNet chain, the sparse indexer's block selection,
the n-gram gather, the hyper-connection and attention launches, and the MTP head.

## The kernel set

The set is built from the Python 0.6.1 engine in a container (`IMAGE`, default tag `tensorfold-qwen38:v0.6.1`) with
the scripts in `zig/tests/cuda/flashnext`, then packed with `tools/zig/triton_aot_pack.py`:

```bash
zig/tests/cuda/flashnext/capture.sh WORK/cap /path/to/hf/snapshot          # the model's own run, about 78 GiB of GPU
zig/tests/cuda/flashnext/capture_rows.sh WORK/rows                          # the batched pass's forms, no checkpoint
zig/tests/cuda/flashnext/capture_mtp.sh WORK/mtp                            # the MTP head's own kernels
zig/tests/cuda/flashnext/capture_sparse.sh WORK/sparse                      # the sparse attention's kernels
zig/tests/cuda/flashnext/pack.sh KERNELS WORK/cap/out/cap WORK/rows/out WORK/mtp/out WORK/sparse/out
```

`capture.sh` also writes the prompts and the Python engine's references the checks below read. The set has
76 kernels; `capture_sparse.sh` writes the selection fixtures `flashnext-select` reads.

## Checks

- `zig build flashnext-prompt -Dflashnext_model=DIR -Dflashnext_kernels=SET -Dflashnext_ids=FILE -Dflashnext_count=32`:
  32 greedy tokens through the batched pass against the Python engine's teacher path
  (`zig/tests/cuda/flashnext/lr7/results.json`, `results-sparse.json`). `TF_FLASHNEXT_PROMPT=rows` runs the row-by-row
  reference instead; `TF_FLASHNEXT_DRAFTS=6` decodes each prompt serially and drafted and compares them token for
  token (`TF_FLASHNEXT_SEED` for keyed sampled draws).
- `zig build flashnext-select -Dflashnext_fixtures=DIR`: the native block selection against Triton's lists.
- `tf-cuda-test experts-rows`: the rows kernel against the decode kernel's one-row launches, byte for byte.
- Served: `tools/bench_concurrent.py --alone --serial` (concurrent equal to solo, drafted equal to `"draft": false`),
  and a prompt sent twice (the second resumed from a kept state, equal to the first).

## Measurements

On a GB10 (DGX Spark, driver 580.178, CUDA 13.0), an MLX 4-bit group-32 Flash Next checkpoint with its MTP head,
the 76-kernel set, greedy unless said, `tools/bench_openai.py` (256 tokens, median of 3) and
`tools/bench_concurrent.py` (256 tokens, aggregate tok/s):

| | tok/s |
| --- | --- |
| One stream, drafted, code / chat | 82.6 / 61.8 |
| One stream, `--no-drafts`, code / chat | 35.0 / 34.9 |
| `--no-drafts`, 1 / 2 / 4 streams, sampled code | 35.0 / 53.0 / 77.3 |
| drafted, 1 / 2 / 4 streams, greedy code | 81.6 / 80.6 / 80.4 |

Every concurrent reply equalled its solo run, and every drafted one its `"draft": false` run. Drafted streams verify
their windows a pass each, so drafted throughput does not grow with streams yet. A cold prompt's first token takes 2.65
s at 3,001 tokens, 10.0 s at 12,001 and 37.9 s at 45,030 (a 65,536-token window, no prompt cache). While a 12,001-token
prompt runs, another stream's decode waits at most 1.3 s between tokens (one prompt chunk). Each stream at a
16,384-token window takes 0.74 GiB with drafts, 0.54 GiB without.

## Not yet

- The set replays the Python engine's Triton kernels for norms, hyper-connections, attention, routing and the head. The
  core's shared row kernels (`cuda/norms.cu`) sum in another order, so moving onto them is its own change.
- The dense projections run a copy of the Python engine's group-32 lane matmul (`qmm.cu`); the core's `qmm_group` and
  `lane_gemv` are built for groups of 64.
- No CUDA graphs or GPU-side rounds; drafted streams verify their windows one pass each, and the MTP head drafts one
  stream at a time.
- Prompts run slower than the Python engine's: a cold 12,001-token prompt takes 10.0 s against its 6.7 s.
- Draws are made on the host from each row's scores.
- Replies equal the Python engine's teacher path and part from its served path after 10 to 36 tokens, from the
  routed-experts kernel's rounding form.
