# Qwen3.8 Flash Next on CUDA

The native CUDA build serves Flash Next (`qwen4_exp`) on one NVIDIA GPU from an MLX affine 4-bit, group-32 checkpoint
with its MTP head, on kernels built in this repository. A set of the Python 0.6.1 engine's Triton kernels, captured
and packed, can stand in for its glue to compare against that engine's teacher path. @AdrianBinDC started the port in
[#476](https://github.com/ashhart/TensorFold/pull/476); it is tracked in
[#472](https://github.com/ashhart/TensorFold/issues/472). It has been run on a GB10 (sm_121) only.

## Run

```bash
tensorfold-native serve /path/to/checkpoint --temperature 0
```

- `--context N` takes up to the model's 262,144 tokens; the default is the whole window. Past 2,048 keys each attention
  layer reads 512 blocks of 4 keys a row, chosen by the model's block indexer, as the Python engine does.
- `--parallel N` admits N streams. Each stream's caches are its own: about 110 MiB of DeltaNet state and the n-gram
  tail, plus 28 KiB a token of keys, values and indexer keys, and with drafts its MTP head's cache.
- `--no-drafts` serves without the MTP head; `"draft": false` on a request does the same for that request.
- `TF_FLASHNEXT_DRAFTS` (1 to 6, default 6) and `TF_FLASHNEXT_CONFIDENCE` (default 0.60) set the head's chain.
- The engine uploads 73.3 GiB of the checkpoint: the n-gram tables stay mapped from the shards, and the vision tower is
  not loaded.
- It refuses another quantization, other shapes, and semantic settings its kernels fix (RMS epsilon, RoPE base, rotated
  share, type and scaling, the gate and activation types, attention bias, the layer roles, the n-gram layer, kernel
  and dilation): config.json is checked at load and the log names the field. It also refuses `--segments`, `--slide`
  and a folder whose shared experts carry Living Weights neurons.
- `TENSORFOLD_CUDA_KERNELS=DIR` swaps the glue for a captured set (see [A captured set](#a-captured-set)) when the set
  holds every entry point the engine launches; a set without them (Nemotron's, say) is passed over, and the startup
  line says which glue runs. With a set loaded, `TF_FLASHNEXT_OWNED` (`all`, or Triton names such as
  `_chunks,_merge`) runs those entry points on our kernels and the rest on the set's, to put a difference down to one
  kernel.

## What the core runs and what the family runs

The family plugs into the core's pieces and adds no scheduler:

- The core's round loop and lane host drive every token. Drafted windows, shared rounds of streams without drafts, and
  prompts a chunk at a time between other streams' rounds (`prefill_step`) all go through them.
- The core's prompt cache keeps and resumes a stream's state at the marks it asks for. The family copies the state
  (`cuda_snap.zig`), with the MTP head's cache rows for a drafting stream; the head reads one token past a row, so
  states keep with a lookahead of 1, as Nemotron's do.
- Shared row kernels: the token rows (`embed_rows.cu`, groups of 32 and a copy a hyper-connection stream), the MTP
  head's RMS rows with their group sums (`norms.cu`), the router logits and softmax top-k (`route_rows.cu`), and the
  routed experts over a window's or a chunk's rows on the core's plan (`cuda/grouped.zig`, `experts_rows.cu`).

The family supplies its config check, weight loading and packing, the layer graph on one batched pass
(`cuda_batch.zig`), and kernels for what is its own: the hyper-connections and the n-gram gate and conv
(`flashnext_hc.cu`), attention with its gate and the sparse indexer's pooling, scores and block selection
(`flashnext_attn.cu`, `flashnext_batch.cu`), the Gated DeltaNet chain, the n-gram gather and the MTP head.

Every kernel keeps a row's own sums in a fixed order, so a row's bits are the same at any width: drafted windows equal
serial decoding, rounds equal lone streams, and resumed prompts equal fresh ones. The glue kernels round to bf16 where
the Python engine's do, with IEEE division and square root and a within-one-ulp exp; the router and the 4-bit
hyper-connection dots run on mma tiles with K in ascending order.

## Checks

- `tf-cuda-test flashnext-hc`, `flashnext-attn` and `route-rows`, and `norm-rows` for the shared rows: each kernel
  against a float64 chain that rounds where it rounds, and every row count's bits equal (tolerances in the test files).
- `zig build flashnext-prompt -Dflashnext_model=DIR -Dflashnext_kernels=native -Dflashnext_ids=FILE
  -Dflashnext_count=32`: 32 greedy tokens through the batched pass, against the Python engine's teacher path
  (`zig/tests/cuda/flashnext/lr7/results.json`, `results-sparse.json`); a captured set's folder in place of `native`
  runs its glue. `TF_FLASHNEXT_DRAFTS=6` decodes each prompt serially and drafted and compares them token for token
  (`TF_FLASHNEXT_SEED` for keyed sampled draws).
- `zig build flashnext-compare -Dflashnext_model=DIR -Dflashnext_kernels=SET -Dflashnext_ids=FILE`: our glue against a
  captured set on one loaded engine: every prompt row's scores (top-1 agreement, KL, each path's perplexity on the
  prompt's own next tokens, dense rows and rows past 2,048 apart), then each path's greedy reply and where they part.
  With `TF_FLASHNEXT_OWNED` it compares the set against the set with those entries swapped.
- `zig build flashnext-select -Dflashnext_fixtures=DIR`: the native block selection against Triton's lists.
- `tf-cuda-test experts-rows`: the rows kernel against the decode kernel's one-row launches, byte for byte.
- Served: `tools/bench_concurrent.py --alone --serial` (concurrent equal to solo, drafted equal to `"draft": false`),
  and a prompt sent twice (the second resumed from a kept state, equal to the first).

## A captured set

A set is built from the Python 0.6.1 engine in a container (`IMAGE`, default tag `tensorfold-qwen38:v0.6.1`) with the
scripts in `zig/tests/cuda/flashnext`, then packed with `tools/zig/triton_aot_pack.py`:

```bash
zig/tests/cuda/flashnext/capture.sh WORK/cap /path/to/hf/snapshot          # the model's own run, about 78 GiB of GPU
zig/tests/cuda/flashnext/capture_rows.sh WORK/rows                          # the batched pass's forms, no checkpoint
zig/tests/cuda/flashnext/capture_mtp.sh WORK/mtp                            # the MTP head's own kernels
zig/tests/cuda/flashnext/capture_sparse.sh WORK/sparse                      # the sparse attention's kernels
zig/tests/cuda/flashnext/pack.sh KERNELS WORK/cap/out/cap WORK/rows/out WORK/mtp/out WORK/sparse/out
```

`capture.sh` also writes the prompts and the Python engine's references the checks read. The set has 76 kernels;
`capture_sparse.sh` writes the selection fixtures `flashnext-select` reads.

## Measurements

On a GB10 (DGX Spark, driver 580.178, CUDA 13.0), an MLX 4-bit group-32 Flash Next checkpoint with its MTP head,
greedy unless said, `tools/bench_openai.py` (256 tokens, median of 3) and `tools/bench_concurrent.py` (256 tokens,
aggregate tok/s), on our kernels and, for comparison, on the 76-kernel captured set:

| | ours | captured |
| --- | --- | --- |
| One stream, drafted, code / chat | 84.2 / 62.6 | 82.5 / 62.1 |
| One stream, `--no-drafts`, code / chat | 35.1 / 35.1 | 35.2 / 35.2 |
| `--no-drafts`, 1 / 2 / 4 streams, sampled code | 35.0 / 53.1 / 77.9 | 35.2 / 53.0 / 77.1 |
| drafted, 1 / 2 / 4 streams, greedy code | 83.8 / 82.7 / 82.2 | 82.3 / 81.1 / 80.7 |
| Cold first token at 3,001 / 12,001 tokens (s) | 2.67 / 10.40 | 2.67 / 10.30 |
| 45,030-token prompt, no cache (s) | 36.5 | 35.8 |

Every concurrent reply equalled its solo run, and every drafted one its `"draft": false` run. Drafted streams verify
their windows a pass each, so drafted throughput does not grow with streams yet. While a 12,001-token prompt runs,
another stream's decode waits at most 1.3 s between tokens (one prompt chunk). Each stream at a 16,384-token window
takes 0.74 GiB with drafts, 0.54 GiB without.

Our kernels' sums are their own, so tokens part from the captured set's (the Python engine's teacher path) where two
candidates are close: 32 greedy tokens equal it on four of `zig/tests/cuda/flashnext/lr7`'s five prompts and part at
token 15 on the 3,001-token one. Over 4,096 tokens of this repository's documentation, the next-token perplexity is
6.168 on the captured set and 6.178 on ours.

## Not yet

- The dense projections run a copy of the Python engine's group-32 lane matmul (`qmm.cu`); the core's `qmm_group` and
  `lane_gemv` are built for groups of 64.
- No CUDA graphs or GPU-side rounds; drafted streams verify their windows one pass each, and the MTP head drafts one
  stream at a time.
- Prompts run slower than the Python engine's: a cold 12,001-token prompt takes about 10 s against its 6.7 s.
- Draws are made on the host from each row's scores.
- Our glue's sums are not the captured kernels' bit for bit, so replies part from the Python engine's teacher path
  after a while, as that engine's own served path does (10 to 36 tokens, from its routed-experts kernel's rounding).
