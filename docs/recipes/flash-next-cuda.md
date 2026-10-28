# Qwen3.8 Flash Next on CUDA

The native CUDA build serves Flash Next (`qwen4_exp`) on one NVIDIA GPU from an MLX affine 4-bit, group-32 checkpoint
with its MTP head, on kernels built in this repository. A set of the Python 0.6.1 engine's Triton kernels, captured
and packed, can stand in for its glue to compare against that engine's teacher path. @AdrianBinDC started the port in
[#476](https://github.com/ashhart/TensorFold/pull/476); it is tracked in
[#472](https://github.com/ashhart/TensorFold/issues/472). Its authors ran the model on a GB10 (sm_121); our kernel tests
also pass on an RTX PRO 6000 (sm_120).

## Experimental ModelOpt and two-rank path

The CUDA family also reads ModelOpt NVFP4 routed experts, including CSF-compressed scale planes, with separate
MXFP8 shared experts and dense projections and the checkpoint's BF16 activations. Two local CUDA ranks partition
expert intermediate channels and gather disjoint vocabulary ranges without reducing their BF16 bits. Each rank
owns its recurrent state, and peer failure makes the serving rank unavailable rather than returning partial results.

This path remains experimental pending independent full-model numerical qualification and complete GPU kernel
contract receipts. Compile success and host tests do not establish checkpoint fidelity. Performance and context
limits must be measured with the exact checkpoint, binary, device topology and request settings. Prefix retention
with MTP is outside this path's qualified scope; evaluate with `--prompt-cache-gib 0`.

## Experimental K32 prompt experts

`TF_FLASHNEXT_PROMPT_EXPERT_TILE=1` selects the K32 staging pipeline for NVFP4 prompt gate-up;
`2` also selects its FP32 down projection. Unset or `0` keeps the existing kernels. Decode retains its existing
16-pair form. `TF_FLASHNEXT_FACT_ROWS` retains its existing prompt threshold, including `0` to disable prompt tiles.
The pipeline adapts MiaAI-Lab's MIT recipe patch0014 and retains the same packed weights, BF16 activations,
16-term dots, scale-FMA order and epilogue. This is an unqualified opt-in pending the GPU gates.

`tf-cuda-test nvfp4-fact` compares both prompt forms with the decode bytes on skewed routes, row and column tails,
odd K32 groups and shared-expert skips. The normal CUDA kernel contract adds both replacement kernels and their
rounding-order controls. Their compiler resource slots remain pending until measured for each served SM.
Before enabling it for serving, run the full contract and compare main/candidate prompt-cut, resumed/fresh,
drafted/plain and concurrent/solo tokens at the same checkpoint. Measure cold 4k, 16k and 128k prefill with cache
disabled, alternating with the flag off. Host checks and compile-only checks do not establish GPU exactness or speed.

NVFP4 PLE gathers sort requested row IDs, dequantize each distinct row once through the parked worker pool,
and copy duplicate BF16 rows after the readers join. Hash history and FP32 row sums retain their existing order.
On Linux these rows use positional reads by default; `TF_FLASHNEXT_PLE_IO=mmap` retains the mapped reader,
and `pread` selects positional reads explicitly. MLX tables retain the mapped path.
`TF_FLASHNEXT_WARM_AHEAD=1` enables next-chunk file hints; unset or `0` leaves them off. Hints derive IDs from the future token
prefix without changing a sequence history, issue on the caller and leave no borrowed worker after return.
These I/O choices need cold-storage and served-token A/B gates before a performance claim.

Raw completion prompts of at least 16 tokens gain a kept end mark after they have been served once, when the
prompt-cache budget fits it. The first pass is unchanged; a second identical pass keeps near its end; later repeats
resume there. Chat endpoints retain priority. `TF_FLASHNEXT_REPEAT_CACHE=0` disables this Flash Next CUDA policy.
Other families keep their existing cache policy. Full token keys and frame identity still decide every resume.

## Run

```bash
tensorfold-native serve /path/to/checkpoint --temperature 0
```

- `--context N` takes up to the model's 262,144 tokens; the default is the whole window. Past 2,048 keys each attention
  layer reads 512 blocks of 4 keys a row, chosen by the model's block indexer, as the Python engine does.
- `--parallel N` admits N streams. Each stream's caches are its own: about 110 MiB of DeltaNet state and the n-gram
  tail, plus 28 KiB a token of keys, values and indexer keys, and with drafts its MTP head's cache.
- `--no-drafts` serves without the MTP head; `"draft": false` on a request does the same for that request.
- `TF_FLASHNEXT_CHUNK` (1 to 16,384, default 1,536) caps the rows of one prompt pass. Every pass reads the weights
  of every expert its rows route to, so fewer, larger passes cost less a token; but a pass runs to its end before
  other streams decode their next token. See [Prompt chunk size](#prompt-chunk-size).
- `TF_FLASHNEXT_DRAFTS` (1 to 15, default 6) and `TF_FLASHNEXT_CONFIDENCE` (default 0.60) set the head's chain.
- `TF_FLASHNEXT_LOAD_THREADS` (1 to 32, default 8) packs the routed NVFP4 experts on that many threads at load, staged
  through at most 192 MiB of pinned memory: every layer on one GPU, and the MTP head's layer on rank 0 at `--tp 2`.
  1 keeps the one-expert loop. Other values are refused.
- The engine uploads 73.3 GiB of the checkpoint: the n-gram tables stay mapped from the shards, and the vision tower is
  not loaded without `--vision`.
- `--vision` (experimental) runs images through the Qwen-VL tower that Qwen's CUDA family runs too
  (`zig/src/vision/qwen_cuda.zig`), on rank zero alone; its rows replace the prompt's image placeholders, M-RoPE
  positions reach attention, the block indexer and the MTP head, and a second rank receives the rows over NCCL. The
  checkpoint's preprocessor and processor files set the image processor. No checkpoint has been checked with images
  yet.
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
- For a last user message of at least 8,192 tokens, the lane cache may also keep one state inside that message.
  It chooses the latest existing cold chunk cut leaving at least 2,048 tokens outside the state's key, for a
  replacement question after the same document. It uses the loaded prompt pass size and adds no chunk split.
  Endpoint, rewind and shared states have priority; the extra copy is skipped if they cannot fit together under
  the byte or snapshot-slot cap. Existing eviction rules still apply. `/metrics` includes it in
  `tensorfold:prompt_cache_states` and `tensorfold:prompt_cache_bytes`; a successful resume counts toward
  `tensorfold:prompt_tokens_cached_total`. The chunk and budget host fixtures exercise this policy; real-model
  resumed/fresh parity and TTFT for this interior state still need the GPU gate.
- Shared row kernels: the token rows (`embed_rows.cu`, groups of 32 and a copy a hyper-connection stream), the MTP
  head's RMS rows with their group sums (`norms.cu`), the router logits and softmax top-k (`route_rows.cu`), and the
  routed experts over a window's or a chunk's rows on the core's plan (`cuda/grouped.zig`, `experts_rows.cu`).

The family supplies its config check, weight loading and packing, the layer graph on one batched pass
(`cuda_batch.zig`), and kernels for what is its own: the hyper-connections and the n-gram gate and conv
(`flashnext_hc.cu`), attention with its gate and the sparse indexer's pooling, scores and block selection
(`flashnext_attn.cu`, `flashnext_batch.cu`), the Gated DeltaNet chain, the n-gram gather and the MTP head.

The text path's kernels keep each row's own sums in a fixed order, so a row's bits are the same at any width; the
authors' text checks below compare drafted, concurrent and resumed replies with their plain, solo and fresh controls.
Images, through the cuBLASLt tower, are not qualified that way yet. The glue kernels round to bf16 where
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

### The ModelOpt checkpoint on one GB10

`nvidia/Qwen3.8-Flash-Next-FP4-Local` (revision `b9ce18a6`) on a GB10 (128 GB, driver 580.178, CUDA 13.3), greedy,
`--prompt-cache-gib 0`, the tools above, three runs of the defaults at a 32,768-token window alternated with other
settings in one session. These are @balancedscorpion's measurements on `beec1aad6`, the #952 branch build before
1.1.0, and have not been remeasured on 1.1.2:

| | defaults |
| --- | --- |
| One stream, drafted, code / chat | 64.1-64.6 / 40.7-41.1 |
| drafted, 1 / 4 / 8 streams, greedy code | 64.1-64.4 / 44.9-45.2 / 66.2-66.8 |
| Cold first token at 2,048 / 8,192 / 32,768 / 65,536 / 131,072 tokens (s) | 1.83 / 7.11 / 29.2 / 60.1 / 131.7 |

Every concurrent reply equalled its solo run and every drafted one its `"draft": false` run. A reply cancelled
mid-stream, or sent again after a restart, equalled a fresh one, and a second conversation turn resumed from the prompt
cache (`--no-drafts`) equalled the same turn computed without it. Prompt rows equal one-token rows bit for bit at seven
fixed histories of 20 to 6,496 tokens. Three streams draft at a time; the rest share plain rounds.

The engine holds 73.2 GiB and keeps a 12.2 GiB reserve. A 32,768-token window then admits 7 or 8 streams of 1.21 GiB
and a 131,136-token window (the cold prompts above) 2 of 4.03 GiB; a 262,144-token stream takes 7.79 GiB, which fitted
in one start of four on this machine. After the 131,072-token prompt the process had peaked at 6.5 GB of host and
83.0 GB of device memory. Absolute speeds on this machine moved by about a tenth between sessions.

### Prompt chunk size

`nvidia/Qwen3.8-Flash-Next-FP4-Local` (revision `b9ce18a6`) on a GB10 (driver 615.71.09, kernels from nvcc 13.0),
main `a05d62003`, `--parallel 3 --context 40960 --no-drafts --prompt-cache-gib 0`, greedy. The two settings ran in
fresh servers in the order 1,536 / 8,192 / 8,192 / 1,536; each cell is the median of 3, both runs of a setting given:

| | default (1,536) | `TF_FLASHNEXT_CHUNK=8192` |
| --- | --- | --- |
| First token at 436 tokens (s) | 0.304 / 0.306 | 0.306 / 0.307 |
| First token at 1,922 tokens (s) | 1.062 / 1.068 | 0.928 / 0.928 (-13%) |
| First token at 8,361 tokens (s) | 4.490 / 4.503 | 3.936 / 3.946 (-12%) |
| First token at 31,901 tokens (s) | 17.81 / 17.94 | 14.95 / 14.95 (-16%) |
| Longest wait between two decoding streams' tokens while an 8,361-token prompt runs (s) | 0.90 | 3.78 |
| Memory available after load (GB) | 44 | 38 |

- **For:** the gain is in the routed experts. In an `nsys` trace of the same build at one stream, a 1,922-token
  prompt took 138 ms less GPU time at 8,192 rows (1,052 to 914 ms), and the two NVFP4 expert kernels account for all
  of it: one pass instead of two, so each routed expert's gate, up and down weights are read once. A 1,536-row pass
  routes to about 350 of the 512 experts, and each takes 1.84 MB of gate and up weights.
- **Against:** a pass is not interrupted, so a stream that is decoding waits for the whole pass, about four times as
  long at 8,192 rows as at 1,536. The prompt scratch takes about 6 GB more, and at 16,384 rows the n-gram tables of
  the MLX 4-bit checkpoint page (see `docs/known-issues.md`).
- One stream at a time (an agent loop, a single user): the larger chunk shortens every prompt past 1,536 tokens. Several
  streams that decode while others send long prompts: the default keeps their waits under a second.
- The 64 greedy tokens after the 436- and 1,922-token prompts were the same at both settings.

## Not yet

- The dense projections run a copy of the Python engine's group-32 lane matmul (`qmm.cu`); the core's `qmm_group` and
  `lane_gemv` are built for groups of 64.
- No CUDA graphs or GPU-side rounds; drafted streams verify their windows one pass each, and the MTP head drafts one
  stream at a time.
- Prompts run slower than the Python engine's: a cold 12,001-token prompt takes about 10 s against its 6.7 s.
- Draws are made on the host from each row's scores.
- Our glue's sums are not the captured kernels' bit for bit, so replies part from the Python engine's teacher path
  after a while, as that engine's own served path does (10 to 36 tokens, from its routed-experts kernel's rounding).
