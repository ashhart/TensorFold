# DeepSeek-V4-Flash

The `deepseek_v4` family serves `mlx-community/DeepSeek-V4-Flash-4bit` on the MLX lane engine, on a Mac with
256 GB and MLX 0.32.2 or later (`serve` refuses an older MLX). The routed experts keep DeepSeek's mxfp4 bytes and
everything else is affine 4-bit in groups of 64; about 151 GiB stays resident.
Packages: `src/tensorfold/families/deepseek_v4/` and `src/tensorfold/kernels/deepseek/v4/`, with GLM-5.3-Flash's
hyper-connection kernels and row linears.

```bash
tensorfold pull mlx-community/DeepSeek-V4-Flash-4bit Vontra/DeepSeek-V4-Flash-DSpark-MLX
tensorfold serve mlx-community/DeepSeek-V4-Flash-4bit
```

## The model

- 43 blocks of hidden size 4,096 mix four residual streams through hyper-connections (20 Sinkhorn steps a boundary).
- Attention: 64 query heads over one shared 512-wide key/value head, a 128-token window with sinks, RoPE on the
  last 64 dims (YaRN on the compressed layers) rotated back on the output, and a grouped low-rank output.
- Compressed pools: 21 layers pool every 4 positions (8 overlapping slots), 20 every 128, and the first 2 see only
  the window. On the 4-position layers an indexer (64 heads of 128) keeps each row's 512 best pool rows once more
  than 512 are visible, past 2,048 tokens.
- MoE: 256 routed experts (top 6 by sqrt-softplus scores; the first 3 layers route by a token-id table) and one
  shared expert, SwiGLU clamped at 10.
- Vocabulary 129,280. The model's window is 1,048,576 tokens.

## Draft heads

The 4-bit checkpoint has no draft head. The family reads two, converted from DeepSeek's MIT-licensed releases and
published in this layout: `model.safetensors` beside a `config.json` whose `model_type` names the head.

- DSpark, `Vontra/DeepSeek-V4-Flash-DSpark-MLX` (10.7 GB, `deepseek_v4_dspark`): three MoE blocks read the target's
  streams after layers 40-42 and draft a 5-token block in one pass. The serve command drafts with it by default once
  it has been pulled.
- MTP, `Vontra/DeepSeek-V4-Flash-MTP-MLX` (3.5 GB, `deepseek_v4_mtp`): the checkpoint's own next-token layer. Serve
  with `--drafter Vontra/DeepSeek-V4-Flash-MTP-MLX` to draft with it.

The converter builds the same folders from DeepSeek's releases: shards 46-48 of `deepseek-ai/DeepSeek-V4-Flash-DSpark`
with the release's `config.json` beside them, or shard 46 of `deepseek-ai/DeepSeek-V4-Flash`. Pass the folder to
`--drafter`.

```bash
python -m tensorfold.families.deepseek_v4.convert dspark model-00046-of-00048.safetensors \
    model-00047-of-00048.safetensors model-00048-of-00048.safetensors ~/models/DeepSeek-V4-Flash-dspark
python -m tensorfold.families.deepseek_v4.convert mtp model-00046-of-00046.safetensors ~/models/DeepSeek-V4-Flash-mtp
```

Without a draft head the engine decodes one row a step. The chat template is DeepSeek's own encoder (vendored,
MIT); thinking is on unless the request or `--no-thinking` turns it off, and DSML tool calls parse into OpenAI
tool calls.

## Exactness

A round verifies its drafts in one forward of up to 16 rows, and every row of a window gets its one-row call's bits:
the dense projections through `simd_qmm` (MMA from 3 rows), the compressors' fp32 projections and the mxfp4
experts through row kernels, attention through a kernel where four simdgroups split each (row, head)'s own pool rows
and window. A load-time check sets the widest exact window and whether several streams' rows can share a forward.
Caches are indexed by position (a 144-row key ring, pool rows by block, a ring of compressor projections), so a
rejected draft only moves an offset.

Prompts prefill in chunks of up to 2,048 tokens, with attention staging keys for eight heads at a time. DeepSeek's
encoder writes the reply prefix itself, so the tokenizer names `<｜Assistant｜>` as the reply marker: chunks start at
replies, a follow-up turn resumes from its last reply, and a resumed prompt gets a fresh prompt's bits.

The reference is mlx-lm's PR #1797 on the same weights, with its two departures from DeepSeek's code switched off.
Teacher-forced over four prompts, the serial path picks the PR's top token at 98-100% of positions and its logits
differ from the PR's by 3-5% relative, less than the PR's own prompt and decode paths differ (5.6-9.0%). Greedy
texts split where the PR's top two logits sit within 0-1.0 of each other.

## Measurements

Measured on an M3 Ultra (60-core GPU, 256 GB) with MLX 0.32.2 and DSpark drafts, against PR #1797's server on the
same machine and weights with its defaults. Cells are the quick fixtures at 64 / 256 tokens, thinking off, median
tok/s:

| Cell | TensorFold | PR #1797 | Ratio |
| --- | --- | --- | --- |
| Chat, greedy | 51.4 / 49.3 | 23.4 / 23.1 | 2.2 / 2.1 |
| Chat, sampled | 48.4 / 48.0 | 29.3 / 28.7 | 1.7 / 1.7 |
| Code, greedy | 69.4 / 70.8 | 23.5 / 23.0 | 3.0 / 3.1 |
| Code, sampled | 65.5 / 67.2 | 29.4 / 28.9 | 2.2 / 2.3 |

Without drafts the engine decodes 44-46 tok/s (22 ms a row). A round of R rows costs about 21 + 7 (R - 1) ms; the
extra rows are mostly the routed experts each one adds. Chat drafts land 2.2-3.1 tokens a round, code 3.2-4.7.

Cold prompts, prefill tok/s (the standard's server failed from 32k):

| Prompt | TensorFold | PR #1797 |
| --- | --- | --- |
| 2k | 444 | 227 |
| 8k | 403 | 225 |
| 16k | 388 | 218 |
| 32k | 362 | failed |
| 64k | 319 | - |
| 128k | 260 | - |

After the 64k and 128k prompts the replies decoded at 53 and 34 tok/s, and the server peaked at 165 GiB.
Drafted replies equal `"draft": false` ones (11 of 11 in each of two server runs), a follow-up turn reuses its
conversation's 8,452-token prefix and matches the same prompt served fresh, and 2 and 4 concurrent streams each
equal their solo replies (63 tok/s together at 4).

## Memory

The family keeps the default allowance, 70% of RAM for the whole process (179.2 GiB on 256 GB). With 151 GiB of
weights resident the admission fits one request of 349,184 tokens; a token costs 6.7 KB of pools after the fixed
rings. `TENSORFOLD_MEMORY_LIMIT_GB` raises the budget on a machine with nothing else loaded.

## Not yet

CUDA on two DGX Sparks and DeepSeek-V4-flash-vision-exp are not in this family yet.

## CUDA GGUF on one DGX Spark

The CUDA adapter runs DeepSeek-V4-Flash-0731 from an existing GGUF through the
[MIT-licensed ds4 engine](https://github.com/Entrpi/ds4), behind TensorFold's HTTP
server, native tokenizer, DeepSeek prompt encoder and DSML tool parser.
It supports one GB10 and one request at a time, with an optional local DSpark
GGUF. MTP, vision, tensor parallelism and concurrent streams are unsupported.

The qualified base is
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`:
80.76 GiB, 1,328 tensors and 129,280 vocabulary entries, using IQ2_XXS routed
gate/up, Q2_K routed down and Q8_0 dense/shared/output weights. `--gguf` selects
the user's quant; it is independent of the native source pin. Other files must
pass schema, tokenizer and memory admission. Routed IQ2_XXS, Q2_K and Q4_K are
recognized; only the named mixed quant has been qualified end to end.

### Build and serve

Use Linux aarch64, CUDA 13 and Python 3.11+ in a dedicated venv with TensorFold's
declared dependencies plus setuptools/wheel. The native build targets GB10
(`sm_121a`); other GPUs are unqualified. Two compiler jobs are the default.

```bash
TENSORFOLD_DEEPSEEK_FLOOR_GIB=4 python tools/build_deepseek_v4_cuda.py \
    --gguf /models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
    --model-dir /models/tensorfold-deepseek \
    --companion-reserve-gib 3 --context 163840 --jobs 2

TENSORFOLD_DEEPSEEK_FLOOR_GIB=4 tensorfold serve /models/tensorfold-deepseek \
    --backend cuda --tp 1 --parallel 1 --context 163840 \
    --drafter /models/DSpark-drafter-Q2K-Q8-0731.gguf \
    --name deepseek-v4-flash --host 127.0.0.1 --port 8000
```

After installation the helper is also `tensorfold-deepseek-build`. It validates
headers and embedded tokenizer metadata, builds a platform wheel in an isolated
copy, installs only into its invoking venv, and prepares the model's sidecars.
The base GGUF remains read-only. No model or tokenizer is downloaded; an external
tokenizer is optional and must match vocabulary and EOS. Add `--preflight-only`
for validation without compilation, installation, directory creation or GPU/model
allocation. Header provenance records its SHA-256 and the source's size, mtime,
device and inode; tensor payloads are not hashed in full.

Omit `--drafter` and select `--no-drafts` for the original serial path. With DSpark
attached, request `"draft": false` disables drafting within the same continuous
bank. CUDA's `--drafter auto` leaves drafting disabled until a local GGUF is
selected explicitly; MLX retains its automatic checkpoint selection.

### Native source reuse

The build pins ds4 at `d183482b413ecd2e3b540b290e6497437e9fbb73`.
`cuda/ds4-source.json` records hashes for the 42 required source files, with
attribution retained in `LICENSES/ds4.txt` and the packaged wheel.

A matching library/build receipt is reused first. Otherwise sources are reused
from `~/.cache/tensorfold/ds4/<revision>`. Select an existing tree or checkout with
`--ds4-source /path/to/ds4` or `TENSORFOLD_DS4_SOURCE`. Modified working trees are
left intact: the builder reads the pinned local Git objects. If neither local
sources nor cache exists, Git fetches the pinned revision once. `--offline`
refuses that fetch. Corrupt cached inputs are rejected. Sources remain outside
the checkout and wheel; the wheel contains the manifest, shim, license and library.

The versioned C shim uses the donor's forward, quantization and packed-cache
operations. An isolated child owns native state; fatal exits become parent-side
errors. Logits travel as float32 binary RPC, with eval and its logits transfer
combined during serial decode. Tokenization accepts embedded NUL bytes without
truncating text. Shutdown and bind failure release the worker.

### Sampling and prefix reuse

TensorFold owns target sampling, keyed by seed, absolute position and token ID,
including temperature, top-k, top-p and min-p. A verified build-copy hook exposes
all target verification rows to this sampler. Only committed tokens reach
callbacks. EOS, stops, cancellation and required/named tools use those callbacks.
The native tokenizer joins token bytes before UTF8 decoding.

DSpark uses 1,024-token prefill chunks and retains one canonical prefill checkpoint,
capped at 131,072 tokens independently of request capacity. The snapshot contains
target raw/packed caches, compressor state and all three injected DSpark KV rings.
Injection is maintained when drafting is disabled, permitting mode switches.
Requests restore the common boundary and replay a nonempty suffix with the same
chunk boundaries as a fresh run. Decode-built frontiers and partial fork rewinds
are not reused. Changed prefixes go cold; reset and shutdown release the cache.
The original serial path retains its 2,048-token prefill boundaries.

Two native width-dependent optimizations are disabled in the DSpark worker so
plain and wider verification rows use the same arithmetic. CUDA graph capture
remains enabled; the final capture scan band is bounded by allocated cache
capacity for contexts between powers of two. The build receipt checks the shim
and hook hashes as well as the pinned source and library hashes.

### Memory admission

Admission precedes model loading and preserves an explicit context without
silently shrinking it. It budgets mapped model and drafter weights, additional
aligned Q8 artifacts, active packed cache/workspace, a fixed checkpoint buffer,
1 GiB runtime reserve, additional companion growth and the host memory floor.
`MemAvailable` already reflects resident companions; their growth is added once.
Inactive F32 cache shells are excluded, while packed rows and scratch are counted.
Native allocator fit checks and bounded boot prewarm remain enabled.

`TENSORFOLD_DEEPSEEK_FLOOR_GIB` defaults to 8 and accepts finite values of at least
4. The qualified shared Spark profile uses a 4 GiB floor, 3 GiB companion growth
reserve, 163,840-token context and 131,072-token retained prefix. The companion
allowance was rounded from a measured 2.91 GiB growth peak; measure it again for
other workloads. Build and serve must use the same selected context.

### Verification and measurements

Focused CPU checks cover malformed GGUF and wire layout, schema/tokenizer
provenance, atomic preparation, verified source reuse, isolated build and native
ownership, early memory/option refusal, seeded callbacks, HTTP streaming and tools.
The real native IQ2 primitive is optional on hosts with a built CPU library.

Real-GGUF CUDA regressions are opt-in:

```bash
TENSORFOLD_DEEPSEEK_FLOOR_GIB=4 \
TENSORFOLD_TEST_GPU_MODEL_DIR=/models/tensorfold-deepseek \
TENSORFOLD_TEST_GPU_DRAFTER=/models/DSpark-drafter-Q2K-Q8-0731.gguf \
TENSORFOLD_TEST_GPU_CONTEXT=163840 \
python -m pytest tests/test_deepseek_v4_cuda_resume.py -k dspark -q
```

Set `TENSORFOLD_TEST_GPU_LONG_PREFIX=1` to exercise the full 128 Ki retained
boundary. The tests compare cold and cached seeded output, drafted and plain
requests, mode switches, aligned lengths, changed history and raw-ring wrap.
At 131,093 initial prompt tokens, both warm modes reused 131,072 tokens and
matched the fresh reply. Follow-up prefill measured 0.80 s cached versus 147.76 s
fresh. These are cache-hit/cold timings, not a cold-prefill throughput comparison.

On one GB10 with the qualified base and Q2K/Q8 DSpark, 256-token replies measured:

| Prompt and sampling | Plain | DSpark | Ratio |
| --- | ---: | ---: | ---: |
| Short vector-module prompt, greedy | 19.45 tok/s | 30.86 tok/s | 1.59× |
| Same prompt, temperature 0.8, seed 123 | 19.24 tok/s | 33.45 tok/s | 1.74× |
| 2,428-token prompt, temperature 0.8, seed 456 | 18.00 tok/s | 33.65 tok/s | 1.87× |

These measurements used 262,144-token capacity before workspace tuning. The
1,024-token workspace at 196,608-token capacity subsequently measured 16.63 s
median cold prefill for 15,613 tokens versus 18.32 s with the prior 512-token
workspace at 262,144 capacity: 10.2% higher throughput, from two runs per case.
A 2,048-token workspace failed memory admission. The deployed 163,840-token
profile fits the doubled prefix buffer; the 0.6.1 update preserved recorded
seeded output and measured 16.82 s cold and 0.45 s cached prefill on the HTTP
follow-up fixture. Historical ds4 workloads were unmatched; a vLLM comparison
and full-capacity/long-duration soak are not claimed.
