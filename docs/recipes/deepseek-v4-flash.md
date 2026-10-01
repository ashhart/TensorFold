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

The CUDA adapter reads the existing DeepSeek-V4-Flash-0731 mixed GGUF without converting its weights.
The qualified checkpoint is
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`:
1,328 tensors, 129,280 tokens, 86,720,111,488 bytes (80.76 GiB), with IQ2_XXS routed gate/up,
Q2_K routed down, Q8_0 dense/shared/output and the checkpoint's F16/F32/I32 tensors.
The native window is 1,048,576; the tested allocation is 262,144.

Support is serial: one GB10, `--tp 1 --parallel 1 --no-drafts`. MTP, DSpark, vision, additional GPUs,
concurrent streams and structured output are not implemented by this CUDA adapter. The Mac support above
uses its existing implementation and checkpoints.

### Build and serve

Use Linux aarch64, CUDA 13 and a qualified Python 3.11+ environment containing TensorFold's declared
Python dependencies plus setuptools/wheel. The helper currently compiles for `sm_121a`, the GB10 target;
other CUDA devices are not qualified by this recipe. Two compiler jobs are the default, four the maximum.
Install into a dedicated venv; its invoking interpreter must be that venv's Python.

```bash
python tools/build_deepseek_v4_cuda.py \
    --gguf /models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
    --model-dir /models/tensorfold-deepseek \
    --companion-reserve-gib 3 --context 262144 --jobs 2

tensorfold serve /models/tensorfold-deepseek \
    --backend cuda --tp 1 --parallel 1 --no-drafts --context 262144 \
    --name deepseek-v4-flash --host 127.0.0.1 --port 8000
```

After installation the same helper is available as `tensorfold-deepseek-build`.
It validates the GGUF header/schema and embedded tokenizer, builds a pinned native library and platform
wheel in isolated directories, installs that wheel only in its invoking venv, and prepares `config.json`,
`descriptor.json` and `build-ready.json`. The GGUF stays read-only; no model or tokenizer is downloaded.
A selected external tokenizer is optional and must match the vocabulary and EOS.
The library cache checks source, shim and library hashes before reuse. First native compilation is required;
subsequent matching builds reuse it. The wheel contains native sources, attribution and the built library.

Add `--preflight-only` to the helper command for header-only validation. It does not compile, install,
create directories, load weights or allocate KV. Artifact validity is separate from current memory fit.
The receipt hashes the 5,333,824-byte GGUF header, not all 81 GiB of tensor payload; it also records file
size, mtime, device and inode. The validated embedded EOS is ID 1, `<｜end▁of▁sentence｜>`.

An independent ml-infra deployment uses `make tensorfold-deepseek-3d`; it owns service lifecycle,
companion readiness and `0.0.0.0` binding for Tailscale. Those local settings do not change TensorFold's
CLI defaults. Its original ds4 service stays down after the operator-authorized replacement; failed
candidate startup does not automatically restart that service.

### Native boundary and serving

The adapter packages unchanged MIT-licensed ds4 sources pinned at
`d183482b413ecd2e3b540b290e6497437e9fbb73`. `cuda/vendor/ds4/tensorfold-source.json` records each file's
SHA-256; notices and license copies retain the ggml/ds4 attribution. CUDA arithmetic, routing,
hyper-connections and packed-cache math are reused from that implementation.

A small versioned C shim exposes engine/session lifecycle, prompt sync, eval, logits and embedded
rendered-chat tokenization. An isolated child owns the native library and session, so a fatal C exit
becomes a bounded parent-side error. Logits use binary float32 RPC; eval and the following logits transfer
share one command during decode. TensorFold owns the HTTP server, request policy, keyed sampling and
committed callbacks. No ds4 HTTP server is started or proxied.

The family lazily exports `cuda_engine` and `CUDA_APP`; family discovery imports neither torch nor MLX.
The app reuses the existing DeepSeek prompt encoder, thinking conventions and DSML parser through shared
CUDA HTTP hooks. Token bytes are joined before UTF8 decoding; instance-local bounded caches avoid
repeated tokenizer RPCs. Required/named tools reuse the existing call gate without assuming the DSML
opener is one token. HTTP shutdown and bind failure close the native engine.

Each request starts a fresh native timeline. Prompt-resume snapshots and prefix reuse are outside this
serial release: the upstream resumed/fresh arithmetic contract needs separate qualification.
Callbacks observe evaluated tokens once, EOS/stop/disconnect terminate generation, and prompt-plus-reply
capacity is checked before streaming. Fatal native failures close the engine instead of silently serving
from damaged state.

### Memory admission and fast kernels

Admission precedes native model loading. It validates prepared/source identity, preserves an explicit
context without shrinking it, and budgets the complete packed file, aligned-artifact overhead, active
cache/workspace geometry, 1 GiB runtime allowance, additional companion growth and an 8 GiB host floor.
`MemAvailable` already includes resident companion occupancy; only their additional growth is reserved.
The local 3 GiB allowance is rounded from an independently measured 2.91 GiB Hunyuan generation peak.
Re-measure it for a different companion workload.

Aligned artifacts replace the IQ2/Q2 expert residency; dense Q8 artifacts add 6,598,885,376 bytes.
They are counted without budgeting another full copy of the experts. The native library builds the
474 artifacts (78.71 GiB) at startup. Full-file host registration is skipped by the donor's complete
replacement policy. Keeping this path enabled is necessary for the fast aligned IQ2/Q2/Q8 kernels.

FP8 compressed KV and FP4 indexer primaries stay enabled. The donor's logical graph estimate includes
inactive F32 primary shells; the shim subtracts only those uncommitted shells while retaining every
packed row and all workspaces in its resident quote. Both logical and resident estimates are reported.
At context 262,144 and prefill batches of 2,048, the quotes are 8,300,789,656 logical bytes and
4,693,498,776 resident-budget bytes. The context remains 262K; the batch size controls temporary memory.
The native allocator's own fit checks remain enabled.

Shared top-k sampling partitions the vocabulary before sorting candidates, including all threshold ties.
It preserves token-ID ordering, nucleus/min-p rules and seed/position-keyed draws. A 129,280-logit CPU
measurement fell from 8.44 to 0.70 ms per sampled token without changing the recorded seeded picks.

### Verification and measurements

Focused CPU tests cover GGUF wire format and malformed input, real schema/provenance, stored Q2/Q8
vectors, a pinned IQ2 primitive, option refusal before allocation, insufficient-memory sentinels,
callback/EOS/cancellation/lifetime, shared HTTP streaming/tools, shutdown and build/preflight boundaries.
Packaged-source hashes are checked without depending on another local donor checkout.

Real GB10 inference passed chat/code, thinking, SSE termination, required tool calls and a tool followup.
With chat decoding at the same time, Hunyuan generated a 15,375,500-byte GLB at 30 steps/octree 256;
CUDA moderation also passed. The API reports context_length=262144. The qualification harness and
machine-specific raw receipts remain in ml-infra and the operator's qualification directory, rather than
shipping personal paths and old board plans in the TensorFold PR.

Historical optimization measurements on that same Spark and GGUF:

| Check | Before | After | Scope |
| --- | --- | --- | --- |
| Sampled decode, 256 tokens | 18.24 tok/s | 20.44 tok/s | Same prompt, seed and output token hash; sampler optimization |
| 16,224-token prefill | — | 927.6 tok/s | Aligned kernels, 1,024-token batches; one cold prompt |
| Thinking decode | 16.07 tok/s | 21.11 tok/s | Same fixture; output lengths differ, before/after aligned kernels |

These are local serial measurements, not evidence of a general speedup over every backend.
The old ds4 log recorded about 1,000 prefill tok/s and about 21 decode tok/s on other requests;
those logs are not a controlled head-to-head comparison. The full 262,144-token prompt, long-duration
soak, independent exhaustive GPU oracle and draft/concurrent exactness are not claimed. Re-run the
[public benchmark fixtures](README.md#measurements) on the final installed artifact before publishing
comparative performance results.

### Contributor / PR notes

Family adapters remain under `families/deepseek_v4/cuda/`; shared HTTP and sampling changes are small
hooks/optimizations. The pinned donor tree is excluded from formatting and must remain byte-identical.
Native ABI changes require matching shim/library receipts; Python-only changes reuse CUDA objects.
TensorFold 0.6.0's Apache-2.0 license is retained, with the donor's MIT notices packaged separately.

The feature branch is `feat/deepseek-v4-gguf-cuda`, based on upstream 0.6.0. The fork remote is `origin`
(`crescit/TensorFold`); `upstream` is `ashhart/TensorFold`. Suggested PR title:
`feat(deepseek_v4 cuda): serial mixed-GGUF inference on one Spark`.
The PR describes the pinned native integration, early memory admission, embedded serving, isolated
wheel builder and exact top-k sampling optimization, with the supported serial limits above.
