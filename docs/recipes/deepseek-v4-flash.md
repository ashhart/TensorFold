# DeepSeek-V4-Flash

The `deepseek_v4` family serves `mlx-community/DeepSeek-V4-Flash-4bit` on the MLX lane engine, on a Mac with
256 GB and MLX 0.32.2 or later (`serve` refuses an older MLX). The routed experts keep DeepSeek's mxfp4 bytes and
everything else is affine 4-bit in groups of 64; about 151 GiB stays resident.
Packages: `src/tensorfold/families/deepseek_v4/` and `src/tensorfold/kernels/deepseek/v4/`, with GLM-5.3-Flash's
hyper-connection kernels and row linears.

```bash
tensorfold pull mlx-community/DeepSeek-V4-Flash-4bit
tensorfold serve mlx-community/DeepSeek-V4-Flash-4bit --drafter ~/models/DeepSeek-V4-Flash-dspark
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

The 4-bit checkpoint has no draft head. The family reads two, converted from DeepSeek's MIT-licensed releases, from
the folder `--drafter` names:

- DSpark, used when the folder holds `dspark.safetensors`: three MoE blocks read the target's streams after layers
  40-42 and draft a 5-token block in one pass. Shards 46-48 of `deepseek-ai/DeepSeek-V4-Flash-DSpark` (10.9 GB)
  convert to a 10.7 GB file; the converter copies the block fields from the release's `config.json` beside them.
- MTP, used when the folder holds `mtp.safetensors`: the checkpoint's own next-token layer, from shard 46 of
  `deepseek-ai/DeepSeek-V4-Flash` (3.5 GB converted).

```bash
python -m tensorfold.families.deepseek_v4.convert dspark model-00046-of-00048.safetensors \
    model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
    ~/models/DeepSeek-V4-Flash-dspark/dspark.safetensors
python -m tensorfold.families.deepseek_v4.convert mtp model-00046-of-00046.safetensors \
    ~/models/DeepSeek-V4-Flash-mtp/mtp.safetensors
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
