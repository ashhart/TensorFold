# Kolibri 1 (`kolibri1`)

Measured on an M5 Max 128 GB, MLX 0.32.3, `velaia/Kolibri-1-MLX-4bit`. Packages:
`src/tensorfold/families/kolibri1/` and `src/tensorfold/kernels/kolibri/v1/`.

## Model

50 layers of Gemma 4-style attention (48 query heads, 4 KV heads, head dim 128; sliding window 513 with RoPE,
full-attention layers without position) and an MoE block: 384 experts, top 6 by `sigmoid + bias`, width 512, SwiGLU,
beside a shared expert of width 512. Hidden size 2,560, vocabulary 128,000. 4-bit affine weights in groups of 64,
8-bit embedding and head. The model code is mlx-lm's port (`vendor/kolibri1.py`, see
[THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES.md)), registered as `mlx_lm.models.kolibri1` until mlx-lm ships it.

## Decode

Prompts run through mlx-lm's forward. Every decode row runs through `RowDecode`: Gemma 4's attention, glue and expert
kernels (the expert gate/up with SwiGLU in place of GeGLU), `rows.qmv` projections, GLM's gemv rows for the bf16
router, and the 8-bit head one row a call. A window of up to 16 rows gives each row the bits of a one-row step,
checked at load, so drafts verify exactly: a drafted reply equals its `"draft": false` reply token for token.
Kolibri 1 has no draft head; drafts are copies of its own context. Up to 8 streams share a forward of up to 32 rows,
each stream's rows with the bits of its own call.

An 8-bit checkpoint (`python -m tensorfold.families.kolibri1.convert` from Aleph Alpha's FP8 one) decodes one row a
step through mlx-lm's forward: the row kernels read 4-bit weights.

## Speed

Server, `--no-thinking`, `tools/bench_openai.py` (512 tokens, 5 reps), median decode tok/s:

| Prompt | Temperature | mlx-lm forward, one row a step | Row kernels with context copies |
| --- | --- | --- | --- |
| fibonacci-raw | 0 | 107.0 | 121.4 |
| gpu-chat-no-think | 0 | 108.4 | 116.4 |
| fibonacci-raw | 1.0 | 107.0 | 118.2 |
| gpu-chat-no-think | 1.0 | 106.7 | 115.8 |

Reasoning repeats itself, so context copies land more often with thinking on. localeval, thinking on, 100 problems
each, one request at a time:

| | mlx-lm forward | Row kernels | Change |
| --- | --- | --- | --- |
| GSM8K score | 0.97 | 0.97 | |
| IFEval score | 0.92 | 0.93 | noise |
| GSM8K decode | 99.0 tok/s | 122.7 tok/s | +24% |
| IFEval decode | 101.5 tok/s | 131.0 tok/s | +29% |
| Drafted tokens accepted | | 78,352 of 100,721 | 78% |

Prefill is mlx-lm's forward on both paths: 2,048 cold tokens in 0.58-0.62 s, 8,192 in 2.35-2.65 s, 65,536 in about
33.5 s.

## Limits

- The router runs GLM's gemv rows, not MLX's matmul. At Kolibri 1's router shape (2,560 -> 384), MLX's bf16 matmul
  gives a row among 6, 15 or 32 other bits than alone, now and then routing it to other experts: a shared forward
  then decoded other tokens for one of its streams, and a 16-row window missed its one-row steps (#434).
- Concurrent requests (server, `tools/bench_concurrent.py --alone --serial`, every reply equal to its solo run):

  | Streams | Aggregate tok/s |
  | --- | --- |
  | 4 | 179-195 |
  | 8 | 207-227 |

- Decode at long context gains nothing yet: at 16.7k tokens the row kernels decode 84.8 tok/s, mlx-lm's forward
  89.5 (within noise). Gemma's attention kernel against MLX's SDPA there is unmeasured.
- The lane matmul backend (`lane_kernels="on"`) loads but is untimed here.
