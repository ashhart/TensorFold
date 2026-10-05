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
kernels (the expert gate/up with SwiGLU in place of GeGLU), `rows.qmv` projections, and the 8-bit head one row a
call. A window of up to 15 rows gives each row the bits of a one-row step, checked at load, so drafts verify
exactly: a drafted reply equals its `"draft": false` reply token for token. Kolibri 1 has no draft head; drafts are
copies of its own context.

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
each, one request at a time (measured with 16-row windows, before the cap below):

| | mlx-lm forward | Row kernels | Change |
| --- | --- | --- | --- |
| GSM8K score | 0.97 | 0.97 | |
| IFEval score | 0.92 | 0.93 | noise |
| GSM8K decode | 99.0 tok/s | 120.4 tok/s | +22% |
| IFEval decode | 101.5 tok/s | 131.8 tok/s | +30% |
| Drafted tokens accepted | | 80,610 of 104,562 | 77% |

Prefill is mlx-lm's forward on both paths: 2,048 cold tokens in 0.58-0.62 s, 8,192 in 2.35-2.65 s, 65,536 in about
33.5 s.

## Limits

- Windows of 15 rows at most. 16-row windows pass the load check, but on real text a row now and then gets other
  bits than its one-row step: 24 of 240 16-row windows at four prompt lengths, none of 1,920 windows of 8 to 15
  rows. A sampled reply with many accepted drafts then differed from its `"draft": false` reply.
- One stream a forward. `hidden_rows` (several streams' windows in one forward) passes the load-time stream check,
  but on some prompts gives one stream other bits than its own call, and which stream depends on the tokens: no fixed
  check can admit it. Concurrent requests take turns, as before the row kernels (about 115 tok/s across 4 or 8
  streams).
- Decode at long context gains nothing yet: at 16.7k tokens the row kernels decode 84.8 tok/s, mlx-lm's forward
  89.5 (within noise). Gemma's attention kernel against MLX's SDPA there is unmeasured.
- The lane matmul backend (`lane_kernels="on"`) loads but is untimed here.
