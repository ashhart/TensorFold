# Float64 reference receipt: LFM3-1.7B NVFP4 on CUDA

- Checkpoint: [LiquidAI/LFM3-1.7B-NVFP4](https://huggingface.co/LiquidAI/LFM3-1.7B-NVFP4) at revision
  `b30025bbbbcbf62f6c110b40d51d36af6579f6a2`.
- Engine: `9265a0f5f`, which is the PR head `08b338c7c` before its rebase (range-diff: the same patches except one
  test-only line in `contract_limits.zig`). Served on a GB10 (sm_121), greedy, drafts off,
  `--parallel 8 --context 32768`. The `collect` step was repeated on 2026-10-28 and returned the same 957 tokens.
- Reference: `ref64.py` on the CPU in float64 (Python 3.12, torch 2.14, safetensors 0.8, tokenizers 0.23), 2026-10-28.

Reproduce (the reference pass takes about a minute and 20 GB of host memory):

```
zig-out/native/bin/tensorfold-native serve MODEL_DIR --port 8080 --parallel 8 --context 32768
python -I tools/lfm3/ref64.py collect http://127.0.0.1:8080 MODEL_DIR engine.json
python -I tools/lfm3/ref64.py MODEL_DIR engine.json --bf16
```

Rows. A prompt row is the engine's one-token greedy pick after the prefix [0, t] of a prompt, one request per prefix.
A reply row is a token of the engine's 32-token greedy reply, teacher-forced on that reply. `t` is the logits row after
token t, so prompt row P - 1 and reply token 0 are the same position. The prompts (`prompts()` in the script) are
capitals (20 tokens), code (40), a chat turn (19) and items (750, a list of 60 items and a question; it crosses the
256-key sliding window).

The float64 pass dequantizes the NVFP4 weights exactly. The `--bf16` pass adds the engine's bf16 storage points (see
the script's docstring). It shows which side of a miss rounding alone lands on; it is not bit-exact.

## Result

- f64: prompt rows 825/829, reply rows 127/128
- bf16: prompt rows 827/829, reply rows 126/128

Float64 top-1 misses (margin = f64 logit of its top minus f64 logit of the engine's token; bf16 margin = the same difference in the bf16-points pass, < 0 means it prefers the engine's token):

| case | row | t | engine | float64 | f64 margin | bf16 margin | bf16 picks |
|---|---|---|---|---|---|---|---|
| capitals | reply | 24 | 11097 " Spain" | 3888 " Japan" | 0.1176 | +0.1250 (pre-round +0.1802) | f64's |
| chat | prompt | 12 | 593 " per" | 22 "." | 0.1530 | -0.0625 (pre-round -0.0671) | engine's |
| items | prompt | 5 | 1008 " first" | 1579 " number" | 0.0996 | +0.0000 (pre-round +0.1009) | engine's |
| items | prompt | 9 | 29 "5" | 654 "10" | 0.0033 | +0.0000 (pre-round +0.0093) | engine's |
| items | prompt | 180 | 28 "4" | 30 "6" | 0.0674 | +0.0000 (pre-round +0.0076) | engine's |

bf16-points misses where float64 agrees with the engine (margin = bf16 logit of its pick minus the engine's token; f64 margin = how far float64 prefers the engine's token):

| case | row | t | engine | bf16 pick | bf16 margin | f64 margin |
|---|---|---|---|---|---|---|
| items | prompt | 4 | 440 " The" | 229 " " | 0.0000 (pre-round -0.0985) | 0.0881 |
| items | prompt | 631 | 32 "8" | 25 "1" | 0.0000 (pre-round -0.0467) | 0.0636 |
| items | reply | 772 | 41501 " Item" | 1672 " So" | 0.0000 (pre-round -0.0814) | 0.1797 |

## FP8 and MLX-4bit (2026-10-29)

- Checkpoints: [LiquidAI/LFM3-1.7B-FP8](https://huggingface.co/LiquidAI/LFM3-1.7B-FP8) at
  `243955da95b1cd68026f293b29de4b4c13f3638f` (block FP8, dequantized exactly: e4m3 times its block's bf16 scale) and
  [LiquidAI/LFM3-1.7B-MLX-4bit](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-4bit) at
  `d0ca73dcfa70ce18733754a492cdc1fb0123262d` (MLX affine, dequantized exactly).
- Served the same way with `--no-drafts`; the same prompts and rows. Only the float64 pass was run: the `--bf16`
  pass models the NVFP4 engine's rounding points.

FP8: prompt rows 824/829, reply rows 128/128.

| case | row | t | engine | float64 | f64 margin |
|---|---|---|---|---|---|
| items | prompt | 131 | 30 "6" | 1325 "24" | 0.0348 |
| items | prompt | 582 | 654 "10" | 1035 "14" | 0.0756 |
| items | prompt | 619 | 25 "1" | 29 "5" | 0.0066 |
| items | prompt | 631 | 27 "3" | 32 "8" | 0.0290 |
| items | prompt | 693 | 9402 " stone" | 17937 " stones" | 0.0773 |

MLX-4bit: prompt rows 825/829, reply rows 128/128.

| case | row | t | engine | float64 | f64 margin |
|---|---|---|---|---|---|
| items | prompt | 10 | 6802 " items" | 93266 " marbles" | 0.0466 |
| items | prompt | 61 | 41501 " Item" | 29155 " Question" | 0.0042 |
| items | prompt | 595 | 27 "3" | 1035 "14" | 0.0300 |
| items | prompt | 748 | 34 ":" | 10494 " holds" | 0.0848 |

## Without torch: the numpy path and the XPU's bf16 points

`ref64.py` runs on numpy when torch is not installed (or with `--numpy`): MLX affine and bf16 weights are dequantized
exactly as on the torch path; NVFP4 and FP8 exports are refused by name and need torch. `--bf16-points` replaces the
`--bf16` pass with the Intel GPU (XPU) engine's storage points: the embedding rows and RoPE cos/sin rounded to bf16 too,
attention P and the logits never. `dump` writes the float64 hidden state after every layer for the XPU forward's
layer-by-layer check, and `check` replays an engine's greedy replies teacher-forced:

```
python -I tools/lfm3/ref64.py dump MODEL_DIR OUT_DIR
python -I tools/lfm3/ref64.py check MODEL_DIR tokens.json      # {"name": {"prompt": [...], "reply": [...]}}
python -I tools/lfm3/ref64.py MODEL_DIR engine.json --bf16-points
```

On LiquidAI/LFM3-1.7B-MLX-4bit (Python 3.12, numpy 2.5, no torch; 3 min 14 s and 4.1 GB for `dump`): the three
prompts' bf16-points gap reaches 0.041 (capitals), 0.025 (chat) and 0.111 (items, worst row of any layer), and the
XPU engine's 64-token greedy replies agree with the float64 top-1 at 192 of 192 positions.
