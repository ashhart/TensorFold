# Float64 reference receipt: LFM2.5-24B-A2B on CUDA

- Checkpoint: [LiquidAI/LFM2.5-24B-A2B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) at revision
  `667408d494bde191c2cb9dd4f4549827816b9275`.
- Engine: this PR's serving commit, before the rebase onto main (`c6645063a`). The rebased head serves the same
  tokens: the token hashes of 8 prompts and of a chat conversation are equal.
  Served on a GB10 (sm_121), greedy, drafts off, `--parallel 8 --context 16384`.
- Reference: `ref64.py` on the CPU in float64 (Python 3.12, torch 2.14, safetensors 0.8, tokenizers 0.23), 2026-10-28.

Reproduce (the reference pass takes about 10 minutes and 17 GB of host memory):

```
zig-out/native/bin/tensorfold-native serve MODEL_DIR --port 8080 --parallel 8 --context 16384
python -I tools/lfm2_moe/ref64.py collect http://127.0.0.1:8080 MODEL_DIR engine.json
python -I tools/lfm2_moe/ref64.py MODEL_DIR engine.json --bf16
```

Rows. A prompt row is the engine's one-token greedy pick after the prefix [0, t] of a prompt, one request per prefix.
A reply row is a token of the engine's 32-token greedy reply, teacher-forced on that reply. `t` is the logits row after
token t, so prompt row P - 1 and reply token 0 are the same position. The prompts (`prompts()` in the script) are
capitals (13 tokens), code (19), a chat turn (19) and items (384, a list of 30 items and a question).

The float64 pass dequantizes the MLX weights exactly. The `--bf16` pass adds the engine's bf16 storage points (see the
script's docstring). It shows which side of a miss rounding alone lands on; it is not bit-exact.

## Result

- f64: prompt rows 421/435, reply rows 123/128
- bf16: prompt rows 424/435, reply rows 123/128

Float64 top-1 misses (margin = f64 logit of its top minus f64 logit of the engine's token; bf16 margin = the same difference in the bf16-points pass, < 0 means it prefers the engine's token):

| case | row | t | engine | float64 | f64 margin | bf16 margin | bf16 picks |
|---|---|---|---|---|---|---|---|
| capitals | prompt | 1 | 589 "'s" | 124901 "<think>" | 0.3389 | -0.1250 (pre-round -0.1026) | engine's |
| capitals | prompt | 3 | 1394 " each" | 278 " the" | 0.0704 | -1.0625 (pre-round -1.0871) | engine's |
| capitals | reply | 39 | 278 " the" | 17755 " Greece" | 0.0095 | +0.0000 (pre-round -0.0276) | engine's |
| code | prompt | 7 | 1288 " return" | 21744 " \"\"\"" | 0.3008 | +0.2500 (pre-round +0.2923) | f64's |
| code | reply | 26 | 4876 "_m" | 71 "_" | 0.3628 | +0.3125 (pre-round +0.2956) | f64's |
| code | reply | 35 | 73 "a" | 2762 "add" | 0.0687 | -0.3750 (pre-round -0.4420) | engine's |
| chat | prompt | 13 | 90978 " Explanation" | 3103 "  \n" | 0.7478 | -0.8125 (pre-round -0.7755) | engine's |
| chat | prompt | 15 | 25 "1" | 3103 "  \n" | 0.1301 | +0.0625 (pre-round +0.0216) | f64's |
| items | prompt | 3 | 22 "." | 34 ":" | 0.9143 | +0.8750 (pre-round +0.8393) | f64's |
| items | prompt | 5 | 2719 " problem" | 3451 " question" | 0.1379 | +0.0625 (pre-round +0.0963) | f64's |
| items | prompt | 6 | 9402 " stone" | 24062 " brick" | 0.1680 | +1.0000 (pre-round +0.9824) | f64's |
| items | prompt | 8 | 229 " " | 267 " a" | 0.0823 | -0.1250 (pre-round -0.0781) | engine's |
| items | prompt | 314 | 27 "3" | 1035 "14" | 0.1292 | -0.3750 (pre-round -0.3704) | engine's |
| items | prompt | 338 | 25 "1" | 26 "2" | 0.3699 | +0.3750 (pre-round +0.4515) | f64's |
| items | prompt | 371 | 437 " are" | 296 " in" | 0.0565 | -0.6250 (pre-round -0.5979) | engine's |
| items | prompt | 375 | 342 " (" | 8187 " item" | 0.3365 | -1.1250 (pre-round -1.1544) | engine's |
| items | prompt | 377 | 342 " (" | 41501 " Item" | 0.6276 | -2.3750 (pre-round -2.3544) | engine's |
| items | reply | 387 | 41501 " Item" | 342 " (" | 0.7665 | +0.6250 (pre-round +0.5957) | f64's |
| items | reply | 396 | 2959 " red" | 6542 " blue" | 0.2816 | +0.5000 (pre-round +0.5314) | f64's |

bf16-points misses where float64 agrees with the engine (margin = bf16 logit of its pick minus the engine's token; f64 margin = how far float64 prefers the engine's token):

| case | row | t | engine | bf16 pick | bf16 margin | f64 margin |
|---|---|---|---|---|---|---|
| capitals | prompt | 10 | 10125 " Italy" | 6398 " Germany" | 0.0000 (pre-round -0.0632) | 0.0079 |
| code | reply | 44 | 1113 " add" | 1619 " main" | 0.6250 (pre-round +0.5908) | 0.1500 |
| code | reply | 47 | 71 "_" | 65296 "_sum" | 0.1250 (pre-round +0.1222) | 0.3751 |
| items | prompt | 7 | 22 "." | 20 "," | 0.0000 (pre-round +0.0882) | 0.0279 |
| items | prompt | 9 | 30 "6" | 27 "3" | 0.5000 (pre-round +0.4381) | 0.1054 |
| items | prompt | 36 | 440 " The" | 41501 " Item" | 0.1250 (pre-round +0.1183) | 0.1051 |
| items | prompt | 379 | 8187 " item" | 229 " " | 0.2500 (pre-round +0.2420) | 0.0420 |
