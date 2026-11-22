# Inkling 2 Mini, native Metal

This recipe serves the original [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
checkpoint, revision `ed40e7da625f594fc9c4990f5f66be24873c22d7`, with its bundled DSpark head
(`mtp.safetensors`), on native Metal. Build with Zig **0.17.0** and Xcode's Metal toolchain, as described in
[the native preview](../zig-preview.md#build). It is experimental and is not release qualification.

```bash
hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 --local-dir "$HOME/models/inkling-2-mini"
zig build native -Dcpu=apple_m1
zig-out/native/bin/tensorfold-native serve "$HOME/models/inkling-2-mini" --port 8888 --context 65552 --parallel 1 \
  --no-drafts --prompt-cache-gib 0
```

Drop `--no-drafts` for DSpark. A DSpark server refuses `--prompt-cache-gib`, so kept prompt states need `--no-drafts`.
`--parallel N` grants as many streams as the weights, the scratch and N states at the chosen `--context` fit in the
load limit; the startup line prints the demand and a named count that does not fit is refused by name.

## What runs

The routed experts stay packed (NVFP4 and MXFP8) and are decoded to BF16 in registers; activations are BF16 and every
sum is FP32. The 24 physical layers run as 48 visits with the loop's gates and cross-pass message. Every output is
byte-equal to the one-row kernels: prompt chunks of 64 rows or more group the route pairs by expert and sum on
simdgroup matrix multiplies in the one-row chain's order, and the kernel contract (`tf-kernel-contract --only inkling2`)
holds every kernel and a planted control for each rule. Images (PNG, JPEG) go through the vision tower.

## Receipt: Apple M4 Pro, 48 GiB

| | |
| --- | --- |
| Build | head of #4337 at `4b3d076ec6`, `zig build native -Dcpu=apple_m1 -Doptimize=ReleaseFast`, Zig 0.17.0, macOS 27.2 |
| Reference | `mlx_lm.server --model "$CKPT" --host 127.0.0.1 --port 8099`, mlx 0.32.4, mlx-lm `0.1.dev1006+g0c4c99192` ("Add Inkling 2 Mini"), same checkpoint, no drafter |
| Native servers | `--context 65552 --parallel 1 --no-drafts --prompt-cache-gib 0` (decode, cold prompts); `--context 8192 --parallel 4 --prompt-cache-gib 0`, with and without `--no-drafts` (concurrency) |
| Requests | greedy, `tools/bench_openai.py --tokens 256 --reps 3 --temperatures 0`; `tools/bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4`; cold prompts of exactly 2,048, 8,192, 32,768 and 65,536 tokens through `/v1/completions`, one token out, cached tokens 0 |
| Conditions | one model process at a time; an unrelated CPU job used about 60 % of a core throughout, so read ratios between rows, not the absolute rates |

Decode, one stream (tokens a second, median of three):

| Prompt | MLX | Native plain | Native DSpark |
| --- | --- | --- | --- |
| `fibonacci-raw` | 22.84 | 29.05 | 35.33 |
| `gpu-chat-no-think` | 22.54 | 29.00 | 35.33 |

Concurrency, `bench_concurrent` (aggregate tokens a second, per stream in brackets; slowest first token in seconds):

| Prompt, streams | MLX | Native plain | Native DSpark |
| --- | --- | --- | --- |
| code, 1 | 22.9, 0.23 s | 30.4, 0.13 s | 37.2, 0.14 s |
| code, 4 | 22.4 (22.8), 34.7 s | 63.3 (16.1), 0.50 s | 63.5 (16.0), 0.53 s |
| chat, 1 | 23.2, 0.37 s | 28.4, 0.24 s | 35.0, 0.24 s |
| chat, 4 | 22.6 (23.1), 34.6 s | 61.5 (15.8), 0.94 s | 62.1 (15.9), 0.99 s |

The reference server answers four requests one after another here. A DSpark server times its verify forward by total
rows at load (2.6 s), so the lane host prices shared rounds: at four streams a round of 4 rows costs 58 ms and each
extra verify row about 10 ms plus a 10 ms head block a stream, so the rule drafts four streams only where that pays and
DSpark matches plain (it drafted 7 deep for every stream before, 213 ms for 32 rows, and ran at 54.7 and 39.8). Rounds
before a stream count's first plain price (about 128 rounds) still draft by the table alone.

Cold first token (seconds, tokens a second in brackets):

| Prompt tokens | MLX | Native |
| --- | --- | --- |
| 2,048 | 4.67 (439) | 8.15 (251) |
| 8,192 | 17.83 (460) | 37.54 (218) |
| 32,768 | 77.67 (422) | 179.76 (182) |
| 65,536 | 180.57 (363) | 422.01 (155) |

Peak physical footprint (`footprint`, after an 8,192-token prompt and a 128-token reply): native about 23 GB, with
`--parallel 1 --context 65552` and with `--parallel 4 --context 8192` and DSpark alike; MLX about 24 GB.

Plain and drafted greedy replies are identical, and these token hashes (the server's `sha=`) match the previous head's:
8 tokens `4c2031fb1535`, 64 tokens `57519a5a167f`, a 416-token prompt and 16 tokens `77bf7c99cc66`, a 1,036-token
prompt and 24 tokens `538175732cc7`.

Native decode is ahead of the reference with and without the drafter, and cold prefill is not: native reaches 0.43 to
0.57 of the reference's rate. Not measured here: other chips, a 36 GB Mac, images served, a quality suite.
