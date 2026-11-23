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
| Build | #4337 at `a0935b5279` (the binary measured), `zig build native -Dcpu=apple_m1 -Doptimize=ReleaseFast`, Zig 0.17.0, macOS 27.2 |
| Reference | `mlx_lm.server --model "$CKPT" --host 127.0.0.1 --port 8099`, mlx 0.32.4, mlx-lm `0.1.dev1006+g0c4c99192` ("Add Inkling 2 Mini"), same checkpoint, no drafter |
| Native servers | `--context 65552 --parallel 1 --no-drafts --prompt-cache-gib 0` (decode, cold prompts); `--context 8192 --parallel 4 --prompt-cache-gib 0`, with and without `--no-drafts` (concurrency) |
| Requests | greedy, `tools/bench_openai.py --tokens 256 --reps 3 --temperatures 0`; `tools/bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4`; cold prompts of exactly 2,048, 8,192, 32,768 and 65,536 tokens through `/v1/completions`, one token out, cached tokens 0 |
| Conditions | one model process at a time; an unrelated CPU job used about 60 % of a core throughout, so read ratios between rows, not the absolute rates |

Decode, one stream (tokens a second, median of three):

| Prompt | MLX | Native plain | Native DSpark |
| --- | --- | --- | --- |
| `fibonacci-raw` | 23.05 | 31.12 | 35.76 |
| `gpu-chat-no-think` | 23.69 | 31.07 | 35.01 |

Concurrency, `bench_concurrent` (aggregate tokens a second, per stream in brackets; slowest first token in seconds):

| Prompt, streams | MLX | Native plain | Native DSpark |
| --- | --- | --- | --- |
| code, 1 | 23.4, 0.25 s | 26.5, 0.13 s | 36.4, 0.14 s |
| code, 4 | 23.3 (23.6), 33.8 s | 64.5 (16.4), 0.50 s | 55.4 (15.5), 0.55 s |
| chat, 1 | 23.3, 0.37 s | 30.6, 0.24 s | 33.8, 0.25 s |
| chat, 4 | 23.3 (23.9), 33.9 s | 64.4 (16.3), 0.97 s | 60.4 (15.5), 1.00 s |

The reference server answers four requests one after another here. A DSpark server times its verify forward by total
rows at load (2.6 s), so the lane host prices shared rounds: it drafted every stream 7 deep at four streams before (213 ms
for 32 rows, 54.7 and 39.8 tokens a second in an earlier run) and now drafts only where a priced round beats plain. At
four streams DSpark is still 6 to 14 % under plain in these runs, and at one stream it is 10 to 37 % over it.

Cold first token (seconds, tokens a second in brackets):

| Prompt tokens | MLX | Native |
| --- | --- | --- |
| 2,048 | 4.68 (437) | 5.49 (373) |
| 8,192 | 19.53 (420) | 22.53 (364) |
| 32,768 | 81.74 (401) | 108.27 (303) |
| 65,536 | 180.27 (364) | 249.21 (263) |

Peak physical footprint (`footprint`, 1 GB resolution, after an 8,192-token prompt and a 128-token reply): native about
24 GB, with `--parallel 1 --context 65552` and with `--parallel 4 --context 8192` and DSpark alike; MLX about 23 GB.

Plain and drafted greedy replies are identical. These token hashes (the server's `sha=`, chat completions at
`--context 8192`) are the same on this binary and on the head before the last prompt-kernel changes: the Fibonacci
question with 8 and 64 tokens out `0b037128ee78` and `b37d8a0e39c4`, a 416-token prompt and 16 tokens
`77bf7c99cc66`, a 1,036-token prompt and 24 tokens `538175732cc7`.

Native decode is ahead of the reference with and without the drafter. Cold prefill is 0.72 to 0.87 of the reference's
rate and falls more with the prompt's length (0.87 at 8,192 tokens, 0.72 at 65,536). Not measured here: other chips, a
36 GB Mac, images served, a quality suite.
