# Inkling 2 Mini, native Metal

This recipe serves the original [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
checkpoint, revision `ed40e7da625f594fc9c4990f5f66be24873c22d7`, with its bundled DSpark head
(`mtp.safetensors`), on native Metal. Build with Zig **0.17.0** and Xcode's Metal toolchain, as described in
[the native preview](../zig-preview.md#build). It is experimental and is not release qualification: the contract and
served runs are on M4-class chips, there is no Linux backend in this change, and the platforms table below says what
was not run.

```bash
hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 --local-dir "$HOME/models/inkling-2-mini"
zig build native -Dcpu=apple_m1
zig-out/native/bin/tensorfold-native serve "$HOME/models/inkling-2-mini" --port 8888 --context 65552 --parallel 1 \
  --no-drafts --prompt-cache-gib 0 --no-thinking
```

Drop `--no-drafts` for DSpark. A DSpark server refuses `--prompt-cache-gib`, so kept prompt states need `--no-drafts`.
Add `--vision` for images. `--parallel N` grants as many streams as the weights, the scratch and N states at the chosen
`--context` fit in the load limit; the startup line prints the demand and a named count that does not fit is refused by
name. A 65,536-token prompt needs `--context 65552`: the receipt's longest cold row is served at that window.
`--context 65536` refused a 65,536-token prompt in an earlier run the PR description records; it was not repeated here.

## The model

The checkpoint's own `config.json` (`inkling2_loop`) and the family's `config.zig` give the shape:

- 24 physical layers, hidden 2,304, 24 query heads over 8 key heads of 128, vocabulary 201,024. Layers 5, 11, 17 and 23
  are global; the other 20 slide over 1,024 keys.
- A loop: layers 6 to 17 run in three passes, so a forward makes 48 layer visits (6 + 3 x 12 + 6) with the loop's gates
  and a cross-pass message between passes. States and taps index by visit: 8 of the 48 are global.
- 128 routed experts a layer, 8 picked a token, width 1,152, no shared expert. Layers 0 to 19 store their experts as
  NVFP4; layers 23 to 20, the last four, store MXFP8 (`hf_quant_config.json`). The weights stay packed in memory:
  18.19 GiB resident, 18.95 GiB with the dense layers and scratch (the startup line).
- A vision tower (23 layers, 20-pixel patches) and a DSpark head of three layers with a block size of 8.

## What runs

The routed experts stay packed (NVFP4 and MXFP8) and are decoded to BF16 in registers; activations are BF16 and every
sum is FP32. The 24 physical layers run as 48 visits with the loop's gates and cross-pass message. The grouped paths
produce the one-row kernels' bytes: prompt chunks of 64 rows or more group the route pairs by expert, decode each
expert's weights once for a tile and sum on simdgroup matrix multiplies in the one-row chain's order; drafted verify
rounds of 2 to 63 rows plan pairs by expert in tiles of four; the dense prompt projections and the loop gates' state and
down chains run on an MMA tile that keeps the one-row chains; prompt chunks' global layers score and weight eight rows
at a time on 8x8 matrices, each row keeping its key blocks, running max and sum. The DSpark head runs every drafting
stream of a round in one pass. Admission counts the weights, the scratch and each stream's state bytes (a cache slot per
visit, at the configured `--context`) against the load limit, and a `warning` memory-pressure level no longer serializes
streams that fit; `critical` still does. The kernel contract (`tf-kernel-contract --only inkling2`) registers every
kernel with a planted control for each rule it states. Images (PNG, JPEG) go through the vision tower.

## Platforms

| Chip | What ran | Where |
| --- | --- | --- |
| M4 Pro, 48 GiB | The served receipt below, the contract and the host tests | Here, at the heads named in the receipt and under Verification |
| M4 Max, 128 GiB | `tf-kernel-contract --only inkling2` only: 86 kernels pass, 132 controls red, 0 missed, 0 limit problems at `e6cb02e09e` | A contributor (@akol1); no served run |
| M5 Ultra | Plain decode 55.5 and 55.4 tok/s over 16 chat prompts, two blocks, at `e6cb02e09e`; DSpark 59.2 and 59.1 against plain 52.1 and 52.0 at `8f4071726b`; the contract at `e6cb02e09e`: 86 pass, 131 controls red, 0 missed | A contributor (@SashaMIT), not reproduced here; the contract line counted 131 controls where the M4 chips count 132 |
| M3 Pro, 36 GB | Contract 86/86, 80 controls red, and plain decode 12.0 to 12.4 tok/s with DSpark 14.4 to 14.6, at `f3a6a992f2` | A contributor (@jschmied); not rerun at any later head |
| M1, M2 | Not run | |
| M5 tensor-unit prompt path | Not written; the text path has no tensor-unit kernel (only the vision tower's GEMMs use them) | |

The grouped MMA paths assume an 8x8x8 `simdgroup_matrix` multiply-accumulate adds its products in k order from the
accumulator. A probe on the M4 Pro found 262,144 of 262,144 bytes equal to an FMA chain, and the contract holds on the
M4 Max and the M5 Ultra rows above.

## Receipt: Apple M4 Pro, 48 GiB

Raw results, the generator of the prompt token arrays, hashes and the build's SHA-256 are in
[`receipts/inkling2-mini-m4pro`](../receipts/inkling2-mini-m4pro/README.md), which says which file each number comes
from. The decode, cold-prompt (table), concurrency and footprint numbers were not measured at the head; the contract,
the hashes and the dense-tile cold first-token pairs below were measured at `61be67917f`.

| | |
| --- | --- |
| Build | Decode, cold prompts, footprint: #4337 at `a0935b5279`, the PR's recipe-time binary; concurrency: the head-pass build that became `05b6fdb5c0`, plain on its parent. `zig build native -Dcpu=apple_m1 -Doptimize=ReleaseFast`, Zig 0.17.0, macOS 27.2 |
| Reference | `mlx_lm.server --model "$CKPT" --host 127.0.0.1 --port 8099`, mlx 0.32.4, mlx-lm `0.1.dev1006+g0c4c99192` ("Add Inkling 2 Mini"), same checkpoint, no drafter |
| Native servers | `--context 65552 --parallel 1 --no-drafts --prompt-cache-gib 0 --no-thinking` (plain decode, cold prompts); `--context 8192 --parallel 4 --prompt-cache-gib 0 --no-thinking`, with and without `--no-drafts` (concurrency, and the DSpark decode column) |
| Requests | greedy, `tools/bench_openai.py --tokens 256 --reps 3 --temperatures 0`; `tools/bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4`; cold prompts of exactly 2,048, 8,192, 32,768 and 65,536 tokens through `/v1/completions`, one token out, cached tokens 0 |
| Conditions | one model process at a time; an unrelated CPU job used about 60 % of a core during the decode and cold runs, so read ratios between rows, not the absolute rates |

Decode, one stream (tokens a second, median of three):

| Prompt | MLX | Native plain | Native DSpark |
| --- | --- | --- | --- |
| `fibonacci-raw` | 23.05 | 31.12 | 35.76 |
| `gpu-chat-no-think` | 23.69 | 31.07 | 35.01 |

Concurrency, `bench_concurrent` (aggregate tokens a second, per stream in brackets; slowest first token in seconds):

| Prompt, streams | MLX | Native plain | Native DSpark |
| --- | --- | --- | --- |
| code, 1 | 23.4, 0.25 s | 31.1, 0.12 s | 37.7, 0.13 s |
| code, 4 | 23.3 (23.6), 33.8 s | 67.0 (16.9), 0.52 s | 65.1 (16.6), 0.52 s |
| chat, 1 | 23.3, 0.37 s | 30.9, 0.23 s | 35.8, 0.23 s |
| chat, 4 | 23.3 (23.9), 33.9 s | 66.8 (17.1), 0.90 s | 66.9 (17.1), 0.91 s |

MLX answers the four requests one after another here, which is why its aggregate stays at one stream's rate and its
slowest first token is 34 s; its four-stream numbers are not a same-width comparison. Native columns are one A-B-B-A
session (plain, DSpark, DSpark, plain; the mean of each pair). A DSpark server prices its shared rounds from a table of
timed verify forwards and drafts only where a priced round beats plain. The head runs the drafting streams of a round in
one pass and a plain round runs no head work, so four DSpark streams are within noise of plain (code 3 % under inside
the 65.6 to 68.3 spread of the two plain runs, chat level) and one stream is 15 to 22 % over it.

Cold first token (seconds, tokens a second in brackets):

| Prompt tokens | MLX | Native |
| --- | --- | --- |
| 2,048 | 4.68 (437) | 5.49 (373) |
| 8,192 | 19.53 (420) | 22.53 (364) |
| 32,768 | 81.74 (401) | 108.27 (303) |
| 65,536 | 180.27 (364) | 249.21 (263) |

The dense projection tile (`61be67917f`, a different build from the table) took the first token from 23.8 s to 22.7 s at
8,192 tokens (4.7 %) and from 104.8 s to 102.0 s at 32,768 (2.7 %): one A-B-B-A session, the mean of each pair, plain
server, same hashes. The table above predates it.

Peak physical footprint (`footprint`, 1 GB resolution, sampled every 3 s through an 8,192-token prompt and a 128-token
reply): native about 24 GB, with `--parallel 1 --context 65552` and with `--parallel 4 --context 8192` and DSpark alike;
MLX about 23 GB.

Plain and drafted greedy replies are identical. These token hashes (the server's `sha=`, chat completions at
`--context 8192`) are the same on four binaries from the session: the Fibonacci question with 8 and 64 tokens out
`0b037128ee78` and `b37d8a0e39c4`, a 416-token prompt and 16 tokens `77bf7c99cc66`, a 1,036-token prompt and 24 tokens
`538175732cc7`. A drafted four-stream run with `--alone --serial` compared 30 replies with their runs alone and 4 with
`"draft": false`: all equal.

Native decode is ahead of the reference with and without the drafter. Cold prefill is 0.72 to 0.87 of the reference's
rate and falls more with the prompt's length (0.87 at 8,192 tokens, 0.72 at 65,536). Not measured here: M1, M2, a 36 GB
Mac, images served, a quality suite.

Sampled decode, `bench_concurrent --temperatures 0.7 --seed 1234 --alone --serial --levels 1,4 --tokens 256 --reps 3`
(aggregate tokens a second), `--context 8192 --parallel 4 --prompt-cache-gib 0 --no-thinking`, on an idle machine, the
build of `61be67917f`: plain (`--no-drafts`), DSpark, DSpark again, plain again; the table gives each server's mean over
its pair of runs, the two runs in brackets:

| Prompt, streams | Plain | DSpark | DSpark against plain | Greedy plain (table above) |
| --- | --- | --- | --- | --- |
| code, 1 | 21.4 (22.0, 20.7) | 20.7 (20.9, 20.5) | 3 % under | 31.1 |
| code, 4 | 33.8 (34.6, 33.0) | 25.9 (25.9, 25.9) | 23 % under | 67.0 |
| chat, 1 | 21.5 (22.1, 20.8) | 21.3 (21.5, 21.1) | 1 % under | 30.9 |
| chat, 4 | 33.8 (35.2, 32.3) | 33.2 (34.2, 32.2) | 2 % under | 66.8 |

Two things stand out. Plain decode with temperature 0.7 runs at 21 tokens a second against 31 greedy at one stream and at
34 against 67 at four, so a sampled token costs about 15 ms more than a greedy one and the cost grows with the streams.
With the draft head on, four code streams are 23 % under plain and the other cells are within the 6 % that separates the
two plain runs. The client's check for sampled replies (each stream's seed is its index plus the base seed, and the same
request must give the same text alone and with `"draft": false`) gave all equal on both DSpark servers: 3 and 12 of
each count against the solo runs, 4 of 4 against `"draft": false`, 0 unequal, 0 failed. Drafts were accepted (467 on
code, 447 on chat), so the check was not vacuous. The raw summaries are `sampled-*.out` in the receipt.

## Verification

Each line names the head it was taken on. The contract line was taken at `61be67917f`; the host-test and sampled-decode
lines at `e90ecce19a`.

- Kernel contract, `tf-kernel-contract --only inkling2` at `61be67917f`: 86 kernels pass gates 1 to 4, 133 controls red
  as planted, 0 missed, 0 limit problems, on the M4 Pro (see [the kernel contract](../kernel-contract.md#coverage)).
- Original checkpoint, device integration (`test-inkling2-metal`, 7 of 7): shared rows, rollback and restore, drafted and
  plain token ids; a 129-row trace matched every FP32 logit across 128-row and 32-row chunking, SHA-256
  `fbbb16ea5bf855e5b106ce740c32232ddb9ce7cd4f200792e3376fd4598956e7` (PR description, measured before `3439d56ae0`).
- Tiny checkpoint `hf-internal-testing/tiny-inkling2-mini@ee242a48af34ff0729518f4ac28cfcb847e470ab` against Transformers
  `6173a2f6` BF16 eager: 35 of 35 top ids, RMS 0.03390 and 0.02735 (PR description, measured before `3439d56ae0`).
- Vision, `test-inkling2-media-metal` with the pinned eager rounding: 83 of 83 (PR description, measured before
  `3439d56ae0`); the pinned public fixture matched 20 of 20 completion ids. At `db6d02593c` the suite gave 79 passed,
  9 skipped (the JPEG tests, for want of the pinned libjpeg-turbo release), 0 failed. On an M5 Ultra a contributor's run
  at `98b1e13bd0` gave 77 passed, 8 skipped, 0 failed.
- Drafted equals plain: the hashes and the `--alone --serial` counts in the receipt above; the contract's grouped kernels
  are byte-equal to the one-row kernels, including NaN, infinity and all-pairs-one-expert inputs.
- Kept prompt states resume as a fresh run only with `--no-drafts`; resumed equals fresh with the drafter on is not
  implemented (see Limits).

## Limits

- Audio parts are refused: the server answers HTTP 400 before the template runs ("content parts must be text or
  image_url; audio and video inputs are unsupported", or "this server accepts text parts only; ..." without `--vision`),
  and the checkpoint's chat template raises "Inkling 2 does not accept audio input" for `audio`, `input_audio` and
  `audio_url` parts. A host test pins both.
- JPEG images need the pinned libjpeg-turbo release (the decoder is core's picture decoder; the version and where the
  server looks are in [image input](../vision.md#jpeg-input)); without it each JPEG is refused by name and PNG and text
  still serve. Images need `--vision`.
- Replies carry the model's raw delimiters (`<|message_model|><|content_thinking|>` and the like): they are not split
  into `reasoning_content`.
- A DSpark server refuses `--prompt-cache-gib`: the draft head keeps per-stream state a kept prompt state does not
  hold, so the served default with drafts never resumes a prompt. `--drafter` is refused (the head is the checkpoint's
  own), as are `--slide`, `--loops`, `--policy` and `--checkpoint-slots`.
- Cold prefill is 0.72 to 0.87 of MLX's rate. The text path has no M5 tensor-unit kernel.
- Four drafted streams are within noise of plain, not faster; one stream is 15 to 22 % faster.
- Admission by state bytes was tested on a 48 GiB Mac only; a 36 GB Mac (where an earlier head serialized four streams
  under memory pressure) has not run it.
- With temperatures above 0 a plain token costs about 15 ms more than a greedy one (21 against 31 tokens a second at one
  stream, 34 against 67 at four), and four drafted code streams are 23 % under plain; serve sampled traffic that
  matters for speed with `--no-drafts` until the sampled path is faster.
