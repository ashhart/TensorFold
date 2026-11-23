# Inkling 2 Mini on native Metal, Apple M4 Pro: served receipt for #4337

Raw results behind the tables in [the Inkling 2 Mini recipe](../../recipes/inkling-2-mini-metal.md#receipt-apple-m4-pro-48-gib).
Measured on 2026-11-23 on one Apple M4 Pro with 48 GiB, macOS 27.2 (26B123), one model process at a time, with the
pinned Zig 0.17.0. The checkpoint is `thinkingmachines/Inkling-2-Mini` at revision
`ed40e7da625f594fc9c4990f5f66be24873c22d7` (with its bundled `mtp.safetensors`); the reference is `mlx_lm.server`
(mlx 0.32.4, mlx-lm `0.1.dev1006+g0c4c99192`) on the same checkpoint with no drafter.
This receipt does not qualify other chips, other checkpoints or a Linux backend. Only the `sampled-*` files were produced
on a binary of the PR's head `e90ecce19a` (built with the table split of `metal.zig`, which changes no kernel). The
recipe's other numbers are from the binary built at `a0935b5279` (decode, cold prompts, footprint) and from the DSpark
head-pass build that became `05b6fdb5c0` (concurrency); the head contains both. The `decode-depth*` files are from a
binary of the PR's head `57de097873` and were taken after the other numbers.
An unrelated CPU job used about 60 % of a core during the decode and cold runs, so read ratios between rows.

## Builds

All are `zig build native -Dcpu=apple_m1 -Doptimize=ReleaseFast`. [provenance.json](provenance.json) holds each binary's
SHA-256 and the commands.

- **Round-three binary** (decode, cold prompts, footprint, the first three hash logs): the PR's recipe names
  `a0935b5279` for it. The run script took the binary as an argument that the logs do not record, so the commit is the
  PR's statement and the SHA-256 is that of the binary the same session's footprint script launches.
- **Concurrency, plain columns:** the base binary of the DSpark head-pass change (its parent, `e6cb02e09e`).
- **Concurrency, DSpark columns:** the fifth build of the head-pass change, before it was committed as `05b6fdb5c0`. A
  later build of the committed tree has another SHA-256, so the DSpark column is not tied to a commit by this receipt.

## Files and the numbers they back

| File | What it is | Recipe number |
| --- | --- | --- |
| `mlx.decode.json`, `native-plain.decode.json`, `native-dspark-p4.decode.json` | `tools/bench_openai.py --tokens 256 --reps 3 --temperatures 0`, one stream; native plain on the `--context 65552 --parallel 1 --no-drafts` server, native DSpark on the `--context 8192 --parallel 4` DSpark server | Decode table: 23.05 / 23.69, 31.12 / 31.07, 35.76 / 35.01 tok/s |
| `mlx.concurrent.json` | `tools/bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4` against `mlx_lm.server` | The MLX column of the concurrency table, and "MLX answers the four requests one after another" (first token 33.8 s at four streams) |
| `abba-1-plain.json`, `abba-2-dspark.json`, `abba-3-dspark.json`, `abba-4-plain.json` | The same client, one A-B-B-A session: plain (`--no-drafts`), DSpark, DSpark, plain, `--context 8192 --parallel 4 --prompt-cache-gib 0 --no-thinking` | Native plain and DSpark columns: the mean of each pair, e.g. code 4 streams 67.0 and 65.1 |
| `abba-1-plain.serve.txt`, `abba-4-plain.serve.txt` | The plain servers' logs (startup line with the admission demand, one line a request). The DSpark servers' logs were 440 KB each with the price-table trace on and are not kept | |
| `dspark-alone-serial.json` | `bench_concurrent ... --levels 1,4 --alone --serial` on the DSpark build | 30 replies equal to their runs alone, 4 equal to `"draft": false`, 0 unequal |
| `mlx.cold.json`, `native-plain.cold.json` | `cold_prefill.py --sizes 2048,8192,32768,65536`: exact token counts through `/v1/completions`, one token out, `cached_tokens` 0, native at `--context 65552 --parallel 1 --no-drafts --prompt-cache-gib 0 --no-thinking` | Cold first-token table |
| `cold-sweep-metal4.native.json`, `cold-sweep-metal4.mlx.json`, `cold-sweep-metal4.*.serve.txt` | The cold-prompt sweep at `4418225fb6` (Metal 4 prompt queue): `cold_prefill.py --sizes 2048,8192,32768,65536`, native then MLX on a quiet machine (load average 2 to 3), one run per size | Native 5.18 / 21.59 / 97.05 / 225.97 s, MLX 4.32 / 17.04 / 75.55 / 169.70 s |
| `native-plain.serve.txt`, `native-dspark-p4.serve.txt`, `mlx.serve.txt` | The servers' logs of the decode and cold session (logs carry `.txt` because the repository ignores `*.log`) | Startup lines: 18.19 GiB weights, 18.95 GiB with dense and scratch |
| `mlx.footprint.txt`, `native-plain.footprint.txt`, `native-p4-dspark.footprint.txt`, `*.mem.cold.json`, `*.memserve.txt` | `footprint -p` sampled every 3 s through an 8,192-token cold prompt and a 128-token chat reply; native `--context 65552 --parallel 1` and `--context 8192 --parallel 4` with DSpark; the early MB-sized samples are the load phase | Peak footprint: the largest GB sample is 24 (native, both) and 23 (MLX) |
| `hash-base.txt`, `hash-plain.txt`, `hash-dspark.txt`, `hash-r3b.txt` | Server logs of `chat_hashes.py` on four binaries of the session (plain, plain, DSpark, DSpark; `hash-r3b.txt` is the round-three binary), each ending in the same four `sha=` values | Token hashes `0b037128ee78`, `b37d8a0e39c4`, `77bf7c99cc66`, `538175732cc7` |
| `chat_hashes.py`, `cold_prefill.py` | The two clients, with local paths removed | |
| `dense-abba.txt` | One quiet-machine session at `61be67917f` (the dense projection tile): the contract SUMMARY line, then A (`e90ecce19a`), B (`61be67917f`), B, A plain servers (`--context 65552 --parallel 1 --no-drafts --prompt-cache-gib 0 --no-thinking`), cold first token at 8,192 and 32,768 tokens and the four chat hashes. The 2,048-token request failed in the harness (no prompt file), so it is not in the file | Dense-tile note: 23.8 to 22.7 s at 8,192, 104.8 to 102.0 s at 32,768; 133 controls; the same hashes on A and B |
| `sampled-plain1.out`, `sampled-drafted1.out`, `sampled-drafted2.out`, `sampled-plain2.out`, `sampled.sh` | One idle-machine session at `61be67917f`: `bench_concurrent --alone --serial --levels 1,4 --tokens 256 --reps 3 --temperatures 0.7 --seed 1234` against a plain (`--no-drafts`), a DSpark, a second DSpark and a second plain server (`--context 8192 --parallel 4 --prompt-cache-gib 0 --no-thinking`), one line of JSON a cell (the `alone` and `serial` counts are the sampled-exactness check); `sampled.sh` is the script with the paths replaced | Sampled table and the Limits bullet: DSpark 1 to 3 % under plain at one stream and 2 % under on four chat streams, 23 % under on four code streams; plain sampled 21 and 34 against 31 and 67 greedy; 3 and 12 of each count equal alone, 4 of 4 equal to `"draft": false` |
| `decode-depth.json`, `decode_depth.py`, `decode-depth-native-A1.serve.txt`, `-native-A2`, `-native-C`, `-native-D`, `decode-depth-mlx-B1.serve.txt`, `-mlx-B2` | Decode rate after a cold prompt, `decode_depth.py` (streamed `/v1/completions`, temperature 0, 192 new tokens, the mean of the inter-token latencies after the first token and the next few, `--rep0` picks the prompt set: set 0 is the arrays of the Cold prompts table below, set 1 the same builder with seed N x 10 + 1). One fresh server a session, in the order native A1, MLX B1, MLX B2, native A2 (prompt sets 0, 0, 1, 1), then native C (set 1) and D (set 0) for the deepest rows; plain native servers as the cold run, `--context 65552 --parallel 1 --no-drafts --prompt-cache-gib 0 --no-thinking`; MLX `mlx_lm.server` at its defaults. A native server at that context takes at most 16 reply tokens after a 65,536-token prompt, so the 65,536 row has 16 and a 65,344-token row has 192 (it ends at 65,536). The `summary` object is the table in the recipe; the logs are the servers' (MLX paths replaced by `$ORACLE`). The generated texts of native and MLX differ (different bits), so `text_sha` is not a cross-server check | Decode at depth table: 28.5 / 28.4 / 25.8 / 23.2 tok/s native and 23.0 / 22.6 / 20.0 / 17.8 MLX at 2,048 / 8,192 / 32,768 / 65,536 tokens |
| `attn-decode-bench.txt`, `attn_decode_bench.swift` | The decode attention kernel timed alone at one query row: GPU time a visit in ms at 2,048 to 65,536 keys (global) and a 1,024-key window (sliding), two passes, then the schedule variants tried (`I2_V_BATCH` 8 and 16, K loads hoisted, the query as float4, V rows double-buffered) and four ablations | The ms-a-token breakdown under the depth table |
| `decode-short-head.json`, `decode-short-head.serve.txt` | `tools/bench_openai.py --tokens 256 --reps 3 --temperatures 0` on a plain native server of the depth build `57de097873` (same flags as the decode table's), taken after the depth runs | Short-prompt decode at the head, 31.3 and 31.2 tok/s, quoted beside the depth table |
| `prefill-stages.txt`, `prefill-stages.timeline.log.gz` | GPU time of a 512-row prompt chunk by kernel family (the table), per chunk at 64 positions, and the raw `metal 4 timeline` log of one 8,192-token and one 32,768-token cold prompt. A local build of `5e75f97db2` with the whole model on a Metal 4 queue, one timeline round named by each chunk's position, `warm = 0` and a stamp window of 4,096 (not committed: the committed hook is `queue.roundBegin("inkling2 pass")`, run with `TF_METAL4_TIMELINE=n`); `--context 65552 --parallel 1 --no-drafts`, one stamp between every two dispatches, so the stage spans add to the first token (21.7 s of 21.9 s at 8,192 tokens, 98.0 s of 98.5 s at 32,768). Floors: 7.7 TFLOP/s on the matrix units, 250 GB/s | The prompt-chunk table and the growth split |
| `prefill-abba.txt`, `prefill-r1-*.serve.txt`, `prefill-r2-*.serve.txt` | Two A-B-B-A sessions (plain server, `--context 65552 --parallel 1 --no-drafts --prompt-cache-gib 0 --no-thinking`): cold first token at 8,192 and 32,768 tokens, the four chat hashes, `bench_openai` decode; A is `5e75f97db2`, B is the prompt-chunk Metal 4 queue (session r1: the build before the timeline hook; r2: the commit's build). Each log's `sha=` lines are the hashes. The all-Metal-4 decode pairs are in the same file | The Metal 4 prompt-queue note |

## Cold prompts

The prompt token arrays are not stored (they are 850 KB). `cold_prefill.py --tokens-only --ck CHECKPOINT --sizes N
--output PREFIX` rebuilds them from the checkpoint's tokenizer (transformers 5.23) and prints each file's SHA-256; the
rebuilt arrays matched the run's byte for byte, and MLX and native were sent the same arrays:

| Tokens | SHA-256 of `{"ids": [...]}` as written by `json.dump` |
| --- | --- |
| 2,048 | `337a24894779f835f34652a61538e424ce014bb80a84955d3d35c57781412199` |
| 8,192 | `aebb44747ca0ae9645c2814445aed1e2f78352c71b7bd6c58413a86433cb5eb0` |
| 32,768 | `5cc69c9aff75c2f4d7109d3e44972aa50faf97ad58a3af79fbdfd479482903b3` |
| 65,536 | `677b8da33b000df205b4ff61e05eff0f396015cded3d644e610c63dc57626daf` |

Request bodies are `{"model": "m", "prompt": TEXT, "max_tokens": 1, "temperature": 0, "stream": false}` with TEXT the
decoded array; their sizes are in each row's `body_bytes`.

## Fidelity against MLX and images (`fidelity/`)

Measured 2026-11-23 on the same machine, one model process at a time, under `/tmp/tf-gpu.lock`. The native binary was
copied from the PR checkout's `zig-out/native` (SHA-256 `fa04538bd2219170aedcc01b354d54831d6d219a099cc7fea16a57ca4374d752`;
built from the head `57de097873` [INFERENCE]). Scripts take paths from their arguments; `run_native.sh` and `run_img.sh`
read `CKPT BIN WORK PY JPEG` from the environment. The pinned libjpeg-turbo release was built from its release tarball.

| File | What it is |
| --- | --- |
| `fid_prompts.py` | Builds the seven text prompts (from this repository's files and docs at the base head; the written `prompts.json` has SHA-256 `bcd9c82e4d4499da1bccf04affaf192c21f496076fd7dac32ddfe37f7f3a4c2b` and is not stored, its token ids are in `mlx.json`; token counts 1,500, 1,500, 900, 70, 62, 7,998, 31,998) |
| `fid_mlx.py`, `mlx.json`, `mlx.step256.json` | MLX greedy 256 tokens (ids, chosen logprob, top 5) with `prefill_step_size` 2048, and 256 on five prompts (the noise control) |
| `fid_native.py`, `run_native.sh`, `native.json`, `native.client.txt`, `native.serve.txt` | Native teacher forcing by restarts (no `prompt_logprobs` on this backend) with `--context 65552 --parallel 1 --no-drafts --prompt-cache-gib 0 --no-thinking` |
| `fid_compare.py`, `compare.txt`, `fid_selfnoise.py`, `selfnoise.txt` | The comparison tables quoted in the recipe, and MLX against itself |
| `img_make.py`, `img/` | The test images (PIL) |
| `img_client.py`, `run_img.sh`, `img.native.json`, `img.mlx.json`, `img.*.serve.txt` | Image requests (streamed, two per image, greedy, 160 tokens) against native `--vision` and `mlx_lm.server`, with each reply's text, usage and time to first token |
| `img-ttft.txt` | The image first-token split and its fix (`a305edd706`, after `5e75f97db2`): the tower's ms per stage at 520 and 1,530 image tokens before and after, the bits check (tower output hash on 14 requests, replies and `sha=` of four sessions, the four text hashes, the contract SUMMARY), the A-B-B-A first tokens and the MLX rerun. The stage numbers come from profile builds that cut the tower's command buffer after every stage (an uncommitted patch), run while other jobs loaded the machine |
| `img-stage-A.serve.txt`, `img-stage-B.serve.txt` | The profile builds' server logs (`VPROF` lines: per stage ms, tower output hash) over `img_client.py`'s seven images |
| `img_client3.py`, `img-abba-1-A.json`, `img-abba-2-B.json`, `img-abba-3-B.json`, `img-abba-4-A.json`, `img-abba-*.serve.txt` | `img_client.py` cut to the scene and the invoice (JPEG and PNG); one A-B-B-A session, A `5e75f97db2`, B `a305edd706` (the `--context 8192 --parallel 1 --no-drafts --prompt-cache-gib 0 --no-thinking --vision` server, quiet machine) |
| `img-eq-1-A.json`, `img-eq-2-B.json`, `img-eq-3-B.json`, `img-eq-4-A.json` | The same four binaries over all seven images, `img_client.py`, with the replies compared byte for byte (their first tokens were taken under load) |
| `img.mlx.rerun.json`, `text_ttft.py`, `text-ttft.mlx.txt`, `text-ttft.native-B.txt`, `hash-vision-ttft.txt` | `mlx_lm.server` on the seven images again, the text-only first-token probe (607 and 1,727 tokens) on MLX and on B, and B's server log of `chat_hashes.py` (the four known `sha=` values) |
| `img-softmax.txt` | The tower's softmax at its memory floor, the standalone and fused variants tried, the bf16 score store (`238f688b59` against `perf/vision-softmax`): the equal tower hashes, replies, `sha=` values and text hashes, the contract totals (86 kernels, 134 controls) and the A-B-B-A first tokens (invoice 6.94 to 6.54 s) |
| `img-sm-bench.m`, `img-sm-bench.metal`, `img-sm-bench.txt` | The standalone harness (kernel texts of both paths, GPU time of one dispatch, scores, probabilities and p v outputs compared bitwise) and its output on seven ragged and full shapes, taken under a load of about 17 (the quiet timings are quoted in `img-softmax.txt`) |
| `img_session.sh`, `img-sm-eq-{1-A,2-B,3-B,4-A}.{json,serve.txt}` | One server and one client run (`BIN TAG CLIENT`; env `CKPT IMG TENSORFOLD_LIBTURBOJPEG`); the four equality sessions over seven images on hash-logging builds (`VHASH` lines are the tower output's Wyhash) |
| `img-sm-abba-{1-A,2-B,3-B,4-A}.{json,serve.txt}`, `img-sm-final-B.{json,serve.txt}` | The A-B-B-A first-token sessions (`img_client3.py`, plain binaries, quiet machine) and the final build's one session |
| `hash-sm-text.txt`, `hash-sm-final-text.txt`, `img-sm-contract.txt` | The `chat_hashes.py` server logs of B and the final build (`sha=` 0b037128ee78 b37d8a0e39c4 77bf7c99cc66 538175732cc7) and the media rows of `tf-kernel-contract --only inkling2` |

## Drafted resume (`drafted-resume/`)

A resumed prompt equals a fresh one with the DSpark head on: `summary.txt` holds the numbers and how each file was made,
`battery.py`, `compare.py`, `ttft.py` and `cancel.py` the clients (they read the recipe at `fe99c81a11` from `NOTES`),
`{plain,drafted}-{cold,cache}.json` the per-request token ids, cached tokens and draft counters, `ttft.jsonl`,
`cancel.jsonl` and `abba-*.dec.json` the first-token, interrupted-pass and decode runs. Taken at a machine load of 45 to
55, so read times as relative.

## Not in this receipt

- Anything greedy at the head `e90ecce19a` except the `decode-depth*` and `decode-short-head*` files (`57de097873`, plain
  servers only); drafted decode and more than one stream at depth; other chips, a 36 GB Mac, a quality suite (task
  accuracy).
- The earlier round-one and round-two results the PR description records; the recipe no longer cites them.
- The M3 Pro, M4 Max and M5 Ultra figures in the recipe's platforms table: they are contributors' runs, quoted from the
  PR's comments and not reproduced here.
