# Inkling 2 Mini on native Metal, Apple M4 Pro: served receipt for #4337

Raw results behind the tables in [the Inkling 2 Mini recipe](../../recipes/inkling-2-mini-metal.md#receipt-apple-m4-pro-48-gib).
Measured on 2026-11-23 on one Apple M4 Pro with 48 GiB, macOS 27.2 (26B123), one model process at a time, with the
pinned Zig 0.17.0. The checkpoint is `thinkingmachines/Inkling-2-Mini` at revision
`ed40e7da625f594fc9c4990f5f66be24873c22d7` (with its bundled `mtp.safetensors`); the reference is `mlx_lm.server`
(mlx 0.32.4, mlx-lm `0.1.dev1006+g0c4c99192`) on the same checkpoint with no drafter.
This receipt does not qualify other chips, other checkpoints or a Linux backend. Only the `sampled-*` files were produced
on a binary of the PR's head `e90ecce19a` (built with the table split of `metal.zig`, which changes no kernel). The
recipe's other numbers are from the binary built at `a0935b5279` (decode, cold prompts, footprint) and from the DSpark
head-pass build that became `05b6fdb5c0` (concurrency); the head contains both.
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
| `native-plain.serve.txt`, `native-dspark-p4.serve.txt`, `mlx.serve.txt` | The servers' logs of the decode and cold session (logs carry `.txt` because the repository ignores `*.log`) | Startup lines: 18.19 GiB weights, 18.95 GiB with dense and scratch |
| `mlx.footprint.txt`, `native-plain.footprint.txt`, `native-p4-dspark.footprint.txt`, `*.mem.cold.json`, `*.memserve.txt` | `footprint -p` sampled every 3 s through an 8,192-token cold prompt and a 128-token chat reply; native `--context 65552 --parallel 1` and `--context 8192 --parallel 4` with DSpark; the early MB-sized samples are the load phase | Peak footprint: the largest GB sample is 24 (native, both) and 23 (MLX) |
| `hash-base.txt`, `hash-plain.txt`, `hash-dspark.txt`, `hash-r3b.txt` | Server logs of `chat_hashes.py` on four binaries of the session (plain, plain, DSpark, DSpark; `hash-r3b.txt` is the round-three binary), each ending in the same four `sha=` values | Token hashes `0b037128ee78`, `b37d8a0e39c4`, `77bf7c99cc66`, `538175732cc7` |
| `chat_hashes.py`, `cold_prefill.py` | The two clients, with local paths removed | |
| `dense-abba.txt` | One quiet-machine session at `61be67917f` (the dense projection tile): the contract SUMMARY line, then A (`e90ecce19a`), B (`61be67917f`), B, A plain servers (`--context 65552 --parallel 1 --no-drafts --prompt-cache-gib 0 --no-thinking`), cold first token at 8,192 and 32,768 tokens and the four chat hashes. The 2,048-token request failed in the harness (no prompt file), so it is not in the file | Dense-tile note: 23.8 to 22.7 s at 8,192, 104.8 to 102.0 s at 32,768; 133 controls; the same hashes on A and B |
| `sampled-drafted.out`, `sampled-plain.out`, `sampled-drafted2.out`, `sampled-plain.serve.txt`, `sampled.sh` | One session at `e90ecce19a`: `bench_concurrent --alone --serial --levels 1,4 --tokens 256 --reps 3 --temperatures 0.7 --seed 1234` against a DSpark, a `--no-drafts` and a second DSpark server (`--context 8192 --parallel 4 --prompt-cache-gib 0 --no-thinking`), one line of JSON a cell (the `alone` and `serial` counts are the sampled-exactness check); `sampled.sh` is the script. The machine was shared with other builds (load average about 7) and the plain log shows a memory-pressure warning | Sampled table and the Limits bullet: DSpark 7 to 14 % under plain at one stream, about 19 % under on four code streams; 30 of 30 equal alone, 4 of 4 equal to `"draft": false` |

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

## Not in this receipt

- Anything greedy at the head `e90ecce19a`, other chips, a 36 GB Mac, images served, a quality suite.
- The earlier round-one and round-two results the PR description records; the recipe no longer cites them.
- The M3 Pro, M4 Max and M5 Ultra figures in the recipe's platforms table: they are contributors' runs, quoted from the
  PR's comments and not reproduced here.
