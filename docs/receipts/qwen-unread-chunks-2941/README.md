# Qwen CUDA unread prompt chunks (#2941)

A non-final prompt chunk whose rows nothing reads stops its last layer at its keys, values and their rotary rows.
This receipt covers the branch's exactness, sampled and drafted replies, the 8K and 16K prompt-cache marks and
cold prefill on one GB10.

## Source and environment

- Device binaries: base `0591c4e85dca440749063d171d4305d65a6edb50` and head `4a7a7f127`, one `zig build native
  -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc` build an arm. Binary md5: base `6720d18b2b1d8875477d5520b95dc26c`,
  head `fe92ed6a93d7ede5c7a6bb6fa42a3cff`. The branch was rebased onto `13d7ce052` after these runs; those three
  commits touch slide rollback only.
- One NVIDIA GB10 (sm_121), CUDA 13.0 (V13.0.88), Zig 0.17.0, Linux 7.0.
- Checkpoint `nvidia/Qwen3.6-35B-A3B-NVFP4`, revision `491c2f1ea524c639598bf8fa787a93fed5a6fbce`.
- Serve flags: `--parallel 1 --context 40960 --no-update-check --prompt-cache-gib 0` for cold, sampled and drafted
  rows; `--prompt-cache-gib 4` for the resumed turns. The GB10 also served an unrelated 86 GiB model, so cold TTFT
  is reported but is not a clean performance gate; exactness is the gate.

## Cold replies

One request each at 2,048, 8,192 and 32,768 tokens, 16 greedy reply tokens, fresh server an arm. Base and head
`token_sha256`/`output_sha256` are equal at every length. The base arm was re-run with no build on the machine so
its TTFT is comparable to the head arm:

| Prompt tokens | Base `token_sha256` / `output_sha256` | Head `token_sha256` / `output_sha256` | Base TTFT | Head TTFT |
| ---: | --- | --- | ---: | ---: |
| 2,048 | `0551b205…` / `4dfc61c7…` | `0551b205…` / `4dfc61c7…` | 0.344 s | 0.337 s |
| 8,192 | `eabb9ee4…` / `bfda9f68…` | `eabb9ee4…` / `bfda9f68…` | 1.282 s | 1.270 s |
| 32,768 | `bfdffc1b…` / `3e1bd53e…` | `bfdffc1b…` / `3e1bd53e…` | 6.006 s | 5.962 s |

The full hashes are in `runs/base-cold.json` and `runs/head-cold.json`. The 32,768-token prompt cuts eight
4,096-row chunks; the seven non-final ones reported `unread=true` in the server log, with no tap, learning, MLP,
pocket, int8 or image reader. That log line came from a temporary `std.log.info` in `cuda_prefill.chunk` for the
run; the diff adds no log line.

The prompts are the server tokenizer's output for Python 3.12.3's standard-library source; the raw arrays were
tokenized once per server start, so a different Python version changes the text but not the equality check.

## Quiet-GB10 A/B (reported by @jschmied)

A separate quiet-GB10 run, base `a3aecd8d0` against head `e7c385e96` (the same `zig/` diff as this head, apart from
hunk offsets), one fresh server an arm in base-head-head-base order, 3 rounds, SM clock capped at 2300 MHz, page
cache dropped before every server, 16 greedy tokens. Binary md5: base `1375faa01aeefc2f9ac40c907d6dab81`, head
`a479e68ff08992d57ff66f15faf52e3f`. Every `token_sha256`/`output_sha256` equals this receipt's:

| Prompt | Base TTFT median [min-max] | Head TTFT median [min-max] | Paired head - base, per round | Adjacent pairs slower |
| ---: | ---: | ---: | --- | ---: |
| 2,048 | 0.3275 s [0.3238-0.3291] | 0.3259 s [0.3224-0.3272] | +1.8 / -4.5 / -2.7 ms | 3 of 6 |
| 8,192 | 1.2360 s [1.2353-1.2437] | 1.2328 s [1.2273-1.2356] | -3.1 / -9.9 / -1.5 ms | 0 of 6 |
| 32,768 | 5.8006 s [5.7908-5.8210] | 5.7695 s [5.7437-5.7825] | -28.4 / -61.8 / -23.0 ms | 0 of 6 |

At 32K the median per-round delta is -28.4 ms and the median of the six adjacent pairs is -34.5 ms (-0.5 to
-0.6 %); 8K gains about 3-4 ms and 2K is level. The raw per-arm rows, gate output and server logs are with
@jschmied.

## Sampled and drafted replies

The same 8,192 and 32,768 token prompts at temperature 1.0 with seed 1234, then the same prompts with the server's
default draft mode; every base and head row has the same `token_sha256` and `output_sha256`:

| Mode | 8,192 base / head | 32,768 base / head |
| --- | --- | --- |
| Sampled, T 1.0, seed 1234 | equal (`eabb9ee4…`) | equal (`bfdffc1b…`) |
| Drafted, default | equal (`eabb9ee4…`) | equal (`bfdffc1b…`) |

Raw rows are in `runs/{base,head}-sampled.json` and `runs/{base,head}-drafted.json`. With the default drafter the
MTP tap on the last layer makes `unread` false, so these rows exercise the unchanged full path.

## Prompt-cache marks

A chat prompt is sent, then the same conversation with a question appended, on base and head with a 4 GiB prompt
cache. The 8K run resumes 8,192 tokens and the 16K run 16,384; every resumed reply is
`token_sha256 e1de42ac…` / `output_sha256 19079cfe…` on both arms. A cache-off run of the appended prompt has the
same reply, so the resumed path equals a fresh one.

| First prompt | Resumed at | Base reply | Head reply |
| ---: | ---: | --- | --- |
| 9,352 tokens | 8,192 | `e1de42ac…` / `19079cfe…` | `e1de42ac…` / `19079cfe…` |
| 17,972 tokens | 16,384 | `e1de42ac…` / `19079cfe…` | `e1de42ac…` / `19079cfe…` |

The runs are `runs/{base,head}-chat8k.json`, `runs/{base,head}-chat16k.json` and `runs/head-chat-fresh.json`.

## Commands

```sh
BIN=zig-out/native/bin/tensorfold-native
G=docs/receipts/qwen-unread-chunks-2941
$BIN serve nvidia/Qwen3.6-35B-A3B-NVFP4 --name gate --port 8081 --context 40960 --parallel 1 \
  --no-update-check --prompt-cache-gib 0
python3 $G/gate.py cold http://127.0.0.1:8081 gate $G/runs/head-cold.json
python3 $G/gate.py sampled http://127.0.0.1:8081 gate $G/runs/head-sampled.json
python3 $G/gate.py drafted http://127.0.0.1:8081 gate $G/runs/head-drafted.json
# restart with --prompt-cache-gib 4
python3 $G/gate.py chat8k http://127.0.0.1:8081 gate $G/runs/head-chat8k.json
python3 $G/gate.py chat16k http://127.0.0.1:8081 gate $G/runs/head-chat16k.json
```

`gate.py` sends the checkpoint's own tokenizer output through `tools/bench_concurrent.stream`, so each row is
the server's own `ttft_s`, cached-token count and token hashes.
