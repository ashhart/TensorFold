# Qwen CUDA unread prompt chunks (#2941)

A non-final prompt chunk whose rows nothing reads stops its last layer at its keys, values and their rotary rows.
This receipt covers the branch's exactness and the cold prefill timings on one GB10.

## Source and environment

- Base `6c5cef8941b27d86b06a986834e13deb9b515e1e` and this branch's head, one `zig build native -Dsm=121
  -Dnvcc=/usr/local/cuda/bin/nvcc` build an arm. Binary md5: base `52d6e03995b1e15b106ea2506cf873b0`, head
  `af995a3c5bcb4e413e0c5421544ad76f`.
- One NVIDIA GB10 (sm_121), CUDA 13.0 (V13.0.88), Zig 0.17.0, Linux 7.0.
- Checkpoint `nvidia/Qwen3.6-35B-A3B-NVFP4`, revision `491c2f1ea524c639598bf8fa787a93fed5a6fbce`.
- Serve flags: `--parallel 1 --context 40960 --no-drafts --no-update-check --prompt-cache-gib 0`, or `4` for
  the resumed turn. The GB10 also served an unrelated 86 GiB model throughout, so cold TTFT is reported but is
  not a clean performance gate; exactness is the gate.

## Cold replies

One request each at 2,048, 8,192 and 32,768 tokens, 16 greedy reply tokens, fresh server an arm. Base and head
`token_sha256`/`output_sha256` are equal at every length:

| Prompt tokens | Base `token_sha256` / `output_sha256` | Head `token_sha256` / `output_sha256` | Base TTFT | Head TTFT |
| ---: | --- | --- | ---: | ---: |
| 2,048 | `0551b205…` / `4dfc61c7…` | `0551b205…` / `4dfc61c7…` | 0.335 s | 0.332 s |
| 8,192 | `eabb9ee4…` / `bfda9f68…` | `eabb9ee4…` / `bfda9f68…` | 1.270 s | 1.281 s |
| 32,768 | `bfdffc1b…` / `3e1bd53e…` | `bfdffc1b…` / `3e1bd53e…` | 5.994 s | 5.987 s |

The full hashes are in `runs/base-cold.json` and `runs/head-cold.json`. The 32,768-token prompt cuts eight
4,096-row chunks; the seven non-final ones report `unread=true` in the server log, with no tap, learning, MLP,
pocket, int8 or image reader.

## Resumed turn

The same 7,857-token chat prompt, then the same conversation with a question appended, on base and head with
`--prompt-cache-gib 4`; a third run repeated the appended prompt on a server with the cache off. Every run's
reply is `token_sha256 e1de42ac…` / `output_sha256 19079cfe…`; the cache-on runs resumed 4,096 tokens.

## Commands

```sh
BIN=zig-out/native/bin/tensorfold-native
$BIN serve nvidia/Qwen3.6-35B-A3B-NVFP4 --name gate --port 8081 --context 40960 --parallel 1 \
  --no-drafts --no-update-check --prompt-cache-gib 0
python3 docs/receipts/qwen-unread-chunks-2941/gate.py cold http://127.0.0.1:8081 gate \
  docs/receipts/qwen-unread-chunks-2941/runs/head-cold.json
python3 docs/receipts/qwen-unread-chunks-2941/gate.py chat http://127.0.0.1:8081 gate \
  docs/receipts/qwen-unread-chunks-2941/runs/head-chat.json
```

`gate.py` sends the checkpoint's own tokenizer output through `tools/bench_concurrent.stream`, so each row is
the server's own `ttft_s`, cached-token count and token hashes.
