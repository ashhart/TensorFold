# DeepSeek V4.1 Flash 1017 Metal experiment

This receipt preserves the experimental native serving source and the device
results from a single Apple M5 Ultra with 512 GiB of unified memory. The
experiment used the TensorFold e2e3-p4-h8 checkpoint at revision
`9012d3064675de44630c529f496e4961b30a73a4` with DSpark enabled. The
final measured server's executable SHA-256 was
`2458ad02d8892c5e1e78aa3eb1a91765f80b8a37e6b4d19e668db4b577787572`.

The patch targets the experimental source revision
`72a552ed09f5962379261e429552d83955cb37ad`. Current main has a different
DeepSeek Metal implementation under `zig/src/families/deepseek_v41/metal/`,
including its own decoder suffix pass. The source below is a reproducible
experiment. Its measurements apply to the pinned build.

## Results

### Prose and concurrency

Greedy, thinking off, 256 generated tokens per reply, six synthetic prompts,
no prompt cache, and four active lanes. C8 and C16 describe offered clients.
Aggregate decode spans the earliest first token to the latest last token
and includes gaps between admission waves. Latencies are measured at the client.

| Offered clients | Aggregate decode tok/s | Median per-request decode tok/s | P50 first token | P95 first token |
| --- | ---: | ---: | ---: | ---: |
| 1 | 77.2 | 77.2 | 0.18 s | 0.19 s |
| 4 | 107.4 | 29.4 | 0.65 s | 1.16 s |
| 8 | 108.2 | 30.0 | 4.56 s | 11.16 s |
| 16 | 109.3 | 28.5 | 13.78 s | 30.50 s |

All 84 concurrent replies matched the SHA-256 token digest of the
corresponding solo reply. The reported 1,048,576-token context was a
configured limit. The largest timed prompt below was about 32K tokens.

### Cold prefill

One active lane. The journal prompts requested one generated token each.
The 17,639-token coding prompts requested 1,024 output tokens and decoded at
98.0 tok/s median. Rates count the tokens actually processed. The server
reported at most two reused template tokens across the cold prompt runs.

| Prompt length | Prefill tok/s | First token |
| --- | ---: | ---: |
| 2,048 | 1,578 | 1.30 s |
| 8,192 | 2,938 | 2.79 s |
| 16,384 | 3,229 | 5.08 s |
| 32,768 | 3,387 | 9.69 s |
| 17,639 coding | 3,408 | 5.19 s |

### Matched controls

- **Ordered attention merge.** The verifier streams adjacent columns of saved attention partials through the original ordered reduction. Six 512-token greedy prose fixtures went from 79.46 and 79.41 tok/s in the bracketing controls to 80.21 tok/s. The served shadow compared 2,907,504,640 BF16 values. All 28 checked replies matched their reference token and text hashes, including four sampled replies. [Decode experiment data](evidence/ordered-attention.json)
- **Encoder lead and decoder suffix.** Preserve encoder and history-projection work. Keep the decoder rows required by the local attention windows, cached state and final logits. Split the wide submission at the decoder boundary. The matched 17,639-token cold coding comparison went from 1,662.5 to 3,401.0 prefill tok/s and 10.621 to 5.198 s to first token. Five paired replies matched token and text hashes. Fresh and cached continuations matched saved-state digests and the first BF16 logits at the audited exports. [Prefill experiment data](evidence/encoder-lead.json)
- **Cache checkpoint and scheduling.** Keep a stable prefill plan across admission changes. Store an exact-token checkpoint at token 20,000 in the existing cache. An edited continuation reused 20,000 tokens instead of two and went from 5.942 to 0.534 s to first token. In the combined two-lane test the queued turn went from 6.791 to 0.475 s to first token. All seven paired replies matched token and text hashes. [Cache and queue experiment data](evidence/prefix-queue.json)

The device evidence is one-chip coverage. Sampled coverage is the stated
attention experiment. The C1/C4/C8/C16 and cold prefill numbers use greedy
decoding with thinking off.

## Replay

[The source patch](metal-1017.patch) preserves the measured source, including
the inactive comparison paths. Use a clean, detached checkout of the exact
revision. The patch uses zero-context hunks. Verify the resulting tree before
building.

```sh
receipt="$PWD/docs/receipts/deepseek-v41-1017-metal"
python3 -B "$receipt/verify.py"
base=72a552ed09f5962379261e429552d83955cb37ad
git fetch --depth=1 origin "$base"
git worktree add --detach ../tf-dsv41-replay "$base"
cd ../tf-dsv41-replay
git apply --check --index --unidiff-zero "$receipt/metal-1017.patch"
git apply --index --unidiff-zero "$receipt/metal-1017.patch"
test "$(git write-tree)" = 6f58f9f38d2b2d23d2a53e5b787013d5fca26727
zig build native -Dcpu=apple_m1 -Dversion=1.3.0 -j4
```

Use the pinned Zig version and an Xcode installation with the Metal toolchain.
Use the same checkpoint and associated DSpark and Engram assets. Load the
[benchmark profile](profile.sh) in an isolated shell with `set -a` and start
the rebuilt native server with the pinned assets, the model name
`deepseek-v4.1-flash-1017`, and thinking off. Configure 1,048,576 context
tokens and disable the prompt cache for throughput measurements. Use one
active lane for the prompt runs and four for the burst runs. For the
cache experiment, the candidate uses two lanes, a 4 GiB prompt cache and
`TF_DS_PROMPT_CACHE_PREFIX=20000`.

Run the scripts from a separate terminal against an idle loopback benchmark
server. Pass its PID to record peak resident memory.

```sh
python3 -B "$receipt/sweep.py" pp http://127.0.0.1:8080 "$server_pid" pp.json
python3 -B "$receipt/coding_shape.py" http://127.0.0.1:8080 coding.json
python3 -B "$receipt/sweep.py" decode http://127.0.0.1:8080 "$server_pid" decode.json
```

[Summary](evidence/summary.json) · [Prose bursts](evidence/decode.json) ·
[Journal prompts](evidence/pp.json) · [Coding prompts](evidence/coding.json).
