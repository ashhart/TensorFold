# XPU Qwen3.8-27B logprobs

`logprobs` and `top_logprobs` on the Intel GPU engine (Qwen3.8-27B GPTQ int4 g128 with the DFlash2 drafter). Source:
`a4556f2fa11bd048872d050bcd33f6408cf6bcb4` on top of main `6c5cef8941b27d86b06a986834e13deb9b515e1e`.

## What was checked

- **Tokens do not move.** For six prompts, greedy and sampled (temperature 1.0, top_k 20, top_p 0.95, seed 7), drafted
  and `"draft": false`, `max_tokens` 160 (prompts 3 and 5 stop earlier, at the end of the reply, which covers the
  alignment of the last rows): the request with `logprobs: true, top_logprobs: 5` and the same request without
  them return the same `token_sha256` and the same text (24 pairs, all equal). The hashes are in `eq-solo.json`.
- **Drafted rows equal plain rows.** For the same 12 prompt and sampling cells, the drafted stream's logprob rows
  (token, logprob, the five best tokens and their logprobs) equal the plain stream's, compared as exact floats: 12 of 12,
  largest difference 0.0, no token or top-list order difference (`eq-solo.json`).
- **Shared rounds.** With `--parallel 2`, three pairs of prompts sent at once, greedy and sampled, drafted and plain
  (12 runs of two streams): each stream's tokens and rows equal what it returns alone (`eq-conc.json`).
- **Host tests.** `zig build test` and `zig build test-draft-xpu-host` pass. The new tests fail if the XPU engine stops
  declaring `logprobs` (lane host or lone driver) or if the drafter wrapper stops offering the target's first row.

## Cost

`bench.json`: decode tokens per second (first to last streamed token, 256 tokens, median of 5 alternating runs after a
warm-up each) on the same server, same binary, with and without `logprobs` and `top_logprobs: 5`. Host work per
logged row (one 248 k-logit row copied to the host, a normalizer and the 20 best) is about 0.5 ms.

| cell | without | with | cost |
|---|---:|---:|---:|
| lone drafted, greedy, 32-token prompt | 123.8 | 116.1 | 6.3 % |
| lone drafted, sampled seed 7, 32-token prompt | 123.4 | 115.7 | 6.2 % |
| lone, `"draft": false`, 32-token prompt | 33.1 | 32.4 | 2.0 % |
| lone drafted, greedy, 7,457-token prompt | 67.8 | 65.5 | 3.3 % |
| two streams at once, drafted, greedy (tokens per second in total, prefill included) | 193.7 | 167.5 | 13.6 % |

The two-stream row is a total rate over both streams, prefill included.

A lone drafted stream logs only the rows of the tokens it accepted. A shared round logs every row of every window,
because the lane core asks for one row per verified row and learns which ones land afterwards. The cost of that is
not isolated here.

**These are an order of magnitude, not the card's speed.** The measurements ran on the second Arc Pro B70 of a host
whose slot is Gen4 x4 behind the chipset (the other card sits in a Gen5 x16 CPU slot). Only the with and without
columns on the same card and server are comparable. One card was used, so no other card's number is claimed.

## Setup

- Intel Arc Pro B70 (Battlemage), one card, `ZE_AFFINITY_MASK` selecting it; Ubuntu Linux, Zig 0.17.0, ReleaseFast,
  `zig build xpu-native`. The binary's SHA256 is in `binary.sha256`; it is the commit above plus a local card-selection
  patch to the VRAM guard that is not part of the change.
- Target `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, drafter `Qwen3.8-27B-DFlash2`, 32,768-token context, 2 streams,
  `--prompt-cache-gib 0`, default KV cache.
- Corpus for the long prompt and one of the six: the first 1,500,000 bytes of the top-level `*.py` files of Ubuntu's
  python3.14 standard library (3.14.4-1ubuntu0.2) in the order of a punctuation- and case-blind sort of their names,
  SHA256 `31b2f903b4a2ea01728927e53436d924994c93cf3114c7b7f35e58d906961e30`.
- `eq.py`, `conc.py` and `bench.py` are the three drivers (`CORP=FILE python3 -I eq.py PORT OUT.json`; `conc.py` reads
  `eq.py` from its own directory).

## What is not covered

- `prompt_logprobs` stays refused. The Nemotron-H, LFM3 and Gemma 4 engines on Intel GPUs are unchanged and still
  refuse `logprobs`.
- The logprobs are the target's raw distribution (before temperature, top_k and top_p), as on the other engines.
- Tokens are checked on six prompts with `max_tokens` 160, not as a sweep.
- A NaN or +inf logit gives a NaN logprob, which the server refuses (host test); the sampled draw already refused
  such logits.
