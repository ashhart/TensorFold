# Flash Next pooled n-gram gather: main against the branch

`TF_FLASHNEXT_GATHER_EARLY`, default `0`, has a decode round's shared pass gather every stream's n-gram rows in one pool call instead of one call a stream. Unset or `0` is main's per-stream call.

## Source and environment

- Main: `077998f7d9b14adebec3a2acc2caa47ff7ae6ec7`. Branch: this pull request at `05e319fca3dcef27d12cdd76373b276765e1c9d7`, one commit on that main.
- Each image built from a verified `git archive` tree of its own commit (checked blob by blob against its commit's `git ls-tree`), the recipe's Zig base image with `tensorfold-native`, its tools and the fatbins.
- One NVIDIA GB10 (sm_121), Zig 0.17.0, CUDA 13.3, one model process.
- Checkpoint `nvidia/Qwen3.8-Flash-Next-FP4-Local` at revision `b9ce18a6887e18c089fdea8509a33e63429a3300`, one checkpoint for every arm.
- A fresh server per arm through the MiaAI-Lab recipe's `start.sh`: `CHECKPOINT=nvfp4`, `VISION=0`, `TENSORFOLD_CUDA_KERNELS=native`, `KV_DTYPE=bf16`, `--context 32768 --parallel 8` on the decode arms, bf16 KV, drafts on.
- The recipe's `scripts/config.sh` exports `TF_FLASHNEXT_GATHER_EARLY=1`, so the branch's off arm sets it to `0` explicitly and main's arm carries the variable and ignores it (main has no such row). The container's own environment is recorded per boot in each arm's `env.txt`.
- Arms, in order (two boots an arm, alternating): M1, B0a, B1a, B1b, B0b, M2. M is main, B0 the branch with the switch off, B1 the branch with it on.

## Cells

- Decode, cache on: `tools/bench_openai.py --tokens 256 --reps 3 --temperatures 0` at one stream and `tools/bench_concurrent.py --tokens 256 --reps 3 --levels 1,2,4,6,8 --alone --serial --temperatures 0` (code and chat prompts), from main's own `tools/` for every arm.
- Cold prompts, cache off (`--prompt-cache-gib 0`, 131,072-token window, 4 slots): 2k, 8k, 32k and 64k tokens at one stream, three salted prompts a size, thinking off, and `tools/bench_cache.py --require-cold`.
- Exactness: the six golden prompts at 1 and 4 streams `--alone --serial` greedy, `served_exactness.py drafts` and `turns` with the cache on, `served_exactness.py fresh` and `drafts` with it off, and a cancel probe on the server's own cancel route.
- Drafts off (`DRAFTS=0`): `bench_concurrent --levels 1,2,4,8 --alone` greedy.

## Decode speed and tokens a round

Aggregate tok/s; tokens a round is the tools' tokens a streamed chunk (a round sends its accepted drafts and one verified token). Change and verdict compare the branch's two runs against main's two runs by the noise rule below; `noise-rule.md` has every cell of every arm and the rerun pairs, `cells.csv` has them as data.

| streams | cells | B0 vs M mean | verdicts | B1 vs M mean | verdicts |
|---|---|---|---|---|---|
| 1 | 2 | -0.52% | 0 faster, 2 level, 0 slower | +0.34% | 0 faster, 2 level, 0 slower |
| 2 | 2 | -0.42% | 0 faster, 2 level, 0 slower | +0.68% | 0 faster, 2 level, 0 slower |
| 4 | 2 | -0.40% | 0 faster, 2 level, 0 slower | +1.32% | 2 faster, 0 level, 0 slower |
| 6 | 2 | -0.12% | 0 faster, 2 level, 0 slower | +1.77% | 2 faster, 0 level, 0 slower |
| 8 | 2 | -0.46% | 0 faster, 2 level, 0 slower | +2.10% | 2 faster, 0 level, 0 slower |

Verdicts (first pair): B0 vs M: 0 faster, 12 level, 0 slower; B1 vs M: 6 faster, 6 level, 0 slower; B1 vs B0: 7 faster, 5 level, 0 slower
Rerun (one per flagged cell; a repeat of slower is a loss):
Losses: 0

## Cold prompts

Seconds to the first token; a fresh salt opens every prompt, so every first request reports 0 cached tokens.

```
| prompt | MC1 | MC2 | B0C1 | B0C2 | B1C1 | B1C2 | B0 vs M | B1 vs M | B1 vs B0 |
|---|---|---|---|---|---|---|---|---|---|
| cold2k-off | 1.093 | 1.072 | 1.078 | 1.076 | 1.091 | 1.067 | +0.5% level | +0.3% level | -0.2% level |
| cold8k-off | 4.051 | 3.967 | 3.989 | 3.980 | 3.981 | 4.001 | +0.6% level | +0.4% level | -0.2% level |
| cold32k-off | 16.248 | 15.924 | 16.018 | 16.052 | 15.992 | 16.003 | +0.3% level | +0.6% level | +0.2% level |
| cold64k-off | 33.098 | 32.902 | 32.926 | 33.028 | 32.928 | 32.890 | +0.1% level | +0.3% level | +0.2% level |
| first_ttft_s | 1.230 | 1.222 | 1.225 | 1.227 | 1.218 | 1.219 | -0.0% level | +0.6% level | +0.7% level |
- B0C1: first ttft 1.2254346199333668 s, 2571 prompt tokens, cached 0 (cold), qualified True
- B0C2: first ttft 1.2271071821451187 s, 2571 prompt tokens, cached 0 (cold), qualified True
- B1C1: first ttft 1.2175115970894694 s, 2571 prompt tokens, cached 0 (cold), qualified True
- B1C2: first ttft 1.2187956660054624 s, 2571 prompt tokens, cached 0 (cold), qualified True
- MC1: first ttft 1.2304023278411478 s, 2571 prompt tokens, cached 0 (cold), qualified True
- MC2: first ttft 1.22166123916395 s, 2571 prompt tokens, cached 0 (cold), qualified True

```

## Exactness

```
| contract | M | B0 | B1 |
|---|---|---|---|
| concurrent equals solo (pass A, 1 to 8 streams, greedy) | 252/252 equal | 252/252 equal | 252/252 equal |
| drafted equals `"draft": false` (pass A, 1 to 8 streams, greedy) | 20/20 equal | 20/20 equal | 20/20 equal |
| resumed equals fresh (pass C, 7 chats, cache on against cache off) | 7/7 equal | 7/7 equal | 7/7 equal |
| cancel and reset leave the next request valid (pass C, the cancel route) | 3/3 contracts | 3/3 contracts | 3/3 contracts |
| every greedy reply equals main's (token_sha over pass A's cells), B0 | 528/528 equal | | |
| every greedy reply equals main's (token_sha over pass A's cells), B1 | 528/528 equal | | |

- B0 cache on golden (6 prompts, 1 and 4 streams): concurrent equals solo 5/0; drafted equals `"draft": false` 8/0
- B0 cache on drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B0 cache on turns: 7 chats, resumed from cached prompt tokens [13, 16, 16, 13, 24, 18, 3406]
- B0 cache on cancel: 3 of 3 contracts, reply after the cancel True equal to the reference (token_sha256)
- B0 cache off drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B0 cache off fresh: 7 of 7 resumed replies equal the fresh ones
- B1 cache on golden (6 prompts, 1 and 4 streams): concurrent equals solo 5/0; drafted equals `"draft": false` 8/0
- B1 cache on drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B1 cache on turns: 7 chats, resumed from cached prompt tokens [13, 16, 16, 13, 24, 18, 3406]
- B1 cache on cancel: 3 of 3 contracts, reply after the cancel True equal to the reference (token_sha256)
- B1 cache off drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B1 cache off fresh: 7 of 7 resumed replies equal the fresh ones
- M cache on golden (6 prompts, 1 and 4 streams): concurrent equals solo 5/0; drafted equals `"draft": false` 8/0
- M cache on drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- M cache on turns: 7 chats, resumed from cached prompt tokens [13, 16, 16, 13, 24, 18, 3406]
- M cache on cancel: 3 of 3 contracts, reply after the cancel True equal to the reference (token_sha256)
- M cache off drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- M cache off fresh: 7 of 7 resumed replies equal the fresh ones
- B0: replies equal over pass A's greedy cells (token_sha): 528 equal, 0 unequal
- B1: replies equal over pass A's greedy cells (token_sha): 528 equal, 0 unequal
- B1 against B0: replies equal over pass A's greedy cells (token_sha): 528 equal, 0 unequal

```

## Pool line, boot time and server acceptance

```
| arm | image | ctx/par | wall s | answered s | gpu at boot (C, MHz, W) | mem GiB | pool line |
|---|---|---|---|---|---|---|---|
| 1/M1 | tensorfold-qwen38:p1-m-077998f7 | 32768/8 | 34.0 | 33 | 42,2463MHz,13.78W | 35 | 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels |
| 1/B0a | tensorfold-qwen38:p1-b-05e319fc | 32768/8 | 34.0 | 33 | 53,2450MHz,15.13W | 34 | 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels |
| 1/B1a | tensorfold-qwen38:p1-b-05e319fc | 32768/8 | 34.0 | 33 | 57,2450MHz,15.69W | 33 | 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels |
| 1/B1b | tensorfold-qwen38:p1-b-05e319fc | 32768/8 | 34.1 | 34 | 57,2411MHz,15.71W | 33 | 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels |
| 1/B0b | tensorfold-qwen38:p1-b-05e319fc | 32768/8 | 34.0 | 34 | 59,2411MHz,15.62W | 33 | 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels |
| 1/M2 | tensorfold-qwen38:p1-m-077998f7 | 32768/8 | 34.0 | 33 | 58,2411MHz,15.76W | 33 | 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels |
| 2/MC1 | tensorfold-qwen38:p1-m-077998f7 | 131072/4 | 34.0 | 34 | 61,2411MHz,15.95W | 34 | 4 streams at once, 4.11 GiB each at a 131072-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels |
| 2/B0C1 | tensorfold-qwen38:p1-b-05e319fc | 131072/4 | 34.0 | 33 | 62,2411MHz,16.47W | 33 | 4 streams at once, 4.11 GiB each at a 131072-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels |
| 2/B1C1 | tensorfold-qwen38:p1-b-05e319fc | 131072/4 | 34.0 | 33 | 64,2411MHz,16.43W | 34 | 4 streams at once, 4.11 GiB each at a 131072-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels |
| 2/B1C2 | tensorfold-qwen38:p1-b-05e319fc | 131072/4 | 34.0 | 33 | 63,2450MHz,16.55W | 34 | 4 streams at once, 4.11 GiB each at a 131072-token window, of 26.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels |
| 2/B0C2 | tensorfold-qwen38:p1-b-05e319fc | 131072/4 | 34.0 | 34 | 64,2431MHz,16.46W | 33 | 4 streams at once, 4.11 GiB each at a 131072-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels |
| 2/MC2 | tensorfold-qwen38:p1-m-077998f7 | 131072/4 | 34.0 | 34 | 64,2411MHz,16.37W | 36 | 4 streams at once, 4.11 GiB each at a 131072-token window, of 28.0 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels |
| 3/M/on | tensorfold-qwen38:p1-m-077998f7 | 32768/8 | 34.0 | 33 | 64,2411MHz,16.45W | 33 | 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels |
| 3/M/off | tensorfold-qwen38:p1-m-077998f7 | 32768/8 | 34.0 | 33 | 56,2431MHz,15.54W | 34 | 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels |
| 3/B0/on | tensorfold-qwen38:p1-b-05e319fc | 32768/8 | 34.0 | 34 | 55,2424MHz,15.38W | 32 | 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels |
| 3/B0/off | tensorfold-qwen38:p1-b-05e319fc | 32768/8 | 34.0 | 34 | 54,2437MHz,15.03W | 33 | 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels |
| 3/B1/on | tensorfold-qwen38:p1-b-05e319fc | 32768/8 | 34.0 | 34 | 54,2405MHz,15.17W | 34 | 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels |
| 3/B1/off | tensorfold-qwen38:p1-b-05e319fc | 32768/8 | 34.0 | 33 | 54,2405MHz,15.07W | 33 | 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels |
| 4/MD | tensorfold-qwen38:p1-m-077998f7 | 32768/8 | 31.0 | 31 | 53,2405MHz,14.65W | 34 | 8 streams at once, 0.98 GiB each at a 32768-token window, of 30.1 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels |
| 4/B1D | tensorfold-qwen38:p1-b-05e319fc | 32768/8 | 30.9 | 30 | 53,2437MHz,14.65W | 35 | 8 streams at once, 0.98 GiB each at a 32768-token window, of 30.1 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels |

```

```
| arm | invocation | accepted | drafted | acceptance | alone equal/unequal | serial equal/unequal | failed | unmeasured |
|---|---|---|---|---|---|---|---|---|
| M1 | openai_t0 | 1488 | 2384 | 0.6242 | - | - | 0 | 0 |
| M1 | conc_t0 | 22760 | 33060 | 0.6884 | 24/0 | 1/0 | 0 | 0 |
| B0a | openai_t0 | 1488 | 2384 | 0.6242 | - | - | 0 | 0 |
| B0a | conc_t0 | 22760 | 33060 | 0.6884 | 24/0 | 1/0 | 0 | 0 |
| B1a | openai_t0 | 1488 | 2384 | 0.6242 | - | - | 0 | 0 |
| B1a | conc_t0 | 22760 | 33060 | 0.6884 | 24/0 | 1/0 | 0 | 0 |
| B1b | openai_t0 | 1488 | 2384 | 0.6242 | - | - | 0 | 0 |
| B1b | conc_t0 | 22760 | 33060 | 0.6884 | 24/0 | 1/0 | 0 | 0 |
| B0b | openai_t0 | 1488 | 2384 | 0.6242 | - | - | 0 | 0 |
| B0b | conc_t0 | 22760 | 33060 | 0.6884 | 24/0 | 1/0 | 0 | 0 |
| M2 | openai_t0 | 1488 | 2384 | 0.6242 | - | - | 0 | 0 |
| M2 | conc_t0 | 22760 | 33060 | 0.6884 | 24/0 | 1/0 | 0 | 0 |

```

## Method and the noise rule

- A cell pairs each arm's two runs (mean): it is slower when both branch runs are under both main runs and the mean gap is over 1%, faster when both are over both and the gap is over 1%, level otherwise; for a latency the same rule with higher meaning slower. A cell slower in the first pair gets one rerun of its invocation on fresh boots an arm, and a repeat of slower is a loss.
- Every probe records `/metrics` (`mtp_drafted_total`, `mtp_accepted_total`) before and after itself and the server's own per-path request counts; an arm that saw a request it did not send, or whose boot or a probe failed, is rerun once on a fresh boot with its first attempt kept as `<dir>.rejected1`.
- The GPU lock was held for the whole window, the production serve was stopped inside it and restored from the runner's EXIT trap with `/health` checked, and the fleet dashboard's poller was stopped before the window and started after it.

## Evidence

- IMAGE p1-m-077998f7 sha256:d00a7156c897a07e09c7f5d29e81e094792e6130ff598e13dd45d5ad925cab65 created=2026-11-16T20:37:06.583481326Z kernels=embedded
- IMAGE p1-b-05e319fc sha256:c28e46c9ebf6379f53c543803bf19273b0cd602d7366b970d4c3ca555dd75aad created=2026-11-16T23:06:14.411234398Z kernels=embedded
- POOL 1/M1 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- POOL 1/B0a 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- POOL 1/B1a 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- POOL 1/B1b 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- POOL 1/B0b 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- POOL 1/M2 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- POOL 2/MC1 4 streams at once, 4.11 GiB each at a 131072-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels
- POOL 2/B0C1 4 streams at once, 4.11 GiB each at a 131072-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels
- POOL 2/B1C1 4 streams at once, 4.11 GiB each at a 131072-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels
- POOL 2/B1C2 4 streams at once, 4.11 GiB each at a 131072-token window, of 26.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels
- POOL 2/B0C2 4 streams at once, 4.11 GiB each at a 131072-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels
- POOL 2/MC2 4 streams at once, 4.11 GiB each at a 131072-token window, of 28.0 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels
- POOL 3/M/on 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- POOL 3/M/off 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels
- POOL 3/B0/on 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- POOL 3/B0/off 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels
- POOL 3/B1/on 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- POOL 3/B1/off 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels
- POOL 4/MD 8 streams at once, 0.98 GiB each at a 32768-token window, of 30.1 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- POOL 4/B1D 8 streams at once, 0.98 GiB each at a 32768-token window, of 30.1 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- MEMLINE 1/M1 model 72.71 GiB; 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- MEMLINE 1/B0a model 72.71 GiB; 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- MEMLINE 1/B1a model 72.71 GiB; 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- RESTORE prod_rc 0 /health 200 2026-11-17T00:29:58+00:00
- FOREIGN-SUMMARY 1/B0a=0 1/B0b=0 1/B1a=0 1/B1b=0 1/M1=0 1/M2=0 2/B0C1=0 2/B0C2=0 2/B1C1=0 2/B1C2=0 2/MC1=0 2/MC2=0 3/B0/off=0 3/B0/on=0 3/B1/off=0 3/B1/on=0 3/M/off=0 3/M/on=0 4/B1D=0 4/MD=0
- DONE 2026-11-17T00:28:27+00:00

Every arm's JSON, boot log, `server.log`, `gets.txt`, `metrics.txt`, `gpu.txt` and `env.txt` are kept with the run, and `cells.csv` has every cell of every arm.
