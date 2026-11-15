# GLM-5.3-Flash FP4-Local on CUDA at TP=2: a receipt on main

Tested commit: `3a56d4eb92` (`commit.txt`), built from a `git archive` of it with an empty `.zig-cache`. Later `main` up
to `964174ea05` changes only DeepSeek V4.1 Metal files and the changelog, which GLM's CUDA path does not read; it was
not rebuilt. Test date: 2026-11-15. Platform: two DGX Spark (GB10, sm_121, 128 GB shared memory), TP=2 over a direct
RoCE link (NCCL NET/IB), Ubuntu 24.04.5, kernel 7.0.0-1025-nvidia, driver 580.178.04 (CUDA 13.0). Toolchain: Zig 0.17.0,
nvcc 13.5 (V13.5.37) for sm_121; runtime NCCL 2.30.7. One `tensorfold-native` (sha256 in `commit.txt`) on both ranks.
Checkpoint: `nvidia/GLM-5.3-Flash-FP4-Local` at `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, its own MTP layer as the
drafter, with the default copies beside it (`drafts: MTP + copies`).

## Scope

This rechecks FP4-Local as #1050 loads it, on current `main` with default settings: drafted = `"draft": false`,
concurrent = alone, resumed = cold, per-rank load and peak memory, decode with and without drafts, and cold prompts from
2,048 to 65,536 tokens. It does not rerun the [1.1.5-era receipt](../glm-fp4local-80517fc3/README.md)'s loader gate,
kernel contract, structured output and tools, media and video, prompt logprobs against BF16 (its BF16 reference is not
published), or TP=3. No baseline binary ran beside it: the earlier receipt's columns below are context, not a
before/after.

## Results

Exactness (`ex/`: `--context 40960 --parallel 4`, every other setting default, so the prompt cache is on):

| Check | Ids | Result |
| --- | --- | --- |
| Golden count (`golden_http.py`, 512 tokens, greedy) | `ba28cfb7c5b9` | PASS drafted = plain, 4.20 tokens a round, 11.6 s |
| Golden prose | `dae7e2a5c84e` | PASS drafted = plain, 1.92 tokens a round, 17.1 s |
| Golden code | `db45a54eb863` | PASS drafted = plain, 3.01 tokens a round, 13.6 s |
| Concurrent (`bench_concurrent.py --levels 1,2,4 --temperatures 0,1 --reps 3 --alone --serial`) | 12 cells | PASS: 84 replies equal to their alone runs and 30 alone runs equal to `"draft": false`, 0 unequal or failed |
| Resume (`resume_http.py`, a 6,595-token chat) | `1e3f89236de5` | PASS: repeated, `"draft": false` and two at once resume from 6,144 cached tokens, each equal to the cold reply |

The three golden ids are the 1.1.5-era receipt's (`tp2/serve/ex-fl`): FP4-Local's greedy replies have not moved since.

Startup and memory (process-owned CUDA buffers from `TF_CUDA_MEMORY_RECEIPTS`, `memory-rank*.json`):

| Launch | Model on each rank | Peak on each rank | After cleanup | Plan |
| --- | --- | --- | --- | --- |
| `ex/`, 40,960 tokens, 4 streams | 90.20 GiB | 103.31 GiB | 0.00 | 324 pages of 512 rows (2.02 GiB), a 10.4 GiB prompt cache |
| `sp/`, 131,072 tokens, 4 streams, `--prompt-cache-gib 0` | 90.26 GiB | 97.33 GiB | 0.00 | 1,028 pages of 512 rows (6.40 GiB), no prompt cache |

Both ranks exited 0 after SIGINT in both launches. The loads read the checkpoint from the page cache (71.5 to 91.5 s to
ready); `ex/`'s first launch of the day timed its windows (`41.07 ms at 1 row, 63.31 at 4, 180.35 at 32, a draft 1.98
ms; experts classic`) and `sp/` kept them.

Decode (`sp/`, 256 tokens, 5 reps after a warm-up, one stream):

| Decode | Prompt | t | Median tok/s | All |
| --- | --- | ---: | ---: | --- |
| `"draft": false` (`openai_aligned.py nodraft`) | fibonacci-raw | 0 | 23.9 | 23.92, 23.53, 23.87, 23.91, 23.43 |
| `"draft": false` | gpu-chat-no-think | 0 | 23.9 | 23.86, 23.62, 23.77, 23.9, 23.89 |
| drafted (`bench_openai.py`) | fibonacci-raw | 0 | 42.3 | 41.94, 42.26, 42.33, 42.87, 40.49 |
| drafted | gpu-chat-no-think | 0 | 34.4 | 35.56, 35.2, 33.11, 34.35, 33.51 |
| drafted | fibonacci-raw | 1 | 42.7 | 44.4, 40.92, 42.68, 46.41, 42.07 |
| drafted | gpu-chat-no-think | 1 | 35.8 | 33.02, 37.72, 35.78, 34.47, 35.8 |

Every greedy reply has one token hash across its reps.

Cold prompts (`sp/`, `bench_cache.py --require-cold --tokens 16`, `--prompt-cache-gib 0`; every row 0 cached tokens and
one token hash):

| Prompt tokens | First token, s (first, two repeats) | Prompt tok/s (median) | Two at once, s |
| ---: | --- | ---: | --- |
| 2,048 | 1.39, 1.83, 1.37 | 1,478 | 1.38, 2.82 |
| 8,192 | 5.69, 5.69, 5.70 | 1,439 | 5.68, 12.40 |
| 9,600 | 6.78, 6.79, 6.76 | 1,417 | 6.96, 14.66 |
| 32,768 | 22.43, 22.88, 22.88 | 1,432 | 23.05, 46.71 |
| 38,400 | 26.91, 26.87, 26.87 | 1,429 | 29.70, 54.59 |
| 65,536 | 45.84, 45.42, 45.91 | 1,430 | 48.26, 91.63 |

Throughput by stream count (`ex/`, `bench_concurrent.py`, drafted, 256 tokens, aggregate tok/s):

| Prompt | t | 1 stream | 2 streams | 4 streams |
| --- | ---: | ---: | ---: | ---: |
| code | 0 | 42.4 | 73.8 | 113.8 |
| chat | 0 | 34.0 | 56.5 | 85.4 |
| code | 1 | 44.9 | 58.0 | 71.9 |
| chat | 1 | 33.9 | 44.7 | 56.3 |

## Beside the 1.1.5-era receipt

The [earlier receipt](../glm-fp4local-80517fc3/README.md) ran `80517fc3` on 2026-10-20 on another GB10 pair (a two-rail
link, containers pinned to the efficiency cores, the page cache dropped before each load). The rows share its inputs:
`p9600.json` and `p38400.json` have its fixture hashes, the golden prompts are its `dc.json`, and the `PROMPTS` lists of
`bench_openai.py` and `bench_concurrent.py` are unchanged since v1.1.1. The machines differ, and so does the clients'
timing code (for example `c874a7e350` and `4a2596c630`), so the columns are side by side, not a before/after.

| Row | 2026-10-20, `80517fc3` | 2026-11-15, `3a56d4eb92` |
| --- | --- | --- |
| Golden ids, count / prose / code | `ba28cfb7` / `dae7e2a5` / `db45a54e` | the same |
| Tokens a round with drafts, count / prose / code | 3.26 / 1.82 / 2.72 | 4.20 / 1.92 / 3.01 |
| `"draft": false` decode (tok/s) | 24.5 to 24.6 | 23.9 |
| Drafted aggregate, 4 streams, code / chat, t=0 (tok/s) | 86.4 / 58.7 | 113.8 / 85.4 |
| Cold prompt of 9,600 and 38,400 tokens (s) | 7.22 and 28.52 | 6.78 and 26.91 |
| Model on each rank at load end | 90.27 GiB | 90.20 GiB (40,960-token launch) |

## Files

- `ex/` and `sp/`: each launch's raw results (`golden.json`, `conc.json`, `resume.json`, `nodraft.json`, `drafted.json`,
  `prefill*.json`), the clients' logs (`check-*.log`, `suite.log`), both ranks' server logs and memory receipts, and the
  host's free memory before the launch and at ready.
- `runs.log`: each launch's start, ready, suite result and end. `commit.txt`: the commit, build, toolchain and binary.
- `inputs.txt`: the sha256 of each client and fixture the suites read, and where it comes from (the tested commit's
  `tools/`, the earlier receipt's `tools/`, or the `openai_aligned.py tokens` command that makes it).
- `tools/`: `build.sh` (the build), `fl-env.sh` (the ranks' environment, with documentation addresses), `run.sh` (one
  launch: rank 1 on the peer, rank 0 here, the suite, then both stopped by SIGINT), `rank.sh` (a rank under
  `tools/memory_receipts.py`), `stop.sh`, `chain.sh` with `queue.txt` (the two launches), `suite-ex.sh`, `suite-sp.sh`,
  and `summarize.py`, which prints this receipt's tables from `ex/` and `sp/` (`python3 tools/summarize.py . ex sp`).
- `SHA256SUMS` covers every file here except itself.
