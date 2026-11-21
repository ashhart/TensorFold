# Flash Next copy window: main against the branch

`TF_FLASHNEXT_WINDOW_CHECK=1`, off by default, runs a boot self-check that sweeps every verify width a window can
hold, compares each row's scores with the scores of a one-token step on the same machine, and serves a lone
stream's copy window at the largest width whose every row matched, past the 32 rows the engine ships and at most
the 48 rows a window's pending-row log holds. Unset or blank is main's 32 rows, and a check that cannot run,
fails, or proves nothing past 32 leaves the window at 32.

## Source and environment

- Main: `d8b947b1ff35ed1492b2a6aa675d30367b6ab777`. Branch: this pull request at `a13efbe9d95fe0dc96d8b4d8f8dddea557388d4f`; the receipt commit adds only this directory.
- Each image built from a verified `git archive` tree of its own commit, blob by blob against the commit's own
  manifest: the recipe's Zig base image, nvcc 13.3, Zig 0.17.0, kernels embedded.
- One NVIDIA GB10 (sm_121), 121 GiB unified memory, one GPU. Host: 7.0.0-1019-nvidia-64k; Ubuntu 24.04.5 LTS; NVIDIA GB10, 580.178.04, [N/A]; RAM 123 GiB.
- Checkpoint `nvidia/Qwen3.8-Flash-Next-FP4-Local`, revision `b9ce18a6887e18c089fdea8509a33e63429a3300`, every
  arm.
- `--context 32768 --parallel 8`, bf16 KV cache, drafts on (`TF_FLASHNEXT_DRAFTS=6`), a fresh server per arm, the
  recipe's own defaults (`TF_FLASHNEXT_WARM_AHEAD=1`, `TENSORFOLD_MEMORY_RESERVE_GIB=9` and the rest of the
  recipe's Flash Next settings), and nothing else. B1 adds `TF_FLASHNEXT_WINDOW_CHECK=1` and nothing else.
- Arms, in order: M1, B0a, B1a, B1b, B0b, M2 for the decode, copy and cold passes; M, B0, B1 for exactness, each
  with a cache-on boot and a `--prompt-cache-gib 0` boot; MD and B1D for the drafts-off pass.

Each arm ran, on its own fresh server:

- `tools/bench_openai.py --tokens 256 --reps 3 --temperatures 0` (one stream, a code prompt and a chat prompt).
- `tools/bench_concurrent.py --tokens 256 --reps 3 --levels 1,2,4,6,8 --alone --serial --temperatures 0`, which is
  both the decode cells and the concurrent-equals-solo contract, on the same code and chat prompts.
- `copy_bench.py`: one text of 1969 prompt tokens sent twice per rep, once with the instruction to repeat it
  verbatim and once with the instruction to change one word, one stream, greedy, 3 reps, the reply's token ids
  and their SHA-256 recorded.
- `/metrics` around every probe (`mtp_drafted_total`, `mtp_accepted_total`), and the server's own per-path request
  counts before and after it, so an arm that saw a request it did not send would be rerun.

The cold pass boots at `--context 131072 --parallel 4 --prompt-cache-gib 0`: `ttft.py` at 2k, 8k, 32k and 64k
(three reps each, thinking off, a fresh salt a prompt) and `tools/bench_cache.py --tokens 64 --repeats 2
--streams 2 --require-cold`.

## Before the first boot: the late-log contract, and the off path's pool line

Two measurements were taken inside the window's lock, with production stopped, before the first timed boot.

The branch serves the 48-row shape of the two late-log kernels `fn_gdn_front_log` and `fn_gdn_rows_late`, so the
kernel contract's gate 3 runs a 48-row window through both against the kernels they replace. One run, from the
branch image:

```
$ docker run --rm --gpus all --entrypoint /bin/sh tensorfold-qwen38:pcw-b-c14485f5 -c \
    'ls -l /opt/tensorfold/bin/tf-cuda-test && /opt/tensorfold/bin/tf-cuda-test kernel-contract fn_gdn_front_log fn_gdn_rows_late'
kernel-contract rc 0
```

`(filtered) SUMMARY 2 kernels: 2 pass gates 1-4, 0 fail, 0 known findings; 0 controls: 0 red as planted, 0 missed; 0 limit problems; gates 112 ms, limits 0 ms`

So 2 kernels passed gates 1 to 4 with 0 failing, and all 0 planted controls were red as planted with 0 missed; exit status 0.

The second measurement is the startup pool line, which is where the off path shows: with the switch unset the
branch must size every buffer a stream's window needs exactly as main does, and with it set the buffers hold the
log's 48 rows. Each boot's own line, from its container's log:

| boot | context/streams | prompt cache | drafts | model GiB | per-stream GiB | GiB left | after reserve GiB |
|---|---|---|---|---|---|---|---|
| 1/M1 | 32768/8 | on | on | 72.71 | 1.29 | 27.8 | 9.0 |
| 1/B0a | 32768/8 | on | on | 72.71 | 1.29 | 27.8 | 9.0 |
| 1/B1a | 32768/8 | on | on | 72.71 | 1.32 | 27.8 | 9.0 |
| 1/B1b | 32768/8 | on | on | 72.71 | 1.32 | 27.8 | 9.0 |
| 1/B0b | 32768/8 | on | on | 72.71 | 1.29 | 27.9 | 9.0 |
| 1/M2 | 32768/8 | on | on | 72.71 | 1.29 | 27.9 | 9.0 |
| 2/M1 | 131072/4 | off | on | 72.73 | 4.11 | 27.8 | 9.0 |
| 2/B0a | 131072/4 | off | on | 72.73 | 4.11 | 27.8 | 9.0 |
| 2/B1a | 131072/4 | off | on | 72.73 | 4.13 | 27.8 | 9.0 |
| 2/B1b | 131072/4 | off | on | 72.73 | 4.13 | 27.8 | 9.0 |
| 2/B0b | 131072/4 | off | on | 72.73 | 4.11 | 27.8 | 9.0 |
| 2/M2 | 131072/4 | off | on | 72.73 | 4.11 | 27.8 | 9.0 |
| 3/M/on | 32768/8 | on | on | 72.71 | 1.29 | 27.9 | 9.0 |
| 3/M/off | 32768/8 | off | on | 72.71 | 1.29 | 27.8 | 9.0 |
| 3/B0/on | 32768/8 | on | on | 72.71 | 1.29 | 27.9 | 9.0 |
| 3/B0/off | 32768/8 | off | on | 72.71 | 1.29 | 27.8 | 9.0 |
| 3/B1/on | 32768/8 | on | on | 72.71 | 1.32 | 27.8 | 9.0 |
| 3/B1/off | 32768/8 | off | on | 72.71 | 1.32 | 27.8 | 9.0 |
| 4/MD | 32768/8 | on | off | 70.84 | 0.98 | 30.1 | 9.0 |
| 4/B1D | 32768/8 | on | off | 70.84 | 0.98 | 30.1 | 9.0 |

- POOL B0 vs M 32768/8 cache off drafts on per-stream: M 1.29 B0 1.29 equal YES
- POOL B0 vs M 32768/8 cache off drafts on left: M 27.8 B0 27.8 equal YES
- POOL B1 vs M 32768/8 cache off drafts on per-stream: M 1.29 B1 1.32 equal NO
- POOL B1 vs M 32768/8 cache off drafts on left: M 27.8 B1 27.8 equal YES
- POOL M vs M 32768/8 cache off drafts on: per-stream 1.29 left 27.8 (1 boots)
- POOL B1 vs M 32768/8 cache on drafts off per-stream: M 0.98 B1 0.98 equal YES
- POOL B1 vs M 32768/8 cache on drafts off left: M 30.1 B1 30.1 equal YES
- POOL M vs M 32768/8 cache on drafts off: per-stream 0.98 left 30.1 (1 boots)
- POOL B0 vs M 32768/8 cache on drafts on per-stream: M 1.29 B0 1.29 equal YES
- POOL B0 vs M 32768/8 cache on drafts on left: M 27.8,27.9 B0 27.8,27.9 equal YES
- POOL B1 vs M 32768/8 cache on drafts on per-stream: M 1.29 B1 1.32 equal NO
- POOL B1 vs M 32768/8 cache on drafts on left: M 27.8,27.9 B1 27.8 equal WITHIN-MAIN-SPREAD
- POOL M vs M 32768/8 cache on drafts on: per-stream 1.29 left 27.8,27.9 (3 boots)
- POOL B0 vs M 131072/4 cache off drafts on per-stream: M 4.11 B0 4.11 equal YES
- POOL B0 vs M 131072/4 cache off drafts on left: M 27.8 B0 27.8 equal YES
- POOL B1 vs M 131072/4 cache off drafts on per-stream: M 4.11 B1 4.13 equal NO
- POOL B1 vs M 131072/4 cache off drafts on left: M 27.8 B1 27.8 equal YES
- POOL M vs M 131072/4 cache off drafts on: per-stream 4.11 left 27.8 (2 boots)

B0's per-stream size equals main's on every group it booted in, so the branch with the switch off allocates
main's buffers and the switch is the only difference. With the switch on, a stream's pool entry carries the
wider size the 48-row window needs: B1 reads 1.32 GiB each against main's 1.29 at the pass A
window.

## The check's line and the width it chose

Every B1 boot printed the same line: `1176 of 1176 rows byte-equal to their one-token steps (widths 1-48)`,
followed by `serving 48-row copy windows after the self-check`. 1176 is the sum of widths
1 to 48, so every row the buffers hold was compared.

M and B0 printed no self-check line at all; the startup-log shapes confirm that the check's line, or its
failure, is the only line the switch adds and that nothing else in the load changed:

- Line shapes (digits normalised) common to M1 and B0a: 8
- Shapes only B1a printed: 2
- Shapes B1a did not print that both main runs did: 0
  - B1a only: info: flashnext: serving #-row copy windows after the self-check
  - B1a only: info: flashnext: window self-check: # of # rows byte-equal to their one-token steps (widths #-#)

The served width is 48, so a lone stream's copy window ramps to 47 copied tokens a
round on this machine, against 31 on main. The width is a measurement, not a target: another GB10 with
another driver may prove fewer rows, and the receipt would then say so.

## Decode speed and tokens a round

Aggregate tok/s, the mean of each arm's two runs; tokens a round is the tools' pooled tokens a streamed chunk.

| invocation | prompt | case | streams | M1 | M2 | B0a | B0b | B1a | B1b | B0vsM | B1vsM | B1vsB0 | tokens a round, before / after: B0vsM, B1vsM, B1vsB0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openai_t0 | fibonacci-raw |  | 1 | 72.9 | 72.8 | 72.6 | 72.7 | 72.1 | 72.7 | -0.3% level | -0.6% level | -0.3% level | 5.22 / 5.22 / 5.22 / 5.22 / 5.22 / 5.22 |
| openai_t0 | gpu-chat-no-think |  | 1 | 46.7 | 46.7 | 46.6 | 46.6 | 46.3 | 46.6 | -0.2% level | -0.5% level | -0.4% level | 2.81 / 2.81 / 2.81 / 2.81 / 2.81 / 2.81 |
| conc_t0 | chat |  | 1 | 46.7 | 46.6 | 46.8 | 46.6 | 46.5 | 46.6 | +0.1% level | -0.2% level | -0.3% level | 2.81 / 2.81 / 2.81 / 2.81 / 2.81 / 2.81 |
| conc_t0 | chat |  | 2 | 86.6 | 86.6 | 87.0 | 86.7 | 86.2 | 86.4 | +0.3% level | -0.3% level | -0.6% level | 2.51 / 2.51 / 2.51 / 2.51 / 2.51 / 2.51 |
| conc_t0 | chat |  | 4 | 156.0 | 156.0 | 156.8 | 156.3 | 156.0 | 156.0 | +0.4% level | +0.0% level | -0.4% level | 2.51 / 2.51 / 2.51 / 2.51 / 2.51 / 2.51 |
| conc_t0 | chat |  | 6 | 210.4 | 210.4 | 211.4 | 210.7 | 210.2 | 210.3 | +0.3% level | -0.1% level | -0.4% level | 2.51 / 2.51 / 2.51 / 2.51 / 2.51 / 2.51 |
| conc_t0 | chat |  | 8 | 257.2 | 257.2 | 258.9 | 257.5 | 256.8 | 257.4 | +0.4% level | -0.0% level | -0.4% level | 2.51 / 2.51 / 2.51 / 2.51 / 2.51 / 2.51 |
| conc_t0 | code |  | 1 | 72.9 | 72.7 | 72.6 | 72.4 | 72.3 | 72.6 | -0.4% level | -0.5% level | -0.1% level | 5.22 / 5.22 / 5.22 / 5.22 / 5.22 / 5.22 |
| conc_t0 | code |  | 2 | 138.5 | 138.3 | 137.9 | 138.4 | 137.8 | 138.1 | -0.2% level | -0.3% level | -0.1% level | 4.83 / 4.83 / 4.83 / 4.83 / 4.83 / 4.83 |
| conc_t0 | code |  | 4 | 242.7 | 242.4 | 242.1 | 242.8 | 242.0 | 242.8 | -0.0% level | -0.1% level | -0.0% level | 4.83 / 4.83 / 4.83 / 4.83 / 4.83 / 4.83 |
| conc_t0 | code |  | 6 | 307.0 | 308.0 | 309.2 | 306.8 | 307.1 | 307.7 | +0.2% level | -0.0% level | -0.2% level | 4.49 / 4.49 / 4.49 / 4.49 / 4.49 / 4.49 |
| conc_t0 | code |  | 8 | 371.5 | 373.0 | 373.5 | 371.7 | 372.1 | 371.6 | +0.1% level | -0.1% level | -0.2% level | 4.49 / 4.49 / 4.49 / 4.49 / 4.49 / 4.49 |
| copy | copy.txt | changed | 1 | 246.8 | 249.2 | 250.2 | 250.7 | 287.9 | 287.9 | +1.0% level | +16.1% faster | +15.0% faster | 23.27 / 23.27 / 23.27 / 28.44 / 23.27 / 28.44 |
| copy | copy.txt | verbatim | 1 | 255.5 | 256.8 | 256.0 | 257.7 | 295.8 | 296.2 | +0.3% level | +15.6% faster | +15.2% faster | 23.27 / 23.27 / 23.27 / 28.44 / 23.27 / 28.44 |

Verdicts, B0 against M, B1 against M and B1 against B0:

| comparison | streams | cells | mean change | faster | level | slower |
|---|---|---|---|---|---|---|
| B0vsM | 1 | 2 | -0.15% | 0 | 2 | 0 |
| B0vsM | 2 | 2 | +0.05% | 0 | 2 | 0 |
| B0vsM | 4 | 2 | +0.16% | 0 | 2 | 0 |
| B0vsM | 6 | 2 | +0.24% | 0 | 2 | 0 |
| B0vsM | 8 | 2 | +0.24% | 0 | 2 | 0 |
| B1vsM | 1 | 2 | -0.35% | 0 | 2 | 0 |
| B1vsM | 2 | 2 | -0.34% | 0 | 2 | 0 |
| B1vsM | 4 | 2 | -0.03% | 0 | 2 | 0 |
| B1vsM | 6 | 2 | -0.05% | 0 | 2 | 0 |
| B1vsM | 8 | 2 | -0.07% | 0 | 2 | 0 |
| B1vsB0 | 1 | 2 | -0.20% | 0 | 2 | 0 |
| B1vsB0 | 2 | 2 | -0.39% | 0 | 2 | 0 |
| B1vsB0 | 4 | 2 | -0.19% | 0 | 2 | 0 |
| B1vsB0 | 6 | 2 | -0.29% | 0 | 2 | 0 |
| B1vsB0 | 8 | 2 | -0.31% | 0 | 2 | 0 |

- B0vsM: 0 faster, 14 level, 0 slower, 0 unmeasured.
- B1vsM: 2 faster, 12 level, 0 slower, 0 unmeasured.
- B1vsB0: 2 faster, 12 level, 0 slower, 0 unmeasured.

The code prompt's cells at 1, 2, 4, 6 and 8 streams are the contract the copy window must not move: a shared
round's copies read `cfg.first_copy` only, because `copyProposal` reads `s.copy_width` under `e.alone`, so the
served width cannot reach a shared round.

## The copy cells

The same text repeated, one stream, greedy, three reps an arm; tokens a round is the server's own MTP figure for
the request, and the first difference is the first character at which the reply leaves the text.

| case | arm | decode tok/s | tokens a round (median / max) | rounds (median) | copied reps | first diff (reps) | prompt tokens |
|---|---|---|---|---|---|---|---|
| changed | M1 | 246.8 | 23.273 / 23.273 | 11 | 3 | [None, None, None] | 1968 |
| changed | M2 | 249.2 | 23.273 / 23.273 | 11 | 3 | [None, None, None] | 1968 |
| changed | B0a | 250.2 | 23.273 / 23.273 | 11 | 3 | [None, None, None] | 1968 |
| changed | B0b | 250.7 | 23.273 / 23.273 | 11 | 3 | [None, None, None] | 1968 |
| changed | B1a | 287.9 | 28.444 / 28.444 | 9 | 3 | [None, None, None] | 1968 |
| changed | B1b | 287.9 | 28.444 / 28.444 | 9 | 3 | [None, None, None] | 1968 |
| verbatim | M1 | 255.5 | 23.273 / 23.273 | 11 | 3 | [None, None, None] | 1969 |
| verbatim | M2 | 256.8 | 23.273 / 23.273 | 11 | 3 | [None, None, None] | 1969 |
| verbatim | B0a | 256.0 | 23.273 / 23.273 | 11 | 3 | [None, None, None] | 1969 |
| verbatim | B0b | 257.7 | 23.273 / 23.273 | 11 | 3 | [None, None, None] | 1969 |
| verbatim | B1a | 295.8 | 28.444 / 28.444 | 9 | 3 | [None, None, None] | 1969 |
| verbatim | B1b | 296.2 | 28.444 / 28.444 | 9 | 3 | [None, None, None] | 1969 |

- `changed`: B0vsM +1.0% level; B1vsM +16.1% faster; B1vsB0 +15.0% faster; tokens a round before / after 23.27 / 23.27 / 23.27 / 28.44 / 23.27 / 28.44.
- `verbatim`: B0vsM +0.3% level; B1vsM +15.6% faster; B1vsB0 +15.2% faster; tokens a round before / after 23.27 / 23.27 / 23.27 / 28.44 / 23.27 / 28.44.

Verdicts for the copy cells are in `noise-rule.md` with the rest; the two cells are the only ones where the
wider window is expected to show, because a copy proposal needs a context the reply can repeat.

## Cold prompts

Time to the first token, s, the median of three salted prompts, thinking off, prompt cache off.

| prompt | M1 | M2 | B0a | B0b | B1a | B1b | B0vsM | B1vsM | B1vsB0 |
|---|---|---|---|---|---|---|---|---|---|
| cold2k-off | 1.061 | 1.055 | 1.055 | 1.074 | 1.061 | 1.072 | +0.6% level | +0.8% level | +0.2% level |
| cold8k-off | 4.007 | 4.009 | 4.010 | 4.039 | 3.999 | 3.982 | +0.4% level | -0.4% level | -0.8% level |
| cold32k-off | 16.022 | 16.054 | 16.008 | 16.086 | 16.076 | 16.055 | +0.1% level | +0.2% level | +0.1% level |
| cold64k-off | 32.968 | 33.065 | 32.977 | 33.168 | 33.068 | 33.070 | +0.2% level | +0.2% level | -0.0% level |
| first_ttft_s | 1.223 | 1.226 | 1.225 | 1.226 | 1.230 | 1.221 | +0.1% level | +0.1% level | +0.0% level |

Every first request reported zero cached tokens. The change reaches decode windows only, and the check runs
before the first prompt, so a prompt cell can move only by boot noise.

## The noise rule

A cell is slower when both runs of the after arm are under both runs of the before arm and the mean gap is over
1% (for a latency: over), faster when both are over both and the gap is over 1%, level otherwise. A flagged cell
gets one rerun on fresh boots an arm, and a repeat of slower is a loss.

- No cell was slower in the first pair, so no rerun was taken.
- Losses: 0

## Exactness

`exactness.md` has every line. The three contracts, on each arm, with the prompt cache on and off:

- B0 cache on concurrent: concurrent equals solo 156/0 (unverified 0); drafted equals `"draft": false` 56/0
- B0 cache on golden: concurrent equals solo 13/0 (unverified 0); drafted equals `"draft": false` 24/0
- B0 cache on drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B0 cache on turns: resumed from cached prompt tokens [13, 16, 16, 13, 24, 18, 3406]
- B0 cache off drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B0 cache off fresh: 7 of 7 resumed replies equal the fresh ones
- B1 cache on concurrent: concurrent equals solo 156/0 (unverified 0); drafted equals `"draft": false` 56/0
- B1 cache on golden: concurrent equals solo 13/0 (unverified 0); drafted equals `"draft": false` 24/0
- B1 cache on drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B1 cache on turns: resumed from cached prompt tokens [13, 16, 16, 13, 24, 18, 3406]
- B1 cache off drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B1 cache off fresh: 7 of 7 resumed replies equal the fresh ones
- M cache on concurrent: concurrent equals solo 156/0 (unverified 0); drafted equals `"draft": false` 56/0
- M cache on golden: concurrent equals solo 13/0 (unverified 0); drafted equals `"draft": false` 24/0
- M cache on drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- M cache on turns: resumed from cached prompt tokens [13, 16, 16, 13, 24, 18, 3406]
- M cache off drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- M cache off fresh: 7 of 7 resumed replies equal the fresh ones
- B0: replies equal over pass A's greedy cells (token_sha): 528 equal, 0 unequal
- B1: replies equal over pass A's greedy cells (token_sha): 528 equal, 0 unequal
- B1/M: replies equal over pass A's greedy cells (token_sha): 528 equal, 0 unequal

Unequal or failed pairs across every contract: 0.

## Boot time and the check's cost

Wall time is the recipe's own launch to its own return, the smoke-test request included, so the difference
between a B1 boot and a B0 boot of the same image is what the check costs.

The boot cells are a second window taken on the head above, after main's load-time page-cache change (#3467),
three boots an arm on a fresh server each, in the order M1 B0a B1a M2 B1b B0b M3 B0c B1c; every other cell in
this receipt is the first window's, whose source files are byte-identical on this head apart from main's own three
edited files.

| arm | image | ctx/par | wall s | server answered s | began (UTC) | ended (UTC) | gpu at boot | check |
|---|---|---|---|---|---|---|---|---|
| B0a | pcw2-b-a13efbe9 | 32768/8 | 37.11 | 36 | 2026-11-19T16:27:10+00:00 | 2026-11-19T16:27:47+00:00 | 53,2470MHz,14.67W | - |
| B0b | pcw2-b-a13efbe9 | 32768/8 | 37.01 | 36 | 2026-11-19T16:32:52+00:00 | 2026-11-19T16:33:29+00:00 | 53,2437MHz,14.85W | - |
| B0c | pcw2-b-a13efbe9 | 32768/8 | 37.03 | 36 | 2026-11-19T16:34:26+00:00 | 2026-11-19T16:35:03+00:00 | 52,2411MHz,14.61W | - |
| B1a | pcw2-b-a13efbe9 | 32768/8 | 49.13 | 49 | 2026-11-19T16:27:58+00:00 | 2026-11-19T16:28:47+00:00 | 54,2405MHz,15.11W | 1176 of 1176, serving 48 |
| B1b | pcw2-b-a13efbe9 | 32768/8 | 49.14 | 49 | 2026-11-19T16:31:53+00:00 | 2026-11-19T16:32:42+00:00 | 56,2405MHz,15.15W | 1176 of 1176, serving 48 |
| B1c | pcw2-b-a13efbe9 | 32768/8 | 49.14 | 49 | 2026-11-19T16:35:14+00:00 | 2026-11-19T16:36:03+00:00 | 54,2411MHz,14.97W | 1176 of 1176, serving 48 |
| M1 | pcw2-m-d8b947b1 | 32768/8 | 37.04 | 36 | 2026-11-19T16:24:13+00:00 | 2026-11-19T16:24:50+00:00 | 45,2444MHz,13.94W | - |
| M2 | pcw2-m-d8b947b1 | 32768/8 | 37.01 | 36 | 2026-11-19T16:31:06+00:00 | 2026-11-19T16:31:43+00:00 | 54,2463MHz,14.90W | - |
| M3 | pcw2-m-d8b947b1 | 32768/8 | 37.03 | 37 | 2026-11-19T16:33:39+00:00 | 2026-11-19T16:34:16+00:00 | 52,2470MHz,14.72W | - |

- Mean wall time: M 37.03 s, B0 37.05 s, B1 49.14 s; B1 minus B0 +12.09 s, B1 minus M +12.11 s, B0 minus M +0.02 s.
- Boot-to-boot spread within an arm: M 0.03 s, B0 0.1 s, B1 0.01 s, so the largest is 0.1 s; the B1 delta is larger than the spread, so it is a cost of the check.
- Mean `Server answered` time: M 36.3 s, B0 36.0 s, B1 49.0 s.
- The same window re-took the copy cells with one probe on every boot; tokens a round read B0 23.273, B1 28.444, M 23.273 (the first window's figures).
- The sweep's own duration is not logged separately; the boot-time difference above is what it costs.
- Boot time is reported, not treated as a regression: it is paid once a server, and the width it buys is the
  lever.

## Drafts off

`MD` (main) and `B1D` (the branch, switch on) booted with `DRAFTS=0`. The copy window is a drafter's path, so
the switch must be inert here: the same tokens and the same speed.

| arm | prompt | streams | aggregate tok/s | tokens a round | alone equal/unequal |
|---|---|---|---|---|---|
| B1D | code | 1 | 26.9 | 1.0 | 3/0 |
| B1D | code | 2 | 50.8 | 1.0 | 6/0 |
| B1D | code | 4 | 91.8 | 1.0 | 12/0 |
| B1D | chat | 1 | 26.9 | 1.0 | 3/0 |
| B1D | chat | 2 | 50.9 | 1.0 | 6/0 |
| B1D | chat | 4 | 91.8 | 1.0 | 12/0 |
| MD | code | 1 | 26.9 | 1.0 | 3/0 |
| MD | code | 2 | 51.0 | 1.0 | 6/0 |
| MD | code | 4 | 91.3 | 1.0 | 12/0 |
| MD | chat | 1 | 27.0 | 1.0 | 3/0 |
| MD | chat | 2 | 51.1 | 1.0 | 6/0 |
| MD | chat | 4 | 91.8 | 1.0 | 12/0 |

- Token hashes B1D against MD: 42 equal, 0 unequal.

## Memory

Each arm's pool line, from its own startup log (the table above has the same lines with the per-stream and
memory-left figures parsed out):

- MEMLINE 1/M1 model 72.71 GiB; 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- MEMLINE 1/B0a model 72.71 GiB; 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- MEMLINE 1/B1a model 72.71 GiB; 8 streams at once, 1.32 GiB each at a 32768-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- MEMLINE 1/B1b model 72.71 GiB; 8 streams at once, 1.32 GiB each at a 32768-token window, of 27.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- MEMLINE 1/B0b model 72.71 GiB; 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- MEMLINE 1/M2 model 72.71 GiB; 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels

## Not established

- One GB10 and one checkpoint. The width is a property of this machine's kernels and driver, which is why it is
  measured at boot; a machine that proves fewer rows serves fewer.
- TP=2, Metal and HIP were not run: the probe runs only with drafts on at TP=1, and every other backend keeps its
  own width, so their behaviour is main's.
- The sampled decode cells are the temperature 1.0 cells of the exactness pass; no speed claim is made on them,
  only that their replies equal the greedy and the solo ones.
