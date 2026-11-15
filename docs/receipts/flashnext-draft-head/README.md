# Flash Next draft head: main against the branch

`TF_FLASHNEXT_DRAFT_HEAD=bf16|affine4` says how the MTP head holds a bf16 language head's rows for the draft vocabulary
(79,591 rows of 2,560). On main, unset is `bf16`, and a blank value stops the start (`BadDraftHead`). On the branch,
unset or blank is `affine4`: 4-bit codes with a bf16 scale and bias for each group of 32, fitted at load from the
group's minimum and maximum, as `TF_FLASHNEXT_DRAFT_HEAD=affine4` already does on main. `bf16` keeps main's rows.

## Source and environment

- Main: `d6c4cce602eef4aff235625a5325bb617a9674ff`. Branch: this pull request's change on that main.
- One NVIDIA GB10 (sm_121), Zig 0.17.0, CUDA 13.3.
- Checkpoint `nvidia/Qwen3.8-Flash-Next-FP4-Local`, revision `b9ce18a6887e18c089fdea8509a33e63429a3300`, every arm. Its
  language head is bf16, the case the setting applies to.
- `--parallel 4`, 131,072-token context, bf16 KV, prompt cache on, a fresh server per arm. `--parallel 8` did not start
  at this context with bf16 KV, so every arm ran with 4.
- Arms, in order: M1 (main), U1 (branch, unset), X1 (branch, `bf16`), M2, X2, U2 (branch, set blank). No cell was
  slower, so the noise rule's rerun did not run.
- Main's tools: `tools/bench_openai.py` (one stream; greedy, 3 reps; temperature 1.0, 9 reps with seeds 1234 to 1242;
  its median decode rate) and `tools/bench_concurrent.py --alone --serial` (1, 2 and 4 streams, 256 tokens, 3 reps;
  greedy, and temperature 1.0 with `--seed` 1234, 2234 and 3234; aggregate tok/s).

## Decode speed and tokens a round

`cells.csv` has every cell of every arm. `tokens_per_round` is the tools' tokens a streamed chunk (a round sends its
accepted drafts and one verified token), `accepted_per_round` is that less one, and `server_acceptance` is the
server's `mtp_accepted_total` over `mtp_drafted_total` across the invocation. `rule5.md` has every cell's runs and
verdicts; there U is the branch unset or blank (`affine4`) and X is the branch with `bf16`. Here the branch unset is
the mean of its two runs against the mean of main's two:

| cells | unset (`affine4`) against main | tokens a round, main to unset |
|---|---|---|
| greedy, code, 1/2/4 streams | +15.9%, +10.5%, +9.3% | 5.22 to 5.45, 4.83 to 4.83, 4.83 to 4.83 |
| greedy, chat, 1/2/4 streams | +9.5%, +8.8%, +8.0% | 2.81 to 2.84, 2.51 to 2.54, 2.51 to 2.54 |
| greedy, one stream (`bench_openai`), code and chat | +16.5%, +9.6% | 5.22 to 5.45, 2.81 to 2.84 |
| temperature 1.0, 9 seeds, one stream (`bench_openai`), code and chat | +12.5%, +7.0% | 4.06 to 4.13, 2.56 to 2.48 |
| temperature 1.0, seeds 1234, 2234 and 3234, code, 1/2/4 streams | +5.9% to +12.2% (9 cells) | -0.14 to +0.10 |
| temperature 1.0, seeds 1234, 2234 and 3234, chat, 1/2/4 streams | +3.7% to +8.2% (9 cells) | -0.11 to +0.01 |

With `bf16` the branch is level with main in every cell (-0.5% to +1.8%), with main's tokens a round in every cell and
main's server acceptance in every invocation.

Server acceptance, main to unset: greedy 0.624 to 0.629 (`bench_openai`) and 0.679 to 0.697 (`bench_concurrent`);
temperature 1.0, 0.691 to 0.685 (`bench_openai`, 9 seeds), 0.722 to 0.715 (seed 1234), 0.700 to 0.678 (seed 2234) and
0.688 to 0.696 (seed 3234). The greedy code cells at 2 and 4 streams keep main's tokens a round (4.83) and run 10.5% and
9.3% faster, so that gain is the cheaper draft pass, not acceptance.

## The noise rule

A cell is slower when both branch runs are under both main runs and the mean gap is over 1%; it gets one rerun, and a
repeat is a loss.

- Unset and blank against main: 0 of 28 cells slower. In all 28 both branch runs are above both main runs, by 3.7% to
  16.5% on the mean.
- `bf16` against main: 0 of 28 cells slower; all 28 level.

## Memory

The server process's device memory (`nvidia-smi --query-compute-apps`), after load and after the benches: main 75,305
and 76,578 MiB (76,582 for M2), `bf16` the same, unset and blank 75,038 and 76,311 to 76,313 MiB. The 267 MiB is the
draft rows: 79,591 x 2,560 weights at 0.625 bytes each against 2.

## Cold prompts

A cold 8k and a cold 32k prompt (new text each rep, 2 cached tokens), three reps an arm, thinking off: medians 4.01 to
4.06 s at 8k and 16.07 to 16.19 s at 32k across the six arms. The draft head is not in prompt processing.

## Off, on and malformed values

- U1 (unset) and U2 (the container's environment holds `TF_FLASHNEXT_DRAFT_HEAD=` empty) take the same rounds: their
  server acceptance is the same in every invocation.
- `TF_FLASHNEXT_DRAFT_HEAD=BF16` stops the branch's start before any load: `TF_FLASHNEXT_DRAFT_HEAD="BF16": expected
  bf16 or affine4 (unset or blank keeps affine4, a bf16 language head's draft rows as 4-bit groups of 32; bf16 keeps
  the rows as stored)`.
- On main, `TF_FLASHNEXT_DRAFT_HEAD=` (blank) stops the start when the head loads, with `BadDraftHead`.

## Exactness

`exactness.md`: drafted against `"draft": false`, resumed against fresh and concurrent against solo, on main, the
branch unset and the branch with `bf16`, each with the prompt cache on and off. Every equality holds in every arm, and
the branch's replies equal main's.

## Not established

GB10 only, one to four streams. Eight streams on this main: `--parallel 8` did not start at 131,072 tokens with bf16 KV;
the run on #1870 covered 8 streams at 16,384 tokens on main `964f98b0b`. A checkpoint whose language head is affine
keeps its stored rows and was not run. Metal and HIP do not read this setting.
