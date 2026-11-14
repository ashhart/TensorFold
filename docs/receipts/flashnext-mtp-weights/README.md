# Flash Next MTP drafter weights: main against the branch

`TF_FLASHNEXT_MTP_WEIGHTS=checkpoint|mxfp8`. Unset or blank is `checkpoint`, main's load. `mxfp8` re-codes the drafter's
BF16 linears to MXFP8 as they load; on this checkpoint the server logs `re-coded 170 MiB of the drafter's BF16 linears`.

## Source and environment

- Main: `06bcb989a277c5300d40454e68020c75c780681f`. Branch: this pull request's commits on that main.
- One NVIDIA GB10 (sm_121), Zig 0.17.0, CUDA 13.3.
- Checkpoint `nvidia/Qwen3.8-Flash-Next-FP4-Local`, revision `b9ce18a6887e18c089fdea8509a33e63429a3300`, every arm.
- `--parallel 4`, 131,072-token context, bf16 KV, prompt cache on, a fresh server per arm.
- Arms, in order: M1 (main), U1 (branch, unset), X1 (branch, `mxfp8`), M2, X2, U2 (branch, set blank). The noise
  rule's rerun: M3 U3 X3 M4 X4 U4, on the one invocation that held a slower cell.
- Main's tools: `tools/bench_openai.py` (one stream; greedy, 3 reps; temperature 1.0, 9 reps with seeds 1234 to 1242;
  its median decode rate) and `tools/bench_concurrent.py --alone --serial` (1, 2 and 4 streams, 256 tokens, 3 reps;
  greedy, and temperature 1.0 with `--seed` 1234, 2234 and 3234; aggregate tok/s).

## Decode speed and tokens a round

`cells.csv` has every cell of every arm. `tokens_per_round` is the tools' tokens a streamed chunk (a round sends its
accepted drafts and one verified token), `accepted_per_round` is that less one, and `server_acceptance` is the
server's `mtp_accepted_total` over `mtp_drafted_total` across the invocation. `rule5.md` has every cell's runs and
verdicts. Here `mxfp8` is the mean of its two runs against the mean of main's two:

| cells | `mxfp8` against main | tokens a round, main to `mxfp8` |
|---|---|---|
| greedy, code, 1/2/4 streams | +2.7%, +3.9%, +3.2% | 5.22 to 5.22, 4.83 to 4.83, 4.83 to 4.83 |
| greedy, chat, 1/2/4 streams | +0.3% (level), +3.1%, +2.8% | 2.81 to 2.75, 2.51 to 2.54, 2.51 to 2.54 |
| greedy, one stream (`bench_openai`), code and chat | +2.8%, +0.5% (level) | 5.22 to 5.22, 2.81 to 2.75 |
| temperature 1.0, 9 seeds, one stream (`bench_openai`), code and chat | +5.3%, +4.9% | 4.06 to 3.94, 2.56 to 2.51 |
| temperature 1.0, seeds 2234 and 3234, code and chat, 1/2/4 streams | +2.2% to +3.6% (12 cells) | -0.12 to +0.07 |
| temperature 1.0, seed 1234, code, 1/2/4 streams | +3.2%, +1.4%, +2.1% | 4.57 to 4.41, 3.77 to 3.77, 3.83 to 3.88 |
| temperature 1.0, seed 1234, chat, 1/2/4 streams | -1.0% (slower), -0.6%, +0.7% | 2.39 to 2.31, 2.45 to 2.38, 2.46 to 2.42 |

Unset and blank are level with main in every cell (-0.3% to +2.1%), with main's tokens a round in every cell.

## The noise rule

A cell is slower when both branch runs are under both main runs and the mean gap is over 1%; it gets one rerun, and a
repeat is a loss.

- Unset and blank against main: 0 of 28 cells slower.
- `mxfp8` against main: 1 of 28 cells slower in the first pair, chat at one stream, temperature 1.0, seed 1234: 43.4 and
  43.4 against 43.6 and 44.1 tok/s (-1.0%). Its rerun: 43.5 and 43.3 against 44.2 and 44.1 (-1.7%), slower again, so
  this cell is a loss. In both pairs its tokens a round are 2.31 against main's 2.39 (server acceptance 0.710 against
  0.722): at this seed the re-coded drafter's proposals are accepted less often, which costs more than its shorter draft
  passes save. The same cell is +2.3% at seed 2234 and +2.9% at seed 3234, and chat over nine seeds at one stream is
  +4.9%.

## Cold prompts

`ttft.py`'s cold 8k and 32k prompts, three reps an arm, thinking off: medians 4.02 to 4.07 s at 8k and 16.07 to
16.37 s at 32k across the six arms. The drafter is not in prompt processing.

## Off, on and a malformed value

- U1 (unset) and U2 (the container's environment holds `TF_FLASHNEXT_MTP_WEIGHTS=` empty) log no re-coding, and every
  reply hash their bench files record equals main's (`exactness.md`).
- `mxfp8` logs `flashnext: TF_FLASHNEXT_MTP_WEIGHTS=mxfp8 re-coded 170 MiB of the drafter's BF16 linears`.
- `TF_FLASHNEXT_MTP_WEIGHTS=MXFP8` stops the start before any load: `TF_FLASHNEXT_MTP_WEIGHTS="MXFP8": expected
  checkpoint or mxfp8 (unset or blank keeps checkpoint, the rows as stored; mxfp8 re-codes the drafter's BF16 linears
  on CUDA)`.

## Exactness

`exactness.md`: drafted against `"draft": false`, resumed against fresh and concurrent against solo, on main, the
branch unset and the branch with `mxfp8`, each with the prompt cache on and off. Every equality holds in every arm, and
the branch's replies equal main's.

## Not established

GB10 only. Metal and HIP refusing `mxfp8`, and a CUDA drafter linear stored as anything but BF16 refusing it, are
covered by host tests, not run on that hardware or on such a checkpoint.
