# Flash Next draft head: main against the branch

`TF_FLASHNEXT_DRAFT_HEAD=bf16|affine4` says how the MTP head holds a bf16 language head's rows for the draft vocabulary
(79,591 rows of 2,560). On main, unset is `bf16`, and a blank value stops the start (`BadDraftHead`). On the branch,
unset or blank is `affine4`: 4-bit codes with a bf16 scale and bias for each group of 32, fitted at load from the
group's minimum and maximum, as `TF_FLASHNEXT_DRAFT_HEAD=affine4` already does on main. `bf16` keeps main's rows.

Two windows, each with both arms built from the same upstream main:

- 1, 2 and 4 streams at a 131,072-token context: main `d6c4cce602eef4aff235625a5325bb617a9674ff` against the branch on
  it (`cells.csv`, `rule5.md`, `exactness.md`).
- 1, 4 and 8 streams at a 65,536-token context: main `41a72fd2578dfbdd73b25663da78d5fc0047bd95` against the branch on it
  (`cells-8.csv`, `rule5-8.md`). `--parallel 8` does not start at 131,072 tokens here; the engine stops with
  `--parallel 8 needs 32.86 GiB for its streams at a 131072-token window, and the CUDA memory budget leaves 27.99 GiB`.

## Source and environment

- One NVIDIA GB10 (sm_121), Zig 0.17.0, CUDA 13.3, Linux.
- Checkpoint `nvidia/Qwen3.8-Flash-Next-FP4-Local`, revision `b9ce18a6887e18c089fdea8509a33e63429a3300`, every arm. Its
  language head is bf16, the case the setting applies to.
- bf16 KV, prompt cache on, a fresh server per arm. Server flags `--kv-cache bf16 --max-tokens 32768 --temperature 1.0
  --top-p 0.95 --top-k 20`; the tools set each request's temperature.
- The serve recipe sets the same environment for every arm. Of the settings main reads: `TF_FLASHNEXT_KEEP_LATE=1` and
  `TF_FLASHNEXT_WARM_AHEAD=1` (main's defaults are unset and 0), `TENSORFOLD_MEMORY_RESERVE_GIB=9`, and
  `TF_FLASHNEXT_CONFIDENCE`, `TF_FLASHNEXT_DRAFTS`, `TF_FLASHNEXT_GDN_LANES`, `TF_FLASHNEXT_SAMPLED_HEADS` and
  `TF_FLASHNEXT_SHARED_SIDE` at main's defaults. The twelve other `TF_FLASHNEXT_*` names it sets have no reader on main.
- Arms, in order: M1 (main), U1 (branch, unset), X1 (branch, `bf16`), M2, X2, U2 (branch, set blank). No cell was
  slower in either window, so the noise rule's rerun did not run.
- Main's tools. First window: `tools/bench_openai.py` (one stream; greedy, 3 reps; temperature 1.0, 9 reps with seeds
  1234 to 1242; its median decode rate) and `tools/bench_concurrent.py --alone --serial` (1, 2 and 4 streams, 256
  tokens, 3 reps; greedy, and temperature 1.0 with `--seed` 1234, 2234 and 3234; aggregate tok/s). Second window:
  `tools/bench_concurrent.py --alone --serial` at 1, 4 and 8 streams, greedy and temperature 1.0 with `--seed` 1234.

## Decode speed and tokens a round

The CSV files have every cell of every arm. `tokens_per_round` is the tools' tokens a streamed chunk (a round sends its
accepted drafts and one verified token), `accepted_per_round` is that less one, and `server_acceptance` is the
server's `mtp_accepted_total` over `mtp_drafted_total` across the invocation. The `rule5` files have every cell's runs
and verdicts; there U is the branch unset or blank (`affine4`) and X is the branch with `bf16`. Below, the branch unset
is the mean of its two runs against the mean of main's two.

First window, main `d6c4cce602`, 131,072 tokens:

| cells | unset (`affine4`) against main | tokens a round, main to unset |
|---|---|---|
| greedy, code, 1/2/4 streams | +15.9%, +10.5%, +9.3% | 5.22 to 5.45, 4.83 to 4.83, 4.83 to 4.83 |
| greedy, chat, 1/2/4 streams | +9.5%, +8.8%, +8.0% | 2.81 to 2.84, 2.51 to 2.54, 2.51 to 2.54 |
| greedy, one stream (`bench_openai`), code and chat | +16.5%, +9.6% | 5.22 to 5.45, 2.81 to 2.84 |
| temperature 1.0, 9 seeds, one stream (`bench_openai`), code and chat | +12.5%, +7.0% | 4.06 to 4.13, 2.56 to 2.48 |
| temperature 1.0, seeds 1234, 2234 and 3234, code, 1/2/4 streams | +5.9% to +12.2% (9 cells) | -0.14 to +0.10 |
| temperature 1.0, seeds 1234, 2234 and 3234, chat, 1/2/4 streams | +3.7% to +8.2% (9 cells) | -0.11 to +0.01 |

Second window, main `41a72fd257`, 65,536 tokens, `--parallel 8`:

| cells | unset (`affine4`) against main | tokens a round, main to unset |
|---|---|---|
| greedy, code, 1/4/8 streams | +16.4%, +9.0%, +8.4% | 5.22 to 5.45, 4.83 to 4.83, 4.49 to 4.49 |
| greedy, chat, 1/4/8 streams | +9.0%, +8.1%, +6.9% | 2.81 to 2.84, 2.51 to 2.54, 2.51 to 2.54 |
| temperature 1.0, seed 1234, code, 1/4/8 streams | +9.0%, +5.9%, +5.5% | 4.57 to 4.57, 3.83 to 3.81, 3.72 to 3.74 |
| temperature 1.0, seed 1234, chat, 1/4/8 streams | +6.2%, +5.6%, +4.4% | 2.39 to 2.35, 2.46 to 2.44, 2.43 to 2.41 |

With `bf16` the branch is level with main in every cell of both windows (-0.5% to +1.8%), with main's tokens a round in
every cell and main's server acceptance in every invocation.

Server acceptance, main to unset. First window: greedy 0.624 to 0.629 (`bench_openai`) and 0.679 to 0.697
(`bench_concurrent`); temperature 1.0, 0.691 to 0.685 (`bench_openai`, 9 seeds), 0.722 to 0.715 (seed 1234), 0.700 to
0.678 (seed 2234) and 0.688 to 0.696 (seed 3234). Second window: greedy 0.686 to 0.713, seed 1234 0.714 to 0.712. The
greedy code cells at 4 and 8 streams keep main's tokens a round and still run faster, so that gain is the cheaper draft
pass, not acceptance.

## The noise rule

A cell is slower when both branch runs are under both main runs and the mean gap is over 1%; it gets one rerun, and a
repeat is a loss.

- Unset and blank against main: 0 of 28 cells slower in the first window and 0 of 12 in the second. In all 40 both
  branch runs are above both main runs, by 3.7% to 16.5% on the mean.
- `bf16` against main: 0 of 40 cells slower; all 40 level.

## Memory

The server process's device memory (`nvidia-smi --query-compute-apps`), after load and after the benches. First window:
main 75,305 and 76,578 MiB (76,582 for M2), `bf16` the same, unset and blank 75,038 and 76,311 to 76,313 MiB. Second
window: main 75,289 and 76,536 to 76,548 MiB, `bf16` 75,289 and 76,546 to 76,558 MiB, unset and blank 75,022 and
76,271 to 76,281 MiB. The 267 MiB is the draft rows: 79,591 x 2,560 weights at 0.625 bytes each against 2.

## Cold prompts

First window: a cold 8k and a cold 32k prompt (new text each rep, 2 cached tokens), three reps an arm, thinking off:
medians 4.01 to 4.06 s at 8k and 16.07 to 16.19 s at 32k across the six arms. The draft head is not in prompt
processing.

## Off, on and malformed values

- U1 (unset) and U2 (the container's environment holds `TF_FLASHNEXT_DRAFT_HEAD=` empty) take the same rounds: their
  server acceptance is the same in every invocation.
- `TF_FLASHNEXT_DRAFT_HEAD=BF16` stops the branch's start before any load: `TF_FLASHNEXT_DRAFT_HEAD="BF16": expected
  bf16 or affine4 (unset or blank keeps affine4, a bf16 language head's draft rows as 4-bit groups of 32; bf16 keeps
  the rows as stored)`.
- On main, `TF_FLASHNEXT_DRAFT_HEAD=` (blank) stops the start when the head loads, with `BadDraftHead`.

## Exactness

`exactness.md` (first window): drafted against `"draft": false`, resumed against fresh and concurrent against solo, on
main, the branch unset and the branch with `bf16`, each with the prompt cache on and off. Every equality holds in every
arm, and the branch's replies equal main's. Second window: `bench_concurrent.py`'s own comparison of each concurrent
reply with the same request alone finds 0 unequal across every request of every arm, at 1, 4 and 8 streams.

## Not established

GB10 only. Eight streams ran at 65,536 tokens, the largest of 131,072 and 65,536 that starts with the recipe's 9 GiB
reserve. A checkpoint whose language head is affine keeps its stored rows and was not run. Metal and HIP do not read
this setting.
