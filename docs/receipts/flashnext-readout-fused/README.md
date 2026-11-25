# Flash Next fused BF16 read-out: main against the branch

`TF_FLASHNEXT_READOUT_FUSED=1`, off by default, runs a CUDA prompt pass's BF16 hyper-connection read-out, from 768
rows, in three launches instead of five: the down projection forms the normed rows in shared memory, the up projection
forms them again in its epilogue and mixes the four streams there, and the writeback before it skips its normalize.
Unset or blank is main's chain.

This receipt was recorded while the setting was off by default. The default is now on; the unset and blank arms below are
the chain only at the commits they name, and their measured numbers are kept as recorded.

## Source and environment

- Main: `41a72fd2578dfbdd73b25663da78d5fc0047bd95`. Branch: this pull request at
  `ee13555c8ef095aaeb9da4786b62af1cdf405dbb`, its commits merged with that main; the commits after it change only this
  receipt, the changelog line and the docs.
- Each image built the same way from its own commit: the recipe's Zig base image with `tensorfold-native`, its tools
  and the fatbins, nvcc 13.3.
- One NVIDIA GB10 (sm_121), Zig 0.17.0, CUDA 13.3.
- Checkpoint `nvidia/Qwen3.8-Flash-Next-FP4-Local`, revision `b9ce18a6887e18c089fdea8509a33e63429a3300`, every arm.
  Its read-out weights are BF16 (`input_mix_weight_down` [320, 10240], `input_mix_weight_up` [10240, 320]).
- `--parallel 4`, 131,072-token context, bf16 KV, a fresh server per arm, `--prompt-cache-gib 0` on every speed arm.
  A discarded boot of main runs first, so the weights are paged in before M1.
- Arms, in order: M1 (main), U1 (branch, unset), F1 (branch, `1`), M2, F2, U2 (branch, set blank).
- Each arm: cold prompts of 2k, 8k, 32k and 64k tokens at one stream (a salted prompt each, three reps, the median),
  `tools/bench_cache.py --require-cold` (its 2,571-token prompt), `tools/bench_openai.py` (greedy, 3 reps), then
  `tools/bench_concurrent.py --alone --serial` greedy and at temperature 1.0 with `--seed 1234` (code and chat, 1, 2
  and 4 streams, 256 tokens, 3 reps), `/metrics` around each. `cells.csv` has every cell, `noise-rule.md` every
  verdict.

## Cold prompts

Seconds to the first token, the median of three salted prompts:

| prompt | M1 | M2 | U1 | U2 | F1 | F2 | unset against main | `1` against main |
|---|---|---|---|---|---|---|---|---|
| 2k | 1.046 | 1.078 | 1.064 | 1.070 | 1.054 | 1.031 | +0.5%, level | -1.8%, level |
| 8k | 4.009 | 3.992 | 3.984 | 3.996 | 3.865 | 3.854 | -0.3%, level | -3.5%, faster |
| 32k | 15.903 | 16.062 | 16.028 | 16.018 | 15.512 | 15.542 | +0.3%, level | -2.8%, faster |
| 64k | 32.795 | 33.108 | 33.115 | 32.959 | 31.866 | 32.024 | +0.3%, level | -3.1%, faster |
| `bench_cache.py --require-cold`, 2,571 tokens | 1.233 | 1.223 | 1.219 | 1.219 | 1.183 | 1.185 | -0.7%, level | -3.6%, faster |

Every first request reported 0 cached tokens. The 2k cell is level by the rule: F1's 1.054 s is above M1's 1.046 s.

## Decode speed and tokens a round

Aggregate tok/s; tokens a round is the tools' tokens a streamed chunk (a round sends its accepted drafts and one
verified token):

| invocation, prompt, streams | M1 | M2 | U1 | U2 | F1 | F2 | unset | `1` | tokens a round, every arm |
|---|---|---|---|---|---|---|---|---|---|
| `bench_openai`, code, 1 | 72.6 | 72.0 | 72.2 | 72.1 | 72.1 | 72.1 | -0.2%, level | -0.2%, level | 5.224 |
| `bench_openai`, chat, 1 | 46.6 | 46.4 | 46.5 | 46.4 | 46.4 | 46.4 | -0.1%, level | -0.2%, level | 2.813 |
| greedy, chat, 1 | 46.7 | 46.5 | 46.5 | 46.4 | 46.5 | 46.4 | -0.3%, level | -0.3%, level | 2.813 |
| greedy, chat, 2 | 85.7 | 85.5 | 85.6 | 85.6 | 85.7 | 85.6 | +0.0%, level | +0.1%, level | 2.510 |
| greedy, chat, 4 | 152.6 | 152.3 | 152.6 | 152.1 | 152.3 | 152.2 | -0.1%, level | -0.1%, level | 2.510 |
| greedy, code, 1 | 72.4 | 72.3 | 72.1 | 72.2 | 72.5 | 72.3 | -0.3%, level | +0.1%, level | 5.224 |
| greedy, code, 2 | 138.0 | 137.4 | 137.4 | 137.4 | 137.5 | 137.7 | -0.2%, level | -0.1%, level | 4.830 |
| greedy, code, 4 | 242.4 | 241.3 | 242.1 | 241.5 | 241.2 | 241.4 | -0.0%, level | -0.2%, level | 4.830 |
| temperature 1.0, chat, 1 | 44.6 | 44.4 | 44.5 | 44.4 | 44.4 | 44.5 | -0.1%, level | -0.1%, level | 2.393 |
| temperature 1.0, chat, 2 | 69.5 | 69.0 | 69.3 | 69.0 | 69.3 | 69.4 | -0.1%, level | +0.1%, level | 2.450 |
| temperature 1.0, chat, 4 | 102.9 | 102.7 | 103.0 | 102.7 | 102.9 | 102.6 | +0.0%, level | -0.0%, level | 2.456 |
| temperature 1.0, code, 1 | 70.3 | 69.9 | 70.0 | 69.8 | 69.9 | 70.2 | -0.3%, level | -0.1%, level | 4.571 |
| temperature 1.0, code, 2 | 88.5 | 88.3 | 88.3 | 88.1 | 88.4 | 88.2 | -0.2%, level | -0.1%, level | 3.765 |
| temperature 1.0, code, 4 | 129.2 | 128.5 | 128.8 | 128.7 | 128.9 | 128.7 | -0.1%, level | -0.0%, level | 3.835 |

The server's `mtp_accepted_total` over `mtp_drafted_total` is the same in all six arms for each invocation: 1,488 of
2,384 (`bench_openai`), 7,934 of 11,682 (greedy) and 8,607 of 11,917 (temperature 1.0).

## The noise rule

A cell is slower when both branch runs are under both main runs (for a time: over) and the mean gap is over 1%; it gets
one rerun, and a repeat is a loss. Unset and blank against main: 0 of 19 cells slower. `1` against main: 0 of 19 cells
slower, four faster (8k, 32k and 64k prompts, and `bench_cache.py`'s). No cell needed a rerun.

## Memory

The read-out reuses the pass's existing scratch (`hc_part`, `act`, `mixed`, `xsm`); the normed rows' buffer is still
allocated. Every arm logged the same pool: `4 streams at once, 4.11 GiB each at a 131072-token window, of 28.0 GiB left
after a 9.0 GiB reserve`. The container's resident memory after each arm: 4.320 to 4.346 GiB in all six.

## Kernel contract

`tf-cuda-test kernel-contract fn_hcn` on the GB10, in the branch's image, exit status 0:

    (filtered) SUMMARY 4 kernels: 4 pass gates 1-4, 0 fail, 0 known findings; 2 controls: 2 red as planted, 0 missed; 0 limit problems; gates 2066 ms, limits 0 ms

- `fn_hcn_down2`, `fn_hcn_down3`, `fn_hcn_down4` and `fn_hcn_up3` pass gates 1 to 3: every output byte equals the
  launches each replaces, planted non-finite rows included, at 768, 769, 1,000, 1,536, 2,053 and 3,072 rows, and five
  runs are byte-identical. The replaced launches run on the engine's lane, with qmmf's BF16 prompt tiles loaded as
  `TF_FLASHNEXT_PROMPT_BF16` leaves them. Gate 4 does not apply: the kernels declare no sums of their own.
- The two controls go red at gate 3 as planted: rounding the normed rows toward zero differs from byte 0, and gating
  the up rows before their bf16 rounding differs from byte 2.
- Gate 5, live on sm_121: each kernel's registers and static shared bytes equal its record (152, 189, 206 and 127
  registers; 0 static shared bytes), within the driver's thread and shared-memory limits.

## The kernels against the chain

`tf-cuda-test flashnext-readout-fused`, exit status 0. At 16 row counts from 64 to 3,072, the five launches run on the
plain lane and on the engine's lane, where qmmf's prompt tiles take the up projection from 512 rows and the down
projection from 1,025 rows on 48 SMs. The two give the same bytes, and the fused path gives them too, with each down
tile. Its times on the engine's lane, the medians of 9 x 10 launches, alternated:

| rows | five launches (us) | fused (us) |
|---|---|---|
| 256 | 224.3 | 251.7 |
| 512 | 346.9 | 325.1 |
| 768 | 590.5 | 406.2 |
| 1,024 | 749.2 | 504.9 |
| 1,536 | 1,091.7 | 667.0 |
| 2,048 | 1,424.0 | 975.2 |
| 3,072 | 2,196.1 | 1,315.5 |

The fused path is slower at 256 rows; the engine takes it from 768 rows. In a served pass the writeback already carries
the normalize (`TF_FLASHNEXT_HC_FUSE`), so a pass saves less than this table's difference.

## Off, on and a malformed value

- U1 (unset) and U2 (`TF_FLASHNEXT_READOUT_FUSED=` empty) take main's chain, and every reply hash their bench files
  record equals main's (`exactness.md`).
- `TF_FLASHNEXT_READOUT_FUSED=2` stops the start: `TF_FLASHNEXT_READOUT_FUSED="2": expected 0 or 1 (default 0)`.

## Exactness

`exactness.md`: drafted against `"draft": false`, resumed against fresh and concurrent against solo, on main, the
branch unset and the branch set to `1`, each with the prompt cache on and off, plus three long prompts (about 1.5k, 5k
and 15k tokens) at 1 and 4 streams, and the long prompts again with prompt passes of 2,048 and 3,072 rows. Every
equality holds in every arm, and the branch's replies equal main's.

With passes of 1,000 rows, main itself fails every request: the first, a 1,685-token prompt, ends `CudaFailed` after
`cuMemsetD8Async: CUDA_ERROR_INVALID_VALUE`, and the branch fails the same way, unset or set. Passes of 768 rows fail
the same way on main `d6c4cce602`. That is not this change.

## Not established

GB10 and this checkpoint only. Metal and HIP refusing the setting, and a CUDA checkpoint whose read-out is MLX affine
refusing it, are covered by a host test, not run on that hardware or on such a checkpoint. The kernel contract's
records for the four kernels are nvcc 13.3's for sm_80 to sm_121; nvcc 13.0 and 13.2 were not available to compile them.
