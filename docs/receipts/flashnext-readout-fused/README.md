# Flash Next fused BF16 read-out: main against the branch

`TF_FLASHNEXT_READOUT_FUSED=1`, off by default, runs a CUDA prompt pass's BF16 hyper-connection read-out, from 768
rows, in three launches instead of five: the down projection forms the normed rows in shared memory, the up projection
forms them again in its epilogue and mixes the four streams there, and the writeback before it skips its normalize.
Unset or blank is main's chain.

## Source and environment

- Main: `d6c4cce602eef4aff235625a5325bb617a9674ff`. Branch: this pull request's first commit on that main. The later
  commits add the kernel contract's entries and controls, tests, docs and pins, and merge main up to `643db94726`. Of
  main's 19 commits since `d6c4cce602`, one touches Flash Next (`6b9de69510`: a lone sampled stream's draws) and none
  touches the read-out, qmmf, the kernel contract or the kernel list; the arms below were not rerun on it.
- One NVIDIA GB10 (sm_121), Zig 0.17.0, CUDA 13.3.
- Checkpoint `nvidia/Qwen3.8-Flash-Next-FP4-Local`, revision `b9ce18a6887e18c089fdea8509a33e63429a3300`, every arm.
  Its read-out weights are BF16 (`input_mix_weight_down` [320, 10240], `input_mix_weight_up` [10240, 320]).
- `--parallel 4`, 131,072-token context, bf16 KV, a fresh server per arm, `--prompt-cache-gib 0` on every speed arm.
- Arms, in order: M1 (main), U1 (branch, unset), F1 (branch, `1`), M2, F2, U2 (branch, set blank).
- Each arm: cold prompts of 2k, 8k, 32k and 64k tokens at one stream (a salted prompt each, three reps, the median),
  `tools/bench_cache.py --require-cold` (its 2,571-token prompt), then `tools/bench_concurrent.py --alone --serial`
  (greedy, code and chat, 1, 2 and 4 streams, 256 tokens, 3 reps), `/metrics` around it.

## Cold prompts

Seconds to the first token, the median of three salted prompts; the noise rule's verdict for the branch set to `1`:

| prompt | M1 | M2 | U1 | U2 | F1 | F2 | unset against main | `1` against main |
|---|---|---|---|---|---|---|---|---|
| 2k | 1.06 | 1.08 | 1.06 | 1.06 | 1.04 | 1.03 | -0.8%, level | -3.0%, faster |
| 8k | 4.03 | 4.00 | 3.98 | 3.99 | 3.89 | 3.86 | -0.8%, level | -3.5%, faster |
| 32k | 16.14 | 16.05 | 16.08 | 16.05 | 15.51 | 15.48 | -0.2%, level | -3.7%, faster |
| 64k | 33.19 | 33.05 | 33.07 | 33.05 | 31.89 | 31.95 | -0.2%, level | -3.6%, faster |
| `bench_cache.py --require-cold`, 2,571 tokens | 1.256 | 1.229 | 1.223 | 1.226 | 1.182 | 1.182 | -1.4%, faster | -4.9%, faster |

Every first request reported 0 cached tokens. The unset arms take main's code path, so their -1.4% on the
`bench_cache.py` cell is that cell's own spread between boots.

## Decode speed and tokens a round

Aggregate tok/s, `tools/bench_concurrent.py`; tokens a round is the tool's tokens a streamed chunk (a round sends its
accepted drafts and one verified token):

| prompt, streams | M1 | M2 | U1 | U2 | F1 | F2 | unset | `1` | tokens a round, every arm |
|---|---|---|---|---|---|---|---|---|---|
| code, 1 | 71.6 | 72.3 | 72.1 | 72.1 | 72.2 | 72.0 | +0.2%, level | +0.2%, level | 5.224 |
| code, 2 | 137.3 | 137.3 | 136.5 | 136.6 | 137.2 | 136.9 | -0.5%, level | -0.2%, level | 4.83 |
| code, 4 | 241.2 | 240.5 | 241.4 | 240.6 | 240.8 | 240.5 | +0.0%, level | -0.1%, level | 4.83 |
| chat, 1 | 46.3 | 46.4 | 46.3 | 46.3 | 46.3 | 46.2 | -0.1%, level | -0.2%, level | 2.813 |
| chat, 2 | 85.3 | 85.5 | 85.2 | 85.4 | 85.6 | 85.2 | -0.1%, level | +0.0%, level | 2.51 |
| chat, 4 | 152.2 | 152.0 | 151.9 | 151.7 | 151.7 | 151.7 | -0.2%, level | -0.3%, level | 2.51 |

The server's `mtp_accepted_total` over `mtp_drafted_total` across this bench: 7,934 of 11,682 in M1 and in F1.

## The noise rule

A cell is slower when both branch runs are under both main runs (for a time: over) and the mean gap is over 1%; it gets
one rerun, and a repeat is a loss. Unset and blank against main: 0 of 11 cells slower. `1` against main: 0 of 11 cells
slower, the five prompt cells faster. No cell needed a rerun.

## Memory

The read-out reuses the pass's existing scratch (`hc_part`, `act`, `mixed`, `xsm`); the normed rows' buffer is still
allocated. Every arm logged the same pool: `4 streams at once, 4.11 GiB each at a 131072-token window, of 28.0 GiB left
after a 9.0 GiB reserve`. The container's resident memory after each arm: 4.263 to 4.280 GiB in all six.

## The kernels against the chain

`tf-cuda-test flashnext-readout-fused` on the branch's image compares every output byte (the down's fp32 rows, `act`,
its sums, the gates, the mixed rows and their sums) of the fused path against the five launches at 64, 65, 193, 255,
256, 767, 768, 769, 1,000, 1,024, 1,536, 2,053 and 3,072 rows, with each down tile the engine can pick: all equal. Its
times, the medians of 9 x 10 launches, alternated:

| rows | five launches (us) | fused (us) |
|---|---|---|
| 256 | 234.1 | 304.4 |
| 512 | 407.0 | 425.1 |
| 768 | 642.8 | 475.3 |
| 1,024 | 833.7 | 547.9 |
| 1,536 | 1,338.0 | 662.2 |
| 2,048 | 1,765.3 | 1,060.8 |
| 3,072 | 2,822.4 | 1,299.2 |

The fused path is slower at 256 rows and about level at 512, which is why the engine takes it from 768 rows. In a
served pass the writeback already carries the normalize (`TF_FLASHNEXT_HC_FUSE`), so a pass saves less than this
table's difference.

## Kernel contract

`tf-cuda-test kernel-contract fn_hcn` on the GB10, in an image of this branch at `25dec090f8` built with nvcc 13.3,
exit status 0:

    (filtered) SUMMARY 4 kernels: 4 pass gates 1-4, 0 fail, 0 known findings; 2 controls: 2 red as planted, 0 missed; 0 limit problems; gates 1908 ms, limits 0 ms

- `fn_hcn_down2`, `fn_hcn_down3`, `fn_hcn_down4` and `fn_hcn_up3` pass gates 1 to 3: every output byte equals the
  launches each replaces, planted non-finite rows included, at 768, 769, 1,000, 1,536, 2,053 and 3,072 rows, and five
  runs are byte-identical. Gate 4 does not apply, as they declare no sums of their own.
- The two controls go red at gate 3 as planted: rounding the normed rows toward zero differs from byte 0, and gating
  the up rows before their bf16 rounding differs from byte 2.
- Gate 5, live on sm_121: each kernel's registers and static shared bytes equal its record (152, 189, 206 and 127
  registers; 0 static shared bytes), within the driver's thread and shared-memory limits.
- `zig build test-kernel-contract`, with no device: 205 of 205.

## Off, on and a malformed value

- U1 (unset) and U2 (`TF_FLASHNEXT_READOUT_FUSED=` empty) take main's chain, and every reply hash their bench files
  record equals main's (`exactness.md`).
- `TF_FLASHNEXT_READOUT_FUSED=2` stops the start: `TF_FLASHNEXT_READOUT_FUSED="2": expected 0 or 1 (default 0)`.

## Exactness

`exactness.md`: drafted against `"draft": false`, resumed against fresh and concurrent against solo, on main, the
branch unset and the branch set to `1`, each with the prompt cache on and off, plus three long prompts (about 1.5k, 5k
and 15k tokens, so every pass of 768 rows or more takes the fused tiles) at 1 and 4 streams, and the long prompts again
with prompt passes of 768 and 3,072 rows. Every equality holds in every arm, and the branch's replies equal main's.

## Not established

GB10 and this checkpoint only. Metal and HIP refusing the setting, and a CUDA checkpoint whose read-out is MLX affine
refusing it, are covered by a host test, not run on that hardware or on such a checkpoint. The kernel contract's
records for the four kernels are nvcc 13.3's for sm_80 to sm_121; nvcc 13.0 and 13.2 were not available to compile them.
