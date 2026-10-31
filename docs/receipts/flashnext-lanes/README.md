# Flash Next attention and DeltaNet in one launch

`TF_FLASHNEXT_ATTN_LANES=1` and `TF_FLASHNEXT_GDN_LANES=1` run a shared decode round's kernels in one launch, off by
default. At short contexts they are 2 to 6% quicker from four streams up and tie at one. At long contexts this run
cannot separate them from prompt-fill noise, so it supports opt-in and not a default.

## Source and environment

- Baseline: main at `94b90c9db6`, one NVIDIA GB10 (sm_121), driver 580.159.03, kernel 6.17.0-1026-nvidia, aarch64.
- `nvcr.io/nvidia/pytorch:26.07-py3`, `sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c`; nvcc 13.3, V13.3.73, CUDA 13.3.
- Checkpoint `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP`, revision `2b170fa6309d5d1ee380b35636075fac7945f286`.
- `tools/bench_concurrent.py`, fresh server per run, alternating builds over two cycles. `cells.csv` has every cell.

Short: `--context 16384 --prompt-cache-gib 0 --parallel 16`, `--tokens 256 --reps 3`, `--temperatures 0,1.0`, levels 1, 4, 8,
16, `--alone --serial`. Long: `--context 45056 --prompt-cache-gib 48`, 16 distinct prompts of about 30k tokens,
`--tokens 128 --reps 1`, same levels; `--alone --serial` in cycle 1 only.

## Tokens

4,616 comparisons equal, 0 unequal, 0 failed requests, across the short and long runs of every build: each concurrent
reply against its run alone, and each alone reply against `"draft": false`, greedy and sampled. Cycle 1 replies match main's in
all 48 short cells. `tools/bench_cache.py`, 2,571-token prompt, four streams: every resumed reply equals the fresh one. Its
`--require-reuse` reports unqualified on main and on every branch, because the first repeat is cold on all of them.

## Speed, short contexts

End-to-end throughput, mean of two cycles, against main. Main's two cycles differ by 0.8% at most.

| Prompt | Temp | Streams | main tok/s | attention | attention + DeltaNet |
|---|---|---|---|---|---|
| code | 0 | 1 | 79.1 | -0.1% | -0.1% |
| code | 0 | 4 | 232.2 | +1.5% | +2.8% |
| code | 0 | 8 | 341.0 | +2.3% | +4.9% |
| code | 0 | 16 | 321.3 | +2.4% | +5.6% |
| code | 1 | 1 | 76.5 | +0.1% | +0.2% |
| code | 1 | 4 | 128.3 | +0.8% | +2.0% |
| code | 1 | 8 | 169.6 | +1.2% | +2.7% |
| code | 1 | 16 | 207.6 | +1.7% | +3.5% |
| chat | 0 | 1 | 65.8 | -0.1% | +0.1% |
| chat | 0 | 4 | 197.9 | +1.5% | +3.7% |
| chat | 0 | 8 | 289.5 | +2.3% | +5.2% |
| chat | 0 | 16 | 275.9 | +2.7% | +5.8% |
| chat | 1 | 1 | 57.3 | -1.5% | -0.0% |
| chat | 1 | 4 | 110.6 | +0.9% | +2.3% |
| chat | 1 | 8 | 142.8 | +1.3% | +3.1% |
| chat | 1 | 16 | 186.7 | +1.8% | +4.3% |

The -1.5% is one run: -3.0% and +0.0% in its two cycles. One stream takes today's path.

## Speed, long contexts

End-to-end tok/s, cycle 1 / cycle 2. `main` and `read-out` run the same attention and DeltaNet code, so together they show the
noise. The first prompt was already cached in cycle 2, so the one-stream cells are not comparable between cycles.

| Temp | Streams | main | read-out | attention | attention + DeltaNet |
|---|---|---|---|---|---|
| 0 | 4 | 8.73 / 8.05 | 8.72 / 7.66 | 7.46 / 8.08 | 7.05 / 8.09 |
| 0 | 8 | 7.75 / 7.06 | 7.76 / 6.47 | 6.75 / 6.81 | 6.58 / 7.48 |
| 0 | 16 | 7.18 / 6.79 | 7.16 / 7.25 | 6.76 / 6.82 | 6.81 / 5.23 |
| 1 | 4 | 8.78 / 8.71 | 8.77 / 8.71 | 8.80 / 8.71 | 8.81 / 8.20 |
| 1 | 8 | 7.74 / 7.69 | 8.91 / 7.71 | 7.77 / 7.70 | 7.79 / 6.76 |
| 1 | 16 | 7.27 / 7.39 | 7.28 / 7.26 | 7.44 / 7.42 | 7.47 / 5.40 |

These cells are mostly prompt fill: a 16-stream cell waits about 270 s for its first token and decodes for about 10 s.
The prompt cache held three states here and evicted during the runs, so the fill varies with what it kept.

Against main and read-out's range, attention falls outside in two cells: 7.46 below 7.66, and 6.76 below 6.79.
Attention + DeltaNet falls outside at 7.05 at temp 0, 4 streams, and in all of its cycle 2 sampled cells from 4 streams up and
at temp 0, 16 streams: 8.20, 6.76, 5.40 and 5.23. Those 16-stream cells have a slower first token (365 to 377 s against 265
to 290 s) and slower decode. Cycle 1 of the same build shows neither, and the cause is not established.

## Not established

- A long-context speed figure. A run with the prompt cache off would remove most of this noise; it has not been done.
- Whether the attention + DeltaNet cycle 2 cells are noise or a real cost. Until that is settled neither switch is a default.
- HIP and Metal were not run. They do not use this path.
