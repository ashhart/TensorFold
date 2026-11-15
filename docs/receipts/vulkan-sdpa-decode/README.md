# Tiled decode attention, Apple M2 Max (14-inch) and M1 (13-inch)

Qwen3.5-2B 4-bit, 128 greedy tokens per run with the token read back each step (`TF_VK_DECODE_BENCH=128`), after the prompt prefill and 8 warm-up tokens. Base = `38458a94d` (tiled prompt SDPA and the k-split decode matmul, one-row decode attention unchanged). Head = `b8b634101` (one-row decode over 512 or more keys takes the tiled path). Clean-tree builds of those commits, run alternately, three rounds, one process per run, no GPU timeouts. Linux 7.3.0-2-20261110.1-lab-ARCH, Mesa honeykrisp.

Median ms per token. A few result lines in the raw files are broken by the GPU-timeout counter print that shares the line (`GPU-TIMEOUTS ...`); those runs are left out of the medians, so a cell may have two samples instead of three.

| Machine | Case | Samples base / head | Base ms/token | Head ms/token | Tokens/s, base to head | Change |
|---|---|---|---:|---:|---|---:|
| 14-inch M2 Max | 24-token prompt (24 keys) | 3 / 3 | 10.55 | 10.78 | 94.7 to 92.8 | within noise (head spread 5.0%) |
| 14-inch M2 Max | 1,024-token prompt | 3 / 3 | 12.96 | 11.53 | 77.2 to 86.7 | 12% faster (head spread 7.1%) |
| 14-inch M2 Max | 6,173-token prompt | 2 / 3 | 23.86 | 17.82 | 41.9 to 56.1 | 34% faster |
| 13-inch M1 | 24-token prompt | 3 / 3 | 24.67 | 24.71 | 40.5 to 40.5 | equal |
| 13-inch M1 | 1,024-token prompt | 3 / 3 | 26.84 | 26.64 | 37.3 to 37.5 | 1% faster |
| 13-inch M1 | 6,173-token prompt | 2 / 3 | 37.80 | 36.23 | 26.5 to 27.6 | 4% faster |

The 24-token case stays on the old path (24 keys is below 512), so it is a control: the head must equal the base there, and it does.

The greedy token hash is equal between base and head in every run (`fc93d24775b1d909`, `3efb3759184db87e`, `fc0ededa74f07707`). The tiled path keeps the decode kernel's order of sums.

Attention contract (`tf-attention-contract`, head build) on both machines: 18 of 18 cases pass, 0 GPU timeouts.

Why: the one-row decode kernel dispatches one workgroup per query head (8) and each walks every key. At 6,173 keys that is 8 workgroups doing all the work. The tiled path runs scores, row softmax and P times V as three dispatches over an f32 score scratch, spread over many workgroups.

The per-kernel time split that the KTIME timestamps showed for the tiled SDPA in #3431 was an attribution artifact, so no claim here rests on it. The numbers above are whole-run decode times.
