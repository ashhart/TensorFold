# Flash Next read-out cut-over at 128 rows

`TF_FLASHNEXT_READOUT_ROWS=128`, off by default, moves the HC read-out's switch to the prompt tiles from 64 to 128 rows. That adds
up to 7% throughput at 16 streams on a GB10. Any cut from 96 to 160 serves the same, so 128 is a value inside a flat range and
not a sharp optimum. These runs set the value at build time. On the same build with the switch unset and at 128, 16 streams,
`--reps 3`, a check gave +7.1%, +3.7%, +2.1% and 0.0%, the same order as below, and replies at 1, 4 and 16 streams were equal.

## Source and environment

- Baseline: main at `94b90c9db6`, one NVIDIA GB10 (sm_121), driver 580.159.03, kernel 6.17.0-1026-nvidia, aarch64.
- `nvcr.io/nvidia/pytorch:26.07-py3`, `sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c`; nvcc 13.3, V13.3.73, CUDA 13.3.
- Checkpoint `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP`, revision `2b170fa6309d5d1ee380b35636075fac7945f286`.
- `tools/bench_concurrent.py`, fresh server per run, alternating builds over two cycles. `cells.csv` has every cell.

The cut-over runs use `--context 16384 --prompt-cache-gib 0 --parallel 16`, `--tokens 256 --reps 3`, `--temperatures 0,1.0`, 16
streams, four builds differing only in `readout_tile_rows`.

## The cut-over, served at 16 streams

End-to-end tok/s, mean of two cycles. 64 is main; 96 and 160 are the cuts either side of this change.

| Prompt | Temp | 64 | 96 | 128 | 160 |
|---|---|---|---|---|---|
| code | 0 | 323.2 | 346.1 | 346.4 | 345.1 |
| chat | 0 | 277.0 | 287.4 | 287.7 | 287.8 |
| code | 1 | 206.4 | 213.5 | 213.5 | 213.4 |
| chat | 1 | 187.3 | 187.0 | 187.8 | 187.6 |

Cycles differ by at most 1.2 tok/s, except 64 at code, temp 1, by 3.2.

This compares cuts, not cells on either side of 128 rows. The tie from 96 to 160 suggests these 16-stream rounds stay below 96
rows, so those three columns take one path and the 64 column takes the other, on the same rounds. No served round at or above
128 rows was measured here, and Flash Next stops at 16 lanes. That side is unchanged: it takes the tiles under either value.

## Main against this branch

Short contexts, 1 to 16 streams, same run as #1739's receipt: 16 streams gain +7.4%, +3.9%, +2.5% and +0.2%, and every other
cell is within 0.5% of main. All token comparisons are equal. Long contexts show no difference beyond the noise in that receipt.

## The timing test disagrees, and the served runs decide

`tf-cuda-test flashnext-readout-cut` times the two paths directly, alternated, on one set of weights. There the tiles are about
twice as fast at every size from 32 to 256 rows: 1.60 ms against 4.21 ms at 128 rows. Served rounds show the opposite between 64
and 127 rows. That test repeats a read-out on warm weights while a served round reads each layer's in turn, which may be
the difference; this is not established. The 29.6 ms against 11.4 ms in the first version of #1741 came from an older
v1.1.7-based build's profile and could not be reproduced on main, so they are withdrawn.

## Not established

- Prompt chunks of 64 to 127 rows now take the row kernels too, and were not timed separately. First-token times at the short
  prompts here are within 2.3% of main, but those prompts are too short to reach that path.
- HIP and Metal do not use this path and were not run.
