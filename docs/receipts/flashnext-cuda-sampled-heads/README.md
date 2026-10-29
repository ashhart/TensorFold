# Flash Next CUDA: sampled streams in the batched MTP head pass

A shared round's streams with a temperature above zero were left out of the batched MTP head pass and drafted alone,
one head forward per draft token (the temperature skip in `cuda_lanes_draft.zig`'s `batched`). With this change they
join the pass: each step, after the shared best-column download, a sampled lane's score row is downloaded and its
draft drawn on the host under the lane's own keyed rule at the drafted position, the draw its lone `pick` makes.
Greedy lanes keep their best column. `TF_FLASHNEXT_SAMPLED_HEADS=0` restores the previous behaviour.

## Qualified scope

TensorFold 1.1.8 (`03d08f1ef5e385f6ecccb49d54e31b44e75ed422`) plus this change (the commit in `commit.txt`), built
from source with Zig 0.17.0 and nvcc 13.3 (`zig build -Doptimize=fast fatbins native install`) in the
`nvcr.io/nvidia/pytorch:26.07-py3` image. One DGX Spark (GB10, sm_121, 128 GB unified memory), kernel driver
580.178.04, CUDA 13.3 forward-compatibility driver 610.43.02. Checkpoint `nvidia/Qwen3.8-Flash-Next-FP4-Local` at
`b9ce18a6887e18c089fdea8509a33e63429a3300`.

Server: `tensorfold-native serve CHECKPOINT --context 131072 --parallel 5 --kv-cache bf16 --max-tokens 32768
--temperature 1.0 --top-p 0.95 --top-k 20 --thinking`, `TENSORFOLD_MEMORY_RESERVE_GIB=9`, no vision, no other
`TF_FLASHNEXT_*` variables. The server reported: model 73.28 GiB; 5 streams at once, 4.05 GiB each at a
131072-token window, of 22.4 GiB left after a 9.0 GiB reserve; a 2.1 GiB prompt cache; own glue kernels.

## Correctness

- `tf-cuda-test flashnext-mtp-parity CHECKPOINT zig/tests/cuda/flashnext/lr7/mid.ids` ([parity.txt](parity.txt)):
  18 shapes pass. The six greedy shapes of 1.1.8 (confidence 0 and 2), and twelve new ones: every stream sampled
  (seed 19 + i, position 5 + 3i, temperature 1, top-k 20, top-p 0.95) and alternate greedy and sampled lanes, at
  confidence 0 and 0.5, over {1,1}, {16,16,16,16} and {1,1,16,16,16,16} rows with four-draft chains. In every shape
  the shared pass's logits equal each lone pass's byte for byte, and the chains' lengths, proposals and probabilities
  are equal; the planted broken-gather control is caught.
- `zig build test-cuda-host`: all host tests pass, including the new `drawPick` test. `zig fmt --check` is clean.
  `tools/zig/lean_check.py`: 428 problems, the same 428 as 1.1.8 (inherited; none added). Every file stays within
  600 lines.
- Served: in every cell of all four runs below, `--alone` found every concurrent reply's token hash equal to its lone
  run's, and `--serial` found every lone run equal to `"draft": false` (0 unequal, 0 failed, 0 unmeasured over the
  64 cells).

## Speed

`tools/bench_concurrent.py http://127.0.0.1:30001 Qwen3.8-Flash-Next --levels 1,2,4,5 --tokens 256 --reps 3
--temperatures 1.0,0 --seed 1234 --alone --serial`, one binary, four runs alternated on one machine in one
session: A = `TF_FLASHNEXT_SAMPLED_HEADS=0` (the previous behaviour), B = the default. Aggregate tok/s, the
median of three repeats; the tool's full reports are [A1.json](A1.json), [A2.json](A2.json), [B1.json](B1.json) and
[B2.json](B2.json). The last column is the two B runs over the two A runs.

| cell | streams | A1 | A2 | B1 | B2 | B/A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code, T=1 | 1 | 66.5 | 67.7 | 67.8 | 67.0 | +0% |
| code, T=1 | 2 | 77.5 | 77.9 | 82.7 | 82.3 | +6% |
| code, T=1 | 4 | 99.9 | 99.9 | 119.2 | 118.4 | +19% |
| code, T=1 | 5 | 107.2 | 107.5 | 132.8 | 132.7 | +24% |
| chat, T=1 | 1 | 41.8 | 42.4 | 42.5 | 42.4 | +1% |
| chat, T=1 | 2 | 60.9 | 61.3 | 65.4 | 65.3 | +7% |
| chat, T=1 | 4 | 82.1 | 82.2 | 94.6 | 94.6 | +15% |
| chat, T=1 | 5 | 87.0 | 87.0 | 103.4 | 103.3 | +19% |
| code, T=0 | 1 | 66.8 | 67.6 | 67.8 | 67.2 | +0% |
| code, T=0 | 2 | 123.5 | 124.3 | 124.5 | 124.4 | +0% |
| code, T=0 | 4 | 210.6 | 211.6 | 212.5 | 210.4 | +0% |
| code, T=0 | 5 | 241.8 | 243.0 | 243.3 | 241.7 | +0% |
| chat, T=0 | 1 | 42.5 | 42.7 | 42.8 | 42.6 | +0% |
| chat, T=0 | 2 | 80.4 | 81.3 | 81.3 | 81.0 | +0% |
| chat, T=0 | 4 | 137.9 | 140.8 | 141.8 | 141.0 | +1% |
| chat, T=0 | 5 | 160.8 | 164.5 | 164.7 | 164.1 | +1% |

Sampled requests at 4 and 5 streams decode 15% to 24% faster in aggregate, 6% to 7% at 2 streams, and within noise
at 1 stream; greedy cells and time to first token are unchanged within noise (at 5 streams, 249 to 289 ms in every
run). The two runs of each side agree within 1%.

What remains of the gap between sampled and greedy cells is the host-side sampling work that is still done a lane
at a time: the score-row download and top-k scan for each draft (79,591 columns) and for each verify row
(248,320 columns). Moving those candidates to the device and waiting once a window is separate work.

## Memory

Available host memory did not fall below 22.6 GiB during the window. The server's admission and reserve are
unchanged by this change: a sampled lane reuses its own host row buffer, and no device allocation is added.
