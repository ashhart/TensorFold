# Flash Next fused split-K form: main against the branch

`TF_FLASHNEXT_FUSED_SPLIT=0|1`. Unset, blank or `0` keeps main's kernels. `1` runs CUDA lane calls that split K over a
cluster and have 2,048 or more output columns on the fused form: one block a 64-column tile adds its K slices in the
cluster's slice order, so the outputs keep the cluster kernel's bits. The server logs
`flashnext: TF_FLASHNEXT_FUSED_SPLIT=1: split-K lane calls of 2048 columns or more run on the fused form`.

## Source and environment

- Main: `41a72fd2578dfbdd73b25663da78d5fc0047bd95`, the base every speed, prompt and exactness arm below ran on.
- Branch: commit `4df765a1c806158f40bdc3a57c0036fe2048bb2d`, that main plus this pull request's four commits before
  this receipt (the change, the earlier receipt, the contract's launch records and the host-test step).
- Main has since moved to `964174ea05879e6dde1de1198a3aae3ef8b46a3b`, which the branch merges at `120123a6c4`. Its 29
  files are DeepSeek-V4.1 and Qwen3.8-27B on Metal, their records and their tests, and none is a file this change
  touches, so the files the arms measured are byte-identical at that merge and the arms describe it too. The five-SM
  limits run below was built from `120123a6c4`.
- Both images were built the same way from their own commit: `zig build -Dnvcc=/usr/local/cuda/bin/nvcc
  -Doptimize=fast fatbins install native` in `nvcr.io/nvidia/pytorch:26.07-py3` (nvcc 13.3, V13.3.73), Zig 0.17.0.
- One NVIDIA GB10 (sm_121), driver 580.178.04, Ubuntu 24.04.5 LTS, Linux 7.0.0-1019-nvidia-64k, 128 GB of unified
  memory (123 GiB as the OS reports it).
- Checkpoint `nvidia/Qwen3.8-Flash-Next-FP4-Local`, revision `b9ce18a6887e18c089fdea8509a33e63429a3300`, with its MTP
  drafter, in every arm.
- Decode boots: `tensorfold-native serve CHECKPOINT --context 32768 --parallel 8 --max-tokens 32768 --temperature 1.0
  --top-p 0.95 --top-k 20 --thinking --kv-cache bf16`, prompt cache on. Prompt boots: the same with `--context 131072
  --parallel 1 --prompt-cache-gib 0`. A fresh server for every boot.
- Every arm had the same environment but for `TF_FLASHNEXT_FUSED_SPLIT`. Of the settings main reads, two differ from
  main's defaults: `TF_FLASHNEXT_WARM_AHEAD=1` and `TENSORFOLD_MEMORY_RESERVE_GIB=9`. The others it reads were at main's
  values: `TF_FLASHNEXT_CONFIDENCE=-0.4`, `TF_FLASHNEXT_DRAFTS=6`, `TF_FLASHNEXT_GDN_LANES=1`,
  `TF_FLASHNEXT_KEEP_LATE=1` (what unset means on one GPU), `TF_FLASHNEXT_SAMPLED_HEADS=1`,
  `TF_FLASHNEXT_SHARED_SIDE=1`, `TENSORFOLD_VIDEO_TOKENS=16384` and `TENSORFOLD_CUDA_KERNELS=native`.
- Arms, in order: M1 (main), F1 (branch, `1`), U1 (branch, unset), M2, U2, F2. Each arm is a decode boot, then a
  prompt boot of the same image and setting.
- Main's tools at main: `tools/bench_openai.py` (one stream, greedy, 256 tokens, 3 reps, its median decode rate),
  `tools/bench_concurrent.py --alone --serial` (1, 2, 4, 6 and 8 streams, 256 tokens, 3 reps; greedy, and temperature
  1.0 with `--seed 1234`; aggregate tok/s) and `tools/bench_cache.py --require-cold`.

## Decode speed and tokens a round

`cells.csv` has every cell of every arm, `rule5.md` every cell's runs and verdict. Here each side is the mean of its
two runs against the mean of main's two:

| cells | `1` against main | unset against main | tokens a round |
|---|---|---|---|
| greedy, code, 1/2/4/6/8 streams | +0.3%, +0.6%, +0.8%, +1.7% (faster), +1.6% (faster) | -0.6%, -0.5%, -0.5%, -0.3%, -0.1% | 5.22, 4.83, 4.83, 4.49, 4.49 |
| greedy, chat, 1/2/4/6/8 streams | +0.3%, +0.6%, +1.1% (faster), +1.8% (faster), +1.8% (faster) | -0.5%, -0.1%, +0.1%, +0.1%, +0.3% | 2.81, 2.51, 2.51, 2.51, 2.51 |
| greedy, one stream (`bench_openai`), code and chat | +0.3%, +0.2% | -0.4%, -0.6% | 5.22, 2.81 |
| temperature 1.0, seed 1234, code, 1/2/4/6/8 streams | +1.0%, +1.1%, +1.1%, +1.3%, +1.8% (all faster) | +0.4%, +0.5%, +0.2%, -0.1%, +0.1% | 4.57, 3.77, 3.83, 3.72, 3.72 |
| temperature 1.0, seed 1234, chat, 1/2/4/6/8 streams | +1.5% (faster), +0.9%, +1.1% (faster), +1.1% (faster), +1.3% (faster) | +1.1% (faster), +0.9%, +0.9%, +0.6%, +0.6% | 2.39, 2.45, 2.46, 2.43, 2.43 |

Tokens a round are main's in every cell, on both branch arms.

## The noise rule

A cell is slower when both branch runs are under both main runs and the mean gap is over 1%; it gets one rerun, and a
repeat is a loss. A cell is faster when both branch runs are over both main runs and the mean gap is over 1%.

- `1` against main: 14 of 22 cells faster, 8 level (+0.2% to +0.9%), 0 slower, so nothing was rerun. The largest gains
  are at 6 and 8 streams: greedy chat 212.8 and 212.7 against 210.0 and 208.1 tok/s (+1.8%) and 260.5 and 260.3
  against 257.2 and 254.2 (+1.8%); greedy code 313.0 and 311.5 against 307.6 and 306.7 (+1.7%) and 378.1 and 377.4
  against 371.8 and 371.6 (+1.6%); temperature 1.0 code 160.9 and 160.8 against 159.7 and 157.9 (+1.3%) and 187.4 and
  187.0 against 184.8 and 182.8 (+1.8%).
- Unset against main, the same binary on main's kernels: 21 level (-0.6% to +0.9%), 0 slower, and 1 faster,
  temperature 1.0 chat at one stream (44.6 and 44.6 against 44.4 and 43.8, +1.1%). That cell shows how large a
  difference with no change behind it can read here: the `1` cells near 1% are at the edge of the noise.

## Prompt speed

A prompt pass takes the fused form only when it has fewer than 256 rows: from 256 rows a split-K call already adds its
slices in one block, the 64-row fused tile, and the prompt tile runs only on plans without a cluster. None of the
prompts below has a pass under 256 rows, so these runs show that the setting leaves long prompts alone; the short
prompts of the decode cells are the ones it reaches.

`tools/bench_cache.py --require-cold --prompt-file FILE --expected-prompt-tokens N --tokens 16 --repeats 1 --streams
1` on the prompt boots, which run prompts in passes of up to 8,192 rows: the prompt is
`zig/tests/cuda/flashnext/lr7/long12k.ids` repeated end to end and cut at N ids. With the prompt cache off, the first,
repeated and concurrent requests are all cold; every run qualified (prompt tokens N, cached 0, the later replies'
tokens equal to the first's). `prompt.csv` has every request. Time to first token in seconds:

| prompt tokens | M1 | F1 | U1 | M2 | U2 | F2 | `1` against main | unset against main |
|---|---|---|---|---|---|---|---|---|
| 2,048, median of 3 | 0.950 | 0.950 | 0.950 | 0.965 | 0.952 | 0.953 | -0.6%, level | -0.7%, level |
| 8,192, median of 3 | 3.435 | 3.428 | 3.427 | 3.469 | 3.429 | 3.438 | -0.6%, level | -0.7%, level |
| 32,768, median of 3 | 14.314 | 14.334 | 14.339 | 14.460 | 14.308 | 14.317 | -0.4%, level | -0.4%, level |
| 65,536, median of 3 | 29.387 | 29.409 | 29.430 | 29.746 | 29.402 | 29.448 | -0.5%, level | -0.5%, level |

On the first requests alone the 2,048 and 65,536 cells are level as well (-0.5% for `1`, -0.2% and -0.6% unset). At
8,192 and 32,768 the first requests read faster on both branch arms (-12.8% and -2.5% for `1`, -12.8% and -2.2%
unset), because M1's first requests at those lengths took 4.417 and 14.853 s against 3.429 and 14.313 s for its own
next ones; the unset arm runs main's kernels, so that difference is not the change.

`ttft.py` (cold prompts of 7,997 to 8,123 and 15,979 to 16,162 tokens behind a fresh salt, thinking off, three a size,
prompt cache on, on the decode boots, which run prompts in passes of up to 1,536 rows), medians: 8k 4.029 and 4.063 s
on main, 4.010 and 3.972 s with `1`, 4.017 and 3.989 s unset; 16k 7.996 and 8.088 s, 8.005 and 7.962 s, 8.000 and
7.974 s. At 8k both branch arms read faster than main (-1.4% with `1`, -1.1% unset), at 16k both level (-0.7%).

## Drafts

One stream, each request with drafts and with `"draft": false` (`tools/bench_concurrent.py`'s own `stream()` and
prompts, 256 tokens, two of each), median decode tok/s with the setting `1`, F1 then F2: greedy code 73.0 and 72.5
drafted against 27.1 and 27.0 plain; greedy chat 46.8 and 46.6 against 27.1 and 27.1; temperature 1.0 code 70.3 and
70.2 against 27.0 and 27.0; temperature 1.0 chat 44.7 and 44.5 against 27.0 and 27.0. Every drafted reply equals its
plain one, in every arm. `prompt.csv` has every arm's figures.

## Memory

The server's own line: model 72.71 GiB on every decode boot and 76.16 GiB on every prompt boot, in all six arms. The
fused form reads the same weights and needs no partial-sum buffer.

## Off, on and a malformed value

- `1` logs the line above on F1 and F2 and their prompt boots; the main and unset arms log nothing about it.
- `TF_FLASHNEXT_FUSED_SPLIT=2` stops the start before any load: `TF_FLASHNEXT_FUSED_SPLIT="2": expected 0 or 1
  (default 0)`.
- Unset and blank (the container's environment holds `TF_FLASHNEXT_FUSED_SPLIT=` with no value) are covered by the
  exactness arms below; unset is also the U arms above.

## Exactness

`exactness.md`: drafted against `"draft": false`, resumed against fresh and concurrent against solo, on main and on the
branch with the setting `1`, unset and blank, each with the prompt cache on and off. Every equality holds in every arm,
and the branch's replies equal main's in every comparison: 32 of 32 concurrent and serial hashes, 8 of 8 golden, 8 of 8
drafted replies' token ids with the cache on and 8 of 8 with it off, and 7 of 7 first and 7 of 7 resumed turns.

## Kernel contract

`tf-cuda-test kernel-contract "fused form" "half tiles"` on the branch's image (`contract.txt`): `(filtered) SUMMARY 8
kernels: 8 pass gates 1-4, 0 fail, 0 known findings; 4 controls: 4 red as planted, 0 missed; 0 limit problems; gates
10632 ms, limits 0 ms`, exit 0. Each fused form is bit-equal to the cluster kernel's K slices at 16 shapes, MXFP8 and
BF16, bf16 and fp32 out; its control, the fused form at a 1.5x tensor scale posing as the cluster kernel, differs from
byte 0 at 1x2560x6144.

The eight launch records (`zig/src/core/contract/lane_fused_records.zig`) hold nvcc 13.0.88's registers, shared bytes
and bounds for sm_80, sm_86, sm_89, sm_120 and sm_121, read from a five-SM `qmmf` fatbin with the contract's own cubin
reader, as the commit that added them says.

`tf-cuda-test kernel-contract-limits` over a five-SM fatbin set (`zig build -Dnvcc=/usr/local/cuda/bin/nvcc
-Doptimize=fast -Dsm=80,86,89,120,121 fatbins kernel-contract-fatbins install`, nvcc 13.3.73, 149 images) on
`120123a6c4` (`limits.txt`) prints `INFO limits fatbins: images built by nvcc 13.3, checked against nvcc 13.3's
records`; 5,345 of its per-SM checks pass, 2,567 are below the chip's minimum, both planted controls go red on all five
SMs, and the run ends `FAIL kernel-contract-limits: 834 problems`, exit 1. Every problem is a registered launch whose
nvcc 13.0 record differs from what nvcc 13.3 compiles, an inherited record of another launch: in this image the
128x128 prompt tiles and the shared-down half-tile launches. None is one of this change's eight, which report `record
current` on all five SMs, and none is a 64-row fused instance, the form a row-tile-64 call of this setting takes.

## Not established

GB10 only. sm_80, sm_86 and sm_89 have no clusters, so their plans never take the fused form, and sm_120 was not run.
The limits run above re-derives this change's records on every SM it serves; the stale nvcc 13.0 records of other
launches are that run's 834 problems and are not this change's. Metal and HIP refusing `1` is covered by host tests,
not run on that hardware.
