# Flash Next fused split-K form: main against the branch

`TF_FLASHNEXT_FUSED_SPLIT=0|1`. Unset, blank or `0` keeps main's kernels. `1` runs CUDA lane calls that split K over a
cluster and have 2,048 or more output columns on the fused form: one block a 64-column tile adds its K slices in the
cluster's slice order, so the outputs keep the cluster kernel's bits. On this checkpoint the server logs
`flashnext: TF_FLASHNEXT_FUSED_SPLIT=1: split-K lane calls of 2048 columns or more run on the fused form`, and the
calls it takes are out_proj and o_proj (MXFP8 [2560, 6144], 8 K slices) and the drafter's 2,560-column BF16 projections.

## Source and environment

- Main: `2ba55ec7d32d0bec446bb012678550be7ef52895`. Branch: commit `dde470ca025a34b28e1a22420e94134355bfda03`, this
  pull request's code on that main (the later commits add the contract records, this receipt and the changelog line).
- One NVIDIA GB10 (sm_121), driver 580.178.04, Ubuntu 24.04.5, Zig 0.17.0, nvcc 13.3.
- Checkpoint `nvidia/Qwen3.8-Flash-Next-FP4-Local`, revision `b9ce18a6887e18c089fdea8509a33e63429a3300`, every arm,
  with its MTP drafter.
- `--parallel 8`, 32,768-token context, bf16 KV, prompt cache on, a fresh server per arm. Every arm ran the same
  launcher environment; of the settings main reads, two differ from main's defaults: `TF_FLASHNEXT_WARM_AHEAD=1` and
  `TF_FLASHNEXT_KEEP_LATE=1` (the others it sets, `DRAFTS=6`, `CONFIDENCE=-0.4`, `GDN_LANES=1`, `SAMPLED_HEADS=1` and
  `SHARED_SIDE=1`, are main's defaults).
- Arms, in order: M1 (main), F1 (branch, `1`), M2 (main), F2 (branch, `1`). Two arms of an unrelated image ran in the
  same window, after F1 and after M2; they are not part of this receipt.
- Main's tools at main: `tools/bench_openai.py` (one stream, greedy, 3 reps, its median decode rate) and
  `tools/bench_concurrent.py --alone --serial` (1, 2, 4, 6 and 8 streams, 256 tokens, 3 reps; greedy, and temperature
  1.0 with `--seed 1234`; aggregate tok/s).

## Decode speed and tokens a round

`cells.csv` has every cell of every arm, `rule5.md` every cell's runs and verdict. Here the branch is the mean of its
two runs against the mean of main's two:

| cells | `1` against main | tokens a round, main and branch |
|---|---|---|
| greedy, code, 1/2/4/6/8 streams | +0.6%, +0.4%, +1.2%, +1.7% (faster), +1.5% (faster) | 5.22, 4.83, 4.83, 4.49, 4.49 |
| greedy, chat, 1/2/4/6/8 streams | +0.5%, +0.3%, +0.3%, +0.6%, +0.8% | 2.81, 2.51, 2.51, 2.51, 2.51 |
| greedy, one stream (`bench_openai`), code and chat | +0.4%, +0.4% | 5.22, 2.81 |
| temperature 1.0, seed 1234, code, 1/2/4/6/8 streams | +0.8%, +0.7%, +0.7%, +0.8%, +1.2% | 4.57, 3.77, 3.83, 3.72, 3.72 |
| temperature 1.0, seed 1234, chat, 1/2/4/6/8 streams | +0.7%, +0.4%, +0.5%, +0.7%, +0.6% | 2.39, 2.45, 2.46, 2.43, 2.43 |

Tokens a round are main's in every cell.

## The noise rule

A cell is slower when both branch runs are under both main runs and the mean gap is over 1%; it gets one rerun, and a
repeat is a loss. A cell is faster when both branch runs are over both main runs and the mean gap is over 1%.

- `1` against main: 0 of 22 cells slower, so nothing was rerun. 2 are faster: greedy code at 6 streams, 312.5 and
  308.2 against 307.0 and 303.5 tok/s (+1.7%), and at 8 streams, 377.4 and 373.1 against 372.4 and 367.2 (+1.5%). The
  other 20 are level, +0.3% to +1.2%.

## Cold prompts

`ttft.py`'s cold 8k and 16k prompts, three reps an arm, thinking off: medians 4.033 and 4.069 s on main against 4.003
and 4.074 s on the branch at 8k, and 8.033 and 8.124 s against 7.989 and 8.084 s at 16k. A prompt call never takes the
fused form: the prompt tile runs only on plans without a cluster, and the setting acts only on cluster plans.

## Memory

The engine allocator's resident model is 72.71 GiB in all four arms. The fused form reads the same weights and needs
no partial-sum buffer.

## Off, on and a malformed value

- `1` logs the line above in F1 and F2; M1 and M2 log nothing about it.
- `TF_FLASHNEXT_FUSED_SPLIT=2` stops the start before any load: `TF_FLASHNEXT_FUSED_SPLIT="2": expected 0 or 1
  (default 0)`.
- Unset and blank (the container's environment holds `TF_FLASHNEXT_FUSED_SPLIT=` empty) are covered by the exactness
  arms below.

## Exactness

`exactness.md`: drafted against `"draft": false`, resumed against fresh and concurrent against solo, on main and on the
branch with the setting `1`, unset and blank, each with the prompt cache on and off. Every equality holds in every arm,
and the branch's replies equal main's in every comparison: 32 of 32 concurrent and serial hashes, 8 of 8 golden, 8 of
8 drafted replies' token ids with the cache on and off, and 7 of 7 first and resumed turns.

## Kernel contract

`tf-cuda-test kernel-contract "fused form" "half tiles"` on the branch's image: `SUMMARY 8 kernels: 8 pass gates 1-4,
0 fail, 0 known findings; 4 controls: 4 red as planted, 0 missed; 0 limit problems`, exit 0. Each fused form is
bit-equal to the cluster kernel's K slices at 16 shapes, MXFP8 and BF16, bf16 and fp32 out; its control, the fused form
at a 1.5x tensor scale posing as the cluster kernel, differs from byte 0 at 1x2560x6144.

## Not established

GB10 only. sm_80, sm_86 and sm_89 have no clusters, so their plans never take the fused form; their launch records come
from `kernel-contract-limits` over five-SM fatbins, not from a run on that hardware. Metal and HIP refusing `1` is
covered by host tests, not run on that hardware.
