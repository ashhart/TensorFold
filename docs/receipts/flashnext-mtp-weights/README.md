# Flash Next MTP drafter weights: main against the branch

`TF_FLASHNEXT_MTP_WEIGHTS=checkpoint|mxfp8`. `mxfp8` re-codes the drafter's BF16 linears to MXFP8 at load, off by default.

## Source and environment

- Base: main at `6c5cef8941b27d86b06a986834e13deb9b515e1e`, one NVIDIA GB10 (sm_121), Zig 0.17.0, CUDA 13.3.
- Checkpoint `nvidia/Qwen3.8-Flash-Next-FP4-Local`, revision `b9ce18a6887e18c089fdea8509a33e63429a3300`.
- `--parallel 4`, 131,072-token context, bf16 KV, a fresh server per arm.
- Arms, in order: main, branch with the setting unset, branch with `mxfp8`, main again.

## Decode speed

`cells.csv` has every cell: `tools/bench_concurrent.py --alone --serial`, greedy, 256 tokens, 3 reps, 1, 2 and 4 streams, aggregate tok/s.
With the setting unset the branch is level with main (-0.5% to +0.2% against the mean of the two main arms). With `mxfp8` code gains
+1.8% at one stream, +3.1% at two and +2.7% at four; chat is -0.2% at one stream, +2.4% at two and +2.3% at four.

## Exactness

`exactness.md` has the table: drafted against `"draft": false`, resumed against fresh and concurrent against solo, on main, on the
branch unset and on the branch with `mxfp8`, each with the prompt cache on and off. Every equality holds in every arm, and the branch's
replies equal main's.

## Not established

Sampled decode speed and `--drafter` against none were not timed. GB10 only.
