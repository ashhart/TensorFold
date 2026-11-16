# k-split decode matmul, Apple M2 Max (14-inch) and M1 (13-inch)

Qwen3.5-2B 4-bit, 128 greedy tokens per run with the token read back each step (`TF_VK_DECODE_BENCH=128`), after the prompt prefill and 8 warm-up tokens. Base = `80b2c090f` (the bench mode only, `qmm_q4_tile` unchanged). Head = `8613ae9aa` (`qmm_q4_tile` with 8 k slices). Both are clean-tree builds of those commits, run alternately, one process per run, no GPU timeouts.

Linux 7.3.0-2-20261110.1-lab-ARCH, Mesa honeykrisp. Cases: `unicode-24` (24-token prompt, context 512), `document-1024` (1,024-token prompt, context 1,536), `prof6173` (6,173-token prompt, context 7,168).

Median ms per token (spread of the head arm over its rounds is in brackets):

| Machine | Case | Rounds | Base | Head | Tokens/s, base to head | Change |
|---|---|---:|---:|---:|---|---:|
| 14-inch M2 Max | unicode-24 | 5 | 18.76 | 10.65 (4.8%) | 53.3 to 93.9 | 76% faster |
| 14-inch M2 Max | document-1024 | 5 | 21.14 | 12.91 (0.9%) | 47.3 to 77.5 | 64% faster |
| 14-inch M2 Max | prof6173 | 5 | 31.96 | 24.23 (4.4%) | 31.3 to 41.3 | 32% faster |
| 13-inch M1 | unicode-24 | 4 | 28.45 | 24.73 (0.3%) | 35.1 to 40.4 | 15% faster |
| 13-inch M1 | document-1024 | 4 | 30.58 | 26.88 (0.5%) | 32.7 to 37.2 | 14% faster |

The greedy token hash is equal between base and head in every run (`fc93d24775b1d909` at unicode-24, `3efb3759184db87e` at document-1024, `fc0ededa74f07707` at prof6173). The f32 sums in the head are added in a different order, so the logits are not bit-identical to the base; the chosen tokens are.

Kernel contract (`tf-vulkan-kernel-contract`), base and head on both machines: 5 kernels pass gates 1 to 4, 0 fail, 9 planted controls caught and 0 missed, 0 limit failures. The `FAIL` lines in the contract files are those planted controls being caught. The `VERDICT ... FAIL` line at the end of those four files comes from the wrapper script, which counted the planted `FAIL` lines; the `SUMMARY` line is the contract's own result.

Why: `qmm_q4_tile` ran one thread per output column and walked all of k, so a 2048-column projection was 8 workgroups and a 6144-column one 24.
