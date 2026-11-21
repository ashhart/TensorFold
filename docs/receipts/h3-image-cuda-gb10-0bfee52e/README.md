# MiniMax-H3-Image's CUDA selftests on a GB10

`tf-h3-image-decoder-cuda selftest` and `tf-h3-image-prompt-cuda selftest`, built at `0bfee52e` with
`-Dsm=121 -Doptimize=ReleaseFast`, on an NVIDIA GB10 (sm_121, driver 580.173.02). The run's checkout was
`0b573dd2` (that commit plus the job script), so the tool code is the same.

- `decoder-selftest.txt`: every check passes. Every block of the 24 is gated at 90% identical to the float64
  reference on the GPU's own input; the lowest is block 9 at 92.037%, worst 2.00 ULP, relative 8.87e-4.
  The pictures pass the Metal decoder's bytes bound (80% identical, at most 2 apart): 88.889% and
  92.164% identical, at most 1 apart. 25 tiles of 16 x 16 take 43.6 ms a tile.
- `prompt-selftest.txt`: every gate passes; the worst row is 4.66e-2 of its row's RMS on
  `condition_proj`, the refiner and the final norm.

Both tools build synthetic models and compare against the host's float64 reference; no weights are read.

The same run's kernel contract passes gates 1 to 4 but exits nonzero on its limits:

```text
SUMMARY 685 kernels: 685 pass gates 1-4, 0 fail, 8 known findings; 422 controls: 422 red as planted, 0 missed; 188 limit problems; gates 3449547 ms, limits 765095 ms
FAIL kernel-contract: 0 gate failures, 0 controls missed, 188 limit problems
FAIL kernel-contract: TestFailed
```

Of the 188, 183 are `main` kernels with no compiled usage recorded (`qwen4_*`, `qwen35*`, `nvfp4_experts_tile`,
`exl3_experts`, `flashnext_mtp`) and 3 are live launches of `main` kernels recorded as 0 (`fn_row_topk`,
`tf_exl3_prompt_gu`, `tf_exl3_prompt_down`). The other 2 are `h3p_norm` and `h3p_heads` on sm_121, measured
at 27 and 40 registers against stale records of 25 and 38. `kernel-contract-limits` exits 1 with 28 stale
records, 10 of them these two kernels'. This branch sets both records to that run's `RECORD` lines. A limits-only run
at `5ef2a963` (that commit plus the job script) checks both current on all five SMs — `h3p_norm` at 31, 35, 35, 27 and
27 registers, `h3p_heads` at 32, 36, 36, 40 and 40 — and exits 1 only on the 18 stale records inherited from `main`
(`fn_row_topk`, `tf_exl3_prompt_gu`, `tf_exl3_prompt_down`, and `tf_q35_gdn_slots_1`, `_2` and `_4` on sm_80), with no
`h3p_*` or `h3d_*` entry:

```text
INFO limits fatbins: images built by nvcc 13.0, checked against nvcc 13.0's records
FAIL kernel-contract-limits: 18 problems
FAIL kernel-contract-limits: TestFailed
```
