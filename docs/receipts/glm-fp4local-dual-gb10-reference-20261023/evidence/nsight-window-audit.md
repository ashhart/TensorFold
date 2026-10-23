# Native FP4-Local v7: all-context Nsight audit

Date: 2026-10-23. Read-only analysis of the 12 SQLite captures and their `*-families.json` / `*-summary.json` files in this directory. No GPU runs or service changes were made for this audit.

## Conclusion

The first capture's conclusion holds across both ranks, both tasks and 8K/32K/128K: traced GPU activity covers **94.51–95.53%** of each approximately 2.1-second window. This is **activity coverage, not SM occupancy**, memory-bandwidth utilization or proof of useful work. Communication kernels can be active while waiting.

Routed NVFP4 expert kernels are the largest identified class (**33.56–43.54%** of the window). Source-matched KDA input/output projection sequences contribute **17.16–19.66%**. Named communication kernels contribute **4.45–5.78%**. These proportions, not isolated launch counts, identify more promising optimization targets than host round bookkeeping.

## Per-capture results

All percentages use that rank's own capture duration. `KDA projections` sums the input-sequence and output-projection interval unions; their recognized sequences do not overlap. Other family sums must not be treated as an exclusive critical-path breakdown.

| Task/context | Rank | Window ms | Activity % | Routed % | KDA projections % | Comm % | Gaps % | Already queued before gap % | Launch started after gap began % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| code / 8K | 0 | 2102.53 | 95.531 | 42.47 | 17.72 | 5.00 | 4.469 | 4.297 | 0.100 |
| code / 8K | 1 | 2118.84 | 95.001 | 41.71 | 17.54 | 5.26 | 4.999 | 4.747 | 0.145 |
| prose / 8K | 0 | 2082.51 | 94.940 | 38.06 | 19.08 | 5.00 | 5.060 | 4.757 | 0.130 |
| prose / 8K | 1 | 2111.35 | 94.964 | 37.23 | 19.15 | 5.48 | 5.036 | 4.830 | 0.091 |
| code / 32K | 0 | 2128.71 | 95.156 | 43.54 | 17.16 | 4.45 | 4.844 | 4.496 | 0.110 |
| code / 32K | 1 | 2129.73 | 95.081 | 42.69 | 17.22 | 5.22 | 4.919 | 4.662 | 0.151 |
| prose / 32K | 0 | 2127.22 | 95.265 | 38.32 | 18.85 | 5.19 | 4.735 | 4.560 | 0.098 |
| prose / 32K | 1 | 2094.33 | 94.509 | 36.18 | 19.16 | 5.26 | 5.491 | 5.203 | 0.171 |
| code / 128K | 0 | 2088.89 | 94.943 | 34.34 | 19.66 | 4.82 | 5.057 | 4.887 | 0.092 |
| code / 128K | 1 | 2108.42 | 94.645 | 33.56 | 19.28 | 5.78 | 5.355 | 4.931 | 0.299 |
| prose / 128K | 0 | 2080.88 | 94.751 | 35.12 | 19.24 | 4.64 | 5.249 | 5.080 | 0.091 |
| prose / 128K | 1 | 2111.02 | 94.516 | 33.76 | 19.24 | 5.49 | 5.484 | 5.299 | 0.071 |

## All-gap check

The original 20-us reporting threshold excludes most gap duration, but **does not exclude it from the activity-union calculation**. Recomputing with every positive gap gives duration-weighted values across the 12 independent captures:

- All gaps: **5.057%** of capture time.
- Gaps below 20 us: **4.948%**; 1–5 us gaps alone: **4.808%**.
- Next kernel's launch API had already returned before the gap: **4.811%**.
- Next launch started after the gap began: **0.129%**, range **0.071–0.299%**.

Thus there are many small gaps, but they overwhelmingly occur with the next kernel already queued. They do not support CPU launch starvation as the dominant bottleneck. They may involve device scheduling/dependencies; this trace does not identify the exact hardware cause. Removing every traced gap would bound the speedup at approximately **4.7–5.8%**, before accounting for unavoidable gaps or profiler effects. CUDA Graph/fusion may have other benefits, but a large improvement cannot be justified from host submission gaps alone.

## Attribution and comparison boundaries

- All **13,402** `tf_glm_kda_prep` sequences matched the source-defined input-projection sequence; zero unmatched. Attribution uses ordered kernel names around KDA, not only launch grid or shape. See `tests/analyze_nsys_families_v7.py` and `upstream/zig/src/families/glm/cuda_kda_block.zig`.
- `df_*` kernels are only a lower-bound named subset of DFlash: generic matrix multiplication and communication kernels are shared names. Do not replace the complete host-round drafter cost with that subset.
- Rank clocks and capture boundaries are independent. Do not subtract rank timestamps or infer network latency from differences between these windows.
- These are approximately two-second slices of different generated sequences, not equal-work, fixed-row microbenchmarks. DFlash acceptance changes the target verification row mix; routed expert overlap changes the work per forward. Compare the same row/shape and projection identity before attributing a percentage change to context length.
- The 128K code fixture emitted 256 reasoning tokens, while the 32K code fixture emitted 22 reasoning tokens. Task/state distribution differs even with the same output-token limit and seed. The event run's identical v5 hashes verify instrumentation parity, not equivalence of different context-length workloads.
- Family activity percentages are not direct speedup promises. Validate actual gains with the unprofiled six-cell fixed-token/hash suite after a precision-preserving kernel change.

Reproduce each capture with `python3 tests/analyze_nsys_families_v7.py PATH_TO_ONE.sqlite` (read-only SQLite connections). Existing per-capture JSONs preserve full gap bins, launch evidence and family data.
