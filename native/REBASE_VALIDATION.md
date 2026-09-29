# Rebase verification — 2026-09-28

Historical milestone `b9a9fc8`. Subsequent investigation found MLX 0.31.2 in the
Python environment, below the repository's 0.32.2 requirement. The version attribution,
precision workaround and remaining Qwen prefill limitation below are superseded by
[PREFILL_VALIDATION.md](PREFILL_VALIDATION.md), which records fresh version-checked runs.

Base: upstream `34bae79` (TensorFold 0.3.6.1), rebased native branch `806291f`,
plus the uncommitted changes described below. Machine: M5 Max, 128 GiB unified
memory. Zig: 0.17.0-dev.2248+3f6a02acd. MLX Python: 0.32.2; mlx-lm: 0.31.3.
Native uses the existing staged MLX/MLX-C libraries under `build/mlx`.

## Passed

- Default fast build and runtime-safety build.
- All 18 host tests, positional checkpoint I/O checks, all 6,105 checkpoint
  tensor contracts, and 38 malformed/missing checkpoint cases.
- Full-model verified rows, partial commits, continuation logits and every cache:
  Qwen (64 layers) and Nemotron (52), both tensor and forced SIMD; Flash (48), tensor.
- GPU-sampled 32-token serial/draft regression: Qwen/DFlash2 on tensor and SIMD;
  Nemotron/MTP depths 1, 3 and 15 on both backends; Flash/MTP depths 1, 3 and 15.
  All 11 serial/draft comparisons match output IDs exactly.
- Fresh Python oracles: 76 CPU/GPU sampling/top-k cases, six long-context tensor
  attention fixtures, 36 PLE normalization fixtures, and three sparse-attention
  boundary fixtures (all eight rows, pooled keys and rollback continuations exact).
- The exporter checks all 86 embedded kernels. The variant suite passes 173 Python
  tests (one invalid group-size combination skipped) and replays 2,190 native
  launches across 54 variants with every output bit exact.
- All 12,288 adaptive draft-depth cases and 360 MTP proposal/retained-position cases.
- All 1,100 injected host allocation failures release their MLX handles, with zero
  retained active MLX memory; intentional MLX API errors recover successfully.
- Short independent Python/native full-model checks: Qwen tensor/SIMD, Nemotron SIMD and
  Flash tensor, four prompt tokens and 16 generated tokens, serial and DFlash2 or
  fixed-depth-15 MTP. All prefill logits match exactly (993,280, 524,288 and 993,280
  values respectively), as do all output token IDs. These short checks do not
  replace the full long-context matrix.
- Flash's repaired intermediate trace matches Python at all 242 saved arrays
  across 48 layers and the final head for the four-token prompt.
- Qwen production engine trace: one prefill and 31 serial forwards, all 64 caches,
  hidden states, logits and 32 emitted tokens equal an independent cache execution.
- Qwen native serial/DFlash2 and the Python lane-tree diagnostic: identical prompt
  and 32 output IDs, hash `f83de6a7f9a7a978e6e3b19a244a70195f8a6ceaf04a0dc08b609b99aeb300a3`.
  Prompt: `Write a short Python function that computes the Fibonacci sequence.`
  Seed 5678, temperature 0.7, top-k 12, top-p 0.8, CPU sampling, context copy off.
  These are smoke checks, not repeated performance measurements.

Commands used:

```sh
.zig-toolchain/zig build
.zig-toolchain/zig build -Doptimize=safe
.zig-toolchain/zig build test test-checkpoint-files test-model-schemas test-schema-failures -Doptimize=safe
.zig-toolchain/zig build test-models -Doptimize=safe
.zig-toolchain/zig build test-drafts -Doptimize=safe -Ddraft-scenario=1
.zig-toolchain/zig build test-metal -Doptimize=safe -Dmetal-tensors=true -j1
.zig-toolchain/zig build test-variants -Doptimize=safe
.zig-toolchain/zig build test-draft-depth test-mtp-positions test-allocation-failures -Doptimize=safe -j1
.zig-toolchain/zig build test-long-context -Doptimize=safe -Dlong-tokens=4 -Dlong-family=0 -Dlong-backend=0 -j1
.zig-toolchain/zig build test-long-context -Doptimize=safe -Dlong-tokens=4 -Dlong-family=0 -Dlong-backend=1 -j1
.zig-toolchain/zig build test-long-context -Doptimize=safe -Dlong-tokens=4 -Dlong-family=1 -Dlong-backend=1 -j1
.zig-toolchain/zig build test-long-context -Doptimize=safe -Dlong-tokens=4 -Dlong-family=2 -Dlong-backend=0 -j1
.venv/bin/python tools/export_native_kernels.py --check
.venv/bin/python tools/native_qwen_engine_trace.py
.venv/bin/python tools/native_flash_trace.py build/native-checks/rebase-flash-trace/python --model build/models/Qwen3.8-Flash-Next-MLX-4bit-MTP
.venv/bin/python tools/native_flash_trace.py build/native-checks/rebase-flash-trace/native --model build/models/Qwen3.8-Flash-Next-MLX-4bit-MTP --native zig-out/bin/tensorfold
.venv/bin/python tools/native_flash_trace.py build/native-checks/rebase-flash-trace/python --compare build/native-checks/rebase-flash-trace/native
```

Benchmark harness smoke reports and trace output are in ignored
`build/native-bench/rebase-*` and `build/native-bench/qwen-engine-trace/`.
The harness records the source version separately from installed distribution
metadata, which can lag behind an editable checkout after a Git update.

## Fixed after the rebase

- Sparse attention differed because the staged native MLX library and the Python
  MLX wheel resolve unqualified Metal math functions differently. Q/K/V, selected
  IDs and attention maxima matched, but partial softmax outputs did not. Explicit
  `metal::precise::exp` restored exact sparse attention. Full-model Nemotron then
  exposed the same issue in other math operations; the exporter now makes
  `exp`, `exp2`, `log`, `log2`, `sqrt`, `rsqrt` and `pow` precision explicit.
  Deliberately fast functions remain fast. No comparison tolerance was added.
- The exporter and variant capture understand current baked constants, module
  locations and threadgroup annotations. Retired interfaces are preserved as
  versioned, independent Python references under `tools/native_legacy`, with their
  original MIT provenance. Expected values still execute the original Python
  kernels; source normalization only identifies the corresponding native kernel.
- Native launch templates now include the current group-size and row-edge fields;
  row attention launches one group per window row. SIMD projections use upstream's
  current reduction split (16 chunks through output width 6,144); the old width
  threshold changed Qwen logits. Sampling passes the current per-row settings
  layout. All 86 embedded kernels have been regenerated.

## Scope and remaining limitations

- Native Qwen and `native_reference.py --production-kernels` retain lane-tree
  arithmetic; the Python serving engine now prefills through the model's regular
  forward. For the English prompt above, native matches the lane-tree oracle but
  differs from the serving engine at token 18. The serving-engine output hash is
  `d68c903d263f5fa5d82e3106cb462aa36b10961c2cd634c3686f4c7c015dccc1`.
  The pre-fix native build matched that output, but this is no longer a
  passing production-engine comparison after the kernel repairs. Exact kernel,
  lane-oracle and serial/draft parity do not establish serving-engine parity.
- The full long-context, resident-PLE, serving and performance matrices were not
  rerun in this verification. Older measurements in the README/COVERAGE are historical.
- No claim of full native coverage for the newly added upstream GLM/Gemma families,
  multi-stream serving, or other physical Mac generations. Native model checks
  here cover the three original downloaded target families and DFlash2.

## Uncommitted changes

- Native reports now separate load, warm-up, startup and calibration timing.
  Startup includes preparation before prefill; load and calibration can overlap
  in Python because calibration happens inside its loader. Do not add them together.
- `native_engine_bench.py` measures Python/native in separate sequential processes,
  records IDs, hashes, draft availability, versions, timings and memory, and rejects
  failed runs, early EOS or nondeterministic repetitions. Qwen uses the current
  family loader's drafter argument. Production and native prefill methods differ.
- `native_qwen_engine_trace.py` now hooks the current family API rather than the
  removed engine callback; it checks caches before/after actual forwards and saves
  mismatching tensors or a successful report.
- `native_reference.py` adds production-loader/fusion diagnostic options and
  unwraps the current family object before using the lane-tree oracle. Its SIMD
  adapter now hooks the current stream-attention entry point.
- Flash PLE/sparse fixtures use the new module locations. Sparse fixtures select
  the original row projection mode and materialize expected results. Sampling
  fixtures materialize each draw before saving: the previous delayed batch produced
  saved draws that did not match fresh draws from the same saved inputs.
- Native sampling failures now identify the failing case and settings.
- Exporter, fixture generators, versioned legacy references, regenerated Metal
  sources and native dispatch fixes implement the rebase repairs described above.
- Nemotron and Flash model oracles use current upstream interfaces while retaining
  the native arithmetic recipe. Flash's sharded PLE adapter avoids recursive lookup.
- This report and the README link distinguish current verification from historical
  coverage claims. No changes were committed or pushed during this verification.
