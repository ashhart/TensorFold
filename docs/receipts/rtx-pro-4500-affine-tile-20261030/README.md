# RTX PRO 4500 affine-tile: draft qualification receipt

**Draft: the full CUDA kernel-contract SUMMARY gate is outstanding.** The complete comparison programs have not yet produced summaries; original and first-overlay resource failures are described below. No installed runtime was changed.

A 128-column/eight-warp MLP affine tile for N=17408, K=5120, one K slice and 1–16 rows, gated to SM120/82 SMs. Existing GEMV, grouped projections and other shapes retain their dispatch.

Base `01e25d3aebc0ebb686670c69e37b7c269f261b3d`; candidate code `aa8ec483b005b94c20dae85a3129a71387ef5f68`. The explicit build label 1.1.6 is retained recipe/client compatibility metadata; source SHAs identify code. See [environment.json](environment.json) for exact checkpoint/drafter revisions, config/tokenizer hashes, driver, libraries and hardware. Native Zig/CUDA inference, no MLX/PyTorch serving runtime.

Normal serving settings remain 131072-token context, FP8 KV, int8 prefill, vision, BF16 DFlash2, 8 GiB host prompt cache and parallel 1. Clean measurements use upstream-supported0.5 GiB reserve in every arm; historical private 0 GiB overlay measurements are not pooled with these.

## Completed decode comparison

Stock/candidate/stock, fresh calibration every boot, identical request history, within-cell medians and a geometric mean of bracketing stock medians. One candidate boot does not establish a confidence interval, particularly for small differences. All rates are tokens/s.

Standard `tools/bench_openai.py`, greedy prose/code, 256 tokens, 3 repetitions:

| Workload | Stock before | Candidate | Stock after | Gain |
|---|---:|---:|---:|---:|
| fibonacci-raw | 214.81 | 216.36 | 214.59 | +0.774% |
| gpu-chat-no-think | 164.19 | 165.32 | 164.08 | +0.724% |

Retained community-v1 client, 256 tokens, 5 repetitions per cell:

| Workload | Stock before | Candidate | Stock after | Gain |
|---|---:|---:|---:|---:|
| code/T1 | 232.32 | 232.52 | 230.48 | +0.484% |
| chat/T1 | 144.84 | 145.00 | 143.80 | +0.468% |
| code/T0 | 216.52 | 216.90 | 215.01 | +0.526% |
| chat/T0 | 165.23 | 165.56 | 164.18 | +0.520% |

Raw client metadata that was unavailable remains unknown. Native manifest and environment sidecars provide separately observed identities; these attached diagnostics are not a leaderboard submission.

## Completed exactness, feature and host checks

Every completed stock/candidate/stock token-hash comparison matches. Drafted/plain, resumed/fresh and queued concurrent/solo outputs match. PNG/JPEG/WebP, required tools, JSON schema, observed reasoning, cancellation recovery and context reuse through 122779 input tokens pass in every serving arm. Reasoning-off speed requests remain separate from reasoning feature checks.

- Build Summary: 478/478 steps succeeded; 311/313 tests passed (2 skipped)
- Build Summary: 167/167 steps succeeded

Zig cache hits are explicit in the logs; these counts describe tests actually reported, not a sum of overlapping runs. Final formatting passes; lean reports 425 inherited findings and 0 new. Boundary/ABI/compiler-selection tests, the failing GEMV regression receipt and its passing correction are attached. A preexisting scratch-memory host fixture crash was reproduced on the untouched baseline and fixed in the test setup.

Four final sanitizer runs pass with zero errors: memcheck and synccheck for actual native 13.3 images with Compute Sanitizer 2026.2.1, and independently compiled 13.0 images with matching 2025.3.1. Each run has 744 byte comparisons plus graph replay, shuffled-page/whole attention, input-preservation and guards. Exact CUDA images were extracted from native ELF headers; identities are attached. A newer-checker/older-image attempt completed all byte checks but returned a selective-recompilation API warning; that nonzero attempt is retained separately and is not counted as clean. No warnings or checks were suppressed.

## Qualification status and remaining gates

The original complete stock/MLP/attention programs were killed under the 24 GiB container budget without a SUMMARY; see [original resource failures](qualification-20261031/original-resource-failures/README.md). A solo MLP retry with only common per-kernel Rig cleanup reached the final standalone suite but also OOM-killed after 6,005 seconds without a SUMMARY. See [first-overlay receipts](qualification-20261031/first-overlay-resource-failure/README.md) for exact source snapshots, hashes, logs and the kernel OOM record. Stock/attention were not repeated unchanged after that failure. A separate local harness extension now tears down registered owners/modules before standalone suites, but is absent from these failed binaries and remains unqualified. No full-suite pass is claimed. Earlier interrupted evidence is historical.

The resumed [complete cold-prefill bracket](qualification-20261031/cold/README.md) contains 60 successful completions at exact 2K/8K/32K/64K/130816-token inputs, with zero cached tokens and matching token/output hashes across stock, MLP, attention and final stock. The pause between completed MLP and restarted attention boots, small observed TTFT differences and telemetry scope are disclosed there. Historical interrupted evidence remains in cold-partial; it is superseded by the complete bracket. The previous-tag/standard-server release comparison remains unrun.

A separate unmodified-baseline edge case—131056 input tokens plus 16 output tokens—returned QwenContextFull after four tokens. Its failed receipt, exact input and server log are retained. The subsequent 130816 input leaves 256 positions for output and speculative lookahead, with configured context 131072 unchanged. This does not qualify the rejected edge case.

Physical Metal/HIP/XPU/other NVIDIA devices, other checkpoints, a full 13.0 served model gate and a previous-tag/standard-server release comparison were not run. SM80/86/89 and SM121 evidence is compile-only. Physical host RAM 31.24 GiB and the 24 GiB model-test container limit are separate facts; this is not a minimum-RAM claim. Per-process and GPU memory observations are retained in served/state.json.

## Reproduction and provenance

Use Zig 0.17.0 and the pinned source/checkpoint revisions. Build with `zig build native -Dsm=120 -Dversion=1.1.6 -Dnvcc=/usr/local/cuda-13.3/bin/nvcc -j6`; provide the matching grammar/media libraries. Set TF_SRC and MODELS, then run reproduction/reproduce_serve.sh. The standard benchmark command is `python3 tools/bench_openai.py http://127.0.0.1:18080 qwen-community --tokens 256 --reps 3 --temperatures 0 --output decode.json`, with spaces between each option and value. Reproduction workers retain exact request order and use conventional /task and /submission container mount points. Use a fresh process-local calibration HOME for each boot.

For device qualification run `zig build -Dsm=120 -Dnvcc=/usr/local/cuda-13.3/bin/nvcc`, then `zig-out/bin/tf-cuda-test kernel-contract` on an isolated/admitted GPU and attach the actual SUMMARY. This is still outstanding here.

[Coordination and supplemental overlap checks](tile-coordination.json). Every file except the manifest itself is covered by [sha256.json](sha256.json). Kernel resource records use live-driver attributes on SM120; raw cuobjdump totals and their accounting difference are retained. SM121 records do not establish a physical device qualification.

## Additional GPU testing

[Independent 5090/RTX PRO 6000 Blackwell testing instructions](testing/REQUEST.md) document the opt-in `TF_CUDA_TEST_MLP128` switch. The override was added after these measurements; defaults are unchanged and no new GPU qualification is claimed.

Local overnight [independent override bracket](qualification-20261031/independent-overrides/README.md) completed with matched tokens/features and all serving settings retained. It reports new matched medians separately from the earlier numbers above; the full CUDA suite is still resource-limited.

Current-main independent [qualification on94b90c9d](qualification-20261031/latest94/README.md) adds fresh single-contribution override measurements, matching tokens/features, and explicitly separated host-fixture failure/repair evidence. This is rebased test-source qualification; the public code branch retains its original01e25d3a ancestry. Neither contribution includes GDN fusion.

Maintainer-requested [alternating noise check](qualification-20261031/alternating-noise/README.md): six fresh0/1/0 groups, all matching tokens; greedy geometric mean +0.825%, approximate95% group-log Student-t interval +0.754% to +0.896%. All six groups positive; raw cells/drift/interruptions retained. This is a separate protocol, not pooled samples.

The [unfiltered numerical/backend contract attempt](qualification-20261031/full-contract/README.md) now has a final SUMMARY:465 numerical kernels pass,322 controls caught,8 known findings; exit1 solely on223 limit-record problems. All three new launches pass their own live limits. This supersedes the older “SUMMARY outstanding” status; it does NOT establish a full backend pass.
