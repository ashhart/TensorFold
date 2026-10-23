# FP4-Local optimization round, 2026-10-22

## Contract

Baseline is exp-v4-20261022 (binary 49047a2ca11147a7c22e363baa8c05a81ad6dea6d6da0e4b810b9cd4e5a422d6), pinned TensorFold f8d254c5 plus PR1050 b960da83 and the recorded local changes. Target: two GB10s, TP2, original NVIDIA FP4-Local revision 76348ddb, DFlash2 bf582e4e, A4 routed prompt / A16 verification, native mixed stored weights, FP8 latent cache, 512K and four streams. Source weights and original serving image remain unchanged.

Goal is higher served speed without lowering precision, changing accumulation order, output tokens, context, concurrency or sampling defaults. Test an existing expert CTA grouping, then explicit DFlash cost/confidence settings, and an optional bit-preserving wider A4 tile. No public release, upload or upstream qualification is implied. CUDA Graph and changed split-K are outside this bounded first round.

Owners: root handles controller, before/after receipts, integration and serving. A4 contributor owns glm_a4_prompt/glm_a4 and its tests. Policy contributor owns cuda_lanes/cuda_native and a tuning helper. Decode contributor supplies an independent 1/4-warp comparison. No simultaneous GPU benchmarks and no model reload while a request is active.

Acceptance: only select candidates with identical operator bits and unchanged served token hashes at fixed seed/settings, plus draft/plain, cache resume, cancellation/retry and concurrent regressions. Ordinary prose/code and sampled cells are measured, cold prompts above 32K tested. Small gains need alternating comparisons; failed or slower candidates remain disabled. Peak memory/swap and exact build identities accompany results. The broader existing qualification gaps (all upstream kernel gates, official-backend teacher-forced fidelity and four full 512K requests) are not silently called passed.

## Receipts


### Operator and build checks

- Candidate image: `local-reference/tensorfold-fp4local:exp-v5-20261022`, ID `sha256:77a6204988f5012919aca93198a267f5cc8faafb0f49ee9848d5479a13d0fb20`.
- Native binary SHA256: `4c75b2b6182560bdbee9eef8267218bfaa975541464761a7ad0476c8fec2e0ad`.
- Frozen source: remote `source-v5.tar.gz`, SHA256 `1cf4f6d3d4afeb3a374f0e967790638ef06619146165da6a4f8746663ad2f1a3`.
- `native test-cuda-host`: 254/254 build steps, 1057 tests passed and 4 skipped. Planted negative tests print FAIL diagnostics; the build summary is the aggregate result. No skipped test is called passed.
- A4 N64→N128 keeps K iteration/MMA order, individual projection scales, GLM clamp and BF16 boundaries unchanged. Same compact pack, no additional full weight copy. Actual TP2 up microbench ratio ~1.51x; down ~1.16–1.17x. These are cache-resident operator measurements, not served speedups.
- Frozen-v4 fatbin replacement comparison: 11 cases including an active minimum N32/K64 case, tail rows, padded strides, separate scales, zeros/extremes and skipped shared slots. Five repeats each; exact bytes, unchanged inputs and intact guards.
- Focused CUDA contract: two A4 kernels passed nonfinite, five-repeat, 13-shape, exact FP32 cancellation stress and physical-resource gates. Four symbols' compiled attributes match records. This focused run does not include all families or planted controls and is not a claim that the entire upstream kernel-contract is qualified.
- WARPS=1 versus 4: 16 real-shaped up/down microbench cases (rows 1–8, dispersed and shared-concentrated routing), five-repeat byte equality. Uniform routing improvement ~0–3.3%; concentrated routing ~4–5%. Full-model impact still requires measurement.

### Scheduling investigation

GLM DFlash2 advertises `node_probabilities=true`; this path bypasses the ordinary live `Rule.depth()` selection. `Config.init` installs `window_costs` then overwrites equal widths with `shared_costs`; `windows.allocate` uses the latter even for one request. The existing `lone_cost_tile=0` means live per-width correction is disabled. Observed overhead cannot lower an overestimated shared table.

Therefore a proposed `41.5,6.3,76,4.1,5` cost override is NOT deployed: its first two numbers would not calibrate the current DFlash allocator. The new override parser is available for future explicit experiments but remains unset. The candidate `tuned` profile changes only the effective cumulative confidence cutoff .7→.5 after the kernel-only comparison. It does not force K=7, change target sampling or alter acceptance checks. The draft block already computes all eight rows; more verification can still cost more than it saves.

### Controlled fixtures

`tests/perf_round.py` uses public Chinese prose and HTML prompts, identical seed=119, temperature 0 and 1, top-p .95, thinking enabled/low, max 384. Two repetitions; full token SHA256 and exact request are retained. Cold public prompts cover 2027, 8108, 32988 and 66429 tokens with a 128-token output cap. These are not the user's original article/HTML fixture. Default serving remains temperature 1 / high reasoning.

V4 baseline median decode: prose t0 ~40.09, HTML t0 ~40.51, prose t1 ~35.66, HTML t1 ~47.05 token/s. The 32988/66429-token cold prefills took 33.742/67.562 seconds. Final candidate receipts follow below.

### Rejected candidates and selected final configuration

1. `kernels`: WARPS=1, A4 N128, confidence=.7. All fixed-seed output hashes match V4. However full-model prose t0 was slower (33.77 vs 40.09 token/s median), and other decode cells were flat or slightly slower. The cache-resident warp microbench did not predict the full-model result. WARPS=1 is **not selected**. The first 2K prefill also included a new-kernel cold-start penalty; it is retained in the raw receipt, not removed to improve the score.
2. `tuned`: WARPS=4, A4 N128, confidence=.5, cost override absent. All tested fixed-seed outputs still match V4. Prose t1 decoded at 34.77 vs 35.66 token/s and HTML t1 at 46.91 vs 47.05; t0 results were effectively flat. More accepted tokens per round did not offset more rejected verification rows. Confidence=.5 is **not selected**.
3. `prefill` (selected for final qualification): WARPS=4, A4 N128, confidence=.7, costs unchanged. This retains only the bit-preserving A4 tile improvement. Default sampling, context, stream count, weights and KV precision remain unchanged.

The second run performed a separately recorded 2K prompt-kernel warmup before timed fixtures; the baseline was already warm from prior service qualification. At 8108/32988/66429 tokens, prefill seconds were 7.318/28.871/57.673 vs V4 8.503/33.742/67.562: approximately 14–15% less time. Final selected-profile measurements and functional regression results follow after qualification. These are small fixed-fixture observations on stock variable clocks, not universal speed guarantees or a vLLM comparison.

### Final selected-profile speed receipts

All 13 timed/warmup response hashes equal their V4 baseline counterparts; the separately added prefill warmup is not a comparison cell. Cold prompts report zero reused tokens.

| Prompt tokens | V4 prefill seconds | V5 final prefill seconds | Time reduction |
|---:|---:|---:|---:|
| 2027 | 2.022 | 1.735 | 14.2% |
| 8108 | 8.503 | 7.334 | 13.7% |
| 32988 | 33.742 | 28.941 | 14.2% |
| 66429 | 67.562 | 57.888 | 14.3% |

The corresponding prefill throughput increase is approximately 16–17%. This is narrower than the operator microbenchmark and does not apply to all workloads.

| Decode fixture | V4 median token/s | V5 final median token/s |
|---|---:|---:|
| Prose, temperature 0 | 40.09 | 39.63 |
| HTML, temperature 0 | 40.51 | 40.18 |
| Prose, temperature 1 | 35.66 | 35.59 |
| HTML, temperature 1 | 47.05 | 46.82 |

There is **no demonstrated decode speedup** in this round; the retained change only targets prefill. Final decode is essentially flat (observed -0.2% to -1.1%), not a claimed improvement.

Parallel short-output fixture, 256 tokens per request, includes prefill in aggregate wall-time throughput: two requests 44.20 token/s; four requests 61.44 token/s. The .5 cutoff measured 42.69/57.62 on the same WARPS=4/N128 kernels, with identical output hashes. This supports retaining .7. No same-fixture V4 parallel speed receipt was taken, so these are not a V4 concurrency speedup claim.

Source builds are authoritative on node-b. The runtime image is identical on both nodes; no source tree on node-a is required for inference. Final control script `tfctl-v5` SHA256: `a3ef96ce6979bca131693607a4eeb16f8bf0501934014e47b236d74f2b7aebba`; unchanged `serve-rank.sh`: `6f5fca1556d164e158cafb34a22172c230cbb6c24dc3b7d6f43628ff20a42dbb`.

### Final correctness and resource receipts

The final regression process exited 0 after `smoke`, `sampled`, `parallel`, `long`, `cancel-long`, `vision`, `api`. Fixed-seed hashes match V4; two temperature-1 seed fixtures match plain decoding; four solo/concurrent pairs match. 41346-token cold, cached and plain outputs are equal, restoring 40960 tokens. The 41363-token cancellation case closed after 20 SSE lines / 7 generated deltas before completion, then drafted/plain 128-token retries matched exactly. Both direct and tool-returned images correctly identified the red square; required tool, JSON schema, Responses include tolerance and SSE completed correctly.

No CUDA failure was observed in these final tests. Both nodes retain ~17 GiB MemAvailable; resident swap ~118/48 MiB (A/B), with zero in/out in the final five two-second intervals. The monitor logs and raw JSONL are retained. No swapoff or page-cache dropping was used. Four simultaneous full 512K requests, broad quality benchmarking and official-backend teacher-forced parity remain outside this bounded round.

The longer final monitor was not all-zero: over 119 five-second intervals, head swap-in peaked at 223 KiB/s and swap-out at 1 KiB/s; worker swap-in peaked at 56 KiB/s and swap-out at 0. Small intermittent paging is distinguished from sustained model-memory thrashing; the final idle intervals above were zero.

The selected `prefill` profile remains running. `tfctl` selects v5; `tfctl-v4` is the independent rollback entry. Refer to IMPLEMENTATION.md for exact commands. No existing 8888 service or OliveTin entry was changed.

## Subsequent decode-only round

See `DECODE_OPTIMIZATION.md` for the v6 KDA-copy, MoE-tail and sampler-gather
experiment. Exactness and API/long-context regressions passed, but repeated
off/on measurements showed no reproducible end-to-end speedup. V6 remains an
opt-in image; v5 and its N128 prefill improvement remain the selected default.

The subsequent v9 same-arithmetic KDA projection round is documented in
`KDA_PROJECTION_V9.md`. Six pipeline/warp variants and an ordered non-cluster O
reduction were tested. The latter passed bitwise operator and fixed-token
served comparisons, but did not demonstrate a reliable full-service speedup;
it remains disabled and does not replace v5.
