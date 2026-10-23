# KDA projection v9 — implementation contract

## Scope and invariants

Optimize only the experimental TensorFold FP4-Local service on port 8890.
The immutable v5 image remains the baseline and rollback target. No changes to
8888, OliveTin, vLLM, EXL3, checkpoint bytes, quantization, split-K, clocks, or swap.

Candidates reuse `tf_qmmf::qmmf_body` for MXFP8, BM=16, BN=64, WM=1:
pipeline STAGES=2/3/4 and WN=4/8. Only launch scheduling, shared-memory staging,
and ownership of independent output columns may differ. Preserve each output's
ordered K32 `mma0` / `mma` / FP32 `fmaf`, scale interpretation, final cast,
and ordered SK=4 peer reduction. Do not reassociate any sum or fold scales.

Target shapes (per rank, rows 1–16):

- KDA QKV: N=12288, K=4096, SK=1, BF16 output.
- KDA O: N=4096, K=4096, SK=4 cluster, FP32 output.

Candidate C symbols: `tf_kda_mxfp8_s{2,3,4}_w{4,8}_{qkv,o}`;
the original qmmf 13-argument ABI is unchanged. Symbols operate on the existing
packed layout; runtime dispatch, if warranted, is opt-in and shape restricted.

## Evidence gates

1. Extract baseline qmmf fatbin from the immutable v5 image and record hashes.
2. Compare candidate vs baseline bytes for real shapes, finite/zero/extreme/
   nonfinite inputs, five repetitions, with output guards and immutable inputs.
   Retain the existing independent-reference projection correctness tests.
3. Time alternating A/B and A/A controls with warm and cache-evicted weights.
   Run GPU tests with the full service stopped to avoid competition.
4. Only promising candidates enter a separate immutable v9 image. Register
   native correctness tests and keep all prior optional experiments disabled.
5. Use unchanged 8K/32K/128K token fixtures, seeds and request settings for full
   model comparisons; compare output hashes and speculative acceptance. A small
   or noisy microbenchmark-only win is not sufficient to enable in service.
6. If correctness fails or repeatable end-to-end gain is absent, restore v5 and
   record the negative result. Report coverage limits explicitly.

## Status

Contract written before candidate implementation. The candidate is **not
promoted**: correctness passed, but a repeatable served speedup over v5 was not
established. The default `tfctl` still selects the immutable v5 baseline. The
v9 controller defaults to `TUNING=off`. V5 restoration and reverse tests are
complete; the original service is running on both nodes at port 8890.

## Additional ordered-reduction candidate

After pipeline/warp screening, also test the already-existing immutable-v5 O
projection's non-cluster path. This **retains SK=4** and all per-slice arithmetic:
write the four FP32 partials, then `reduce_kernel<true>` adds slices 0, 1, 2, 3
in the same order and applies the same final scale. The change is only the
location/synchronization of partial sums (global scratch and a second kernel
instead of cluster shared memory), not their precision or mathematical grouping.
At M=8 this needs 512 KiB scratch; guard checks and the same bytewise gate apply.
No production dispatch change is authorized by a microbenchmark result alone.

## Candidate provenance and completed gates (2026-10-23)

- Candidate image: `local-reference/tensorfold-fp4local:exp-v9-kda-20261023`
- Image ID: `sha256:5d9f6b9e6251f809cfff64013998075ca18483ac7bb464d1ddb62b6bf45b4225`
- Native binary SHA-256: `f48721fd65c368e5de899efab9cdc377a507b08d2a7d1f1ef1ceceb6cc3e3496`
- Final embedded qmmf SHA-256: `0680e5c151fed49bcca2ba33eef528ebd719d480d96879e8fefe919300ce14c3`
- Final qmmf blob was located byte-for-byte in that binary at offset 4188016.
- Immutable v5 qmmf SHA-256: `930bd57940100a0459bca930f1dd5a7fac975c313fa8e312c9bed57a545d785d`

The six pipeline/warp variants were screened but not integrated: QKV gains were
small/inconsistent, while O gains were mainly a warm-cache effect. Their source
and fatbin are retained under `results/`, not included as new runtime entrypoints.
Use only timing-revision-2 receipts for final microbenchmark interpretation;
the first timing method was superseded by events inside the captured graph.

The integrated candidate is only `TF_GLM_KDA_O_SPLIT=1`: on SM121, decode rows
1–16, MXFP8 O projection N=K=npad=4096, existing SK=4. Other shapes, prefill,
formats and projections retain their original dispatch. Existing `mm_part`
scratch is sufficient; no new persistent allocation was added. Startup validates
the flag, GPU architecture and agreement between ranks.

Completed correctness gates:

- Final compiled candidate vs frozen v5, all rows 1–16, seven input fixtures,
  five repeats, output/partial guards and input immutability: byte-equal PASS.
- Independent sampled FP64 projection checks and existing native projection-f32
  tests: PASS.
- Native build plus host tests: PASS. Host dispatch tests distinguish FP32 vs
  BF16 reducers, non-unit final scale, scratch bounds and fallback boundaries.

Evidence: `results/kda-v9-final-build-correctness.jsonl`,
`results/kda-v9-native-projection-f32.log`, `results/kda-v9-host-final.log`,
`results/build-image-v9.log`. The final noncluster kernel uses 62 registers,
versus 60 in the frozen-v5 split microbenchmark. Its served performance must be
measured directly; old microbenchmark speedups are not its qualification.

## Served comparison method

Reuse `results/decode-v7-fixtures.json` unchanged: prose/code at exactly 8192,
32768 and 131072 rendered prompt tokens; thinking enabled (`low`), temperature
1.0, top-p 0.95, seed 119, 256 completion tokens, two repeats per cell. The
256-token code test at 128K remains a fixed-token continuation (entirely
reasoning in these receipts), not a completed HTML-generation benchmark.

`tests/compare_kda_served_v9.py` compares canonical full-request hashes, exact
prompt-token hashes, full output-token hashes, speculative rounds/drafted/
accepted counts, token counts, thinking state and error flags before reporting
decode timing. Different cached-prefix lengths prohibit cold-prefill or total
wall-time speedup claims. All 12 initial v9-off requests matched v5 on the
correctness and speculation checks. Timing showed materially more variability
than the previous v5 session, so a small on/off difference is not sufficient
for promotion. The tail-only GPU monitor observed active SM clocks 2405–2437
MHz and temperatures 74–84 C; these observations do not establish the cause
of the timing variation.

## Initial served results and decision

Decode tok/s means (two repeats per cell; all 24 new v9 requests, 12 off and
12 on, match v5's requests/settings and full output-token hashes; speculative
counts also equal):

| Fixed-token workload | Prior v5 | v9 off | v9 on |
|---|---:|---:|---:|
| Prose 8K | 33.948 | 33.002 | 33.622 |
| Code 8K | 44.372 | 42.516 | 44.144 |
| Prose 32K | 30.182 | 29.581 | 29.718 |
| Code 32K | 47.660 | 47.650 | 47.238 |
| Prose 128K | 30.306 | 29.442 | 30.193 |
| Code 128K | 29.280 | 28.426 | 28.645 |

Some on/off means favor the candidate, but gains are inconsistent and overlap
the sequential-run variation. The candidate did not exceed the prior v5 mean
in any of these six cells (approximately 0.4–2.2% lower). This is not proof
that the new transport itself causes those differences: v9-off also contains
prior opt-in source changes, and the runs are sequential. It is sufficient to
reject promotion under the user's "enable only with actual benefit" condition.
The final rollback additionally repeats 8K/32K cells on the immutable v5.

No repeatable served gain was established; the warm-cache microbenchmark
advantage did not translate into demonstrated end-to-end benefit in these runs.
One possible explanation is weight traffic: O's packed codes/scales occupy
about 16.5 MiB per rank, while the preceding QKV reads about 49.5 MiB, larger
than the GB10's 24 MiB L2. This may limit O's cross-step cache reuse; intervening
layers also compete for cache. This is an inference from footprint and warm/cold
tests, not cache-hit-counter evidence or a measured DRAM-bandwidth saturation
claim. No precision, split count, reduction ordering, weight, sampling, context
or concurrency change was used to seek a win.

Because promotion failed on performance, broad new v9 four-request, vision,
512K and cancellation qualification was not run. Do not call v9 a fully qualified
replacement based on the six fixed-token cells. Existing v5 qualification stays
attached to v5, not transferred automatically to this experiment.

Receipts: `results/decode-v9-{off,on}.jsonl`,
`results/kda-v9-served-comparison.json`, `results/gpu-kda-v9-*-a.csv`.
The comparison script's CPU synthetic tests include deliberately mismatched
tokens/settings and missing evidence, not just passing fixtures.

## Final restoration and reverse check

The immutable v5 image was restarted after the v9-on session. All 12 reverse
requests (8K/32K prose/code, three repeats each) match the original complete
request, prompt/output token hashes, speculative counts and thinking state,
with zero request errors. Reverse-run decode means:

| Workload | V5 after experiment | V9 on |
|---|---:|---:|
| Prose 8K | 33.938 | 33.622 |
| Code 8K | 44.388 | 44.144 |
| Prose 32K | 30.315 | 29.718 |
| Code 32K | 47.779 | 47.238 |

The reverse reference is v5 (the actual rollback target), not a third v9-off
session. These small sequential differences do not prove kernel causality,
but give no reason to enable the candidate. In total this round added 36 served
requests: 24 v9 on/off requests and 12 reverse-v5 requests. All matched the
prior v5 correctness/speculation evidence. No second reverse 128K run is implied.

Final state:

- Both containers `glm53-tf-fp4local-exp` run image ID
  `sha256:77a6204988f5012919aca93198a267f5cc8faafb0f49ee9848d5479a13d0fb20`.
- Original profile, `http://192.0.2.1:8890/v1`, default v5 `prefill` tuning.
- 512K context, four streams, FP8 KV, A4 prefill/A16 decode, DFlash2 confidence
  .7 and existing NFS/view arrangements unchanged. No new experiment enabled.
- At final check, MemAvailable was approximately 16.95/17.95 GiB on a/b;
  resident swap approximately 119/180 MiB. The two measured one-second idle
  intervals after vmstat's since-boot line had zero swap-in/out on both nodes.
  This is not a claim of zero swap occupancy or a full-run paging trace.
- `tfctl` remains unchanged. V9 source, controller and independent images are
  retained as default-off experiments. Existing 8888/OliveTin deployments
  were not changed.

Final comparison: `results/kda-v9-served-comparison.json`; reverse receipts:
`results/decode-v5-after-v9.jsonl`. Source archive and exact binary/image hashes
are pinned in `provenance-v9.json`.
