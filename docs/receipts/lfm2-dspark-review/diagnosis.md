# DSpark diagnosis: model inputs, fixed GPU costs, and M5 dispatch limits

**A real model-input bug is corrected; M5 target acceleration has a separate arithmetic obstacle.**
Requested proposal depth changed the drafter's bidirectional input. That correction and its maximum-context
capacity fix passed actual M4/M5 checks. Simply enabling the existing target tensor route is unsafe:
the private dispatch-only experiment failed 50 of 72 unchanged target forward/cache checks.
Neither the corrected model input nor a component timing establishes the full MLX-LM gap.

Historical causal findings below retain their original source bindings. This is not a release or performance-qualification claim.

## Serving policy after the M5-only review

Following [review6633416133](https://github.com/ashhart/TensorFold/pull/1394#issuecomment-6633416133),
serving keeps neural DSpark only for recognized Apple M5-or-newer device names. Pre-M5 and unknown names
stand aside before the LFM2 family allocates or calibrates a head; canonical target forwarding and cheap
copies remain. Other families and independent FP32 diagnostics are unchanged. Normal CLI path validation
still applies. The neural-enabled M4 matrices below remain historical failures, not new-policy clearance.
The eligible M5 path requires a full source-bound matrix with a separately declared plain/plain A/A;
[the PR](https://github.com/ashhart/TensorFold/pull/1394) links those results without relabelling older packets.

## Request-enrollment and copy-backoff findings

The [complete `750bca95c` qualification](https://github.com/BlivionIaG/TensorFold/blob/0cece53cc7be9fa67c9c792945ae3b2f17fbc6bf/docs/receipts/lfm2-dspark-review/750bca95c/README.md)
is exact on both devices but fails the maintainer's unchanged every-cell-at-least-plain gate.
Losses below 1% are still losses under that gate. Later component or smoke results do not replace this snapshot.
The following evidence and commands are recorded in [enrollment-diagnosis.json](https://github.com/BlivionIaG/TensorFold/blob/1b968d6039af141acc4876fc9c81a5fdd48dde69/docs/receipts/lfm2-dspark-review/enrollment-diagnosis.json).

- **Zero holds was not a bare-target route.** A private M4 probe based on `0b2e9336e` observed no neural
  holds for sampled code at eight streams, while the loaded wrapper still performed reservation and
  captured-context work. Reservation took 8.998 ms out of 19,551.438 ms of step wall time, about 0.046%.
  These are elapsed host spans, potentially including GPU waits, not CPU thread cycles. The quote counter
  has no duration measurement. Optimizing quote lookups alone cannot explain the percent-scale loss in this probe.
  Five diagnostic arms completed; the sixth hit real memory pressure and was excluded, without an override.
- **Copy lookup changed the decision after funding.** Asking the suffix proposer once for a reservation
  and again before any commit advanced its silence/backoff twice. The consumer regression reproduced four
  verify rows where the funded head required three. Reservations now own their decision, including empty
  results and borrowed source metadata, until consumed or invalidated by a real state change.
- **Neural context now has an explicit request-level route.** Enrollment prices full nine-query physical
  blocks, legal retained prefixes and conservative context debt against target rows. A declined request
  keeps cheap copies without a neural lane or taps. An enrolled zero-funded request retains its context
  and existing in-request recovery. Recent head-only rates are model/precision-local and greedy/sample-separated;
  they seed only fresh conditional rates, never another request's grants, confidence or probe state.
- **Native lone calibration is sparse.** The initial enrollment implementation treated one unsupported
  retained-prefix width as an unknown whole calibration and fell back to tracking the request. A regression
  through the real `Config.init` reproduced this unwanted enrollment. Like the existing depth rule, admission
  now skips unpriced candidates while retaining other complete prices; an entirely unpriced candidate set
  remains unknown. A separate before/after consumer regression preserves legacy feature-prefill capabilities:
  context stand-aside is opt-in, not an implicit change to Gemma, Minimax or CUDA providers.

At implementation snapshot `10091fa26`, the original M4 numerical/resource guards passed, followed by
27 measured 512-token HTTP requests across plain, BF16-weight and 8-bit-weight arms, all exact with clean exits.
The same source passed the original M5 cache-resume/fresh comparison and sampled equality at 1/4/8/16 streams,
plus the original cancellation workloads:

| Drafter weights | Concurrent streams | Stress duration (s) | Completed | Cancelled | Wrong replies | Errors |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 6 | 60 | 456 | 250 | 0 | 0 |
| BF16 | 16 | 90 | 941 | 508 | 0 | 0 |
| 8-bit | 6 | 60 | 458 | 250 | 0 | 0 |
| 8-bit | 16 | 90 | 930 | 503 | 0 | 0 |

These results remain bound to that snapshot, before the sparse-calibration and legacy-capability corrections.
They prove exercised behavior, not zero regression on the final source. The native FP32 comparison also remains
a reported numerical diagnostic, not proof that every 8-bit pick equals FP32. Current full performance receipts
are linked from #1394; neither these counts nor the private probe establish the MLX-LM gap or GPU occupancy.


## What is actually wrong

We have been optimizing drafter kernels before establishing how much **additional useful progress** the neural drafter buys over the existing cheap speculation. Numerical correctness and faster component kernels do not establish that adding the drafter reduces work per emitted token.

The earlier complete serving evidence proves a regression against TensorFold's own no-drafter arm on that frozen source, not on the newer correctness fixes. It does **not** establish the cause of a gap against MLX-LM: a matched MLX-LM execution breakdown has not been supplied by this investigation. Those are distinct questions.

## 1. The baseline already gets speculative tokens

The plain arm is not a naive one-token-per-round baseline. Existing copy speculation is enabled. Its accepted/drafted counters must not be mistaken for DSpark neural acceptance.

One concrete example from frozen source `7eb317eda495891a5b6d1525e566d746bd249937`: rotation 0, code, greedy, four concurrent requests, each emitting 512 tokens:

| Arm | Emitted tokens | Sum of per-request verification rounds | Accepted speculative tokens | Proposed speculative tokens |
|---|---:|---:|---:|---:|
| No neural drafter | 2048 | 1660 | 384 | 786 |
| BF16 neural drafter | 2048 | 1654 | 390 | 795 |
| 8-bit-weight neural drafter | 2048 | 1656 | 388 | 806 |

The BF16 arm records six additional accepted tokens in this example; the 8-bit arm records four. These counters combine proposal sources; they do not isolate neural acceptance or prove actual neural execution. Summed per-request rounds are not a count of physical batched GPU command buffers. This example bounds incremental aggregate progress, not every workload or the full speed gate.

The immediate question is why the neural path contributes so little beyond the cheap path: poor proposal quality, incorrect model inputs, conservative or inaccurate pricing, or work spent on recovery trials. The available aggregate counters do not yet distinguish these causes. Disabling copies to make neural acceptance look better would answer the wrong question.

## 2. A hold is substantial GPU work, not a cheap token lookup

In [dspark_block.zig](../../../zig/src/families/lfm2_moe/dspark_block.zig), `hold` first projects staged context, runs the drafter transformer, projects the hidden rows into the full 128,000-token vocabulary (`hold`, approximately lines 410–460), and applies the Markov chain and confidence work. This must save enough target execution to repay its cost.

Owned context cannot simply be skipped on zero-grant rounds: it is required by later proposals. The opportunity is to eliminate genuinely redundant work or improve useful proposals, not corrupt context retention.

The existing M5 serving traces show additional compute-encoder-active time in drafter arms, not a growing gap between encoders. That is evidence of added GPU-side work. It is **not** proof of full ALU occupancy, tensor-unit utilization or memory-bandwidth saturation.

### Actual critical-path localization

A private actual-model probe now separates the hold's phases on the M5. This is the existing 64-token
native fixture (63 prior context rows), with three real captured rows staged for absorption, one complete
nine-row query block, and either one or nine retained proposals. Activations remain BF16 in both weight
modes. Each cell below is the median of three warm captures after discarding the first:

| GPU work (milliseconds) | BF16 weights, retain 1 | BF16 weights, retain 9 | 8-bit weights, retain 1 | 8-bit weights, retain 9 |
|---|---:|---:|---:|---:|
| Context absorption | 0.190 | 0.189 | 0.130 | 0.130 |
| Five-layer transformer | 1.586 | 1.580 | 0.831 | 0.825 |
| Vocabulary head | 0.314 | 0.311 | 0.313 | 0.314 |
| Markov chain and confidence | 0.088 | 0.501 | 0.056 | 0.358 |
| Embedding and final normalization | 0.009 | 0.009 | 0.009 | 0.009 |
| Matching uninstrumented command buffer | 2.167 | 2.574 | 1.385 | 1.627 |

**The transformer is the dominant measured phase.** Reducing the retained prefix does not remove its
fixed cost or the full vocabulary-head projection; it mainly shortens the Markov chain. A scheduler
cannot price a one-proposal hold as one ninth of the backbone. The fixed-query correction already
updates the clean quote grid and physical-group funding rather than assuming linear cost per proposal.

Changing the weight format nearly halves this fixture's transformer cost while the tied head remains
similar. That establishes sensitivity to weight format and its selected kernels, not a proof of bandwidth
saturation. Source inspection found that at nine physical rows, BF16 projections with at most 2,048
outputs still use the row kernel; wide gate/up projections already admit NAX at eight.

**Tested, not assumed:** a private follow-up changed only the narrow BF16 NAX threshold from 16 to
nine. It retained 9/9 FP32-reference picks and all original native controls, but the warm captures
regressed:

| Retained proposals | Transformer GPU-time change | Matching uninstrumented hold GPU-time change |
|---|---:|---:|
| 1 | +53% | +40% |
| 9 | +54% | +34% |

The existing threshold is retained. More tensor dispatch is not automatically more throughput.
The [BF16 dispatcher](../../../zig/src/families/lfm2_moe/dspark_bf16.zig) launches only 16 NAX
threadgroups for a nine-row, 2,048-output product (four for 512 outputs). **[INFERENCE]** This small
grid may limit parallelism; core occupancy was not measured. A smaller column tile or shape-fixed
split-K is a candidate to investigate after isolating the individual expensive products. Simply
lowering the threshold is rejected. These sequential fixture captures are not a new speed qualification.

**Measurement limits:** the M5 exposes stage-boundary counters, not dispatch-boundary counters.
The private probe preserves concurrent encoders and inserts fenced boundaries at existing phase
dependencies. Two-point calibrated stage timestamps are used; stage rotation can perturb execution.
Every profiled base-logit, requested-pick and confidence byte matched its paired uninstrumented hold.
The instrumented command-buffer total timestamp was invalid/absolute-looking and is excluded;
the last table row comes from separate uninstrumented buffers. Phase medians are not additive exact
totals. This is one native fixture, not serving qualification, matched MLX-LM attribution, or occupancy.

## 3. Target M5 routing crosses an arithmetic-compatibility boundary

In [forward.zig](../../../zig/src/families/lfm2_moe/forward.zig):

- `tile_rows` is 33.
- `segments` selects prompt/tiled mode with `(!record and head == .last) or total >= tile_rows`.
- `Encoder.nax` returns null outside that mode.
- `Encoder.projectAll` takes `encodeTwin` early when outside that mode.

The target admits at most 32 verification rows (`state.zig`), so the 33-row threshold makes this M5 NAX route unreachable for every admitted recorded verification call. `encodeTwin` can itself select an older SIMD-group MMA window kernel; bypassing NAX does **not** mean every such operation is pure scalar arithmetic.

The drafter uses NAX from five rows for its vocabulary head and nine for other wide products. Copying
those thresholds into the target is **not a safe correction**: production source
`174a74681920cbaefc3d3e65a9cd18c5e0f15a5a` passed the unchanged actual M5 target oracle **72/72**;
private dense-only candidate `b99a794df` passed its build/family checks but only **22/72** target
forward/cache checks. Expert and prompt routing were retained. The candidate is excluded from publication.

The installed comparison is MLX-LM 0.32.2 with MLX 0.32.4. Version-matched [MLX dense dispatch](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L89-L125) selects by shape and architecture, rather than a verification flag. Static inspection also found native target serialization of independent projections, eager state/logits commits, attention dispatch tied to prompt mode and a redundant CPU argmax in the one-token queue path. Their runtime contributions are not measured; rollback state cannot be deleted, and the redundant argmax is not on the greedy verification path.

The first newly eligible seven-row window changes only the six-bit tied head. A real-input localization
replayed its exact final-normalization BF16 rows and released head weights through one-row NAX,
seven-row NAX and the established one-row route, with a separately unpacked FP64 reference:

| Head computation on identical captured inputs | Differences from canonical BF16 outputs | Printed worst normalized FP64 error |
|---|---:|---:|
| Canonical one-row route | 0 / 896,000 | 1.151e-3 |
| NAX, original epilogue, one/seven rows | 57 / 896,000 | 1.151e-3 |
| NAX, canonical nested-FMA epilogue, one/seven rows | 4 / 896,000 | 1.151e-3 |

Within each NAX variant, the two widths match byte-for-byte: this exposes a **cross-route arithmetic
mismatch**, not a seven-row packing failure. The error is normalized by
`sum_groups(abs(s * D) + abs(b * T))`; the printed summaries have only three significant figures.
They do not make the BF16 outputs identical.

A second private experiment changed only the group64 epilogue to the canonical nested FMA order.
Its reduced mismatch count isolates accumulation order as a major cause, but not the entire cause.
The remaining first difference is immediately across a BF16 rounding boundary: FP64
`-2.976563190`, canonical `-2.984375`, NAX `-2.96875`. This neither establishes a target token change
nor licenses waiving the original target oracle.

The source has different group evaluation graphs
([canonical twin](../../../zig/kernels/metal/core/tile_qmv.metal),
[tensor product](../../../zig/kernels/metal/core/affine_mm.metal)):

```cpp
// Established decode group:
acc = fma(b, sum, fma(s, dot, acc));
// Current tensor group:
acc += s * dot + b * sum;
```

The matrix primitives also differ. Six-bit codes have no four-bit `+128` offset, so that offset cannot
explain this first head-only case. Passing ordinary FP64-error bounds is not permission to replace the
target's canonical per-row arithmetic. MiniMax explicitly maintains the same target/drafter separation;
Gemma's fixed tensor schedule is a useful design precedent, not a drop-in equivalent to LFM2 decode.
An additional captured-input check found **all 224 GPU row sums bit-identical to canonical ordered
FP32 FMA sums**, while the four final head differences remained after epilogue alignment. This excludes
row-sum accumulation for this case; the remaining dot/matrix primitive is the next discriminator.
The shared shader experiments remain diagnostic only and are not published.

## 4. Fusion can make utilization worse

The drafter already encodes independent gate/up projections in a concurrent encoder, then places barriers at dependencies. Eliminating one dispatch is not automatically an improvement: a fused kernel can increase live registers, reduce available parallel work or serialize previously overlapping products.

Two new fusion candidates passed actual M5 numerical suites and completed six alternating real-weight pairs each. Their current general policies are rejected:

| Private fusion policy | Measured eligible shapes | Paired median GPU change |
|---|---|---:|
| 8-bit low-row fusion | All nine shapes slower in all six pairs | +10.5 to +65 microseconds |
| BF16 partial-tile fusion | Rows 8/9/12/24 slower in all six pairs | +69.5 to +93.5 microseconds |
| BF16 full-tile cases | Some 32-row shapes improve; not a general-policy win | −24 to −31 microseconds |

These are component timings, not a serving or MLX-LM comparison. They establish that reducing dispatch count made important paths slower. Register pressure, padded work and lost projection overlap remain possible mechanisms, not measured occupancy findings. Neither experiment is in the published fixes.

## 5. Confirmed released-model input discrepancy: the query block is shortened

The released [checkpoint configuration](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark/blob/e052e5b1dd103c38cdd4f8ddc068d7dea4586f55/config.json) has **nine query rows**. Its five layers use bidirectional/full attention. The pinned official [proposer](https://github.com/sgl-project/sglang/blob/488869c2d/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L354-L411) constructs the configured query block independently of a scheduler's eventual retained prefix.

Before correction `e06ce301`, native `hold` set its query width to the largest requested proposal depth in the current physical pass. A one-proposal request got only the pending row; a three-proposal request got pending plus two masks. The same short request beside a nine-proposal request got all nine query rows. Earlier queries attend these later mask rows, so shortening the block changed the attention computation and could make a proposal depend on its batch companions.

This is a **reproduced model-input discrepancy**, not a demonstrated cause of the acceptance regression. On actual M4 released weights, private before-source `da8029b5b09317c5b668a3b7e33cbf41c92c95c9` failed the full first-row base-logit prefix invariant (`DraftPrefixDependsOnGrant`). Official SGLang permits an explicitly configured alternative block size; what is unsupported is treating a dynamically shortened noncausal block as prefix-preserving pruning of the configured nine-row computation.

Correction `e06ce30116c0a77017136ebacc4e8f15ce671def` separates configured query width from returned proposals and updates physical packing, clean/context funding and the reference together, without increasing grants. The funding regression rejected the fourth short head when four fixed blocks require a second physical pass. The core suite changed from one new failing regression to 391 passing tests and one platform skip. On M4, requested prefixes 1/3/9 in both precisions preserve first-row base-logit bytes, first proposal and confidence, and all FP32 picks matched in these cases. Real 8/16-logical-cohort controls passed. M5 passed the original numerical/resource contracts and both native precision scenarios, including the new invariants; its BF16 reference matched 9/9 picks, while 8-bit matched 8/9. These are source-bound correctness checks, not serving or MLX-LM speed evidence.

BF16 activations are compared with a separately implemented FP32 host reference; explicit 8-bit mode changes weights, not activation dtype. Previously, that reference used native captures and reproduced the shortened query block. It now runs the full configured query block and compares the requested proposal prefix. It still does not independently establish every target-feature value against an official target implementation.

The official audit found no indexing, raw-versus-normalized capture, prompt/verified temporal shift, RoPE-layout, matrix-orientation or Markov-input mismatch in the inspected paths. It did find a drafter RMSNorm rounding-order difference and a different sampled-proposal policy; their effect remains unproven. Those are separate from the query-block correction, and do not justify silently changing target mathematics.

**Maximum-context capacity defect reproduced and corrected:** the full query block can extend beyond
the target's 131,072-key span. Partials and merge-local arrays previously held only 32 chunks of 4,096
keys. On private before-source `a6164114c3e5cc57cbfa83fa7a072eb9d295732b`, an actual GPU fixture at
131,077 visible keys overwrote the next allocated scratch row (`0x7fc12345` became zero). Correction
`174a74681920cbaefc3d3e65a9cd18c5e0f15a5a` reserves a drafter-only 33rd chunk; target attention and
the advertised context window remain unchanged. M4 and M5 passed the guard-row and analytic tail-value
regression, original attention/BF16 numerical and resource bounds, and actual released-model scenarios
in both precisions. This boundary fixture is not a new full-model 131k-context qualification.

The corrected-capacity source also passed all six M4 requested-prefix scenarios (1/3/9, BF16/8-bit):
every requested pick matched its FP32 reference in these cases. M5 BF16 matched 9/9; 8-bit matched 8/9,
with largest differing-pick reference margin 0.075 and worst confidence difference 0.00225. These are
drafter-reference results, not permission to tolerate different final target tokens.

## Work order from here

1. Compare independently captured official/native target features and drafter intermediates. The FP32
   arithmetic reference alone cannot establish that every model input is the intended one.
2. Develop a target tensor replacement compatible with the established one-row computation; preserve
   M4 behavior and the original byte/token/cache checks. The simple threshold-copying experiment is rejected.
3. Split the measured dominant transformer phase into its projection/attention/MLP work, then test
   narrow-product tiling or shape-fixed split-K. Merely lowering the existing narrow BF16 threshold
   already regressed. Separately isolate neural incremental acceptance from cheap copies. Hardware
   occupancy and matched MLX-LM phase attribution remain unmeasured; stage timestamps establish neither.
4. Retain only changes that remove identified waste. Resume full qualification after a causal fix,
   rather than using repeated full matrices as diagnosis. Publication does not mean merge-ready.

## Source and publication provenance

- Earlier complete serving evidence: `7eb317eda495891a5b6d1525e566d746bd249937`; all token checks passed, but 21 of 32 speed comparisons failed. This report does not relabel that evidence to newer heads.
- Initial execution-path inspection: `8fb1c35c417ffc1011ad62220ad6b23a4141accb`. Fixed-query implementation and M4/M5 checks belong to `e06ce30116c0a77017136ebacc4e8f15ce671def`; subsequent documentation or comment-only changes do not inherit a different source label.
- Maximum-context before/after: `a6164114c3e5cc57cbfa83fa7a072eb9d295732b` → `174a74681920cbaefc3d3e65a9cd18c5e0f15a5a`. The before M4 log SHA-256 is `27d50388cf2fe6fa8b7f16e39015758879acace88c55df8a6a08119357dc02c6`; the after M4 build/units log is `54418ea32f23fd47d695095bac21fac8bf406476ad495780667d2cee228f0bf8`, and the M5 build/units log is `6f6079c30e6f82c7f7f67d3a3a99017f9b0b0394533450d72d624b931c01a5ec`. Original numerical bounds and controls were not relaxed.
- #1391: `caf9282b8c1f85d4a8dd7f66c09feaf89053d977`; #1392: `5d81437d0f16c9c52ed60f095e4c26a16c498592`; #1393: `5e7ae48da075f4e318c4d137aa8d3f53cd80f707`. All three were fast-forward pushed after their own 76/76 scoped build/check steps (251 passed, five platform skips), original attention/BF16 contracts and actual M4 BF16/8-bit native scenarios. Each mode matched 9/9 FP32 picks and passed prefix/split/lifecycle controls; the depth-only stage does not claim confidence coverage. Their code-parent CPU suites passed 341/366/365 respectively, each with one skip.
- #1394 correctness/report publication: `5215a8b3588e9ad8992846111250e505081a9935`. Its own native/CPU build passed 60/60 steps, 391 consumers and one platform skip, plus actual CLI help. Production GPU receipts retain `174a7468`; that source also passed the unchanged M5 target oracle 72/72. All four retained 424 inherited lean findings and zero new.
- Exact-head CI on those four publications was observed cancelled, not green: #1391 has one successful formatting check and ten cancellations; the other three have eleven cancellations each. Workflow concurrency is scoped by PR/ref, so the configuration does not explain cross-PR cancellation. Cause is unproven. No gate, workflow, test skip or permission retry was introduced.
- Target arithmetic localization: private probe `751656d464635bd38127b39bfc1e660b034be033`, log SHA-256 `4969548bc69dc94b7e8812101b55f8c3d1ecb9e991d468837ebcd52c46724b76` on the M5. Input: the first seven tokens of the existing exact fixture, zero prior context, canonical target forward, its final-normalization BF16 rows; head 128000×2048, six-bit/group64. The original independent FP64 unpack/reference was reused. This is localization, not a replacement for the failed full oracle.
- Epilogue-only localization: private `5b90d6232f3ef1b6c640b80419ad23dba185e8e5`, log SHA-256 `944eda8b134563196c9c3e66798a2254e8f2d1a5d7163e2e55d38664f5a6ad68`, binary SHA-256 `0187669099bb50aa72b3143cc598441681b724f398e21f01c3ad1384fc8c9a8a` on the M5. Same captured inputs, every vocabulary output checked, no source-threshold or oracle change.
- Row-sum exclusion: private `e5718a4c53f18eeef72899b58c1128ff9bb40cf5`, log SHA-256 `241eed316cbe253e0fa221acfb9a5eea22e3d08cdec92da3a99102f75986c103` on the M5.
- Actual phase attribution: private `d6ae603903e91aecd146386bf6214078e2b292d8` on the M5; BF16 log SHA-256 `1188eab48ab4b1e71ec4f47a4046a4dec4d061d6cc08cf85488fcc72b5a7cdf3`, 8-bit log `1769ea2b6d1e702c0a184bbd32e9cfe6ed377c7ab6bf065bf337ed717581a09e`. Original native controls ran first and passed, including zero host allocations/resizes in the 20 steady rounds; BF16 matched 9/9 FP32 picks and 8-bit matched 8/9 with the previously stated bounds. An initial empty-context counter attempt (`4d3e1df7d`) returned `InvalidCounterSample` and supplies no phase timing; the published table uses only valid nonempty-stage captures.
- Narrow BF16 tensor follow-up: private `0fdcedf12733466ba7aa28d45b225d4a3bacda2b` on the M5; actual native log SHA-256 `6292b344fe16e7b2cde13834e0ad2601b3ec2697b21d85d75c9e190c99ae601f`. Same staged real-input fixture and phase sampler; FP32 picks 9/9, worst confidence difference 0.00129, all profiled outputs identical to their matched unprofiled runs. Rejected and not published.
- No new full serving matrix is being used as a substitute for the missing causal explanation.

## Scheduled-cohort follow-up: one enrollment defect fixed, performance still open

The complete `cf23762368785b470b3bf7485026e2ea4837ce82` matrices supersede the older
qualification numbers above, not their historical source labels. M4 has **12/32** strict losses
(worst **-5.322375682894121%**); M5 has **16/32** (worst **-1.4341478593879908%**).
Both completed 144/144 cells and 306/306 token oracles without exactness failures.
The [immutable full packet](https://github.com/BlivionIaG/TensorFold/blob/4488fea6ba35edef9271931d0b0678f9bb0fa27d/docs/receipts/lfm2-dspark-review/cf2376236/qualification/README.md)
contains all 64 comparisons and the unchanged zero-tolerance verdicts.
The [GPU/cohort diagnosis](https://github.com/BlivionIaG/TensorFold/blob/2a61f7a96b8a47fd1b92712e366fab81296d93bc/docs/receipts/lfm2-dspark-review/cf2376236/prefix-attribution/README.md)
separates observed work from unproven kernel and MLX-LM hypotheses.

**Confirmed defect:** LFM2 prefills a whole prompt during `admitOne`, before taking the next queued
job. Pricing only the drafter's registered stream map can therefore price the first request of a
known queued cohort as a lone request. The host now supplies its already admitted/queued lane bound
before prefill; enrollment uses the larger count. No new allocation, mutex, artificial delay,
stream-count cutoff, activation dtype or target arithmetic is introduced.

The [source-bound receipt](https://github.com/BlivionIaG/TensorFold/blob/1b968d6039af141acc4876fc9c81a5fdd48dde69/docs/receipts/lfm2-dspark-review/scheduled-cohort.json) records the three-file runtime delta from CF,
the actual M5 serving replay and M4 guards. In the measured four-stream M5 case, BF16 changed from
one tracked context to none; 8-bit also declined all four. All three diagnostic servers exited 0
without forced termination. Each completed 46 phases and retained the original request/token
contracts; measured target tokens matched plain. A final diagnostic parser failed after the
third server exited because its validation tuple shadowed `fields()`. Offline recovery used the
saved logs and requests, exactly reproduced both already-persisted arm records, and did not rerun
the models. The executed controller failure is retained in the receipt.

**Not solved:** M5 four-stream target GPU time remains higher with every context declined.
Plain used 1,660 rounds / 786 copied drafts / 384 accepted drafts; both loaded-head modes used
1,662 / 775 / 382. Those are different target workloads, so the remaining loss is not attributable
solely to feature capture. At wider widths a first request can still arrive before its peers:
an already-queued hint cannot predict future arrivals. These fixed-history diagnostics are not
a replacement for either full qualification matrix.

M4's candidate passed the original 82-step scope: 763 of 771 tests passed, eight skipped,
all six original commands exited 0, and both native weight modes matched 9/9 FP32 picks.
Activations remain BF16; 8-bit changes weights only. The FP32 reference consumes BF16 target taps,
not an FP32 target server.

The profiled M4 long-context/cache functional run is **incomplete**: six-slot stress completed
149 requests and 87 cancellations without wrong answers/errors, then the original memory-pressure
guard aborted sixteen-slot stress. The owned server exited 0 without a forced kill. Sampled and
fresh-cache checks were not reached; neither a retry nor a memory override is counted as success.
Its 30.617045-second Metal System Trace recorded 505 native compute intervals totaling
28.519554947 seconds of union activity (93.1493% of the trace), while other processes' recorded
GPU activity totaled only 0.014343135 seconds of union activity. Native Metal allocation stayed
between 14.120925903320312 and 14.144302368164062 GiB during the captured interval.
This is evidence against a large CPU/GPU idle gap in that profiled workload, **not** SIMD occupancy,
achieved bandwidth, or proof of the memory-pressure cause. The intrusive trace is not performance
clearance. The earlier clock capture retained 306 complete samples and an incomplete tail, but lost
its wrapper clock anchors and has no per-process GPU fields or complete three-arm coverage.

Review readiness still requires both unchanged full matrices on the new exact head and green
exact-head CI. The CF GitHub metadata was DIRTY/no-CI despite verified pinned-base ancestry.
No tolerance, re-review request, force push or merge is implied by this follow-up.

## Admitted-residency calibration: corrected measurement context, performance still open

The [causal copy-price and GPU-class packet](https://github.com/BlivionIaG/TensorFold/blob/0c87e639ac2f65809ad30a4a422c2ba2a0024d48/docs/receipts/lfm2-dspark-review/2acdb56a2/copy-gpu-attribution/README.md)
diagnoses pushed source `2acdb56a2`, not a new qualification. In the captured M4 four-stream case,
every neural context is declined and all copy offers/probability bits agree. The real allocator chooses
three drafts per stream with plain startup prices and six with BF16 prices. Swapping the price curve
alone reproduces that decision; swapping measured fixed overhead does not. BF16 accepts more drafts
but its wider batches spend more target GPU time. This explains that workload divergence without
claiming the older M5 all-declined case or every remaining performance loss is explained.

Startup target prices previously preceded normal serving residency and loaded-head enrollment.
Calibration now follows admission and residency, with temporary cache buffers following the same owner.
The existing idle-flush callback commits pending buffers before timings; cache teardown unregisters them.
Tap destinations are prepared afterward, preserving bare-target calibration instead of capturing into
temporary caches that have no stream destination. Target math, BF16 activations, admission limits,
deterministic calibration history, warmup and timing-trial counts are unchanged.

The first local ordering candidate, `5d9469fe5`, passed original kernel/reference guards on both devices
but failed actual native startup with `BadLfmTapDestination`; it was never a published PR tip.
Corrected source `79837c4e2` passes the original 82-step scope on both devices and the unchanged
long-context/cache, fresh-cache, sampled and cancellation workflows in both head-weight modes.
The [source/binary-bound receipt](https://github.com/BlivionIaG/TensorFold/blob/1b968d6039af141acc4876fc9c81a5fdd48dde69/docs/receipts/lfm2-dspark-review/serving-calibration.json) retains exact commands, skips, sample hashes,
FP32 comparisons, cache counts and all eight clean non-forced exits. M4's build summary counts
355/362 newly run tests, seven skipped, with the shared-allocation test cached; M5 reports 767/771,
four skipped. M4 matches 9/9 FP32 picks in both modes; M5 matches BF16 9/9 and 8-bit 8/9, with
0.075 maximum margin for the different 8-bit pick. That result is not relabelled as universal equality.
Stress completes 818 requests and 468 cancellations on M4, 2,788 and 1,514 on M5, without wrong
answers/errors. These counts are observed correctness workload totals, not speed comparisons.

Calibrated operator-class captures on the preceding source put expert gate/up and down projections
first in the larger target batches. All 138 original counter-off/on phases per device retain identical
target tokens. Serial encoder fences are intrusive; invalid command-buffer stopwatches are excluded.
Neither GPU occupancy, bandwidth saturation nor a matched MLX-LM comparison is established.

Correcting the measurement context and passing native correctness do not establish the zero-regression
gate. The later frozen full qualification is the
[`f7eb5f0bc` packet](https://github.com/BlivionIaG/TensorFold/tree/f615b6d2fc611550952651fec1d24541b06968bf/docs/receipts/lfm2-dspark-review/f7eb5f0bc),
not the earlier CF matrix. Both devices completed all 144 cells and 306 original oracle checks, but both
original performance summaries still exited 3 (`FAIL`). M4 put 2/32 comparisons inside the owner's observed
less-than-1% budget, with a worst median loss of 8.7149%; M5 put 32/32 inside it, with a worst loss of 0.3289%.
These are not zero-regression passes under the maintainer's rule.

The predeclared controlled M4 study completed all 24 launches and 8,352 requests. Its observed medians put
31/32 comparisons inside that budget, but only 2/16 A/A intervals established the prescribed quiet baseline;
the conclusion remains **INCONCLUSIVE**. Neither extra warmups nor a point median replace the frozen failures.

## Unpaid-admission follow-up

The financial admission path still allowed two unfunded routes: an unknown quote enrolled the serving
request, and an unpaid recent history could spend a recovery probe when perfect acceptance would pay.
That is a policy defect, not a claim that the remaining timing differences are measurement error.

Serving enrollment now requires a complete quote that pays at the existing conservative acceptance rates
and unchanged margin. Cold paid trials retain their bounded request cadence; unknown or unpaid requests
keep the target-only route without a neural lane or taps. A later price change or funded head observation
must actually repay the work before enrollment resumes. Fully unbound direct diagnostic callers remain
explicitly available; their availability no longer makes missing serving calibration free.

This trades unpaid exploration for conservative stand-aside. It does not change BF16 activations, stored
weight modes, target arithmetic, the pricing margin, memory admission or the original qualification protocol.
The [new frozen packet](https://github.com/BlivionIaG/TensorFold/blob/9bb1c385d5d663ef800ad2dd231c3dfe43ed1e9c/docs/receipts/lfm2-dspark-review/5fdbfb874/README.md)
records the completed qualification workflow for runtime source `5fdbfb8749b14bc20bc9021dbe463cd56206d53a`, after integrating pinned main
`7eae7dc008ca8f280b3557596e7f8d425693b12d` and moving raw JSON to the evidence branch.
Both chips completed all 144 cells and 306 token oracles, with no token/scenario failures.
All 11 CI checks passed on that source; neither matrix passed the original speed gate.

| Device | Negative comparisons | Inside the owner's observed <1% loss budget | Worst median loss |
|---|---:|---:|---:|
| M4 Pro | 13/32 | 20/32 | 4.489518% |
| M5 Ultra | 10/32 | 32/32 | 0.445179% |

The owner's budget is not the maintainer's zero-regression rule. Recorded variability is not a measured
noise floor. The packet preserves all 64 comparisons, raw requests, byte-identical verdict replays and failures.
Both chips passed cached/fresh equality, all 58 sampled-hash triples and served contexts through 130,816.
Stress completed 820 requests and 465 cancellations on M4, 2,776 and 1,506 on M5, without wrong replies/errors.
The original M4 native 130,816 diagnostic failed GPU OOM before the FP32 comparison; its 65,536 case
and both M5 native long cases passed. This is not a serving long-context failure or a numerical pass.

### Native diagnostic cache lifetimes

The native diagnostic retained independent target caches after their last forward, overlapping later
capture/stage controls and the main comparison cache. Diagnostic-only source
`763c8678cc680a89ff0104cde4bb8b59cd9fa766` releases those controls' caches earlier and creates the main
cache after capture controls. It preserves every row count, capture layout, comparison and numerical bound.
Source763's short BF16 and 8-bit runs each match 9/9 FP32 picks, with worst confidence differences
0.00085 and 0.00152, but its unchanged maximum-context case still failed GPU OOM after 2,097.326 seconds.
Follow-up diagnostic source `b57b839119a10eedfd1cb0909c1e660d4a5abffe` also defers the main tap buffer
until controls finish and completes table-only stage controls before allocating the other stage buffer.
It retains the complete fixture and labels the active control. Both short weight-mode cases exit 0,
match 9/9 FP32 picks and retain the same confidence differences. The unchanged maximum-context case
still fails GPU OOM after 2,341.274 seconds, at the first full eight-column staged forward in twin mode.
The full-length bare target, five/three-column controls and eight-column table-only control finish first.
The FP32 drafter comparison is not reached. Production sources and the actual M4 production binary remain
identical to the full-matrix runtime; no full diagnostic pass or measured peak-memory reduction is claimed.
