# Native runtime optimization audit

Kernel coverage and production scheduling are separate claims. The embedded catalog
executes all 86 original Metal kernels; this does not imply that every Python scheduling
mode is integrated into the native completion driver.

## Buffer ownership

Alternating buffers are implemented for all three targets and both MTP heads.
The isolated donation check now drains the stream before asserting address reuse:
MLX's Metal `eval.cpp` retains input Data in command-buffer completion callbacks,
while `array::is_donatable` requires sole ownership. Host-readable outputs alone
do not establish callback cleanup. Without the drain this test failed intermittently.
Production inference has no added drain and may legitimately copy a busy buffer.

All 508 drained writes reuse their exact allocations. Another 256 generated
histories compare against independent concatenation while retaining snapshots;
partial commits, growth and rollback to empty caches pass. Real-model observations
reuse 992/992 Qwen and 384/384 Nemotron buffers on each backend, plus 1,152/1,152
Flash buffers. These observed counts are not a guarantee of donation on every run.
All 15 independent buffered/unbuffered prefill-cache and verification-logit
comparisons pass at 0/31/2,044 tokens, across five model/backend combinations.
Another 11 long-context comparisons pass: Qwen/Nemotron tensor and SIMD at
9,999/10,007 tokens, and Flash at 2,044/2,051/2,063 tokens.
All 904 injected host allocation failures pass with zero retained MLX memory.

All 1,056 short accepted-prefix checks, 23 buffered/concatenated CLI pairs and
414 MTP state checks pass. Serial pipelining passes 48 real-model comparisons and
35 synthetic EOS/budget cases. The 640 cache and 256 pipeline reset cycles retain
flat active MLX memory. All 400 long-context accepted-prefix/cache/continuation checks
also pass, including the 128-row Qwen window and Flash's sparse threshold rollback.

## Committed runtime status

| Original behavior | Native status | Evidence / remaining work |
| --- | --- | --- |
| Reduced MTP vocabulary | Default for both families; full-head override | All 32,768 Nemotron and 79,592 padded Flash IDs and packed head rows match original Python selection. Both samplers map columns back to original IDs. |
| Queued dependent MTP chain | Default with Metal sampling; per-token host override | A whole chain is evaluated before reading proposal IDs. `test-mtp-runtime` checks proposal hashes as well as final tokens. |
| Early MTP speculation and reuse across rounds | Default with Metal sampling; late override | Target samples feed batched MTP before the verification host read. The kept cache prefix, hidden state and first draft are reused. Real MTP batch/serial and all-prefix checks pass through 10K. |
| Adaptive MTP depth | Default; fixed-depth override | Native startup measures target windows and a head step; the original acceptance/cost policy updates from observed rounds and probes deeper every eight choices. All 12,288 policy fixtures match Python. |
| MTP sampling positions | Corrected to the original convention | The first draft after a pending token at position P uses P+1; early speculation from target row P uses P+2. All 360 fixtures from original speculate/settle pass. The former native chain used P+2 too early, reducing acceptance without changing target output. |
| Pipelined serial decode | Default for Qwen/Nemotron with Metal sampling and no neural drafts; enabled for Flash with resident PLE | The next target step is submitted before reading the current token. Deferred cache graphs are drained before returning; queued EOS suffixes stay uncommitted. Qwen/Nemotron pass 96 full-model comparisons through 10K, 32 CLI pairs and 35 terminal GPU fixtures. Flash's bounded PLE default remains synchronous; its resident path also drains PLE history, convolution and sparse caches. |
| GPU proposal handoff to the next target pass | Default for queued Nemotron Metal MTP and resident Flash; host override | Both feed lazy proposal arrays directly into target verification and read all draws together. Nemotron passes 48 focused completion comparisons and 36 scheduling comparisons on tensor/SIMD. Resident Flash computes n-gram IDs and token history on the GPU; bounded Flash retains positional file reads. |
| Alternating/preallocated attention cache writes | Default for all three targets and both MTP heads; concatenation override | Buffers grow in 2,048-row blocks and protect snapshots through MLX ownership. Flash also buffers raw index keys. Qwen branched commits compact accepted paths. Full-capacity inputs avoid hidden contiguous-prefix copies in Qwen/Flash attention. `test-kv-buffers` and `test-kv-runtime` compare against the old concatenation path. |
| Stacked SIMD Qwen projections | Diagnostic coverage; native production uses original unstacked weights | Shape-dependent SIMD reduction splits make these different rounding configurations. Both kernel variants execute; the native unstacked configuration matches its original Python oracle at short and 10K context. Do not label it byte-identical to the stacked Python serving configuration. |
| MTP projection arithmetic | Native uses row-exact projections | Original Nemotron MTP deliberately uses plain MLX matmuls; Flash switches its hidden projection to MLX above 32 stream rows. Native keeps row-exact kernels, splitting Flash's 64-stream-row windows. This preserves native proposal parity across batching; do not claim bit-identical proposal logits to those Python matmul configurations. Every emitted token is still checked by the target. |
| Flash PLE resident tables | Optional original eight-group packed lookup; bounded positional reads remain the default | Donated chunk updates avoid holding source shards and a second concatenated 32 GB table. Resident mode enables GPU hashing/history and pipeline scheduling. The MLX wired budget covers the measured loaded weights, capped by Metal's recommendation; the previous process budget is restored at shutdown. |

Sources: `engine/lane_family.py`, `engine/lane_engine.py`,
`families/nemotron_h/model.py`, `families/qwen4_exp/runtime.py`,
`families/qwen4_exp/draft_head.py`, and `families/qwen3_5/__init__.py`.
The ID lists are the original repository files, embedded directly by `build.zig`;
there is no independently maintained duplicate or runtime Python dependency.

Resident Flash qualification passes 34 serial/MTP comparisons and 26 exact proposal
schedule comparisons, 51 resident/bounded state comparisons across sparse thresholds,
40 short/long full-model pipeline comparisons, and eight CLI budget pairs. All
384 integer GPU hash windows and 640 resident lookup rows match their oracles.
The packed loader donates all 14,976 updates, peaks 944,140 bytes above its table
size, and retains zero MLX bytes after cleanup. All 956 injected allocation failures
pass. Bounded Flash and Qwen/Nemotron pipeline regressions also pass; detailed
counts and reproduction commands are in [COVERAGE.md](COVERAGE.md) and [README.md](README.md).

Final performance comparisons must run the original engine entry points with their
production optimizations enabled and disclose these differences. Correctness oracle
scripts use controlled prefill grids and are not the performance baseline. Physical
M1–M4 qualification and failures inside MLX/Metal's private allocators remain outside
the execution evidence available on this Mac.
