# Native runtime optimization audit

Kernel coverage and production scheduling are separate claims. The embedded catalog
executes all 86 original Metal kernels; this does not imply that every Python scheduling
mode is integrated into the native completion driver.

| Original behavior | Native status | Evidence / remaining work |
| --- | --- | --- |
| Reduced MTP vocabulary | Default for both families; full-head override | All 32,768 Nemotron and 79,592 padded Flash IDs and packed head rows match original Python selection. Both samplers map columns back to original IDs. |
| Queued dependent MTP chain | Default with Metal sampling; per-token host override | A whole chain is evaluated before reading proposal IDs. `test-mtp-runtime` checks proposal hashes as well as final tokens. |
| Early MTP speculation and reuse across rounds | Remaining integration work | Python `FamilyRounds._family_round` feeds target samples into MTP before reading target output, then `settle` retains the accepted prefix. Zig currently commits MTP context after verification and constructs the next chain separately. |
| Adaptive MTP depth | Remaining integration work | Python `_depth` combines per-depth acceptance with measured window/MTP costs and periodically probes deeper. Zig currently uses the configured fixed budget, bounded by remaining output capacity. |
| Pipelined serial decode | Remaining integration work | Python `_queue_next` submits the next target step before reading the current token. Zig serial decode currently reads each token before building its next step. |
| Alternating/preallocated attention cache writes | Remaining optimization work | Original family runtimes adopt alternating KV buffers. Native concatenation and accepted-prefix slicing have exact cache/rollback coverage but different allocation and copying costs. |
| Stacked SIMD Qwen projections | Diagnostic coverage; native production uses original unstacked weights | Shape-dependent SIMD reduction splits make these different rounding configurations. Both kernel variants execute; the native unstacked configuration matches its original Python oracle at short and 10K context. Do not label it byte-identical to the stacked Python serving configuration. |
| Flash PLE resident tables | Native uses bounded positional packed-row reads | All 128 shard boundaries and long-context integration checked. This is an intentional memory strategy allowing the 113 GB checkpoint to run on this 128 GiB Mac. |

Sources: `engine/lane_family.py`, `engine/lane_engine.py`,
`families/nemotron_h/model.py`, `families/qwen4_exp/runtime.py`,
`families/qwen4_exp/draft_head.py`, and `families/qwen3_5/__init__.py`.
The ID lists are the original repository files, embedded directly by `build.zig`;
there is no independently maintained duplicate or runtime Python dependency.

Final performance comparisons must run the original engine entry points with their
production optimizations enabled and disclose these differences. Correctness oracle
scripts use controlled prefill grids and are not the performance baseline. Physical
M1–M4 qualification and failures inside MLX/Metal's private allocators remain outside
the execution evidence available on this Mac.
