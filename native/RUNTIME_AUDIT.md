# Native runtime optimization audit

Kernel coverage and production scheduling are separate claims. The embedded catalog
executes all 86 original Metal kernels; this does not imply that every Python scheduling
mode is integrated into the native completion driver.

| Original behavior | Native status | Evidence / remaining work |
| --- | --- | --- |
| Reduced MTP vocabulary | Default for both families; full-head override | All 32,768 Nemotron and 79,592 padded Flash IDs and packed head rows match original Python selection. Both samplers map columns back to original IDs. |
| Queued dependent MTP chain | Default with Metal sampling; per-token host override | A whole chain is evaluated before reading proposal IDs. `test-mtp-runtime` checks proposal hashes as well as final tokens. |
| Early MTP speculation and reuse across rounds | Default with Metal sampling; late override | Target samples feed batched MTP before the verification host read. The kept cache prefix, hidden state and first draft are reused. Real MTP batch/serial and all-prefix checks pass through 10K. |
| Adaptive MTP depth | Default; fixed-depth override | Native startup measures target windows and a head step; the original acceptance/cost policy updates from observed rounds and probes deeper every eight choices. All 12,288 policy fixtures match Python. |
| MTP sampling positions | Corrected to the original convention | The first draft after a pending token at position P uses P+1; early speculation from target row P uses P+2. All 360 fixtures from original speculate/settle pass. The former native chain used P+2 too early, reducing acceptance without changing target output. |
| Pipelined serial decode | Remaining integration work | Python `_queue_next` submits the next target step before reading the current token. Zig serial decode currently reads each token before building its next step. |
| GPU proposal handoff to the next target pass | Remaining integration work | Native chains are queued, but their IDs are still read before building target verification. Python can feed a GPU token array directly. Flash's bounded PLE reader needs host token IDs for positional file reads; preserve its memory benefit when evaluating this path. |
| Alternating/preallocated attention cache writes | Remaining optimization work | Original family runtimes adopt alternating KV buffers. Native concatenation and accepted-prefix slicing have exact cache/rollback coverage but different allocation and copying costs. |
| Stacked SIMD Qwen projections | Diagnostic coverage; native production uses original unstacked weights | Shape-dependent SIMD reduction splits make these different rounding configurations. Both kernel variants execute; the native unstacked configuration matches its original Python oracle at short and 10K context. Do not label it byte-identical to the stacked Python serving configuration. |
| MTP projection arithmetic | Native uses row-exact projections | Original Nemotron MTP deliberately uses plain MLX matmuls; Flash switches its hidden projection to MLX above 32 stream rows. Native keeps row-exact kernels, splitting Flash's 64-stream-row windows. This preserves native proposal parity across batching; do not claim bit-identical proposal logits to those Python matmul configurations. Every emitted token is still checked by the target. |
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
