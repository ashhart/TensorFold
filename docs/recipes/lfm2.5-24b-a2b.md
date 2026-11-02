# LFM2.5-24B-A2B, native Metal

This recipe covers [`LiquidAI/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit)
at revision `667408d494bde191c2cb9dd4f4549827816b9275`. The numbers below are measured on this 4-bit export only.

```bash
hf download LiquidAI/LFM2.5-24B-A2B-MLX-4bit \
  --revision 667408d494bde191c2cb9dd4f4549827816b9275 --local-dir "$HOME/models/lfm2.5-24b-a2b-4bit"
zig build native -Dcpu=apple_m1
zig-out/native/bin/tensorfold-native serve "$HOME/models/lfm2.5-24b-a2b-4bit" --port 8080 --parallel 16
```

A Mac with less memory may not hold 16 streams at the checkpoint's 131,072-token window; the server then names how
many fit. `--context 40960` holds 16 on a 48 GB M4 Pro.

## Shared rounds

Every stream's tokens go through the lane rounds. A round of several streams is one forward over all their rows.
In it each layer's short conv (30 layers) and each attention layer's qk, attention and merge (10 layers) run once
over every row (`lfm2_segments.metal`), from a table of each stream's cache addresses, row offsets and key counts
(`segments.zig`). Each row's arithmetic is the per-stream kernel's, so a reply equals its solo run. Prompt calls and
one-stream windows keep the per-stream launches. A greedy row takes the GPU's pick of its logits (`lfm2_argmax`);
sampled and masked rows draw on the host. The dense twin takes its MMA window only for matrices of 4,096 outputs or
more (the short conv's input and the head); narrower ones keep the scalar passes.

`zig build test-lfm2` runs the family's tests. `tf-kernel-contract --only lfm2` checks the shared-round kernels and
the pick, and `tf-lfm2-exact` compares windows and shared rounds with one-row decode on the real weights.

## Measured

Greedy, 256 tokens, 16 public prompts, a different one a stream, 3 reps in each of 3 alternated rounds on an M5 Ultra;
aggregate tok/s, the range of the rounds' medians. Before is main at 0d24ecb3 (each stream's own launches), after adds
the shared-round kernels on it. The M4 Pro is not yet measured on a quiet machine.

| Streams | before | after |
| --- | --- | --- |
| 1 | 258.5-259.2 | 266.5-267.7 |
| 4 | 483.5-486.5 | 565.2-566.1 |
| 8 | 551.5-551.9 | 732.2-735.6 |
| 16 | 660.7-661.9 | 922.5-923.5 |

Every reply was token for token equal before and after, and equal to its solo run and to `"draft": false`.
Cold prefill at 2k, 8k and 32k tokens and peak memory did not change.

## DSpark drafts

LiquidAI's DSpark drafter for this model is
[`LiquidAI/LFM2.5-24B-A2B-DSpark`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark/tree/e052e5b1dd103c38cdd4f8ddc068d7dea4586f55)
at revision `e052e5b1dd103c38cdd4f8ddc068d7dea4586f55` (327.7M bf16 parameters, 655 MB). Its weights are under the
[LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark/blob/e052e5b1dd103c38cdd4f8ddc068d7dea4586f55/LICENSE);
TensorFold does not ship them, so read the license before use.

```bash
hf download LiquidAI/LFM2.5-24B-A2B-DSpark --revision e052e5b1dd103c38cdd4f8ddc068d7dea4586f55 \
  config.json model.safetensors LICENSE README.md --local-dir "$HOME/models/LFM2.5-24B-A2B-DSpark"
zig-out/native/bin/tensorfold-native serve "$MODEL_DIR" --port 8080 --parallel 16 \
  --drafter "$HOME/models/LFM2.5-24B-A2B-DSpark"
```

Without `--drafter` the engine serves as before. `"draft": false` on a request, or `--no-drafts`, turns the drafter
off; the lane core verifies every draft, so a drafted reply equals its `"draft": false` reply token for token.

The drafter has no embedding and no head of its own:

- Context: the target's rows after layers 2, 10, 18, 26 and 34 (captured by the fused normalization kernels), through `fc` and
  `hidden_norm`, give each of its five layers' context keys and values (k_norm, then RoPE at the row's position).
- Block: the pending token and mask tokens run five Qwen3 layers with non-causal attention over the context and the
  block, then the target's head. A block is as long as the deepest chain its pass holds, at most the nine rows the
  drafter was trained on.
- Markov head: row i's logits plus `W2 W1[previous token]` (rank 256); draft i is their argmax.

The drafter runs the released bf16 matrices as they are (`--drafter-bits 0`, the default). `--drafter-bits 8` instead
quantizes them at load to 8-bit MLX affine codes in groups of 64: about 361 MiB of weights
against 625 MiB. Weight storage does not guarantee serving throughput; the server logs the mode, and 4 bits is refused.
Both run on the family's kernels
(bf16 on `lfm2_bf16_rows`, `lfm2_bf16_mma`, the M5 GPU tensor path and `lfm2_markov_scan_bf16`,
8-bit on `core/tile_qmm`); the q and k rows are reordered so that the target's half-split RoPE kernel gives the
checkpoint's interleaved RoPE. Its own kernels include `lfm2_attn_block` and `lfm2_attn_block_merge`
(non-causal block attention, the target's attention kernels with a template flag), `lfm2_markov_scan` and
`lfm2_markov_pick` (a Markov step) and `lfm2_draft_ids` (a hold's drafts into the next verify's rows on the GPU, so
the host does not wait for a hold that fits one pass). Each kept prompt state keeps the drafter's context too, so a
drafting stream resumes prompts. Hold lengths follow the lane core's per-round depth pricing; no stream-count guard disables drafting.

Priced shared rounds use the existing lane allocator and the measured block-count/depth grid. Only neural blocks
incur hold cost; copies and forced rows still occupy verify capacity. The native 32-row hold limit can split a
cohort into several passes, priced in execution order rather than as one impossible padded batch.

A new stream waits for its first funded hold while retaining its prompt context. Each lane owns incoming prompt
and verified taps before the target overwrites them, projecting them only for a hold, snapshot or stage overflow.
Zero-grant rounds therefore retain contiguous context for later drafts; they do not disable drafting by stream count.
Prompt, decode and verify forwards capture raw BF16 taps during normalization through each row's destination
straight into the lane's separate stage allocation, without a per-round host copy. The stage retains up to
2,048 rows, clipped to lane capacity, and admission counts those bytes independently of K/V storage. Its GPU
base belongs to the live allocation; snapshots contain bytes and state, never virtual addresses. A target
consumer waits on the pending hold's GPU event; checked host consumers settle before reading. Only accepted
rows become staged context. Overflow projects only the retained prefix before the next capture's anchor.
Declined snapshots leave the retained final anchor intact, including across other streams' forwards.

`tf-kernel-contract --only lfm2` checks these kernels. `tf-lfm2-dspark MODEL DRAFTER TOKENS [ROWS]` runs the drafter
against an fp32 host reference of the released checkpoint (the same taps and the target's embedding and head),
prints each draft against the reference's pick with the margin between the reference's two best tokens, and times
the drafter's passes; a near tie can pick the other token.
The native diagnostic covers 64 context rows and a nine-row block, deferred context, the 2,143-row
zero-row-rewind case, complete clean/context calibration grids, and NaN/+Inf/-Inf rejection on all three
held-token consumer paths. Exact FC bytes and alternating-two-live-buffer raw-tap oracles remain separate
from its floating-point reference comparison.

Row-owned capture uses `lfm2_norm_tap`, `lfm2_combine_norm_tap` and the sorted-combine variant;
the original diagnostic `lfm2_tap` interface is unchanged. The fused norms launch 512 threads with
144 bytes of threadgroup storage. Sorted combine retains a 512-thread source reservation. Resource
declarations are not physical-device or throughput qualification; `--only` checks remain scoped diagnostics.

### Verify schedule

The drafter also carries a confidence head: a linear map of a draft's final block row and `W1[previous token]`
(2,304 inputs) to one logit, through a sigmoid (`lfm2_dspark_confidence`). It does not change the drafts.
The verify rows follow it: a chain's chance at draft j is the product of its first j confidences. Each round the
lane core's window allocator takes the likeliest drafts over all streams while the expected tokens a ms rise, so
each stream gets its own verify length, and the next hold drafts as deep as the stream's grant. A hold still
running on the GPU is priced by its lane's last confidences, so the host does not wait for it. A checkpoint
without the head verifies by the core's depth rule, from each stream's measured acceptance by depth.

A stream with a copy ahead verifies the copy and asks no hold, and a drafting stream prices its copy
windows by its own copies. Clean holds are calibrated at every legal depth for 1, 2, 4, 8 and 16 blocks,
within the 32-row pass limit. Context-only held passes use the same encoder, barriers, partitioning and
gathering at powers-of-two row counts 1 through 2,048 for those five counts: 60 context quotes. Every shape
warms once, then records the best of three valid CPU-encode + GPU measurements on finite nonzero raw taps;
both grids publish only when complete. The newly retained context debt is staged before funding, and the
matching calibrated context estimate is subtracted before updating the clean-hold EMA. Confidence still
allocates verify rows unchanged, but a hold's expected gain is capped by its actual head acceptance. If the full hold
cannot pay, the controller can retain a shorter prefix rather than discard all its useful drafts. Its separate
hold-value estimate starts each newly tried conditional from the observed outcome; later trials use the
existing depth learning rate. The reservation fallback keeps its original prior-initialized rates, and untested
conditionals do not discount the confidence gain. One-stream and shared rounds use the same fresh reservations
and physical-pass pricing, without disabling drafts at any stream count.

A hold suppressed by this acceptance cap retains its context and uses the existing periodic probe backoff to
try again. A probe requires a positive fresh reservation and funding in the full round's verify and physical
hold costs, including companion rows, before other holds are selected. An impossible trial does not spend its
probe cadence; cheaper live costs can fund the retained trial immediately. Actual profitable trials restore
ordinary holds; copies do not train head acceptance. These rules do not establish throughput qualification:
that still needs the complete same-head no-drafter/BF16/8-bit matrix, including sampled one-stream replies.

Observed fixed overhead excludes completed hold encoding CPU at submission and successful GPU time at
settlement, each once. A pending pass retains its encoding receipt only for clean-hold repricing, not a second
charge. Metadata-only absorption and zero-row rewind create no command buffer or autorelease pool. Startup
calibration is discarded before serving. Reading these receipts never submits work or waits; backends
without actual timing retain modeled pricing.

[Controller regression and native API receipts](../receipts/dspark-chain-restack/hold-value-controller.json)
record the constructed timing cases and source hashes; they are not device-throughput qualification.

`tf-lfm2-dspark` also prints the confidence
head against its fp32 reference and times shared holds of 2 to 16 streams.

### Shared native paths in the confidence stage

The shared native cutover leaves this stage's confidence/depth decisions, hold grants, acceptance-cap
pricing, probe cadence and owned context unchanged. It does not import the later allocation-stage
controller or the top candidate's subsequent pricing policies.

The loader leaves matrices in the checkpoint until needed. Ordinary BF16 matrices, `W1` and the
optional confidence projection read directly into their final Metal buffers. Paired Q/K matrices use
one reusable staging matrix and reorder rows directly into their destination. Coded matrices apply
the same row mapping during quantization. The overwritten 8-bit clear is removed; 4-bit packing's
required clear remains in the quantization helper, although 4-bit serving is still refused. Spawned
quantization threads are joined on failure.

Context absorption rotates keys and stores values without computing discarded query heads. Target
and drafter Q projections write `queries`, then normalize and rotate into the existing padded `x`
after its projection consumers finish. Attention reads `x`, keeping Metal input/output arguments
disjoint. Target scratch no longer retains unused Q or sort-order backing. BF16 drafter sums retain
only the quantized target-head requirement; 8-bit drafters keep their original sums capacity.

Target Q/K/V and gate/up projections share row sums only when input width, group, sums regime and
ordinary-versus-tensor route agree. Each matrix retains its own code width, and down projections
recompute their sums. Sorted expert combine can feed the next or final RMSNorm and raw taps in one
dispatch, preserving expert-pick order, both BF16 boundaries, duplicate columns and null destinations.
The native twin path keeps its original arithmetic.

BF16 projections use M5 GPU tensor instructions from 8 physical rows for wide matrices and 16 for
other matrices. Capability comes from the GPU architecture, not Apple Neural Engine availability.
The tensor numerical contract retains the predecessor's scalar-row bound through 32 rows and the
128-thread, 128-column geometry. Without that pipeline, this confidence branch retains its original
scalar projections through 32 rows, then 64x64 tiles. On non-tensor devices only, wide BF16 gate/up
projections fuse with SwiGLU at 4 through 16 physical rows. Projection and activation BF16 rounding
remain canonical; smaller or narrow shapes and tensor-capable devices use separate projections.
The 8-bit routing is unchanged.
Context-FC gathering stays strictly below the shared BF16 tensor threshold when that pipeline exists;
tensor-sized pieces retain their original independent calls. This preserves exact projection bytes rather
than changing the numerical oracle to accommodate a different projection regime.

Only a successfully completed native all-row verification marks its exact scratch owner. Keeping
that full identity frame skips redundant convolution-state and logit copies. Partial, reordered,
generic and last-row frames keep their copy path; snapshots and release order remain unchanged.
Chain windows build positions directly without unused parent/depth arrays; tree handling is unchanged.
These are source behavior and storage/work changes, not performance qualification of this head.
