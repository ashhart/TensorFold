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

- Context: the target's raw rows after layers 2, 10, 18, 26 and 34, captured by the next/final normalization,
  pass through `fc` and `hidden_norm` into each drafter layer's context keys and values. `lfm2_dspark_kv` applies
  k_norm and RoPE without computing discarded queries.
- Block: the pending token and mask tokens run five Qwen3 layers with non-causal attention over the context and the
  block, then the target's head. A block is as long as the deepest chain its pass holds, at most the nine rows the
  drafter was trained on.
- Markov head: row i's logits plus `W2 W1[previous token]` (rank 256); draft i is their argmax.

The drafter runs the released bf16 matrices as they are (`--drafter-bits 0`, the default). `--drafter-bits 8` instead
quantizes them at load to 8-bit MLX affine codes in groups of 64: about 361 MiB of weights
against 625 MiB. Weight storage does not guarantee serving throughput; the server logs the mode, and 4 bits is refused.
Both run on the family's kernels: BF16 uses `lfm2_bf16_rows`, `lfm2_bf16_mma`, the GPU tensor-unit route
`lfm2_bf16_nax` when supported, and `lfm2_markov_scan_bf16`; 8-bit uses `core/tile_qmm`. The q and k rows are
reordered so that the target's half-split RoPE kernel gives the checkpoint's interleaved RoPE. Block kernels
include `lfm2_attn_block` and `lfm2_attn_block_merge`
(non-causal block attention, the target's attention kernels with a template flag), `lfm2_markov_scan` and
`lfm2_markov_pick` (a Markov step) and `lfm2_draft_ids` (a hold's drafts into the next verify's rows on the GPU, so
the host does not wait for a hold that fits one pass). Each kept prompt state keeps the drafter's context too, so a
drafting stream resumes prompts. Hold lengths follow the lane core's per-round depth pricing; no stream-count guard disables drafting.

Priced shared rounds use the existing lane allocator and the measured block-count/depth grid. Only neural blocks
incur hold cost; copies and forced rows still occupy verify capacity. Each hold runs the checkpoint's complete
bidirectional query block, including unused mask rows, regardless of the requested proposal prefix. The final
pointwise projections retain that full width too. Markov steps and returned proposals still follow the grant.
The native 32-row hold limit splits larger logical cohorts into bounded passes, priced in execution order using
the configured query width rather than the retained depth. All logical stream counts remain supported.
The target retains its original 1–16-row waited-wall calibration and wall-clock round measurement, including
real host residual work. This stage does not add later shared-width calibration or per-row host-draw pricing.

A new stream waits for its first funded hold while retaining its prompt context. Each lane owns up to 2,048
contiguous incoming prompt and verified tap rows, clipped to its lane capacity, before the target overwrites
them. Projection is deferred until a hold, snapshot or stage overflow. Zero-grant rounds retain context at every
admitted stream count. Prompt, decode and verify forwards capture raw BF16 taps directly into the owned stage.
The serving target allocates only its destination-pointer table; generic capture retains raw scratch storage.
A forward waits on a pending hold before replacing rows that hold still reads; only accepted rows become staged
context. Overflow projects only the retained prefix before the new anchor. A declined final snapshot leaves
the final anchor intact across other streams' forwards. Snapshots retain bytes and state, never GPU addresses;
each live allocation caches its own base address.

Shared rounds stage newly verified rows before pricing the next hold. The prepared request then consumes
neither its features nor its first-token handle twice. Actual staged debt reaches the existing depth and
allocation inputs, including rounds with no grant. Metadata-only absorbs and zero-row rewinds create no
command buffer or autorelease pool and accrue no GPU time.

`tf-kernel-contract --only lfm2` checks these kernels. `tf-lfm2-dspark MODEL DRAFTER TOKENS [ROWS]` runs the drafter
against an fp32 host reference of the released checkpoint (the same taps and the target's embedding and head).
The reference always evaluates the full configured query block and compares only the requested retained prefix.
The diagnostic checks that changing the grant preserves first-row raw logits, first pick and confidence, and
exercises independent same-regime references for 8- and 16-lane split cohorts. It prints each draft against the
reference's pick and its margin, and times the passes; a near tie can pick the other token.

The sorted expert combine preserves pick order and both BF16 rounding boundaries while feeding the next/final
RMSNorm and raw taps; decode's twin combine follows the same ownership rules. Duplicate tap columns and null
row destinations remain supported. The sorted fused kernels retain a 512-thread cap and a 144-byte source
threadgroup reservation; these are resource contracts, not physical-device or throughput qualification.

Target Q/K/V and gate/up projections reuse row sums only when K, group, sums mode and ordinary/tensor regime
agree, while each matrix retains its own code width. Down projections recompute sums. Target Q and routing-order
backing are removed: Q projects into `queries`, then normalization and RoPE write into the existing padded `x`.
Attention reads `x`; the Metal input and output arguments remain disjoint. The drafter uses the same Q lifetime,
omits discarded context queries, and sizes BF16 sums for the target head while retaining 8-bit sums capacity.

`dspark_bf16.zig` owns the BF16 projection and row-GLU routes. Actual GPU architecture capability, not a model
name, enables M5 GPU tensor acceleration (not Apple Neural Engine): wide matrices use it from eight physical
rows, other matrices from sixteen, with 128-thread/128-column geometry. Through 32 rows the tensor kernel keeps
the predecessor scalar-row numerical bound. This allocation stage's non-tensor callers retain their original
32-row scalar limit and existing MMA fallback; 8-bit routing is unchanged. Only non-tensor devices use the
wide gate/up row-GLU route at 4–16 physical rows, preserving projection and activation BF16 rounding.
The context-FC gather uses the same `tensorRows` threshold as BF16 projection and stays strictly below it
when the tensor pipeline exists. Tensor-sized pieces retain their independent calls, preserving the exact
BF16-byte and nonfinite oracle; gathering never selects a different projection regime.

A successful native all-row verify marks its completed cache frame, allowing a full identity keep to skip
recopying conv state and logits. Partial/reordered keeps, generic and last-only forwards, snapshots and release
retain their existing semantics. Chain windows construct positions directly; trees still validate their parents.
These shared native changes do not import later controller policies into this allocation-stage recipe, and no
performance or merge-readiness claim is made for this source head.

### Verify schedule

The drafter also carries a confidence head: a linear map of a draft's final block row and `W1[previous token]`
(2,304 inputs) to one logit, through a sigmoid (`lfm2_dspark_confidence`). It does not change the drafts.
The verify rows follow it: a chain's chance at draft j is the product of its first j confidences. Each round the
lane core's window allocator takes the likeliest drafts over all streams while the expected tokens a ms rise, so
each stream gets its own verify length, and the next hold drafts as deep as the stream's grant. A hold still
running on the GPU is priced by its lane's last confidences, so the host does not wait for it. A checkpoint
without the head verifies by the core's depth rule, from each stream's measured acceptance by depth. Its lone
depth decision stages the newly kept target rows first and includes their context quote in modeled hold work;
observed whole-round prices already include that work and are not charged a second time.

A stream with a copy ahead verifies the copy and asks no hold, and a drafting stream prices its copy
windows by its own copies. The lane core also weighs the hold pass itself: clean holds are calibrated at
every requested depth for every legal physical block count up to 16, using the configured query width within
the 32-row pass (1/2/3 blocks at the released nine-row width). Each shape is warmed, then takes the best of
three valid, same-attempt CPU-encode plus GPU totals. Clean quotes remain keyed by the requested depth.
The same concurrent encoder, barriers, piece partition and gather path calibrate context-only passes at
powers-of-two row counts from 1 through 2,048 for all five stream counts: 60 context quotes. Temporary K/V-only
lanes share deterministic finite nonzero raw taps.
Both grids publish only when complete, and startup work is not charged to a serving round.

Each selected physical group pays context once using its largest staged debt and the next covering measured
regime. This is a conservative uniform-row estimate, not a hardware bound; copies and forced rows incur no
neural fee. Clean-hold EMA adaptation subtracts the matching context estimate from measured CPU-plus-GPU work;
invalid residuals do not invent clean observations. Confidence still allocates verify
rows unchanged, but a hold's expected gain is capped by that stream's actual head acceptance. If the full hold
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
This allocation stage retains probability-priority row allocation, gain-per-row hold ordering, grant/slack
rules and probe cadence. Context prices correct the existing normal, group and physical-pass prices and the
optimistic `canExplore` bound; they do not import later cost-aware reordering or cheap-peer cohort searches.
Due probes still advance their clocks, but their own selected context debt must be fundable before their trial
is spent, even when no ordinary hold remains. An unselected zero-row companion cannot lend them free context.

Observed fixed overhead excludes encoding in its launch interval and GPU work in its settlement interval,
each once. Pending encoding metadata is retained only for matching EMA repricing, never charged again.
Accountable startup or out-of-round work is discarded before the clock starts. Actual timing does not also
subtract modeled extra-step work; backends without the timing callback retain modeled pricing. Reading these
receipts never allocates, submits work or waits.

[Controller regression and native API receipts](../receipts/dspark-chain-restack/hold-value-controller.json)
record the constructed timing cases and source hashes; they are not device-throughput qualification.

`tf-lfm2-dspark` retains the actual 64-context/9-row FP32 reference, confidence reference and shared-hold
diagnostics. Its deferred-context case crosses the 2,048-row stage and performs the zero-row rewind at 2,143.
Waited, signalled target-held-ID and checked host consumers all reject planted NaN and both infinity signs.
The first-normalization oracle alternates two live destination buffers and checks untouched peer bytes,
masked columns, dark rows and nonfinite raw bits. Context-FC tests retain exact bytes, capacity bounds and
all NaN-separated shape/input cases. The steady native loop requires zero host allocations/resizes and
20 unwaited holds.

The loader reads unpaired BF16 matrices directly into final Metal backing; paired Q/K use one reusable matrix
stage and reorder into their destination. Quantization maps paired source rows directly into codes, clears only
the required 4-bit packing storage, and joins stack-held worker handles on failure. Markov W1 and the existing
confidence projection remain BF16. This allocation stage retains one combined Metal allocation for each lane's
K/V and bounded stage; the full bytes participate in stream-memory admission. The earlier depth and confidence
stages retain their separate stage-buffer ownership. At load the drafter logs weights, scratch and lane bytes.

These are source contracts, not new qualification receipts. The last frozen full serving gate at `7eb317eda`
remains FAIL 21/32 with the original no-tolerance per-cell comparison; no historical receipt is relabelled as
evidence for this source head.
