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
For sampled `top_k` from 1 through 128, a bounded stack selector orders finite BF16 words by value and token id,
then uses the unchanged FP64 keyed sampler without heap allocations or a second candidate sort. Larger and
unbounded cuts retain the general path. Verification uses the GPU's whole-row finite flag; a nonfinite value
still errors even when a grammar mask hides it.

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

Neural DSpark serving is eligible only on recognized Apple M5 or newer Metal device names. Earlier or unknown
names log that DSpark is standing aside and take the ordinary target-only path before neural weight loading,
calibration, taps or admission charges. Cheap copies and canonical target arithmetic remain enabled. Normal
CLI drafter-path resolution and validation still apply; omit `--drafter` on older devices.
Independent `tf-lfm2-dspark` FP32 diagnostics remain available there, but are not serving qualification.
Use the following drafter setup on an eligible M5-or-newer device:

```bash
hf download LiquidAI/LFM2.5-24B-A2B-DSpark --revision e052e5b1dd103c38cdd4f8ddc068d7dea4586f55 \
  config.json model.safetensors LICENSE README.md --local-dir "$HOME/models/LFM2.5-24B-A2B-DSpark"
zig-out/native/bin/tensorfold-native serve "$MODEL_DIR" --port 8080 --parallel 16 \
  --drafter "$HOME/models/LFM2.5-24B-A2B-DSpark"
```

Without `--drafter` the engine serves as before. `"draft": false` on a request, or `--no-drafts`, turns the drafter
off; the lane core verifies every draft, so a drafted reply equals its `"draft": false` reply token for token.

The drafter has no embedding and no head of its own:

- Context: the target's captured rows after layers 2, 10, 18, 26 and 34, through `fc` and
  `hidden_norm`, give each of its five layers' context keys and values (k_norm, then RoPE at the row's position).
- Block: the pending token and mask tokens run five Qwen3 layers with non-causal attention over the context and the
  block, then the target's head. Every hold keeps the configured nine query rows and full pointwise tail,
  including unused masks; a smaller grant retains a proposal prefix rather than shortening the backbone.
- Markov head: row i's logits plus `W2 W1[previous token]` (rank 256); draft i is their argmax.

Full-block attention reserves one tail chunk beyond the target's 131,072-key span for query masks.
The target context window is unchanged. A GPU boundary oracle checks both the next scratch row and the
extra keys' contribution to the result.

The drafter runs the released bf16 matrices as they are (`--drafter-bits 0`, the default). `--drafter-bits 8` instead
quantizes them at load to 8-bit MLX affine codes in groups of 64: about 361 MiB of weights
against 625 MiB. Weight storage does not guarantee serving throughput; the server logs the mode, and 4 bits is refused.
Both run on the family's kernels
(bf16 on `lfm2_bf16_rows`, `lfm2_bf16_mma`, the M5 GPU tensor path and `lfm2_markov_scan_bf16`,
8-bit on `core/tile_qmm`); the q and k rows are reordered so that the target's half-split RoPE kernel gives the
checkpoint's interleaved RoPE. Its own kernels are `lfm2_tap`, `lfm2_attn_block` and `lfm2_attn_block_merge`
(non-causal block attention, the target's attention kernels with a template flag), `lfm2_markov_scan` and
`lfm2_markov_pick` (a Markov step) and `lfm2_draft_ids` (a hold's drafts into the next verify's rows on the GPU, so
the host does not wait for a hold that fits one pass). Each kept prompt state keeps the drafter's context too, so a
drafting stream resumes prompts. The lane core prices drafts across the admitted stream counts.

Priced shared rounds use the existing lane allocator and the measured block-count/depth grid. Only neural blocks
incur hold cost; copies and forced rows still occupy verify capacity. The native 32-row hold limit can split a
cohort into several passes, priced in execution order rather than as one impossible padded batch.

Before prompt processing, native serving prices whether to retain a neural context for the request.
It compares complete physical holds and conservative prompt/context debt against bare target rows,
using recent head-only acceptance for this model and weight mode, separately for greedy and sampled draws.
Only calibrated lone target widths are candidates; an unmeasured prefix does not invalidate other priced shapes.
A target-only request retains cheap copies but owns no drafter lane or taps. Its route stays fixed until release.
Tracked requests receive only an initial conditional-rate seed; grants, confidence, probe clocks and context are fresh.
Cold admissions use the same complete, paying quote and retain the 64-to-1,024-request backoff if no head
outcome is observed. Unknown serving prices or an unpaid quote stand aside: hypothetical perfect acceptance
does not fund a recovery probe. Later measured prices or head observations must actually fund enrollment.
Only fully unbound direct diagnostics preserve the tracked route without serving calibration.
Other providers retain their legacy feature-prefill lifecycle unless they offer the enrollment capability.


A tracked stream waits for its first funded hold while retaining its prompt context. Each lane owns incoming prompt
and verified taps before the target overwrites them, projecting them only for a hold, snapshot or stage overflow.
Zero-grant rounds therefore retain contiguous context for later drafts; they do not disable drafting by stream count.
Decode and verify norms write taps directly into the lane's stage, without a per-round host copy. A forward
waits on a pending hold before replacing rows that hold still reads; only accepted rows become staged context.
Each shared round stages its newly verified rows before pricing the next hold. The allocator keeps the
confidence and grant inputs captured before staging; prepared requests neither reread nor absorb those rows twice.

`tf-kernel-contract --only lfm2` checks these kernels. `tf-lfm2-dspark MODEL DRAFTER TOKENS [ROWS]` runs the drafter
against an fp32 host reference of the released checkpoint (the same taps and the target's embedding and head),
prints each draft against the reference's pick with the margin between the reference's two best tokens, and times
the drafter's passes; a near tie can pick the other token.

### Verify schedule

The drafter also carries a confidence head: a linear map of a draft's final block row and `W1[previous token]`
(2,304 inputs) to one logit, through a sigmoid (`lfm2_dspark_confidence`). It does not change the drafts.
The verify rows follow it: a chain's chance at draft j is the product of its first j confidences. Each round the
lane core's window allocator takes the likeliest drafts over all streams while the expected tokens a ms rise, so
each stream gets its own verify length, and the next hold returns that proposal prefix while keeping the
full configured query block. A hold still running on the GPU is priced by its lane's last confidences, so
the host does not wait for it. A checkpoint without the head verifies by the core's depth rule, from each
stream's measured acceptance by depth.

Without the BF16 GPU tensor pipeline, wide gate/up projections fuse with SwiGLU for 4 through 16 physical rows.
Each projection retains its original accumulation order and rounds to BF16 before the canonical SwiGLU
arithmetic; only the final activation is stored. Other shapes and tensor-capable devices retain separate
projections and activation. The 8-bit path is unchanged.

A stream with a copy ahead verifies the copy and asks no hold, and a drafting stream prices its copy
windows by its own copies. The lane core also weighs the hold pass itself: the drafter measures clean
holds at every legal physical block count and requested proposal depth. Every block still runs the
checkpoint's configured nine bidirectional query rows, including unused masks; a smaller grant retains
only its proposal prefix, not a shorter attention problem. Larger logical cohorts run as bounded
32-row passes, without a stream-count cutoff. Both clean and context funding use those physical groups.
A reserved copy decision, including silence and source evidence, belongs to the next logical window.
Planning consumes it without advancing suffix-lookup backoff a second time; changed history or lifecycle invalidates it.
Each quote selects the best of three same-attempt precommit CPU-encoding plus GPU totals after a warm
pass. Neighboring measured depths interpolate, preserving the cost changes at kernel transitions.

A separate 60-cell calibration uses the same cost currency for context projection: 1/2/4/8/16 streams at powers-of-two row
counts from 1 to 2,048. Temporary K/V-only lanes share one nonzero raw fixture and are freed before serving;
this adds startup measurements, not a claim of lower startup latency or peak memory. A physical selected
group pays context once, using its largest actual staged debt and the next measured covering regime.
Mixed debts use a conservative uniform-row estimate, not a proven hardware bound; forced and copy-only
companions owe no neural context fee. Missing or invalid coverage cannot fund a context-bearing hold.
CPU spans use native monotonic elapsed time through command preparation, ending immediately before commit.
They exclude GPU execution, waits and pick readback; they are elapsed spans, not thread-cycle measurements.
Only clean-price adaptation subtracts the matching estimated context quote; this residual is not a separately
measured clean segment, and invalid residuals do not invent a clean observation.
Confidence still allocates verify rows unchanged, but a hold's expected gain is capped by that stream's actual
head acceptance. If the full hold cannot pay, the controller can retain a shorter prefix rather than discard
all its useful drafts. Its separate hold-value estimate starts each newly tried conditional from the observed
outcome; later trials use the existing depth learning rate. The reservation fallback starts from its request's
independent initial rates, and untested conditionals do not discount the confidence gain. One-stream and shared
rounds use the same fresh reservations and physical-pass pricing, without a fixed stream-count cutoff.

A hold suppressed by this acceptance cap retains its context and uses the existing periodic probe backoff to
try again. A probe requires a positive fresh reservation and funding in the full round's verify and physical
hold costs, including companion rows, before other holds are selected. An impossible trial does not spend its
probe cadence; cheaper live costs can fund the retained trial immediately. Actual profitable trials restore
ordinary holds; copies do not train head acceptance. These rules do not establish throughput qualification:
that still needs the complete same-head no-drafter/BF16/8-bit matrix, including sampled one-stream replies.

Context-aware recovery prices the actual selected trial group's debt once while retaining the existing
optimism over allowed deeper clean shapes. An expensive due peer cannot veto a funded cheaper prefix.
Zero-context and opt-out backends retain their original recovery semantics.

Native target calibration is separate from backend construction. It runs after admission, with
the normal serving residency and loaded head buffers in place, but before preparing tap destinations.
Temporary calibration caches follow that residency; its existing idle-flush callback commits pending
head and cache buffers before timing. Cache teardown removes those buffers before their owner leaves.
The original deterministic history, warmup, per-width timing trials and target arithmetic are unchanged.
The [source-bound original-contract proof](https://github.com/BlivionIaG/TensorFold/blob/1b968d6039af141acc4876fc9c81a5fdd48dde69/docs/receipts/lfm2-dspark-review/serving-calibration.json)
records actual cache, sampled and cancellation workflows on both devices; it is not a performance pass.

Before prefill, the lane host snapshots admitted and already queued requests under its existing mutex,
bounded by the advertised and memory-admitted lane limits. Context enrollment prices the larger of
that cohort and the drafter's registered streams. This matters for LFM2: whole-prompt prefill runs
inside admission, before queued peers reach the drafter's stream map. Direct engine callers without
a host hint retain their registered-stream estimate. No request is delayed to await future arrivals,
and declining neural context does not disable copies or user-provided drafts.
The [scheduled-cohort receipt](https://github.com/BlivionIaG/TensorFold/blob/1b968d6039af141acc4876fc9c81a5fdd48dde69/docs/receipts/lfm2-dspark-review/scheduled-cohort.json) records actual
queued serving and its limits: correctly declining a known cohort is not a complete performance pass.

The no-confidence depth fallback stages newly committed context before choosing a lone head depth.
An unobserved positive-depth round includes its context quote once; an observed whole-round cost is
already inclusive and is returned unchanged. Recorder and CUDA-bound adapters retain the optional
preparation hook and the timing observer's own context.

The working [causal diagnosis](../receipts/lfm2-dspark-review/diagnosis.md) separates incremental
neural usefulness over copies, target M5 dispatch eligibility and unproven MLX-LM differences.

Observed fixed overhead excludes encoding in its launch interval and GPU work in its settlement interval, each
once. Pending CPU metadata remains only for later repricing, not a second charge at settlement. Accountable
startup or out-of-round work is discarded before the clock starts. Actual timing does not also lose modeled
extra-step work; absent callbacks retain modeled pricing. Reading these receipts never submits or waits.

For stable, unmasked LFM2 verify routes, each successful window quotes its host draw time per verified
row. The existing allocator charges pending, copied, forced and neural rows at their stream's quote,
including companion rows when it prices a physical hold cohort. Actual quoted draw spans leave fixed
overhead once; predicted row prices are never subtracted from observed time. Greedy GPU picks quote zero.
Failed and first-draw work cannot leak into a later verify span. Grammar or masked routes retain their
host work in residual overhead rather than publishing a reusable quote. Unknown target costs keep the
existing exploration path; confidence-only controls and complete measured lone-round prices do not add
this surcharge. These accounting changes still require the complete throughput qualification above.

[Controller regression and native API receipts](https://github.com/BlivionIaG/TensorFold/blob/1b968d6039af141acc4876fc9c81a5fdd48dde69/docs/receipts/dspark-chain-restack/hold-value-controller.json)
record the constructed timing cases and source hashes; they are not device-throughput qualification.

`tf-lfm2-dspark` also prints the confidence
head against its fp32 reference and times shared holds of 2 to 16 streams.

Independent target-control caches are released after their last forward. The main tap buffer and comparison
cache are created after capture controls finish; independent stage buffers do not overlap. Context lengths,
capture layouts, the FP32 reference and numerical bounds are unchanged. Serving cache lifetimes are unaffected.

The loader reads the checkpoint one matrix at a time and keeps the released bf16 matrices by default, or the
8-bit codes with `--drafter-bits 8`; a stream's drafter lane is one
Metal buffer (its layers' keys and values, then its staged rows), and its bytes join the streams the load limit
admits. At load the drafter logs its weights, scratch and a stream's bytes. `tf-lfm2-dspark` also counts host
allocations over twenty served drafter rounds, which must be none.
Ordinary BF16 weights, `W1` and the confidence head load directly into their final Metal buffers. Paired
Q/K matrices use one staging matrix and write their reordered rows directly to the destination. Coded
weights apply that row mapping during quantization; 8-bit output needs no pre-clear, while 4-bit packing
retains its required clear. These reduce load work and peak backing, not retained model payload.
Context absorption rotates only keys and stores values with the target's existing numerical helper; it does
not compute unused query heads. This removes an 8 MiB write-only scratch buffer at the released geometry.
The KV bytes, surrounding cache positions and later drafts remain unchanged. Target and drafter project
queries into `queries`, then normalize and rotate into the existing `x` buffer after its projection consumers
finish. Attention reads `x`; its padding moves with that role. The Metal input/output arguments stay disjoint,
without another buffer. Target scratch no longer retains unused Q or sort-order backing. BF16 drafter sums
retain only the quantized target-head requirement; coded drafters retain their original larger capacity.
These are storage/work reductions, not a throughput claim.

Target Q/K/V and gate/up projections reuse row sums only when their input width, group size and sums
regime agree; each projection retains its own code width. A sorted expert combine can feed the next
normalization, or the final normalization and raw taps, in one dispatch without changing expert-pick order
or BF16 rounding boundaries. This applies to prompt and sorted fallback execution; the qualified native
twin path is unchanged.

A hold runs on one concurrent encoder with a barrier only where a step reads an earlier step's writes: q, k and v,
and gate and up, share one sums pass, and the blocks of a shared hold attend side by side. An 8-bit projection takes
the scalar twin up to 8 rows and the window kernel past them where the GPU has no tensor units. With tensor units,
wide projections use them from 9 rows, the head from 5, and projections no wider than the hidden state from 33.
BF16 projections use M5 GPU tensor instructions from 8 physical rows for wide matrices and 16 for
matrices no wider than the hidden state. This is GPU acceleration, not Apple Neural Engine execution.
Without that pipeline, held blocks retain the original row kernel through 16 rows and context ingestion
through 32, followed by 64x64 tiles. A hold gathers adjacent small staged pieces into preallocated scratch:
at most 128 rows per hold and 32 per call, or 8 per call on the coded scalar twin without tensor units.
With the BF16 tensor pipeline, context-FC gathers stay below its tensor threshold; tensor-sized pieces
retain separate calls. This keeps each piece's projection arithmetic unchanged. Lone, larger or
out-of-room pieces use their original calls. A lane retains every verified raw tap in its own buffer before
the target overwrites its taps. At most 2048 rows, clipped to the lane's capacity, wait for that lane's next
hold, snapshot or overflow; an unrelated lane's absorb no longer flushes them. A later hold projects the
entire retained prefix first, including rows from zero-grant rounds. The released five-tap checkpoint uses
at most 40 MiB of raw staging per lane; this larger bounded storage is included in memory admission.
Captured target residuals use the same canonical bf16 arithmetic as plain forwarding; capture adds a masked
store of those bits, including duplicate columns in the same dispatch. Small staged gathers copy on the GPU
before projection. Stage overflow projects only the retained prefix before the next capture's anchor, so a
reused anchor can replace its old row without losing contiguous context.
Serving captures prompt, decode and verify rows directly into that lane storage. The shared capture buffer
contains only its row-address table: 16 KiB at 2048 rows, without a second 40 MiB raw-tap allocation.
The prompt's last row remains in the lane across unrelated forwards, avoiding a copied 20 KiB anchor per stream.
A declined final snapshot leaves that anchor in place; a valid prefix snapshot still materializes its context.
Generic diagnostic capture retains its raw scratch storage.
Tap destinations cache their live allocation's GPU base; snapshots retain bytes and state, not virtual addresses.
Metadata-only absorb keeps every context update without creating an autorelease pool, and a zero-row rewind
encodes no command buffer. Encoding paths retain their pool through command completion and error cleanup.
A completed native all-row verification marks the exact scratch owner. Keeping that unchanged full frame
does not recopy already-committed convolution state and logits. Partial, reordered, generic and last-row
frames retain the copy path; snapshots, context and release ordering are unchanged.
A successful shared hold preserves its completed picks for surviving peers even if its timing receipt is unusable.
Shared scheduling avoids constructing unused stream descriptors. Chain-only windows construct positions
without temporary parent/depth arrays; tree windows retain their parent and depth handling.
These changes do not establish throughput qualification.

Shared verification past the 16-row qualified window uses measured increments
from the shared forward path through 32 rows, anchored at the 16-row window
price. This calibrates the batch cost without widening a stream's qualified
window. Older wide-batch receipts retain their recorded source, device and measurement provenance; their
four-stream tradeoff does not establish the current no-slowdown gate.

### Requalifying the drafter

On an eligible M5-or-newer device, run the complete matrix from one frozen native binary, with its source commit and pinned model revisions:

```sh
python3 tools/bench_lfm2_dspark.py run \
  --binary zig-out/native/bin/tensorfold-native \
  --source-revision "$SOURCE_REVISION" \
  --target "$TARGET" --drafter "$DRAFTER" \
  --target-revision "$TARGET_REVISION" --drafter-revision "$DRAFTER_REVISION" \
  --output-dir dspark-qualification
```

Defaults: 512-token replies, code completion and chat, greedy and sampled, 1/4/8/16 streams, three balanced
rotations of no external drafter, released BF16 and 8-bit. The plain arm retains normal core proposals such as
copies. Each launch verifies the binary hash and drafter mode. Every solo, `draft:false` and concurrent token
hash must match across arms; each cell's median aggregate decode throughput must be at least its plain cell's.
Negative cells fail individually, without a tolerance or pooled average. Reduced runs require `--smoke` and
never establish qualification.

For a plain/plain A/A control, use the same plain binary, model, ordinary copy behavior, workloads, seeds,
lengths and warmups in both arms; predeclare the ordering and publish its separate raw data. A/A observations
do not alter the original performance verdict or establish a tolerance. Source-bound protocols and results
are linked from [the DSpark PR](https://github.com/ashhart/TensorFold/pull/1394). Pre-M5 stand-aside checks are
not an active-drafter matrix: the frozen driver requires active DSpark startup for its neural arms.

The driver requires and records those revisions for qualification. Keep the source commit/patch with the binary hash.
Publish the metadata, raw request and cell records,
events and summary, not `private-logs/`. Device receipts remain device-specific; a new short smoke cannot
replace historical long-context, cancellation, load-memory or other-device evidence.
