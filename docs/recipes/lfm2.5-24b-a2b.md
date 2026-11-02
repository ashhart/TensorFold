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

- Context: the target's rows after layers 2, 10, 18, 26 and 34 (the backend's raw-state norm taps), through `fc` and
  `hidden_norm`, give each of its five layers' context keys and values (k_norm, then RoPE at the row's position).
- Block: the pending token and mask tokens run five Qwen3 layers with non-causal attention over the context and the
  block, then the target's head. A block is as long as the deepest chain its pass holds, at most the nine rows the
  drafter was trained on.
- Markov head: row i's logits plus `W2 W1[previous token]` (rank 256); draft i is their argmax.

The drafter runs the released bf16 matrices as they are (`--drafter-bits 0`, the default). `--drafter-bits 8` instead
quantizes them at load to 8-bit MLX affine codes in groups of 64: about 361 MiB of weights
against 625 MiB. Weight storage does not guarantee serving throughput; the server logs the mode, and 4 bits is refused.
Both run on the family's kernels. BF16 projections use GPU tensor units when the device reports that capability:
wide matrices (more than 2,048 outputs) from eight physical rows, other matrices from sixteen. The 128-thread,
128-column tensor launch retains the predecessor's scalar-row numerical bound through 32 rows. M4 and other
non-tensor devices retain this depth stage's scalar route through 32 rows, then the existing MMA tiles. This is GPU
acceleration, not the Apple Neural Engine. Wide gate/up projections alone fuse with row SwiGLU at 4–16 rows on
non-tensor devices, preserving both projection and activation BF16 rounding; M5 and 8-bit routing do not use it.
The 8-bit matrices use `core/tile_qmm`; Q/K rows are reordered so that the target's half-split RoPE kernel gives the
checkpoint's interleaved RoPE. Context absorption computes only keys and values, without discarded query work.
The drafter also uses `lfm2_attn_block` and `lfm2_attn_block_merge`
(non-causal block attention, the target's attention kernels with a template flag), `lfm2_markov_scan` and
`lfm2_markov_pick` (a Markov step) and `lfm2_draft_ids` (a hold's drafts into the next verify's rows on the GPU, so
the host does not wait for a hold that fits one pass). Each kept prompt state keeps the drafter's context too, so a
drafting stream resumes prompts. Hold lengths follow the lane core's per-round depth pricing; no stream-count guard disables drafting.

Priced shared rounds use the existing lane allocator and the measured block-count/depth grid. Only neural blocks
incur hold cost; copies and forced rows still occupy verify capacity. The native 32-row hold limit can split a
cohort into several passes, priced in execution order rather than as one impossible padded batch.

Clean hold prices cover every legal depth at 1, 2, 4, 8 and 16 streams. Context projection has a separate
60-point grid: those five stream counts at 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 and 2048 retained rows.
Each point uses the serving encoder and deterministic nonzero BF16 inputs, one warmup and the best of three
valid nonoverlapping CPU-encode plus GPU measurements. Both grids publish together; startup receipts are drained.
Before either the lone depth decision or shared allocation, committed target rows are staged exactly once so
the existing controller sees the actual deferred-context debt. Context quotes round up to covered row/count
regimes and charge only neural holds, once per physical pass. The live clean-hold EMA subtracts the matched
context quote; an already observed whole-round price is not charged context a second time.

A new stream waits for its first funded hold while retaining its prompt context. Each lane owns incoming prompt
and verified taps before the target overwrites them, projecting them only for a hold, snapshot or stage overflow.
Zero-grant rounds therefore retain contiguous context for later drafts; they do not disable drafting by stream count.
Decode and verify forwards capture raw-state taps in the canonical normalization through each row's destination
straight into the lane's stage, without a separate tap pass or per-round host copy. A forward waits on a pending
hold before replacing rows that hold still reads; only accepted rows become staged context. Stage overflow projects
only the retained prefix before the next capture's anchor, so a reused anchor can replace its old row without
losing contiguous context.
The raw stage is bounded by 2048 rows and the lane capacity, separate from the per-layer K/V buffers in this
depth stage. Its allocation caches the GPU base used for row pointer tables. Owned-stage serving retains only
that table in target scratch; missing drafting-lane destinations are errors, and plain-stream rows are null.
Snapshots count persistent K/V bytes only, and a declined snapshot must not consume the final owned prompt anchor.

`tf-kernel-contract --only lfm2` checks these kernels. `tf-lfm2-dspark MODEL DRAFTER TOKENS [ROWS]` runs the drafter
against an fp32 host reference of the released checkpoint (including a real 64-context/nine-draft case), prints
each draft against the reference's pick and margin, and reports CPU-encode, GPU and summed pass costs separately.
A near tie can pick the other token. Native diagnostics and kernel contracts also exercise table-only and
alternating two-buffer raw capture, null/dark destinations and uneven FC gather regimes with unchanged exact bytes,
deferred zero-grant context through a rewind to 2143, snapshot lifecycle and allocation-free unwaited holds. Checked host consumers
reject planted NaN and positive/negative infinity independently; neither completion waits nor timing receipts
replace those numerical oracles. The depth stage has no confidence-head oracle.

Row-owned capture uses `lfm2_norm_tap`, `lfm2_combine_norm_tap` and `lfm2_combine_sorted_norm_tap`; the original
`lfm2_tap` interface remains the reference copy. Sorted expert outputs stay live until the next operator norm or
final norm, preserving pick order, residual BF16 boundaries, duplicate columns and null destinations. The sorted
norm launches retain the 512-thread cap and 144-byte threadgroup reservation.
FC batching never crosses the per-matrix tensor threshold: wide BF16 projections switch at eight rows and other
BF16 projections at sixteen, while non-tensor BF16 and 8-bit routes retain the original 32-row boundary. Each
span still projects all rows; gathering changes neither separators nor BF16 results.

The loader reads unpaired BF16 matrices directly into their final Metal buffers. Paired Q/K matrices use one
reusable matrix stage and reorder into the destination; coded matrices map paired source rows during quantization.
W1 stays BF16 in both modes. Quantization retains the required 4-bit clear, omits only overwritten 8-bit clears,
and joins stack-held worker threads even on spawn failure.

Target scratch no longer holds unused Q or sorted-order buffers. Q projects into `queries`, then normalization
and rotation write disjoint padded `x`, which attention reads. Compatible target Q/K/V and gate/up projections
reuse input row sums only when K, group, sums regime and ordinary/tensor path agree; each matrix retains its code
width and down projections recompute sums. BF16 drafter sums cover only the target head; 8-bit capacity is unchanged.
Only a successfully completed native all-row frame marks cache ownership for skipping a redundant full identity
keep copy. Partial or reordered paths, generic/last-only forwards, snapshots and release retain their semantics.
Chain windows construct positions directly; tree windows keep their parent/depth validation.

These are source behaviors for the depth stage, not performance qualification. Its depth controller, measured
block-count/depth pricing, owned context, zero-grant capture and completion-before-consumption rules remain in
place. No confidence head or later allocation policy is added. Builds, numerical/resource contracts, model
oracles and serving gates must be run on this head before drawing qualification or throughput conclusions.
