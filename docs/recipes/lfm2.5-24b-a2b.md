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

- Context: the target's captured rows after layers 2, 10, 18, 26 and 34, through `fc` and
  `hidden_norm`, give each of its five layers' context keys and values (k_norm, then RoPE at the row's position).
- Block: the pending token and mask tokens run five Qwen3 layers with non-causal attention over the context and the
  block, then the target's head. A block is as long as the deepest chain its pass holds, at most the nine rows the
  drafter was trained on.
- Markov head: row i's logits plus `W2 W1[previous token]` (rank 256); draft i is their argmax.

The drafter runs the released bf16 matrices as they are (`--drafter-bits 0`, the default). `--drafter-bits 8` instead
quantizes them at load to 8-bit MLX affine codes in groups of 64, about 5% faster on one stream and 361 MiB of weights
against 625 MiB; the server logs which mode runs, and 4 bits is refused. Both run on the family's kernels
(bf16 on `lfm2_bf16_rows`, `lfm2_bf16_mma` and `lfm2_markov_scan_bf16`, 8-bit on `core/tile_qmm`); the q and k rows are reordered so that the target's half-split RoPE kernel gives the
checkpoint's interleaved RoPE. Its own kernels are `lfm2_tap`, `lfm2_attn_block` and `lfm2_attn_block_merge`
(non-causal block attention, the target's attention kernels with a template flag), `lfm2_markov_scan` and
`lfm2_markov_pick` (a Markov step) and `lfm2_draft_ids` (a hold's drafts into the next verify's rows on the GPU, so
the host does not wait for a hold that fits one pass). Each kept prompt state keeps the drafter's context too, so a
drafting stream resumes prompts. The lane core prices drafts across the admitted stream counts.

`tf-kernel-contract --only lfm2` checks these kernels. `tf-lfm2-dspark MODEL DRAFTER TOKENS [ROWS]` runs the drafter
against an fp32 host reference of the released checkpoint (the same taps and the target's embedding and head),
prints each draft against the reference's pick with the margin between the reference's two best tokens, and times
the drafter's passes; a near tie can pick the other token.

### Verify schedule

The drafter also carries a confidence head: a linear map of a draft's final block row and `W1[previous token]`
(2,304 inputs) to one logit, through a sigmoid (`lfm2_dspark_confidence`). It does not change the drafts.
The verify rows follow it: a chain's chance at draft j is the product of its first j confidences. Each round the
lane core's window allocator takes the likeliest drafts over all streams while the expected tokens a ms rise, so
each stream gets its own verify length, and the next hold drafts as deep as the stream's grant. A hold still
running on the GPU is priced by its lane's last confidences, so the host does not wait for it. A checkpoint
without the head verifies by the core's depth rule, from each stream's measured acceptance by depth.

A stream with a copy ahead verifies the copy and asks no hold, and a drafting stream prices its copy
windows by its own copies. The lane core also weighs the hold pass itself: the drafter times holds of 1 to 16
blocks at every admissible depth at load and corrects those prices toward each settled hold's GPU time.
Prices interpolate neighboring measured depths, preserving the cost changes at kernel transitions.

`tf-lfm2-dspark` also prints the confidence
head against its fp32 reference and times shared holds of 2 to 16 streams.

The loader reads the checkpoint one matrix at a time and keeps the released bf16 matrices by default, or the
8-bit codes with `--drafter-bits 8`; a stream's drafter lane is one
Metal buffer (its layers' keys and values, then its staged rows), and its bytes join the streams the load limit
admits. At load the drafter logs its weights, scratch and a stream's bytes. `tf-lfm2-dspark` also counts host
allocations over twenty served drafter rounds, which must be none.

A hold runs on one concurrent encoder with a barrier only where a step reads an earlier step's writes: q, k and v,
and gate and up, share one sums pass, and the blocks of a shared hold attend side by side. An 8-bit projection takes
the scalar twin up to 8 rows and the window kernel past them where the GPU has no tensor units. With tensor units,
wide projections use them from 9 rows, the head from 5, and projections no wider than the hidden state from 33.
Held blocks use one bf16 16-row group, then 64x64 tiles past 16 rows. Context ingestion
keeps the original row kernel through 32 rows, preserving the persistent context projections. A hold gathers
adjacent small staged pieces into preallocated scratch: at most 128 rows per hold and 32 per call, or 8 per call
on the coded scalar twin without tensor units. This preserves each piece's projection regime; lone, larger or
out-of-room pieces use their original calls. Every context row is still projected.
When one lane's stage of 128 verified rows fills, its absorb pass takes
every lane's staged rows, so the lanes' stages fill together.

Shared verification past the 16-row qualified window uses measured increments
from the shared forward path through 32 rows, anchored at the 16-row window
price. This calibrates the batch cost without widening a stream's qualified
window. Older wide-batch receipts retain their recorded source, device and measurement provenance; their
four-stream tradeoff does not establish the current no-slowdown gate.

### Requalifying the drafter

Run the complete matrix from one frozen native binary, with its source commit and pinned model revisions:

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

The driver requires and records those revisions for qualification. Keep the source commit/patch with the binary hash.
Publish the metadata, raw request and cell records,
events and summary, not `private-logs/`. Device receipts remain device-specific; a new short smoke cannot
replace historical long-context, cancellation, load-memory or other-device evidence.
