# Qwen3.8-27B dense (`qwen3_5`)

Measured on an M5 Max with 128 GB, MLX 0.31.2 and mlx-lm 0.31.3, with the 4-bit checkpoint. Package:
`src/tensorfold/families/qwen3_5/`, engine `engine/lane_engine.py`, kernels in `src/tensorfold/kernels/`,
drafter in `src/tensorfold/drafters/`.

## What decides the speed

- 64 layers: 48 Gated DeltaNet layers and 16 full-attention layers (every fourth). Hidden size 5,120 and a
  dense SwiGLU MLP of width 17,408 in every layer.
- Attention: 24 query heads over 4 KV heads, head dim 256, rotary on a quarter of each head. The query
  projection also produces an output gate.
- DeltaNet: 16 key heads and 48 value heads of dim 128, conv kernel 4. The fp32 recurrent state is about 3 MB
  a layer.
- Vocabulary 248,320. 4-bit affine weights in groups of 64: about 14.4 GB read per forward.
- One row is bandwidth-bound: the chip reads about 569 GB/s, MLX's one-row 4-bit kernel reaches 470-560 GB/s.
  So serial decoding tops out in the 20s of tok/s, and everything above that comes from verifying several
  drafted tokens per forward.
- Extra rows are nearly free up to 16 on the tensor units. Past 16 each row costs about 1.35 ms: a 32-row
  forward takes 60-63 ms against 44 for 16, and 64 rows take 117 ms.

## What worked

Speeds through the server, each byte-identical to the same server decoding serially:

| Step | HTML via a tool call | Code | Story | Whole-file edit |
| --- | --- | --- | --- | --- |
| MLX kernels, exact only up to 9 rows | 24.3 | 25.5 | 20.9 | |
| lane matmul, DFlash2 chains, 8-bit drafter, greedy | 93.3 | 69.0 | 35.0 | |
| plus pipelined graph build, 4-bit drafter, lane attention, greedy | 113.8 | 79.8 | 43.3 | |
| same, sampled (T 1.0, top-k 20, top-p 0.95) | 99.6 | 76.5 | 37.3 | |
| plus draft trees, sampled | 120.6 | 99.1 | 50.0 | |
| plus half-precision P, calibrated trees, in-place commits | 134 | 116 | 54 | 295 |
| plus exact prompts through the lanes and the 8-token copy rule | 129 | 109 | 57 | 339 |
| plus tiled weights and a draft vocabulary | 152 | 126 | 66 | 388 |

In agent sessions with thinking on and 21-23k tokens of context, short turns ran 75-94 tok/s on a cool machine
(4.0-5.2 tokens a round, 45-55 ms rounds) and 60-69 once hot. A 4,567-token copy task ran 319 tok/s. Prompt
processing that stays exact to token-by-token feeding runs 464-572 tok/s.

The pieces, in the order they paid:

1. Lane matmul (`kernels/lane_qmm.py`): 4-bit weights times bf16 activations on the M5 tensor units, one
   arithmetic for every row count. Whole-model forward at 1/2/4/8/16 rows: 46.8/47.1/48.1/49.7/53.7 ms,
   against MLX's 32.9/35.2/49.2/86.8/112.6. One row costs about 1.35x MLX's vector kernel; 16 rows cost what
   one does.
2. Pipelined graph build: building the 64-layer graph in Python took about 8 ms with the GPU idle. Handing it
   to the GPU every 4 layers took the forward from 46.6 to 38.7 ms at one row, same graph and bits.
3. The DFlash2 block drafter: 5 layers, trained on blocks of 8, reading the target's hidden states at layers
   5, 19, 33, 47 and 61. At 4 bits a block costs about 7.5 ms against 12.5 at 8 bits, same acceptance.
4. Lane attention (`kernels/qwen/dense/v1/lane_attention.py`): decode attention for 1-128 queries with fixed arithmetic. At
   20k keys an 8-row window cost 0.42 ms a layer against 1.36 for MLX run query by query.
5. Draft trees: best-first over the drafter's candidate lattice, 4 children a node, up to 15 nodes, the scores
   carrying the target's own Gumbel noise, all verified in one forward. Offline tokens a pass: code 4.35 to
   5.57, HTML 3.33 to 4.36, story 2.89 to 3.73. Calibrating the scores on about 1,500 traced rounds gave 6.5%
   more accepted tokens.
6. One 32-row tensor op for 17-32 rows: a 17-row forward went from 75 to 60 ms, so copy chains run up to 31
   nodes (whole-file edits 239 to 293 tok/s).
7. Commits that no longer copy the KV cache: 7 ms a round at 20k context and 12 at 40k, down to 2.7.
8. Fused glue kernels (`kernels/lane_glue.py`): residual add plus RMSNorm plus the next matmul's group sums,
   the DeltaNet conv, SiLU, q/k norms and gates, the gated norm, SiLU(gate) times up. Small gain, as noted in
   the method.
9. Exact prompts through the decode path in 128-row chains (`LaneEngine.lane_prefill`). MLX's chunked prefill
   gives a prompt row different bits depending on where the chunk boundaries fall, and those followed the
   cached prefix: 4 of 12 replayed requests differed between two servers.
10. MLX's buffer cache capped at 8 GB: a 33k-token prefill had left a 103 GB process.
11. The copy rule: a verbatim copy backed by 8 or more matching tokens replaces the tree. Such a match existed
    in about 25% of tree rounds and its next token was right 94% of the time.
12. Tiled weights: each 4-bit weight regrouped in place into contiguous 1 KB blocks per 32-column tile and
    64-input group, same bytes and bits. Rounds 58.2 to 49.7 ms.
13. A draft vocabulary: 99.64% of committed tokens have ids below 98,304, so the drafter reads only those head
    rows (40% of the head). Verification still reads the whole vocabulary.
14. A session n-gram prior on tree scores at weight 0.1: 4.49 to 4.62 tokens a pass, kept on only while it
    helps the stream.
15. Serving: pinned system-block snapshots (a fresh session reused 21,415 of 21,481 prompt tokens and started
    in 0.42 s) and session-title requests as preemptible background jobs (first token 1.3 s to 0.27 s). One
    request decodes at a time: a second request sharing rounds turns drafting off.

## Tried and rejected

- MLX's own quantized matmul for verify windows: rows match only up to 9, and a 9-row check costs 4.3x a
  one-row step.
- An exact replica of MLX's 8-bit one-row kernel for 32 rows on an M3 Ultra: bit-identical, but 1.84 ms
  against 0.81 for MLX's non-exact batched kernel, and a 32-position round cost 307-363 ms.
- Int8 activations on the tensor units: 27% faster per op, not faster end to end, 7x the error.
- Wider windows and trees: 64-row windows cost twice 32; 31-node trees gave 7% more tokens for 1.6x the
  matmul time; 127 nodes gave 5.47 tokens a round against 4.57 at 15, at about 4x the cost.
- Matmul variants that were faster alone and slower live: a pipelined group loop (+5% alone, -2.7% live), a
  per-shape split-K table (17% faster in a microbenchmark, rounds 53.3 to 69.3 ms live), software prefetch,
  tile-major scales, unpacking each tile once per threadgroup (0.72-0.88x).
- Memory-system tricks: hardware async copies into threadgroup memory cannot be reached from Metal source; a
  last-level-cache prefetch pass costs more than it saves.
- Attention: 1,024-key chunks win above about 40k keys, lose at 2-20k, and change the bits. Keys staged in
  threadgroup memory were 33-45% slower.
- Drafting: the checkpoint's MTP head as a sequential chain gave more tokens a pass (5.33 against 4.55) but a
  step costs 1.5-2.9 ms, and with about 7 ms of drafting per 50 ms round a sequential guess must cost under
  0.5 ms. A relay drafter (the MTP layer over a lane tree) guessed 13-19% more tokens a round but ran slower
  live, 72.3 against 76.0 tok/s. Fine-tuning DFlash2 on 372k of the target's tokens gave no gain. Drafters
  trained from scratch on rented H200 GPUs reached 3.8-5.1% first-token acceptance.
- The ceiling is the drafter's candidates, not the tree search: the truth is in the top 16 for 6.5-7.0 tokens
  a round, and no scorer picked the right candidate more than about 75% of the time past depth 1.
- Parallel lanes without a drafter (Jacobi decoding): 1.07 tokens a pass.

## Exactness

- Lane matmul arithmetic: for each 64-input group the tensor op multiplies the bf16 rows by the raw 4-bit
  weights into fp32; each output adds scale times that product plus bias times the fp32 sum of the group's
  inputs, groups in order. K splits into a fixed number of slices chosen from the weight shape alone. A
  column's bits depend only on K and the slice count, so projections that share an input stack into one call
  (`kernels/lane_fuse.py`).
- Lane attention arithmetic is fixed by absolute key position: 512-key chunks, 64-key tiles, 16-row tiles of
  query-head rows, fp32 online softmax, chunks merged in order.
- Rollback to exactly the kept prefix: attention layers are trimmed, and each DeltaNet layer re-runs its
  recurrence over the kept tokens from the recorded start state (`kernels/gdn_capture.py`).
- Trees: per-row RoPE positions, each node attending to the committed keys and its own path, the recurrence
  visiting nodes in row order one step from the parent's state (`kernels/lane_tree.py`).
- Tests: rows of any call equal the same rows alone at every projection shape; attention window rows equal
  single queries; tree nodes equal serial steps along their paths; a 16-token window through all 64 layers
  equals 16 single steps bit for bit; end to end, drafted replies equal `"draft": false` replies byte for
  byte.
- Traps: the tensor op is exact only up to 128 output columns. Its destination fragment layout depends on the
  operand types, and assuming the wrong one looked exactly like "half-precision P breaks row independence". The
  chat template's highest reasoning effort writes into the system prompt, so no system-block snapshot matched;
  medium writes nothing.

## Next

- More tokens a round: a drafter distilled at scale from the target's own tokens (top-32 logits per token),
  gated at 6.5 tokens a round offline.
- Cheaper rounds: hide drafting behind verification (6-9 ms a round), attention toward its tensor-op ceiling
  (the biggest win at 40-70k context), matmuls toward 90% of bandwidth.
