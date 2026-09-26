# Qwen3.8-27B dense (`qwen3_5`)

Measured on an M5 Max with 128 GB, MLX 0.31.2 and mlx-lm 0.31.3, with the 4-bit checkpoint. Package:
`src/tensorfold/families/qwen3_5/`, engine `engine/lane_engine.py`, kernels in `src/tensorfold/kernels/`,
drafter in `src/tensorfold/drafters/`. The same checkpoint and drafter on NVIDIA GPUs (one or two DGX Sparks)
are at the end: [DGX Spark (CUDA)](#dgx-spark-cuda).

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

## DGX Spark (CUDA)

The CUDA engine (`src/tensorfold/families/qwen3_5/cuda/`, kernels listed in its README) reads the same
`Vontra/Qwen3.8-27B-MLX-4bit` checkpoint and drafts with the same `z-lab/Qwen3.8-27B-DFlash2`. Measured on
DGX Spark (GB10, 128 GB unified memory, about 240 GB/s measured read) in NVIDIA's `pytorch:26.07-py3` container, one
Spark and two Sparks linked by their 200 Gb/s ports.

```bash
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --host 0.0.0.0 --port 8080
# two Sparks, the same command on each (rank 1 first); rank 0 serves HTTP
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --tp 2 --rank 1 --master 192.168.100.1
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --tp 2 --rank 0 --master 192.168.100.1 --host 0.0.0.0 --port 8080
```

Pull both checkpoints first on every Spark (`tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit
z-lab/Qwen3.8-27B-DFlash2`): with two Sparks both ranks draft, each with half the draft model. The ranks check
at start that they were given the same settings.

### Against vLLM with MTP

Decode tokens per second after the first token, one stream, 64-token replies, median of 5 seeds (1234 to
1238), through the same OpenAI client (`tools/bench_openai.py`) for both engines, with this release installed by pip and started by `tensorfold serve`. Code prompt: "Write a short
Python function that computes the Fibonacci sequence and explain it." as a raw completion. Chat prompt:
"Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example." through
the chat template with thinking off. Sampled means temperature 1, top-k 20, top-p 0.95.

| | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| vLLM, MTP=3, one Spark | 17.7 | 15.0 | 17.7 | 17.0 |
| TensorFold, one Spark | **49.6** | **45.8** | **49.2** | **45.9** |
| ratio | 2.80x | 3.05x | 2.78x | 2.70x |
| vLLM, MTP=3, two Sparks (TP2) | 33.1 | 30.4 | 31.6 | 30.5 |
| TensorFold, two Sparks | **82.4** | **58.9** | **76.2** | **71.1** |
| ratio | 2.49x | 1.94x | 2.41x | 2.33x |

vLLM ran NVIDIA's NVFP4 ModelOpt quantization of the same model (NVFP4 weights, FP8 DeltaNet projections and
KV cache, 22.1 GiB) with `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--max-model-len 32768 --max-num-seqs 16 --enable-prefix-caching --enable-chunked-prefill
--attention-backend TRITON_ATTN`, and across two Sparks `--tensor-parallel-size 2 --nnodes 2`. Without drafts
it decodes 10.5 tok/s on one Spark; with its own DFlash2 support at 7 drafts, 28.3 and 22.9. vLLM's drafted
output is not byte-identical to its serial output. Ours is: every run compares the drafted token ids with
serial decoding on the same engine by SHA-256.

Serial decoding runs 13.1 tok/s on one Spark and 22.5 on two. Single seeds vary a lot for both engines (ours
on two Sparks, code: 70.0 to 135.6 tok/s over the five seeds), which is why the table uses medians.

### What paid on CUDA

| Step | Effect |
| --- | --- |
| A row-invariant 4-bit lane matmul in Triton: per 64-input group a tensor-core dot, then scale and bias, groups in order, the K split fixed by the weight's shape | exact windows of 1 to 128 rows; the arithmetic of the Metal lane matmul with CUDA's own bits |
| The 4-bit words regrouped once at load into contiguous blocks per program and group, scales group-major (same bits) | weights stream at 200-220 GB/s instead of 107-130; one-row forward 123 to 75 ms |
| DFlash2's projections at 4 bits through the same matmul, the context's keys and values cached per layer | drafting 19.5 to 9.6 ms a round, same acceptance |
| DFlash2 on fused kernels (stacked q/k/v, Triton dynamic convolution and norm plus rotary, one mask and rotary table a round): 918 to 288 kernels | drafting 9.4 to 7.4 ms a round |
| Commits write the accepted rows in place; one launch replays every DeltaNet layer's accepted path | no cache copies at long context; 3.6 to 2.3 ms a round on two Sparks before the one-launch replay |
| Two Sparks: tensor parallel with fp32 partial sums gathered and added in rank order, the head split by vocabulary, the drafter split over both ranks | two-Spark code 89.2 to 95.3, chat 69.8 to 75.3 tok/s on the engine bench |
| 12-row windows | the same drafts accepted as at 16 rows on these prompts, about 3 ms less a round |

### Tried and rejected on CUDA

- Per-shape matmul settings picked from isolated sweeps: 3-8% faster alone, 1.5-2 ms slower per forward in
  the whole model.
- `NCCL_PROTO=LL` for the 128 all-gathers of a two-Spark forward: 16-row forwards went from 52.6 to 82.0 ms.
- Stacking the DeltaNet gate projections into one matmul: no gain at 16 rows.
- Wider trees: 32-64 rows accept about one more token a round but add 30-40 ms of verification, so 12 rows
  stays fastest (49.2 tok/s against 44.7 at 32 rows and 43.7 at 64 in a 256-token sweep).
- Longer DFlash2 blocks (26 or 32 positions, the checkpoint was trained on 8): fewer accepted tokens than
  block 16 at the same width, 5.8 against 6.2 a round at 32 rows.
- More of the target's sampling noise in the tree scores (weight 1.0 instead of 0.7): fewer tokens a round.

### Where the time goes

On one Spark a 12-row round takes about 95 ms: 83 ms verifying (the weights stream near the read limit), 8 ms
drafting, under 1 ms committing. On two Sparks it takes about 60 ms: 52 ms verifying (each rank's 4-bit
matmuls about 42 ms at about 175 GB/s on half-size shards; the 128 all-gathers and the small kernels the
rest), 6 ms drafting, 2 ms committing. Half of the 12-row rounds end because every draft on the accepted branch was right
and the tree had nothing deeper, and at two thirds of the misses the right token was among the drafter's
16 candidates for that position: a tree that spends its rows on depth before siblings is the next thing to
try.

