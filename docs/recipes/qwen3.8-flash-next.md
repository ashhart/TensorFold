# Qwen3.8 Flash Next (`qwen4_exp`)

Measured on an M3 Ultra with 256 GB and MLX 0.32.0, with a 4-bit conversion that keeps the MTP weights.
Package: `src/tensorfold/families/qwen4_exp/` (`model.py` reference forward, `kernels.py` and `decode.py` fused
decode, `mtp.py` draft head, `runtime.py` what the engine serves).

## What decides the speed

- 125B total and 6B active parameters; 51B of the total is a hashed n-gram embedding.
- Hidden size 2,560. 48 layers: 12 repeats of 3 Gated DeltaNet layers and 1 sparse-attention layer, each
  followed by an MoE.
- Four residual streams joined by hyper-connections: each block reads a learned mix of the streams through a
  low-rank bottleneck of 320 and writes back to each stream with its own gate.
- DeltaNet: 48 value heads and 16 query-key heads of dim 128.
- Sparse attention: 24 query heads over 2 KV heads, head dim 256. An indexer scores blocks of 4 keys, and once
  the context passes 2,048 tokens each query reads its best 512 blocks plus its unfinished tail.
- MoE: 512 routed experts, top 10, expert width 640, plus a gated shared expert in every layer.
- A hashed 2- and 3-gram embedding (PLE) added before layer 1, hashed on the host from token ids it already
  holds.
- MTP: one draft layer (sparse attention over its own cache plus an MoE over four streams) that reads the main
  model's streams after the last layer and the next token's embedding.
- Vocabulary 248,320. 4-bit affine weights in groups of 32; about 4.26 GB of weights read per token, a floor of
  about 5.3 ms.

## What worked

The first port decoded at 30 tok/s with about 4,500 small kernels a token: Python built and encoded about 31
ms of graph against about 5 ms of weight reads. The fused decode brought that to about 690 kernels and 79
tok/s serial; drafting did the rest.

Through the server with exact sampling (T 1.0, top-k 20, top-p 0.95), thinking on, byte-identical to serial:

| Request | tok/s |
| --- | --- |
| short answer | 88-92 (79 serial) |
| code | 110 (80 serial) |
| file edit (a rename in a 5k-character file) | 170-176 |
| 18k-token context | 77-84 |
| 23k-token agent prompt, 512 thinking tokens, then a long tool call | 91-99 |

The pieces:

- One kernel for each step between two weight reads (`kernels.py`): the hyper-connection norm with the
  previous block's write-back; the hyper-connection projections; the whole DeltaNet step after its
  projection (conv window, conv, SiLU, fp32 L2 norms, g and beta, the delta-rule update and read-out, the gated
  RMSNorm); the router; expert gate/up and down with in-kernel routing; attention prep with q/k norms and RoPE;
  the attention gate. A step is GPU-bound: 6.5 ms of host against 12 ms of GPU.
- Projections that read the same input stacked into one matrix: DeltaNet q/k/v, z, b and a; attention q, k,
  v and the indexer; the router rows plus the shared expert's gate row as one matvec.
- Grouped expert kernels for windows of 3 or more rows, reading each distinct expert once for all the rows
  that picked it. 8 consecutive tokens pick about 40 distinct experts across their 80 slots.
- Tokens drawn on the GPU: the host sampler's argpartition over 248k logits cost about 2 ms a draw, twice per
  drafted round.
- Sparse-attention block selection in kernels: each finished block's indexer key is pooled once, one kernel
  scores every block for all rows, another picks each row's top 512 by radix select, and one attention kernel
  serves every row. Forward time in ms at 1/2/4/8 rows:

  | Context | Reference selection, MLX attention per row | Kernels |
  | --- | --- | --- |
  | 2k | 12.6 / 17.1 / 26.5 / 43.2 | 12.4 / 16.6 / 25.9 / 42.0 |
  | 18k | 15.0 / 18.4 / 29.3 / 48.3 | 13.1 / 17.3 / 27.3 / 44.5 |
  | 50k | 16.1 / 18.8 / 31.1 / 49.2 | 13.4 / 17.9 / 28.8 / 45.3 |

- MTP drafting: each round verifies the pending token and up to 3 MTP drafts in one forward and keeps them up
  to the first mismatch; the MTP head then absorbs the kept rows and chains the next drafts. The depth follows
  a running average of acceptance: 1 draft below 80%, 2 below 90%, otherwise 3. Acceptance runs about 87% on
  code and 73% on prose. A chained draft step costs about 1.3 ms and its verify row about 5 ms, so depth pays
  only at high acceptance. On the agent prompt above, a cap of 3 ran 91.1 tok/s against 86.8 with 1.
- Copy windows: when the context holds the text being written (file edits), a round verifies up to 7 copied
  tokens, and the MTP head only absorbs that round's rows instead of chaining drafts nobody will use.

## Tried and rejected

- Folding the hyper-connection norm into the split down projection: every threadgroup redid the norm, GPU 12.0
  to 13.2 ms a step.
- Hyper-connection kernels that read their weights once for all rows by looping over rows inside a
  threadgroup: 1 row 12.48 to 12.82 ms, 4 rows 25.82 to 26.43. Rows in parallel threadgroups stay. Padding
  their threadgroup arrays against bank conflicts saved about 0.1 ms and stays.
- Grouped expert kernels in which each simdgroup loops over its rows: gate/up at 4 rows 97.6 to 108.2 us.
- Converting a weight word's nibbles once for all rows in the matvec: same bits, 124 to 165 us at 4 rows
  (register pressure).
- Entering copy windows on short matches: in fresh code, coincidental matches (indentation, "self.") failed 56
  of 70 copied tokens and cost 5%, so entry needs 8 matching tokens.
- MLX's quantized matmul for multi-row windows on this GPU and MLX version, and bf16 router logits (see below).

## Exactness

- Numerics follow the checkpoint's training framework (PyTorch with the FLA kernels): fp32 math, one bf16
  rounding per op where a tensor is stored in bf16, fp32 L2 norms in the delta rule. Serial decoding goes
  through the kernels, so they are the reference. Teacher-forced NLL over 1,500 tokens matched the MLX
  reference forward within 0.003 nats; at 18k context, 150 tokens, 1.5443 against 1.5544 (noise).
- Every kernel treats each of R consecutive rows on its own, in a fixed order, and takes the row count from a
  small buffer.
- Projections through `project` (attention and DeltaNet projections, their output projections, the head): one
  row goes through MLX's quantized matmul; 2 or more rows through `qmv_rows`, one simdgroup per row running MLX's
  one-row loop (strategy B in the [recipe book](README.md)). MLX 0.32.0's quantized matmul on the M3 Ultra
  sums a row differently when 2-4 rows ride together. `qmv_rows` copies even MLX's bf16 partial sums in its
  vector load.
- Router logits in fp32: bf16 logits over 512 experts tied at the top-10 cut in about a third of a token's
  layers, and the tie decided the expert. Ties go to the lower id.
- Attention: one kernel for all rows. Each row's key list is cut into 16 parts by that row's own length, the
  parts run in parallel and merge in order, so a row's result does not depend on the other rows.
- Block selection: fp32 scores (the sum over index heads of relu(q . pooled block) over sqrt(d)), radix select
  of the top 512 with the lowest block id among ties, keys in position order, then the tail.
- Rollback: DeltaNet conv and recurrent states are kept for every row of the last call; the n-gram history and
  the PLE conv tail are restored; attention caches are trimmed, including pooled blocks no longer complete.
- At load, windows of 2, 3 and 4 rows are compared with one-row steps from a 48-token prompt, logits exactly;
  drafting is off if they differ. On the real model, serial bits held in every window at the 2,048-key boundary
  and at 18k, including a window of 8 cut to 3 and then continued.
- Traps: this conversion stores the centred norms around 1 instead of 0; `norms_stored_around_one` detects it
  and subtracts 1 at load. Derived weights (stacked matrices and their row views) must be evaluated at load:
  the server's engine thread cannot evaluate lazy arrays made on the main thread.

## Next

- Expert kernels that dequantize a weight tile once and multiply it for every row of the window with
  simdgroup matrices, the same kernel serving one row. The expert kernels cost about 15 us per extra row per
  layer whatever the number of distinct experts, so they are bound by the dequantize-and-multiply work each row
  repeats. That is the 2.4 ms of the roughly 5 ms each extra verify row costs at 18k; the rest is the
  hyper-connection projections (1.3 ms), attention and the other projections.
- n-gram ids computed on the GPU, so decode can run one step ahead as Nemotron does.
