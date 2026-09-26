# Qwen3.8 Flash Next (`qwen4_exp`)

Measured on an M3 Ultra with 256 GB and MLX 0.32.0, with a 4-bit conversion that keeps the MTP weights.
Package: `src/tensorfold/families/qwen4_exp/` (`model.py` reference forward, `decode.py` fused decode,
`mtp.py` draft head, `runtime.py` what the engine serves). Fused kernel version `v1` lives in
`src/tensorfold/kernels/qwen/flash_next/v1/kernels.py`. The same checkpoint on NVIDIA GPUs (one or two DGX Sparks) is
at the end: [DGX Spark (CUDA)](#dgx-spark-cuda).

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
| short answer | 105-107 (79 serial) |
| code | 112 (80 serial) |
| file edit (a rename in a 5k-character file) | 190 |
| 18k-token context | 98.5 |
| 23k-token agent prompt, 512 thinking tokens, then a long tool call | 103-115 |

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

  | Context | Reference selection, MLX attention per row | Kernels | Kernels, without the overheads below |
  | --- | --- | --- | --- |
  | 2k | 12.6 / 17.1 / 26.5 / 43.2 | 12.4 / 16.6 / 25.9 / 42.0 | 11.0 / 14.5 / 21.7 / 36.7 |
  | 18k | 15.0 / 18.4 / 29.3 / 48.3 | 13.1 / 17.3 / 27.3 / 44.5 | 11.6 / 15.1 / 22.9 / 38.7 |
  | 50k | 16.1 / 18.8 / 31.1 / 49.2 | 13.4 / 17.9 / 28.8 / 45.3 | 11.8 / 15.5 / 23.9 / 39.7 |

- MTP drafting: each round verifies the pending token and up to 3 MTP drafts in one forward and keeps them up
  to the first mismatch; the MTP head then absorbs the kept rows and chains the next drafts. The depth follows
  a running average of acceptance: 1 draft below 80%, 2 below 90%, otherwise 3. Acceptance runs about 87% on
  code and 73% on prose. A chained draft step costs about 1.3 ms and its verify row about 5 ms, so depth pays
  only at high acceptance. On the agent prompt above, a cap of 3 ran 91.1 tok/s against 86.8 with 1.
- Copy windows: when the context holds the text being written (file edits), a round verifies up to 7 copied
  tokens, and the MTP head only absorbs that round's rows instead of chaining drafts nobody will use.

### Removing overheads (+15-17%, byte-exact, the same NLL)

Each change was measured in one process through the serial engine on fixed prompts and seeds (thinking budget
512), with the output compared between variants. tok/s:

| Change | prose | code | agent prompt | file edit |
| --- | --- | --- | --- | --- |
| before | 92.4 | 100.2 | 96.6 | 119.5 |
| MLX command-buffer limits raised, each layer submitted as it is built | 95.6 | 103.4 | 99.7 | |
| expert gate/up hands its routing to expert down | 98.6 | 105.4 | 101.0 | |
| per-slot expert kernels at every window width | 99.6 | 107.6 | 103.6 | 128.5 |
| the MTP head's first draft for every verify row queued behind the verify | 100.4 | 109.5 | 105.8 | 129.0 |
| `a[0]` indexing, per-layer `arange` and `zeros` removed | 101.6 | 110.6 | 108.0 | 130.1 |
| the n-gram embedding in one lookup kernel | 105.9 | 113.8 | 111.2 | 133.6 |
| expert down without its in-kernel combine | 106.9 | 115.7 | 113.3 | 137.1 |

- MLX ends a command buffer once the bytes bound in it pass `MLX_MAX_MB_PER_BUFFER`. Every expert kernel binds
  the 420 MB expert stacks, so with the default each one ended a command buffer: an empty kernel binding them
  cost 28 us a launch against 12 with the limit raised. The family now sets `MLX_ENV` as Nemotron does, and the
  fused decode submits each layer with `mx.async_eval` (every 4 layers lost about 1%).
- Routing: every simdgroup of the down kernel (3,520 a call) repeated the top-10 selection over 512 logits
  before reading a weight. Gate/up already selects each slot's expert, so it now writes the picks and the
  renormalised weights for down to read: same bits, 0.5 ms a one-row step.
- With the routing handed on, the per-slot kernels beat the grouped ones at every width the fused path takes
  (8-row copy windows included), so no window is grouped. The down kernel writes each slot's output and the next
  hyper-connection norm combines them (the grouped path's write-back, the same arithmetic), so no threadgroup
  waits for its slowest expert.
- `a[0]` on an MLX array is a gather that copies: the DeltaNet state (3.1 MB a layer) was copied every step
  through `cache.ssm[0]`. `a.reshape(a.shape[1:])` is a view. `mx.export_to_dot` on a forward lists every
  primitive; that is how the stray gathers, fills and aranges were found.
- MLX's quantized embedding is three gathers and a dequantize per lookup, and the n-gram embedding looked up up
  to 16 shards a row: about 70 small operations. Its 128 shards are now 8 concatenated groups (the shards keep
  views into them, so memory does not grow), read by one kernel that matches `mx.dequantize` bit for bit. The
  token embedding and its tiling into the four streams are one kernel too.
- The MTP head absorbs every verify row and draws each row's first draft in the same GPU pass as the verify,
  before any token is read; the rows past the kept ones are then dropped from its cache. One host round trip
  less a round. Its input projections go through MLX's matmul, whose bits depend on the row count on this GPU,
  so a draft can differ from the old path's by rounding: acceptance moved by under 1%, and the output did not.

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
- Hyper-connection kernels with the rows of a window in one threadgroup and the weights read once, with the
  rows looped and with a single set of barriers: both slower than rows in parallel threadgroups.
- Expert down tiles of 4 or 16 model dims (8 is best at 1 and 2 rows), both rows of a 2-row window in one
  threadgroup, split-K expert gate/up, and the up projection with 16 dims a threadgroup: slower or level.
- Drafts drawn with temperature 0.8, 0.9 or 1.1, top-p 1.0 or top-k 40 (the target's rule is unchanged): within
  0.5%. Deeper draft-depth policies: 0.5-2% slower.
- fp32 8x8 simdgroup matrices: 25.8 TF/s against 23.7 for scalar FMA on the M3 Ultra, so padding one row to 8
  costs 8x the arithmetic. They do not make verify rows cheap on this GPU.

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

- A forward is about 680 dependent dispatches. A trivial dependent dispatch costs 3.2 us in raw Metal, and MLX
  adds about 3 us of GPU time (and 12-18 us of host time) to each one when the GPU is the limit: about 2 ms a
  forward. Encoding the whole fused forward in one custom primitive would remove it.
- An extra verify row costs about 3.5 ms at 2k context: the expert down projection about 1.3 ms, gate/up 0.9
  (most of the two is the new experts' weight reads, about 1.1 ms), the hyper-connection projections about 1.0.
  With rows near 2 ms, three drafts a round pay: about 3.2 tokens a round at 86% acceptance, which puts agent
  turns near 150 tok/s.
- n-gram ids computed on the GPU, so decode can run one step ahead as Nemotron does.

## DGX Spark (CUDA)

The CUDA engine (`src/tensorfold/families/qwen4_exp/cuda/`) reads the same
`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` checkpoint, unchanged, and drafts with its MTP head. Measured on DGX
Spark (GB10, 128 GB unified memory) in NVIDIA's `pytorch:26.07-py3` container (PyTorch 2.13, Triton 3.7.1,
NCCL 2.30.7), one Spark and two Sparks linked by their 200 Gb/s ports.

```bash
tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --host 0.0.0.0 --port 8080
# two Sparks, the same command on each (rank 1 first); rank 0 serves HTTP
tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --tp 2 --rank 1 --master 192.168.100.1
tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --tp 2 --rank 0 --master 192.168.100.1 --host 0.0.0.0 --port 8080
```

A round verifies the pending token and up to 6 MTP drafts, and a chain stops before any draft the head gives
less than 30%. `--mtp-drafts N` sets the most drafts a round, and `--no-drafts` decodes one token a round, the
serial reference; a request with `"draft": false` does the same for itself. The caches hold 8,192 tokens of
prompt and reply. The server keeps the state after the last request's prompt and after its reply, and a prompt
that extends either resumes from it (a second chat turn, a longer completion). With two Sparks both need the
checkpoint, and the ranks refuse to start when they were given different settings.

The first start builds the DeltaNet kernel with the container's `nvcc` and compiles the Triton kernels. Every
start then reads the hashed n-gram tables into the page cache and captures a CUDA graph for each decode window
size: about 80 s to ready on two Sparks and 90 s on one. One Spark holds 80.4 GB of weights on the GPU, and
each of two Sparks 40.7 GB. The 32 GB of n-gram tables stay in the checkpoint files, memory-mapped, and a
token reads 16 rows of 100 bytes from them. They have to stay in the page cache, or every token waits about
8 ms on the disk, so a single Spark has little memory to spare.

### Against vLLM with MTP

Decode tokens per second after the first token, one stream, 64-token replies, median of 5 seeds (1234 to
1238), through the same OpenAI client (`tools/bench_openai.py`) for both engines. TensorFold ran as released:
the package installed with pip into a fresh `pytorch:26.07-py3` container and started with `tensorfold serve`. Code prompt: "Write a short
Python function that computes the Fibonacci sequence and explain it." as a raw completion (14 tokens). Chat
prompt: "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."
through the chat template with thinking off (31 tokens). Sampled means temperature 1, top-k 20, top-p 0.95.

| | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| vLLM, MTP=3, one Spark | 42.4 | 33.2 | 40.9 | 37.6 |
| TensorFold, one Spark | 68.3 | 58.5 | 73.1 | 60.2 |
| ratio | 1.61x | 1.76x | 1.79x | 1.60x |
| vLLM, MTP=3, two Sparks (TP2, expert parallel) | 46.4 | 41.4 | 55.2 | 50.7 |
| TensorFold, two Sparks | 103.8 | 84.0 | 96.2 | 100.2 |
| ratio | 2.24x | 2.03x | 1.74x | 1.98x |

Greedy cells decode one text, so their five timings agree within about 2%. Sampled cells decode a different text per
seed: two-Spark code ranged from 92 to 112 tok/s over the five seeds, which is why the table uses medians. The
engine before packaging measured 69.9, 58.9, 73.7 and 60.4 on one Spark and 105.5, 85.5, 96.7 and 99.7 on two,
with a draft vocabulary built from our development tree; the package's own list is within 2.2% of that in every
cell.

vLLM ran Qwen3.8 Flash Next NVFP4 checkpoints in the `vllm/vllm-openai:qwen38-flash-next` build (vLLM
0.1.dev20073+g8e685d198), with `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` and
chunked prefill:

- One Spark: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` at revision `7c4f1bc1` (105.9 GB), with
  `--max-model-len 65536 --max-num-seqs 4 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.714`.
- Two Sparks: `RadixArk/Qwen3.8-Flash-Next-NVFP4` with `--tensor-parallel-size 2 --enable-expert-parallel
  --nnodes 2 --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.835` and a bf16 KV cache. Its NCCL needs the link's adapter whose RoCE v2 IPv4
  GID sits at the index the configuration names; check `show_gids` on both machines.

Both used the same client settings as ours: `ignore_eos`, one warm-up request, then the five seeds. The page
cache was dropped on both machines before the two-Spark vLLM start.

### Exactness

Drafted output is byte-identical to serial decoding on the same engine, token for token. It is the Mac
engine's contract with CUDA's own bits: every kernel on the verify path gives a row the same bits whether it
runs alone or as one row of a window, and sampling is the keyed rule of `engine/exact_sampling.py`, so a draft
is kept exactly when it is the token serial decoding samples there.

- Matmuls: each output is the same chain of tensor-core steps over the same 32-input groups in the same
  order at any row count. The K split is a constant of the weight's shape.
- Experts: a row and expert pair gets the same arithmetic whatever other rows share the expert. The combine
  adds a row's ten slots in slot order, then the shared expert.
- DeltaNet: a window's rows run in order inside one kernel from the committed state. Keeping a prefix replays
  those rows with the same update routine, compiled without FMA contraction.
- Attention: fixed 512-key chunks by absolute position; past 2,048 keys, a row's sparse key list depends only
  on its own indexer scores (the 512 best blocks, lower block id on ties).
- Two Sparks: each rank's fp32 partials are all-gathered and summed rank 0 first, then rounded once, so both
  ranks hold the same activations and draw each token from the same gathered candidates. Their tokens differ
  from one Spark's by rounding, and each is exact against its own serial decoding.
- CUDA graphs replay the eager kernels with the same arguments: eager and graph decoding give the same tokens.

What was checked:

| Check | Result |
| --- | --- |
| During development, every drafted run of the engine's benchmark against serial decoding, both prompts, sampled and greedy, one and two Sparks | the same token-ID SHA-256 in every run |
| The released server: each of 3 prompts with 2 seeds and greedy, 96 tokens, drafted against `"draft": false`, one and two Sparks | 9 of 9 equal on each |
| The released server: a reply resumed from the kept state (after a finished reply, and after a prompt) against the same request served cold, one and two Sparks | equal |
| Windows of 2, 3, 4 and 8 rows keeping 1, half or all rows, then continuing, at a short context and at 2,200 tokens (sparse rows) | bit-identical to serial steps in logits, streams and the next step |
| Teacher-forced over 150 tokens against a plain fp32 PyTorch forward | NLL 1.7539 against 1.7641; top-1 agreement 96.1% (99.2% at 2,200 tokens) |
| `tests/cuda/test_flashnext_*.py`, small random models with the real head sizes | 61 tests: row invariance of every kernel, windows against serial steps, graphs against eager, drafted against serial greedy and sampled, prefixes resumed against fresh prefills, two ranks as threads with the real loader slicing, the server engine on one and two ranks |

### The recipe

- The 4-bit matmul for groups of 32 in Triton, one program per column tile and K slice, tensor-core dots of
  the bf16 rows with the integer-valued weights. The loader regroups the words once so a program's group is
  one contiguous block. Launch settings are tuned per weight shape.
- Experts grouped by expert: one program per distinct expert and column tile, so a window reads each selected
  expert once. The shared expert rides in the same table as expert 512, the eleventh slot of every row.
- The hyper-connection read-out in three kernels for decode windows: the norm inside the down projection, the
  mix inside the up projection.
- A CUDA graph for every decode window size and for every MTP step size. Host work before a forward (token ids,
  the n-gram rows) goes into static pinned buffers, so a step is one graph launch.
- The MTP head drafts over 79,591 token ids (`cuda/draft_vocab.txt`): every id below 65,536 (the tokenizer's
  earliest merges), the added tokens, every id in CPython 3.14's standard library and in this repository, and
  every id that occurs at least 10 times in the Python sources and documentation of about 210 open-source
  packages from PyPI (337 million characters). A draft step reads a third of the full head. A token outside the
  list can never be a draft, which costs speed, never correctness. A list with every id below 98,304 instead
  (98,755 ids) was 7.7% and 8.6% slower on sampled code, one and two Sparks, and 1-2% slower in the other cells.
- The chain stops before a draft under 30%: a rejected draft costs a verify row, 3 to 4 ms on one Spark.
- Two Sparks: DeltaNet and attention split by heads, every expert split by its 640-wide intermediate (each rank
  reads half of each selected expert, so the load is even for any routing), the head split by vocabulary, and
  hyper-connections, router, embeddings, n-gram tables and the MTP input layers replicated. Two reductions a
  layer, after the branch's output projection and after the MoE. NCCL's all-gathers run on the current CUDA
  stream, inside the graphs.

What paid, in order:

| Step | Effect |
| --- | --- |
| Loading with large sequential reads instead of memory-map page faults | 391 to 64 s |
| n-gram tables read into the page cache at start | the forward's page faults cost 8 ms a token; serial 27.7 ms a token, 36.1 tok/s |
| CUDA graphs and launch settings per weight shape | one-row forward 26.4 ms; serial 38.1 tok/s; 5 drafts a round 71.8 code / 69.8 chat sampled |
| The draft head over part of the vocabulary (71,475 ids then; the shipped list has 79,591) instead of all 248,320 | a draft step 2.3 to 1.1 ms; 5 drafts a round 76.9 / 77.1 sampled |
| Chains that stop under 30%, the rule chosen by replaying recorded draft chains offline | greedy code 66.2 to 72.4-73.3 (the replay predicted 72.2) |
| Two Sparks, tensor parallel | one-row forward 20.1 ms; serial 50.5 tok/s |

### Tried and rejected on CUDA

- The 4-bit operand built as bf16 bits `0x4300 | q` (exactly 128 + q) instead of an integer-to-float
  conversion: no gain beyond run-to-run noise, at 1 row or at 5 to 8 rows.
- Depth that follows the last round (one deeper after a fully kept round, two past the kept run after a miss):
  no better than a fixed depth when replayed on recorded chains.
- Stopping when the chain's product of probabilities falls under a threshold instead of each draft's: no
  better on either Spark count.
- Two Sparks: sampling candidates gathered inside each step's graph and the n-gram rows read with one gather
  per checkpoint file gave 0-3%. A CUDA profile of whole drafted runs finds the GPU busy 94% of the time, so
  host work was not the limit.
- Two Sparks: `NCCL_PROTO` set to `LL`, `LL128` or `Simple`, 1, 2 or 4 channels, and forced GPUDirect RDMA
  (the platform reports no support) moved decode by 2% or less. `NCCL_GRAPH_MIXING_SUPPORT=0` gained 0-4% for
  three runs, then both ranks hung.

### Where the time goes

On one Spark a one-row forward takes 26.4 ms in its graph, against 21.3 ms to stream its 4.25 GB of weights at
200 GB/s. The big matmuls stream at 200-222 GB/s and the expert kernels at 194-207, but the 96 hyper-connection
matrices, 2 MB each, stay at 104-148 GB/s. Each extra verify row adds 3 to 4 ms, because each row brings about
five new experts a layer across 48 layers. A draft step is 1.1 ms with a 71,475-id head. A greedy code round at up to 6 drafts takes
54 ms: 45.8 ms verifying about 7 rows, 7.5 ms drafting, under 1 ms on the host.

On two Sparks a one-row forward takes 20.1 ms and a 7-row window 34.0 ms. Each rank reads 2.4 GB a token instead
of 4.25, but the forward makes 97 rank-ordered all-gathers of 25 to 77 us each, 2.5 to 5 ms a forward. At 7 rows
the expert kernels take half the forward. A greedy code round at up to 6 drafts takes 38.4 ms.

Greedy code on two Sparks is the weakest cell: a greedy draft must match an exact argmax, while a sampled
draft shares its position's Gumbel noise with the target and agrees whenever the two distributions are close.
The next steps are fewer bytes per verify row and cheaper hyper-connection kernels, which are latency-bound.

### Limits

- One stream: requests decode one at a time.
- Prefill runs 64-row chunks through the decode kernels, not tuned for long prompts. Prefix reuse keeps two
  states, the last prompt's and the last reply's, and the caches hold one sequence: a prompt that extends
  neither starts over.
- The context capacity is fixed at start.
- With two Sparks a request decodes to its end even when its client stops reading, so the ranks stay in step.
