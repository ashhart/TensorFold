# GLM-5.3-Flash on CUDA: source notes

The longer descriptions these files carried as multi-line header comments, kept here as written.

## `zig/kernels/cuda/flashnext/sample.cu`

Flash Next port: device-side greedy candidates. One block per bf16 logits row: the row's first maximum (lowest column among equal maxima, as argmax picks it), mapped to a global token id, and the row's log-sum-exp in fp32. out[row] = {id, max (fp32 bits), lse (fp32 bits), 0}. Build: nvcc -cubin -arch=sm_121 -O3 -o fn_sample.cubin fn_sample.cu

## `zig/kernels/cuda/glm/dflash.cu`

DFlash2 block drafter kernels (our own, after z-lab/dflash's reference, MIT): the target's layer taps, the drafter's per-head norms with rotary positions, its grouped two-tap dynamic convolutions, SwiGLU, and its attention over a stream's context ring (the target's features, a 2048-token window) and the 8-row block (bidirectional). Layout: hidden 4096, 32 query heads and 8 key/value heads of 128, a block of 8 rows a stream.

## `zig/kernels/cuda/glm/exl3.cu`

GLM-5.3-Flash's EXL3 routed experts (4-bit "mcg" trellis, ExLlamaV3's format: MIT, Copyright (c) 2025 Turboderp) on CUDA: the grouped trellis GEMV/GEMM and the Hadamard rotations around it. From TensorFold's python-0.6 src/tensorfold/families/glm5_next/cuda/exl3.cu, without torch, over the routed pairs alone (the shared expert runs on the dense linears), with items of 64 pairs for prompt chunks (each expert's tiles read once a chunk). A 16x16 tile is 32 little-endian 32-bit words; lane L of a warp decodes the tile's values 8L..8L+7 from words L-1 and L, and those eight values are exactly the B fragments of two mma.m16n8k16 (columns 0-7 and 8-15 of the tile). Every output depends only on its own pair: pairs share an mma tile but mma keeps rows independent, the K range of every warp and split is fixed by the shape, and warps and splits are summed in a fixed order, never with atomics.

## `zig/kernels/cuda/glm/experts_sym.cuh`

GLM-5.3-Flash's symmetric routed experts: flashnext/experts.cuh with a group's block holding its scales alone (the bias of a symmetric int4 group is -8 x its scale, exact in bf16): 16 words a 32-row tile instead of 32. Shared by experts_sym.cu (decode form) and experts_sym_prefill.cu (prefill form).

## `zig/kernels/cuda/glm/experts_sym_prefill.cu`

GLM-5.3-Flash's symmetric routed experts, prefill form: flashnext/experts_prefill.cu reading experts_sym.cuh's blocks, each group's bias pair -8 x its scale pair (exact in bf16). Prefill: w = bf16(fma(q, s, b)), one fp32 chain over K; chunk-invariant bits, not decode's (replies prefill again).

## `zig/kernels/cuda/glm/index.cu`

GLM-5.3-Flash on CUDA: the DSA indexer, ours. A row writes its index key (LayerNorm) and gate into its stream's rings, the row that completes a pool of 4 writes the pooled key, each row scores its visible pools, keeps its 512 best (the lower pool on ties) and lists their tokens ascending. Rows name their own rings, pools and positions, so one launch serves any mix of streams, and a row's choice never depends on the rows beside it.

## `zig/kernels/cuda/glm/kda.cu`

From src/tensorfold/families/glm5_next/cuda/kda.cu (python-0.6), with the chain's body a device function and a segmented chain (any number of streams' windows in one launch) added. GLM-5.3-Flash's KDA (Kimi delta attention) on CUDA: one block of 1024 threads per head, a chain of R rows. Per row, following the Hugging Face definition: the depthwise conv over [conv state; q|k|v rows] (4 taps, fp32) with SiLU and one bf16 rounding; fp32 L2 norms of q and k (eps inside the sum, q times DK^-0.5); the per-channel decay g_i = exp(lower * sigmoid(exp(A_log) * (a_i + dt_bias_i))) in fp32; beta = bf16(sigmoid(b)); the delta rule in fp32 (decay along the key channel, read with k, correct toward v, read out with q); read-out rounded to bf16; then the gated RMSNorm bf16(w * (y * rsqrt(mean(y^2) + eps)) * sigmoid(gate)). Warp w owns value rows 4w .. 4w + 3, lane l the key columns 4l .. 4l + 3. The state update is one routine (``update``) shared with ``replay``, compiled without FMA contraction, so a replayed prefix of a window gives the bits of the serial steps.

## `zig/kernels/cuda/glm/latent.cu`

GLM-5.3-Flash on CUDA: MLA attention on the 512-wide latent cache, ours. A program is one row's 32 heads over one chunk of 512 keys (online softmax in 16-key tiles, tensor-core scores and outputs), and the chunks are merged in order; nothing a program computes depends on another row, so a row has the same bits alone, in a decode window, in sixteen streams' windows or in a prompt chunk. Every row names its own cache (``lc``, the latent rows of its stream) and position (``pos``), so one launch serves any mix of streams.

## `zig/kernels/cuda/glm/mm.cu`

GLM-5.3-Flash on CUDA: the dense linears, ours. One tensor-core kernel for every row count (a decode window's few rows, sixteen streams' windows at once, a prompt chunk's 2048): each group of 64 inputs is summed from zero by the bf16 mma, then acc = fma(P, scale, acc) in group order, and the K slices (fixed by the shape) are added in slice order, so a row's bits never depend on the rows beside it. Weights are tiled at load (glm_mm_tile*): a column tile of 8 outputs and a group of 64 inputs is 32 lanes' chunks of 16 values in mma B-fragment order (lane l: column l / 4; value 4 kt + j at input 16 kt + {2t, 2t+1, 2t+8, 2t+9}[j], t = l % 4), each chunk BITS / 2 words (ops.cu's 16-value packing; BITS 16: the bf16 values themselves). Scales are fp16 [N / 8][K / 64][8].

## `zig/kernels/cuda/glm/rows.cu`

GLM-5.3-Flash on CUDA: the row steps between the linears, ours (embedding, RMSNorm, the hyper-connections, the residuals, SwiGLU, the router and its top-8, the experts' combine, the KDA conv window). Each row is one block's, with fp32 sums in a fixed order (threads strided over the row, then a fixed tree), so a row's bits depend on its own inputs alone, never on how many rows the window holds.

## `zig/kernels/cuda/glm/vision.cu`

GLM-5.3-Flash's vision tower on CUDA (families/glm_cuda/vision.zig): the elementwise and row steps around its cuBLASLt products, as mlx-vlm's glm5_next/vision.py defines them. The residual stream stays in fp32; a step's bf16 output is rounded once, to nearest even.

## `zig/src/families/glm_cuda/api.zig`

GLM-5.3-Flash (glm5_next) on CUDA, two ranks: the pieces the weight loader, the forward and the native host share. Weights are a name -> device tensor store whose names are the Python engine's prepared-weight names (its weight tree: "layers.3.dsa.q_b.weight", "layers.5.moe.experts.up", ...), so every tensor the checkpoint loader builds can be checked byte for byte against the Python engine's (tools: capture_ref.py's weights.json hashes). Kernels: the fatbins embedded by the build (this family's, zig/kernels/cuda/glm; the shared grouped experts and the greedy candidates, zig/kernels/cuda/flashnext).

## `zig/src/families/glm_cuda/blocks.zig`

GLM-5.3-Flash on CUDA: the weights' views and the layers of a step (forward.py's blocks): KDA and DSA attention, the dense MLP and the MoE, the hyper-connections around them, the final norm and the head; and the MTP head. Row-wise work covers the step's rows at once; the KDA recurrences, the attention and the indexer read the step's tables (forward.prepare).

## `zig/src/families/glm_cuda/copy.zig`

Copy drafts (prompt lookup): code, JSON and edits repeat what came before, often longer than the MTP head drafts with confidence. A stream keeps its prompt and every committed token, and an index of each 8-token run's latest end; when the history's last 8 tokens ran before, the tokens that followed them are a draft. A run inside the reply itself must match 16 tokens (a reply's own short repeats mislead more often). The drafts are verified like the head's, so a wrong copy costs rows, never the reply.

## `zig/src/families/glm_cuda/dflash.zig`

DFlash2 block drafts (opt-in: TF_GLM_DRAFTER=dflash2, TF_GLM_DFLASH2 the checkpoint's folder). IncoAI's drafter (incoai/GLM-5.3-Flash-DFlash2, CC BY-NC-ND 4.0: non-commercial use; read from its own download, never changed on disk or redistributed) after z-lab/dflash's DFlash2 architecture (MIT); this implementation is our own. The drafter is a 5-layer Qwen3-style decoder (hidden 4096, 32/8 heads of 128, SwiGLU 12288) that drafts a whole block at once: rows [anchor, MASK x 7] at positions start .. start + 7 attend, bidirectionally, to each other and to the stream's context: the target's completed residuals after layers 5, 14, 24, 33 and 42 (each the mean of its 4 mHC streams), concatenated, through fc and a norm, then each layer's own k and v projections (a ring of the last 2048 positions). Two-tap dynamic convolutions along the block wrap both sublayers. The block's last 7 rows go through the target's head; a path selector picks each draft among the row's 16 likeliest tokens by the previous draft's codebook row. Drafts never change a reply: the target verifies them. Both ranks run the drafter on their own copy (the head is each rank's vocabulary half, its candidates gathered); rank 0 alone runs the selector, and the verify frame carries its drafts to rank 1.

## `zig/src/families/glm_cuda/draw.zig`

A sampled row's token from both ranks' gathered candidates (fn_cand_topk): each rank's 64 largest logits, global ids, its log-sum-exp and its log-sum-exp at the row's temperature. The merged 128 hold the vocabulary's top 64 exactly, so a rule within them draws as lanes.sampling.choose over the whole vocabulary would: top_k 1..64 by choose itself (only the top k take part), a wider rule when its nucleus (top_p) or floor (min_p) ends within the top 64, keyed the same way (seed, position, global id). A rule that reaches past them asks for the full row.

## `zig/src/families/glm_cuda/forward.zig`

GLM-5.3-Flash (glm5_next) on CUDA, one rank of two: the forward on this family's own kernels, for any mix of streams. A step is a list of segments (a stream and its rows, in order): the row-wise work (projections, norms, hyper-connections, router, experts, head) runs over all the step's rows at once, and the attention, the indexer and the KDA recurrences read the step's tables (each row's position and its stream's caches). Every kernel keeps a row's bits its own, so a stream's tokens are the same alone and beside fifteen others. Both ranks make the same calls in the same order. Tokens are drawn from both ranks' vocabulary shards (an all-gather of each rank's best candidates), so both ranks reach the same tokens with no host decision that could differ.

## `zig/src/families/glm_cuda/guard.zig`

A rank's watchdog. The GB10's memory is the host's: when a rank's caches, graphs and buffers take the last of it the kernel stalls the whole machine (no OOM kill comes), and the peer then waits in a collective forever. So a thread reads MemAvailable every 100 ms and the rank exits at the floor (TF_MEM_FLOOR_GIB, 2 GiB), which frees everything it holds; below the floor and 3 GiB more it is ``tight`` and rank 0 admits no new request. It also exits when its peer is gone (the control link hung up or timed out, or NCCL reports an error): a rank alone can only hang, and an exited one can be restarted.

## `zig/src/families/glm_cuda/kern.zig`

GLM-5.3-Flash on CUDA: every launch the forward makes, on this family's own kernels (glm/mm.cu, rows.cu, latent.cu, index.cu, kda.cu, ops.cu, vision.cu) and the shared grouped experts. Each is written so that a row's bits depend on its own inputs alone: a window's rows, sixteen streams' windows and a prompt chunk's rows go through the same arithmetic. The attention and indexer launches read a step's per-row tables (``Rows``): each row's position and its stream's caches, so one launch serves any mix of streams.

## `zig/src/families/glm_cuda/native.zig`

GLM-5.3-Flash (glm5_next) on two GB10s behind the native family interface: rank 0's lane core drives the engine through a lane backend that first sends each call to rank 1, which replays it on its own half of the model (``follow``). Every token is drawn from both ranks' vocabulary shards (an NCCL exchange), so both ranks reach the same tokens and rank 1 needs nothing back. Up to 16 streams decode together, each with its own caches sized to its request out of one memory pool; a released stream's kept prompt state stays for a later prompt that extends it while the pool has room.

## `zig/src/families/glm_cuda/seq.zig`

GLM-5.3-Flash on CUDA: a stream's committed caches (forward.py's State), sized to its request. KDA conv windows and recurrent states (two parities), the main and MTP latent caches, the indexer's keys and gates (rings: only a window's rows and the 3 before it are read) and pools, and the kept prompt state a later prompt can resume.

## `zig/src/families/glm_cuda/vision.zig`

GLM-5.3-Flash's vision tower on CUDA (rank 0 encodes; rank 1 receives the features), as mlx-vlm's glm5_next/vision.py defines it: the patch embedding (a linear over a patch's 2 x 14 x 14 x 3 values), 24 blocks (RMSNorm, attention with RMSNorm'd q and k and a 2-D rotary over each frame's patches, a limited SwiGLU), the post-norm, the 2 x 2 downsample (a linear over each merge block's four patches) and the merger (a projection, LayerNorm + GELU, a limited SwiGLU) to the language model's 4096. The linears are cuBLASLt products (bf16 in, fp32 accumulate, bias added before the one rounding); the residual stream stays fp32; glm/vision.cu does the rest. Checked against the torch reference (tools/vision_ref.py) by tf-cuda-test glm-vision.

## `zig/src/families/glm_cuda/weights.zig`

GLM-5.3-Flash's prepared weights for one rank of two, read from the converted checkpoint (glm-affine-experts-v1: MLX affine 4-bit routed experts in groups of 64 with fp16 scales and biases, everything else BF16 / FP32), as the Python engine prepares them (families/glm5_next/cuda/weights.py, its affine-experts path, split.py's rank rules): every tensor under the name, shape and bytes of the Python engine's prepared weights (its weight tree, the names tools/capture_ref.py hashes), uploaded to this rank's GPU. The derivations are exact: rank rows and columns cut on the host, BF16 rows stacked, the kv_b blocks split per head (the latent path's absorb weights), fp16 expert scales rounded to bf16 (round to nearest even, as torch), and the experts' blocks in experts_pack.cu's index map. The draft head is the verification head itself (no 4-bit copy).

## `zig/tests/cuda/glm_forward.zig`

GLM-5.3-Flash on CUDA: the family's kernels against float64 references and the row-invariance contract (a row's bits alone equal its bits in any window, any mix of streams), on one GPU; and the two-rank engine's contract on the real checkpoint: a stream's greedy reply is the same alone, beside 15 other streams, drafted, and resumed from a kept prompt state. usage: tf-cuda-test glm-kcheck tf-cuda-test glm-batch <rank> <rank 0 address> <port> <converted dir> [tokens] [sync]

## `zig/tests/cuda/glm_weights.zig`

GLM-5.3-Flash on CUDA: the native weight loader (src/families/glm_cuda/weights.zig) against the Python engine's prepared weights (tools/capture_ref.py's weights.json: each tensor's name, bytes and sha256). usage: tf-cuda-test glm-weights <converted dir> <rank> <capture rank dir>
