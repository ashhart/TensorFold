# Qwen3.8 dense on CUDA

The CUDA engine for Qwen3.8-27B (`model_type` `qwen3_5`), written in PyTorch, Triton and one small CUDA
extension. It reads the MLX 4-bit checkpoint (`Vontra/Qwen3.8-27B-MLX-4bit`, affine 4-bit, groups of 64) as
stored and drafts with `z-lab/Qwen3.8-27B-DFlash2`. Measured on DGX Spark (GB10): see
[the recipe](../../../../../docs/recipes/qwen3.8-27b.md#dgx-spark-cuda).

Every kernel on the verify path gives a row the same bits whether it runs alone or as one of up to 128 rows of
a window. Serial decoding runs through the same kernels, so a drafted token is the token serial decoding
produces on this machine. The bits differ from the Mac engine's; each engine is its own reference.

## Kernels

| File | Kernel | What it computes | Why the bits do not depend on the row count |
| --- | --- | --- | --- |
| `qmm.py` | `_qmm`, `_reduce`, `_group_sums` | the lane matmul on MLX's stored layout: per 64-input group a tensor-core dot of bf16 inputs and integer-valued bf16 weights, then `acc + p * scale + xs * bias`, groups summed in order | rows run as one block of 16, 32, 64 or 128; the K split depends only on the weight's shape (`split_k`), and split slices are added in slice order |
| `qmm_fast.py` | `_qmm_tiled`, `_reduce` | the same arithmetic on weights regrouped once at load to `[N/64][K/64][64][8]` words with group-major scales, so each program reads one contiguous block per group | same arithmetic and splits as `qmm.py`; tile width, unroll depth, warps and stages change speed only |
| `glue.py` | `_embed` | a 4-bit embedding row, dequantized | one program per row |
| | `_add_rmsnorm` | residual add, RMSNorm, and the fp32 group sums the next matmul needs | one program per row |
| | `_gdn_pre` | GDN's depthwise convolution over each node's own path (a tree window), q/k/v split and norms, the decay `g` and `beta` | each node reads its path's inputs only |
| | `_gated_norm`, `_swiglu`, `_attn_prep`, `_gate_mul` | gated RMSNorm with silu(z); SwiGLU; q/k norms and rotary; the attention output gate | elementwise or one row at a time |
| `gdn_tree.cu` | `tree_kernel`, `preorder_kernel` | the gated delta rule over a draft tree: each node's state is its parent's state updated with the node's k, v, g and beta, in fp32; chains take a one-state path | a node's state depends on its ancestors only, updated in path order |
| | `replay_kernel`, `replay_many_kernel` | the commit: the accepted path replayed into the cached state, one layer or all 48 GDN layers in one launch | the same update order as serial steps |
| `attention.py` | `_paths`, `_shared`, `_tail`, `_merge` | tree attention: every query sees the committed keys, then its own root-to-node path; 512-key chunks of eight 64-key tensor-core tiles, merged in key order | a query's chunks and merge order depend on its own key range, never on other nodes |
| `sampling.py` | `sample_rows` | position-keyed sampling (the rule in `engine/exact_sampling.py`): top-k candidates on the GPU, the draw on the host in float64 | one row at a time |
| `distributed.py` | `row_partial`, `gather_rank_partials` | two GPUs: column-parallel projections keep whole output rows; row-parallel ones return fp32 partials that both ranks all-gather and add rank 0 then rank 1, rounding once | a fixed summation order instead of NCCL's all-reduce |
| `dflash2.py` | `_dconv_kernel`, `_prep_kernel` | the DFlash2 draft model with 4-bit projections through the same lane matmul, fused dynamic convolution and norm plus rotary; on two GPUs each rank holds half the heads, MLP and draft vocabulary | drafts only propose; the target verifies every token |

## The rest of the package

- `weights.py` loads the checkpoint as stored (the vision tower is skipped).
- `forward.py` has `tree_forward` (one verify window), `commit` (accepted rows into the caches) and `State`.
- `decode.py` has `prefill` (128-row chunks), `serial_decode`, `draft_decode` (trees from DFlash2 or copies from the context) and an optional per-round trace.
- `decode_tp.py` is the two-GPU loop: rank 0 decides each window and shares it; both ranks run the same forwards.
- `engine.py` is what `tensorfold serve` runs, with prefix reuse and the rank-1 follow loop.
- `reference.py` is a plain fp32 PyTorch forward for the quality tests.

The tests are in `tests/cuda/test_qwen27_*.py`. They cover row invariance of every kernel at 1 to 128 rows, a tree window against serial steps on every path, commit against serial state, the regrouped layout against the stored one, and the two-rank share protocol.
