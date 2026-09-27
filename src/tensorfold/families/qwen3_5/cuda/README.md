# Qwen3.8 dense on CUDA

The CUDA engine for Qwen3.8-27B with model type `qwen3_5`, written in PyTorch, Triton and shared CUDA
extensions. It reads the MLX 4-bit checkpoint (`Vontra/Qwen3.8-27B-MLX-4bit`, affine 4-bit, groups of 64) as
stored and drafts with `z-lab/Qwen3.8-27B-DFlash2`. See
[the recipe](../../../../../docs/recipes/qwen3.8-27b.md#cuda) for setup and public benchmark fixtures.

Every kernel on the verify path gives a row the same bits whether it runs alone or as one of up to 128 rows of
a window. Serial decoding runs through the same kernels, so a drafted token is the token serial decoding
produces on this machine. The bits differ from the Mac engine's; each engine is its own reference.

## Kernels

| File | Kernel | What it computes | Why the bits do not depend on the row count |
| --- | --- | --- | --- |
| `qmm.py` | `_qmm`, `_reduce`, `_group_sums` | the lane matmul on MLX's stored layout: per 64-input group a tensor-core dot of bf16 inputs and integer-valued bf16 weights, then `acc + p * scale + xs * bias`, groups summed in order | rows run as one block of 16, 32, 64 or 128; the K split depends only on the weight's shape (`split_k`), and split slices are added in slice order |
| `qmm_fast.py` and `tensorfold/cuda/kernels/qmm.cu` | `qmm_kernel`, `reduce_kernel` | decode matmul on weights packed once at load into tensor-core fragment order | the same group arithmetic and weight-shape K split as `qmm.py` |
| `glue.py` | `_embed` | a 4-bit embedding row, dequantized | one program per row |
| | `_add_rmsnorm` | residual add, RMSNorm, and the fp32 group sums the next matmul needs | one program per row |
| | `_gdn_pre` | GDN's depthwise convolution over each node's own path (a tree window), q/k/v split and norms, the decay `g` and `beta` | each node reads its path's inputs only |
| | `_gated_norm`, `_swiglu`, `_attn_prep`, `_gate_mul` | gated RMSNorm with silu(z); SwiGLU; q/k norms and rotary; the attention output gate | elementwise or one row at a time |
| `tensorfold/cuda/kernels/gdn.cu` | `tree_kernel`, `replay_kernel` | GDN tree updates and accepted-path commits for one or several streams | every tree node and replay uses the same state-update routine in path order |
| `tensorfold/cuda/kernels/attention.py` | `_paths`, `_shared`, `_tail`, `_merge` | tree attention over each stream's committed keys and its root-to-node path | a query's chunks and merge order depend on its own key range, never on other rows or streams |
| `tensorfold/cuda/sampling.py` | `sample_rows`, `sample_streams` | position-keyed sampling (the rule in `engine/exact_sampling.py`): top-k candidates on the GPU, the draw on the host in float64 | one row at a time |
| `distributed.py` | `row_partial`, `gather_rank_partials` | default two-rank decode: column-parallel projections keep whole output rows; row-parallel ones return fp32 partials that both ranks all-gather and add rank 0 then rank 1, rounding once | a fixed summation order instead of NCCL's all-reduce |
| `b16.cu` | `b16_linear` | a plain fp16/bf16 projection (`x @ W.T + bias`) for the tensors an EXL3 pack stores at 16 bits, one warp per output | each output sums its own row in a fixed order, whatever the row count |
| `exl3_load.py` | the shared module | `Exl3Linear` for the pack's EXL3 tensors: the rotation, the trellis GEMV of any codebook and width, and the split-K plan the weight's shape decides | the shared module's kernels are row-invariant by construction |
| `dflash2.py` | `_dconv_kernel`, `_prep_kernel` | the DFlash2 draft model with 4-bit projections through the same lane matmul, fused dynamic convolution and norm plus rotary; on two GPUs each rank holds half the heads, MLP and draft vocabulary | drafts only propose; the target verifies every token |

## The rest of the package

- `weights.py` loads the checkpoint as stored (the vision tower is skipped).
- `forward.py` has `tree_forward`, `multi_tree_forward`, `commit`, `commit_streams` and `State` for one or several streams.
- `prefill.py` processes up to 4,096 prompt tokens per chunk through shared `qmm_prefill8.cu`, `gdn_prefill.cu` and `prefill_attention.py` kernels. `prefill_glue.py` produces FP8 activations with row scales and group sums. Two-rank prefill gathers bf16 partials and adds them in fp32. Prefill is chunk invariant but differs from decode, so the prefix cache retains prompt ends and a follow-up prefills the reply again.
- `decode.py` has `prefill` through `prefill.py`, `serial_decode`, `draft_decode` (trees from DFlash2 or copies from the context) and an optional per-round trace.
- `decode_tp.py` is the two-GPU loop: rank 0 decides each window and shares it; both ranks run the same forwards.
- `engine.py` is what `tensorfold serve` runs, with startup memory admission, prefix reuse and the rank-1 follow loop.
- `multi.py` shares verify rounds between requests when an explicit `--parallel N` exceeds one, on one or two ranks. It uses `tensorfold/cuda/scheduler.py` for request admission and `draft_tree.py` to allocate draft windows. CUDA `--parallel auto` serves one request at a time.
- `reference.py` is a plain fp32 PyTorch forward for the quality tests.

The tests in `tests/cuda/test_qwen27_*.py` cover row invariance, tree windows against serial paths, accepted-state commits, weight layouts, chunked and resumed prefill, shared streams and the two-rank share protocol. Shared kernel tests are in `tests/cuda/test_qmm.py`, `test_gdn.py` and `test_attention.py`.
