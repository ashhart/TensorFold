# MiniMax H3 / FastH3 transformer blocks on CUDA (GB10)

This is a video model's denoising transformer, not a token model. It has no lane round and no drafted tokens, and
it runs its projections and attention in int8, which trades precision for speed. There is no bf16 path. It sits
beside the engine as a library a host pipeline calls; nothing in the lane core or the server changes.

## What it is

`zig build tf-h3 -Dnvcc=/usr/local/cuda/bin/nvcc` builds `zig-out/lib/libtf_h3.so` (it links libibverbs). The host
owns the text encoder, the sampler, the patch projections, the final layer and the decoders, and shares its CUDA
context (the device's primary context and the legacy default stream). Each denoising pass it hands over the packed
stream (rows of the model's hidden width, bf16) and gets all 50 blocks applied in place.

| Call | Does |
|---|---|
| `tf_h3_open(path)` | Maps the one-file checkpoint and quantizes the block projections on the GPU, or reads the saved copy; a `.tf-int8` path opens the saved copy alone |
| `tf_h3_forward(model, inputs)` | Every block over the stream; returns the seconds taken |
| `tf_h3_set_weight(model, block, projection, w, rows, cols)` | One block projection from bf16 on the device (a LoRA merged by the host), quantized as at load |
| `tf_h3_tp_init`, `_info`, `_connect`, `_close`, `_stats` | Join, wire and leave a sequence-parallel world of 2 to 8 ranks over RoCE |
| `tf_h3_forward_host`, `tf_h3_host_alloc`, `tf_h3_host_free` | The forward for inputs in host memory, for a rank with no framework |
| `tf_h3_profile`, `tf_h3_spent` | Per-stage seconds, each stage waited for |
| `tf_h3_close(model)` | Frees the model |

The checkpoint is FastVideo's `fastvideo_fasth3_8step_v2_pruned_bf16.safetensors` (50 blocks, hidden 5376, attention
width 7168, MLP 14336). The int8 copy is written beside it as `<checkpoint>.tf-int8` (21 GB) on first use.

## Arithmetic

- Projections: int8 weights with a scale per output channel; activations int8 with a scale per row (per row and 1024
  columns for the MLP's wide rows). Products on `mma.sync m16n8k32`, accumulated in int32.
- Attention: FastH3's video sparse attention. Rows are laid out in 64-row tiles; a video tile attends the prefix tiles
  and its top 20% of video tiles by pooled score, and the pooled branch is gated in. Scores and values are int8,
  softmax weights 8-bit per key tile with an online softmax across tiles.
- Rounding to bf16 where the reference's bf16 arithmetic rounds.

## Several GPUs

A rank runs the rows of its own routing tiles (`Inputs.tile0`, `tile1`, `cuts`). Every block it writes its tiles'
int8 keys, values, scales and pooled means into every peer's registered region over RDMA, takes theirs, and attends
as on one GPU. The kernels are the one-GPU kernels, so N ranks give the bits of one. The host moves each pass's rows
to the ranks and back over its own channel; that orchestration is not part of the library.

## Receipt

Measured 2026-10-16 on DGX Spark (GB10, compute capability 12.1, driver 580.173.02, CUDA 13.0.88 toolkit, Ubuntu
24.04.4, kernel 6.17.0-1029-nvidia), host ComfyUI at `a4b5a045` (0.39.0), checkpoint above, 864x480, 124 frames,
8 passes, sparsity 0.8, seed 7 (gate) and 8 (timed), the same machine for every row.

| Blocks | A pass | Sampler, 8 passes | Clip |
|---|---|---|---|
| ComfyUI's own blocks in bf16, dense attention | 10.6 s | 85.1 s | 103.0 s |
| ComfyUI's own blocks in bf16, its block-sparse attention 0.8 | 11.5 s | 92.2 s | 110.5 s |
| This library, one GPU | 6.68 s | 54.1 s | 72.6 s |
| This library, two GB10s over RoCE (two rails) | 4.06 to 4.14 s | 32.8 s | 51.1 s |
| This library, three GB10s | 3.02 to 3.06 s | 24.9 s | 43.5 s |

| Check | Result |
|---|---|
| This branch against the fork's engine (the same kernels before the port), 8 passes, same seed | 8 of 8 pass streams sha256-identical |
| Same seed after a ComfyUI restart | 8 of 8 identical |
| Two and three ranks against one | 8 of 8 identical at each |
| One block, dense attention, against ComfyUI's bf16 block, first pass | cosine 0.99989, relative error 0.0150, max abs 512 of values up to 1.08e5, worst row 0.077 |
| All 50 blocks, dense attention, first pass | cosine 0.99669, relative error 0.0813, max abs 7.9e5 of values up to 6.2e6, worst row 0.561 |
| All 50 blocks, sparse attention, against ComfyUI dense, first pass | cosine 0.641: a different attention, and at the first pass the tiles' routing is decided by noise |

The kernel contract (`zig build kernel-contract -Dnvcc=/usr/local/cuda/bin/nvcc` on the GB10) registers the four
int8 products and `h3_spans`; gate 5's records come from `kernel-contract-limits` over sm_80, 86, 89, 120 and 121.

## Not done

- No bf16 path: the projections and attention are int8 only.
- The tile attention, row, routing, pooling and quantize kernels are not registered with the kernel contract.
- One GPU model tested (GB10) and one checkpoint (FastH3 8-Step V2); no other size than 864x480 in this receipt.
- Four or more ranks, and a rank dying mid-pass (the host's pass fails; there is no fall back).
