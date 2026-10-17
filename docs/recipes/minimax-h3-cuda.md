# MiniMax H3 / FastH3 transformer blocks on CUDA (GB10)

This is a video model's denoising transformer, not a token model. It has no lane round and no drafted tokens. It runs
in the checkpoint's own precision: bf16 weights as shipped, bf16 activations, fp32 sums, softmax and routing. It sits
beside the engine as a library a host pipeline calls; nothing in the lane core or the server changes.

## What it is

`zig build tf-h3 -Dnvcc=/usr/local/cuda/bin/nvcc` builds `zig-out/lib/libtf_h3.so` (it links libibverbs). The host
owns the text encoder, the sampler, the patch projections, the final layer and the decoders, and shares its CUDA
context (the device's primary context and the legacy default stream). Each denoising pass it hands over the packed
stream (rows of the model's hidden width, bf16) and gets all 50 blocks applied in place.

| Call | Does |
|---|---|
| `tf_h3_open(path)` | Maps the one-file checkpoint and copies the blocks' bf16 tensors to the GPU as stored |
| `tf_h3_forward(model, inputs)` | Every block over the stream; returns the seconds taken |
| `tf_h3_set_weight(model, block, projection, w, rows, cols)` | One block projection from bf16 on the device (a LoRA merged by the host) |
| `tf_h3_tp_init`, `_info`, `_connect`, `_close`, `_stats` | Join, wire and leave a sequence-parallel world of 2 to 8 ranks over RoCE |
| `tf_h3_forward_host`, `tf_h3_host_alloc`, `tf_h3_host_free` | The forward for inputs in host memory, for a rank with no framework |
| `tf_h3_profile`, `tf_h3_spent` | Per-stage seconds, each stage waited for |
| `tf_h3_close(model)` | Frees the model |

The checkpoint is FastVideo's `fastvideo_fasth3_8step_v2_pruned_bf16.safetensors` (50 blocks, hidden 5376, attention
width 7168, MLP 14336). Its block tensors take 42 GB of device memory.

## Arithmetic

- Projections: the runtime's cuBLASLt `Linear` (`zig/src/cuda/cublaslt.zig`), bf16 in and out, fp32 compute, no TF32.
  Each (outputs, inputs) shape runs the algorithm cuBLASLt picks for 16,384 rows, whatever a rank's row count, so a
  row's sums take the same order on one GPU and on several.
- The rows around them (`h3.cu`): the modulated norm, the gated adds, the SwiGLU and the head layout round to bf16
  where torch's bf16 ops round in the reference (`1 + scale`, the norm's product, `silu(gate)` before the product).
- Attention (`h3_attention.cuh`): FastH3's tile-routed attention on `mma.sync m16n8k16`, bf16 q, k and v, fp32 scores
  and an fp32 online softmax. The softmax weights are rounded to bf16 for the value product, as flash attention does.
- Routing: tile means, tile scores, the top-k and the pooled branch in fp32. A video tile attends the prefix tiles and
  its top 20% of video tiles; the pooled branch is gated into its rows.

## Several GPUs

A rank runs the rows of its own routing tiles (`Inputs.tile0`, `tile1`, `cuts`). Every block it writes its tiles'
bf16 keys and values (and, with routing, their fp32 key and value means) into every peer's registered region over
RDMA, takes theirs, and attends as on one GPU. The kernels are the one-GPU kernels and the products keep one algorithm
a shape, so N ranks give the bits of one. The host moves each pass's rows to the ranks and back over its own channel;
that orchestration is not part of the library.

## Receipt

Measured 2026-10-16 on DGX Spark (GB10, compute capability 12.1, driver 580.173.02, CUDA 13.0.88 toolkit, cuBLASLt
13.1.1.3, Ubuntu 24.04.4, kernel 6.17.0-1029-nvidia), host ComfyUI at `a4b5a045` (0.39.0), checkpoint above, 864x480,
124 frames (15,433 rows, 288 tiles), 8 passes, sparsity 0.8, seed 7 (gate) and 8 (timed), rank 0 on the same machine
for every row, ranks over RoCE on two rails.

| Blocks | A pass | Sampler, 8 passes | Clip |
|---|---|---|---|
| ComfyUI's own blocks in bf16, dense attention | 10.6 s | 85.1 s | 103.0 s |
| ComfyUI's own blocks in bf16, its block-sparse attention 0.8 | 11.1 s | 92.8 s | 111.4 s |
| This library, one GPU | 10.27 to 10.41 s | 82.8 s | 101.1 s |
| This library, two GB10s | 6.26 to 6.42 s | 51.2 s | 69.4 s |
| This library, three GB10s | 4.92 to 5.06 s | 40.3 s | 58.5 s |

| Check | Result |
|---|---|
| Two and three ranks against one, 8 passes, same seed | 8 of 8 pass streams sha256-identical at each |
| Same seed after a ComfyUI restart | 8 of 8 identical |
| One block, dense, against ComfyUI's bf16 block, first pass | cosine 0.99999, relative error 0.0032, worst row 0.020, worst column 0.015 |
| All 50 blocks, dense, against ComfyUI's bf16 blocks | cosine 0.99910, relative error 0.042, worst row 0.547, worst column 0.220 |
| All 50 blocks against an fp32 forward of the same blocks | this library 0.0354 relative error; ComfyUI's bf16 blocks 0.0445 |

The fp32 forward is an independent torch implementation of the blocks (fp32 weights, fp32 activations, no TF32) on the
same first-pass rows; it shares no code with ComfyUI or this library.

Memory, the library alone (`tf_h3_open` on the checkpoint, then three passes over the same 15,433-row geometry from
host inputs, sampled every 0.5 s):

| | Device memory (nvidia-smi, peak) | MemAvailable drop |
|---|---|---|
| One GPU, every tile | 44,253 MiB, of which the bf16 block weights are 40,426 MiB | 45,746 MiB |
| One of two ranks, its 144 tiles, 3 GiB registered region | 42,870 MiB | 47,322 MiB |

The process maps the checkpoint while it copies the tensors; its resident pages peak at 36 to 38 GB during the load.

The kernel contract (`zig build kernel-contract -Dnvcc=/usr/local/cuda/bin/nvcc` on the GB10) registers every kernel
the family ships; gate 5's records come from `kernel-contract-limits` over sm_80, 86, 89, 120 and 121.

## Not done

- One GPU model tested (GB10) and one checkpoint (FastH3 8-Step V2); no other size than 864x480 in this receipt.
- Four or more ranks, and a rank dying mid-pass (the host's pass fails; there is no fall back).
