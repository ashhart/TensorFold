# MiMo-V2.6-Flash

The `mimo_v2` family serves an EXL3 pack of MiMo-V2.6-Flash (exllamav3's MiMo-V2 module, mul1 codebook) on the
native CUDA server, on one GPU or two DGX Sparks as tensor-parallel ranks. The pack's DFlash drafter sits in its
`dflash/` folder (`config.json` and `model.safetensors`); the server drafts with it when it is there.
Package: `zig/src/families/mimo_v2/`, kernels in `zig/kernels/cuda/mimo_v2/`, EXL3 launches in `zig/src/cuda/exl3.zig`.

The port follows Jay Leaton's DeepSeek-V4.1 method: every launch mirrors the Python 0.6 engine's binding (same
kernel copies, grid, block and instance choice), Triton kernels run as the Python engine's own captured cubins, and
the two ranks exchange through the tensor-parallel session and RoCE mailbox of his two-Spark work.

## The model

- 48 layers of hidden size 4,096: 9 full-attention layers and 39 sliding-window layers (128 tokens, sink logits).
- Attention: query/key head dim 192 (RoPE on the first 64), value dim 128; the full and sliding layers have their
  own key/value head counts and RoPE bases.
- MoE: 256 routed experts, sigmoid routing with a bias, top 8; the first layer is dense.
- The drafter (DFlash) reads the target's residuals after its tap layers and drafts its whole block in one pass.

## Kernels

The family needs the Python engine's Triton kernels as a captured set (a folder with `aot.json` and the cubins),
recorded from the Python engine for sm_121 with `tools/zig/triton_aot_manifest.py` and packed with
`zig/tests/cuda/nemotron/aot_pack.py`. Point `TENSORFOLD_CUDA_KERNELS` at the folder on every rank; without it the
server refuses the checkpoint (`NoCapturedTritonSet`). EXL3, the torch glue and the 8-bit cache are built in.

## Two Sparks

Rank 1 follows rank 0 over the plan link; start it first, then the server. Both need the same build, checkpoint
path and kernel set. With rank 0's fabric address 192.0.2.1:

```bash
# rank 1
TF_TP_WORLD=2 TF_TP_RANK=1 TF_TP_MASTER=192.0.2.1 TF_TP_PORT=29661 TF_COMM_BACKEND=roce \
  tf-mimo follow /models/mimo-v2.6-flash-exl3 /kernels/mimo-sm121
# rank 0
TF_TP_WORLD=2 TF_TP_RANK=0 TF_TP_MASTER=192.0.2.1 TF_TP_PORT=29661 TF_COMM_BACKEND=roce \
TENSORFOLD_CUDA_KERNELS=/kernels/mimo-sm121 \
  tensorfold-native serve /models/mimo-v2.6-flash-exl3 --host 0.0.0.0 --parallel 4 --context 608256 \
  --top-k 20 --top-p 0.95
```

`--parallel` is the number of streams; each owns a cache slot of the whole `--context`. A lone stream's draft depth
comes from its own acceptance rate, priced with a round cost that grows with its context. Prompts of 256 tokens or
more run on the prompt path (EXL3 prompt GEMM, the full-attention prompt kernel, experts grouped up to 3,072 rows a
chunk and per expert beyond); shorter ones run as decode rows.

## Policy

`--policy kv=fp8` keeps the 9 full-attention layers' keys and values in e4m3 with an fp32 scale a token and key
head, half the bytes a decode step reads at long context. It is not bit-equal to the 16-bit cache; `kv=fp16`, the
default, is. `/health` shows the active policy.

## Checks

- `tf-cuda-test mimo-exl3 FILE PREFIX` and `mimo-exl3-fixtures MODEL DIR`: the EXL3 kernels against the format
  reference and against the Python engine's bytes (`zig/tests/cuda/oracle/mimo_exl3_fixtures.py`).
- `tf-mimo m1 MODEL ORACLE AOT_DIR` on each rank: residuals, logits and argmax byte-equal to the Python engine's.
- `tf-mimo kv8check MODEL AOT_DIR`: the 8-bit decode attention against Triton's on the same values.
