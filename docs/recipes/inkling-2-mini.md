# Inkling 2 Mini on CUDA

The `inkling2` family serves a standard exllamav3 EXL3 pack of Thinking Machines' Inkling 2 Mini (`model_type`
`inkling2_loop`) on the native CUDA server, on two DGX Sparks (GB10, sm_121) as tensor-parallel ranks. It is the
Small runtime of [inkling-2-small-cuda.md](inkling-2-small-cuda.md) with Mini's loop added; the pack format, the rank
split and the serving flags are the same. The server picks this path only when two ranks are placed and the pack uses
Small's tensor names (`model.llm.layers.N.attn.wq_du`). A one-GPU run, Mini's own NVFP4 and MXFP8 checkpoint and EXL3
packs in other layouts open on the one-rank engine in `cuda_native.zig`.

## The model

- 24 layers of hidden size 2,304: 20 sliding-window layers (1,024 keys) and 4 global layers, no RoPE, so the served
  window is the model's own 1,048,576 tokens.
- A looped core: layers 6 to 17 run three times with shared weights, so a forward makes 48 layer visits. Each visit
  keeps its own caches.
- From the second pass on, a core visit mixes a message from the previous pass into its attention and MLP inputs
  through four rank-64 gates.
- Attention: 24 query heads and 8 key heads of 128, with the relative-position bias, sinks and gate of Small.
- MoE: 128 routed experts, 8 a token, softcapped SwiGLU.
- A vision tower (`vision.safetensors` beside the pack) turns pictures into rows that replace the image placeholders.

## Checkpoint

| | Repository | Revision |
|---|---|---|
| Pack (EXL3 mul1, 4.05 bpw, head 8 bits) | `ajensenwaud/inkspark-mini-2dgx` | `a2707bbdbfae47a66154dbd89603c546bf1fe634` |
| Source | `thinkingmachines/Inkling-2-Mini` | `ed40e7da625f594fc9c4990f5f66be24873c22d7` |

## Files

| File | What it does |
|---|---|
| `visits.zig` | The visit plan from the shared `loop_config`: the layer, pass and cache slot of each visit |
| `state.zig`, `forward.zig` | A cache slot per visit, and a call's rows through every visit with the messages between passes |
| `loop_ops.zig` | The message attention, the four gates and their norms on `zig/kernels/cuda/inkling2/loop.cu` |

Small's plan is its layers in order, so Small's allocations and launches do not change. Mini's prompt rows run on the
decode kernels, so a prompt pass gives the same bits as decode rows.

## Serving

Both hosts run the server with the same flags; rank 0 answers the API.

```bash
# on each host, RANK 0 on the head and 1 on the worker
TF_INKLING2_RANK=$RANK TF_INKLING2_WORLD=2 NCCL_COMM_ID=<head address>:<port> TF_ROCE=1 \
  tensorfold-native serve /model --name Inkling-2-Mini --context 1048576 --parallel 1 --vision --no-drafts
```

`--context 131072 --parallel 4` gives four streams of 128K. The KV cache is bf16. The bundled DSpark drafter
(`mtp.safetensors`) is not read yet, so every stream decodes one token a step.
