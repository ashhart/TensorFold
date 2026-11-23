# Inkling 2 Small on CUDA

The `inkling2` family serves a standard exllamav3 EXL3 pack of Thinking Machines' Inkling 2 Small on the native CUDA
server, on two DGX Sparks (GB10, sm_121) as tensor-parallel ranks. Package: `zig/src/families/inkling2/`, kernels in
`zig/kernels/cuda/inkling2/`. It reads the pack exllamav3 writes (`config.json` with `quantization_config`, an index
and shards), so any exllamav3 1.x pack of the model on the `mul1` codebook loads.

## The model

- 48 layers of hidden size 4,608: 40 sliding-window layers (1,024 keys) and 8 global layers, no RoPE. The global
  layers keep every key, so the served window is the model's own 1,048,576 tokens.
- Attention: 48 query heads and 8 key heads of 128, a learned relative-position bias, sinks and a sigmoid output gate.
  Short convolutions sit on the keys, the values and both residual branches.
- MoE: 384 routed experts, 8 a token, softcapped SwiGLU.
- A vision tower (`vision.safetensors` beside the pack) turns pictures into rows that replace the image placeholders.

## Checkpoint

| | Repository | Revision |
|---|---|---|
| Pack (EXL3 mul1, 2.2 bpw, head 8 bits) | `ajensenwaud/inkspark-2dgx` | `aa29f3a011b011896826416f03b8ac55d7ec7c6f` |
| Source | `thinkingmachines/Inkling-2-Small` | `aac974122acd33ddb0d3e1f54433a9c5dd2249ce` |

## Files

| File | What it does |
|---|---|
| `exl3.zig`, `pack_weights.zig`, `shard.zig`, `shape.zig` | The pack's reader: EXL3 projections, grouped experts, the head, each cut to its rank |
| `forward.zig`, `prompt_dense.zig` | A call's rows through every layer: decode rows of several streams, or a prompt chunk |
| `engine.zig`, `kept.zig`, `sampler.zig` | One rank's engine: slots, graphed decode rows, prompt chunks, kept prompt states, draws |
| `follow.zig`, `lanes.zig`, `native.zig`, `admit.zig` | Rank 0 serves and sends each step to rank 1; the lanes backend behind `serve` |
| `pictures.zig` | The server's image offer and each slot's image rows on every rank |
| `engine_main.zig` | `tf-inkling2-engine`: graph replay against eager rows, and greedy decode, on two ranks |

A row's bits do not depend on the other streams in its forward, the graph or how its prompt is cut into chunks, so
graphed decode equals eager decode and a resumed prompt equals a fresh one.

## Serving

Both hosts run the server with the same flags; rank 0 answers the API.

```bash
# on each host, RANK 0 on the head and 1 on the worker
TF_INKLING2_RANK=$RANK TF_INKLING2_WORLD=2 NCCL_COMM_ID=<head address>:<port> TF_ROCE=1 \
  tensorfold-native serve /model --name Inkling-2-Small --context 1048576 --parallel 1 --vision --no-drafts
```

`--context 131072 --parallel 4` gives four streams of 128K. The KV cache is bf16. The bundled DSpark drafter
(`mtp.safetensors`) is not read yet, so every stream decodes one token a step.

The family refuses any other setup by name: one rank, a GPU other than GB10, another codebook or exllamav3 version,
and a pack whose tensors differ from what the reader expects.
