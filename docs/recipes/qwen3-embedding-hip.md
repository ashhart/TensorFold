# Qwen3-Embedding-8B on native HIP

The native HIP engine serves `Qwen/Qwen3-Embedding-8B` as published (BF16
safetensors, revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`) through
`POST /v1/embeddings`. It is qualified on one Radeon AI PRO R9700 (gfx1201).
gfx11 builds run the same code; FP16 builds (RDNA2) refuse the checkpoint.

| Responsibility | Files |
| --- | --- |
| Config, BF16 weights, batched prompt pass | `zig/src/families/qwen3/` (`config.zig`, `model.zig`, `forward.zig`) |
| Prompt-only engine, lane backend, embedding head | `zig/src/families/qwen3/` (`engine.zig`, `prompt.zig`, `native.zig`) |
| BF16 dense GEMM (dot2 tile, BF16 weight decoder) | `zig/kernels/hip/quant/bf16_decoder.hpp`, `dense_tiles.hpp`, `zig/src/hip/launches/dense.zig` |
| Head-major attention rows | `tf_heads_rows` in `zig/kernels/hip/ops/attention.hpp` |
| Stored-dtype dense weights | `zig/src/core/quant/dense.zig` (`readStored`) |
| Device contracts | `zig/tests/hip/contract_dense.zig`, `contract_heads_rows.zig` |
| Pinned model-card oracle | `tools/zig/oracle_qwen3_embedding.py` |

The weights stay BF16 on the GPU (14.1 GiB). Activations are BF16 and every
product sums in fp32. The head comes from the sentence-transformers folder:
`1_Pooling/config.json` must pool the last token only, and `modules.json`'s
`Normalize` makes every full-width vector unit length. A folder without
`1_Pooling` is refused.

## Build and serve

```sh
zig build native-hip -Dhip=true -Dhip-arch=gfx1201 -Dhipcc=/opt/rocm/core/bin/hipcc
zig-out/native-hip/bin/tensorfold-native serve MODEL_DIRECTORY \
  --name Qwen3-Embedding-8B --host 127.0.0.1 --port 8090
```

With ROCm 10.1, pass the hipcc under `/opt/rocm/core`: the build takes the
ROCm root from the hipcc path, and `/opt/rocm` lacks the device math headers.

The server sizes one pass to `--context` (32,768 tokens by default, at most
the config's 40,960) and shrinks it by halves to fit the memory budget, down
to 2,048; the startup line names the window. A request holds at most 128
inputs and the window's tokens in all. `dimensions` crops to any width from
32 to 4,096 (the card's MRL range) and renormalizes. The chat and completion
routes are refused.

Queries take the card's instruction prefix; documents are sent as they are:

```sh
curl -s http://127.0.0.1:8090/v1/embeddings -H 'Content-Type: application/json' -d '{
  "input": ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:Explain gravity",
            "Gravity is a force that attracts two bodies towards each other."]}'
```

Text inputs end in `<|endoftext|>` through the tokenizer's post-processor,
which last-token pooling reads. Token-ID inputs are used as sent, so a client
must append 151643 itself.

## Gates

The oracle runs the model card's Transformers example in a venv with
Transformers 5.22.1. It refuses a snapshot or a Transformers install whose
files differ from its pins before any model code runs:

```sh
python3 tools/zig/oracle_qwen3_embedding.py sources --model MODEL_DIRECTORY
python3 tools/zig/oracle_qwen3_embedding.py ref --model MODEL_DIRECTORY --dtype float32 --device cpu --out ref32.json
python3 tools/zig/oracle_qwen3_embedding.py ref --model MODEL_DIRECTORY --dtype bfloat16 --device cuda --out ref16.json
python3 tools/zig/oracle_qwen3_embedding.py check --ref ref32.json --bar ref16.json --url http://127.0.0.1:8090
```

`check` passes when every vector lies within `min(0.001, 2 x transformers
BF16's worst distance)` of the fp32 vector (distance = 1 - cosine), has unit
norm, is the same fp32 bytes alone, in a batch and from token IDs, and the
card's 2 x 2 scores are within 0.01. The same gate runs on the GPU without
a server:

```sh
TF_QWEN3_EMBED_DIR=MODEL_DIRECTORY TF_QWEN3_EMBED_REF=ref32.json TF_QWEN3_EMBED_BAR=ref16.json \
  zig build qwen3-hip-gpu-test -Dhip=true -Dhip-arch=gfx1201 -Dhipcc=/opt/rocm/core/bin/hipcc
```

Results on the R9700 (ROCm 10.1):

| Check | Result |
| --- | --- |
| Card example, distance to fp32 (4 inputs) | 1.32e-4, 9.30e-5, 7.76e-5, 9.27e-5 (limit 2.26e-4; transformers BF16: 1.13e-4) |
| Card 2 x 2 scores | [[0.7479, 0.0759], [0.0886, 0.6324]] (card: [[0.7493, 0.0751], [0.0880, 0.6318]]) |
| Batch, alone and token-ID vectors | identical fp32 bytes |
| `hip-kernel-test`, `dense_gemm_block` | fp32 within the dot2 chain's bound against float64; BF16 out is the fp32 sum rounded once |
| `hip-kernel-contract` | dense GEMM (fp32 and BF16 out) and `tf_heads_rows` pass gates 1 to 4; their gate 5 records are current (gfx1030 does not serve the dense GEMM) |
| Load | 4 s from the page cache |
