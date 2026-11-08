# Qwen3-Embedding-8B on native HIP

The native HIP engine serves `Qwen/Qwen3-Embedding-8B` as published (BF16
safetensors, revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`) through
`POST /v1/embeddings`. It is qualified on one Radeon AI PRO R9700 (gfx1201).
gfx11 builds run the same code; FP16 builds (RDNA2) refuse the checkpoint.

| Responsibility | Files |
| --- | --- |
| Config, BF16 weights, batched prompt pass | `zig/src/families/qwen3/` (`config.zig`, `model.zig`, `forward.zig`) |
| Prompt-only engine, lane backend, embedding head | `zig/src/families/qwen3/` (`engine.zig`, `prompt.zig`, `native.zig`) |
| BF16 dense GEMM (dot2 tile, BF16 weight decoder; gfx12's WMMA tile) | `zig/kernels/hip/quant/bf16_decoder.hpp`, `dense_tiles.hpp`, `zig/kernels/hip/tiles/wmma12_gemm.hpp`, `zig/src/hip/launches/dense.zig` |
| Head-major attention rows | `tf_heads_rows` in `zig/kernels/hip/ops/attention.hpp` |
| Stored-dtype dense weights | `zig/src/core/quant/dense.zig` (`readStored`) |
| Device contracts | `zig/tests/hip/contract_dense.zig`, `contract_dense_wmma12.zig`, `contract_heads_rows.zig` |
| Pinned model-card oracle | `tools/zig/oracle_qwen3_embedding.py` |

The weights stay BF16 on the GPU (14.1 GiB). Activations are BF16 and every
product sums in fp32. The head comes from the sentence-transformers folder:
`1_Pooling/config.json` must pool the last token only, and `modules.json`'s
`Normalize` makes every full-width vector unit length. A folder without
`1_Pooling` is refused.

On gfx12 the BF16 projections can run on a WMMA tile, opt-in with
`--policy matrix=on`. The default `matrix=auto`, `matrix=off` and every other
chip run the dot2 tile and keep its bits. The two tiles sum in different
orders: on Qwen3-8B's layer-0 projections WMMA's fp32 sums sit 8 to 41
times further from float64 than dot2's, while the BF16 outputs the engine
writes are as close (`TF_QWEN3_*` GPU checks below, `projection_test.zig`).
A less exact default output needs an opt-in, so WMMA is not the default.
gfx1200 builds the same tile but is unmeasured: no run, and no gate 5
record (the records cover gfx1201).

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
files differ from its pins before any model code runs. Its fidelity set is
the card's 4 inputs and 5 long inputs of 2,044 to 32,759 tokens, each the
longest whole-line prefix of a pinned file within 2K, 4K, 8K, 16K or 32K
tokens (the card's `LICENSE` and `README.md`, three Transformers sources);
every input's text and token IDs are pinned by SHA256. The card's inputs run
as the card runs them, one padded batch; each long input runs alone. The
fp32 reference takes about 20 minutes on 32 CPU cores:

```sh
python3 tools/zig/oracle_qwen3_embedding.py sources --model MODEL_DIRECTORY
python3 tools/zig/oracle_qwen3_embedding.py ref --model MODEL_DIRECTORY --dtype float32 --device cpu --out ref32.json
python3 tools/zig/oracle_qwen3_embedding.py ref --model MODEL_DIRECTORY --dtype bfloat16 --device cuda --out ref16.json
python3 tools/zig/oracle_qwen3_embedding.py check --ref ref32.json --bar ref16.json --url http://127.0.0.1:8090
```

`check` sends the inputs in requests that fit the server's window (the card's
and the 2K to 16K inputs together, the 32K one alone). It passes when every
vector lies within transformers' own BF16 distance of the fp32 vector,
rounded up to two digits and at most 0.001 (distance = 1 - cosine), has unit
norm, is the same fp32 bytes alone, in a batch and from token IDs, and the
card's 2 x 2 scores are within 0.01. At MRL widths 32, 1024 and 4096 (`dimensions`) every vector
must have unit norm and lie within the same bound, taken at that width, of
the fp32 vector cropped and renormalized. The full-width gate runs on the
GPU without a server too:

```sh
TF_QWEN3_EMBED_DIR=MODEL_DIRECTORY TF_QWEN3_EMBED_REF=ref32.json TF_QWEN3_EMBED_BAR=ref16.json \
  zig build qwen3-hip-gpu-test -Dhip=true -Dhip-arch=gfx1201 -Dhipcc=/opt/rocm/core/bin/hipcc
```

The GPU checks take `TF_QWEN3_MATRIX=auto|off|on` to run either tile, and `TF_QWEN3_ATTENTION=wide|f32` for the attention path.

The WMMA rows below come from a server started with the WMMA tile:

```sh
zig-out/native-hip/bin/tensorfold-native serve MODEL_DIRECTORY \
  --name Qwen3-Embedding-8B --host 127.0.0.1 --port 8090 --policy matrix=on
```

Results on the R9700 (ROCm 10.1); the distance rows are dot2 (the default), the WMMA rows `matrix=on`:

| Check | Result |
| --- | --- |
| Card example, distance to fp32 (4 inputs) | 1.32e-4, 9.30e-5, 7.76e-5, 9.27e-5 |
| Long inputs, distance to fp32 (2,044, 4,043, 8,177, 16,383, 32,759 tokens) | 1.16e-4, 5.24e-5, 7.00e-5, 9.90e-5, 9.22e-5 |
| Full-width limit (transformers BF16's worst, 1.572e-4, rounded up) | 1.6e-4 |
| MRL width 32, 1024, 4096: worst distance (limit; transformers BF16's worst) | 7.92e-5 (1.9e-4; 1.884e-4), 1.18e-4 (1.7e-4; 1.624e-4), 1.32e-4 (1.6e-4; 1.572e-4) |
| Card 2 x 2 scores | [[0.7479, 0.0759], [0.0886, 0.6324]] (card: [[0.7493, 0.0751], [0.0880, 0.6318]]) |
| Batch, alone and token-ID vectors (all 9 inputs) | identical fp32 bytes |
| `hip-kernel-test`, `dense_gemm_block` | fp32 within the dot2 chain's bound against float64; BF16 out is the fp32 sum rounded once |
| `hip-kernel-contract` | dense GEMM (fp32 and BF16 out) and `tf_heads_rows` pass gates 1 to 4; their gate 5 records are current (gfx1030 does not serve the dense GEMM) |
| WMMA: distance to fp32 (9 inputs, card's then long) | 1.18e-4, 9.57e-5, 7.40e-5, 7.44e-5, 1.15e-4, 6.39e-5, 7.62e-5, 1.04e-4, 1.31e-4 |
| WMMA: MRL width 32, 1024, 4096 worst distance (limit) | 1.06e-4 (1.9e-4), 1.35e-4 (1.7e-4), 1.31e-4 (1.6e-4) |
| WMMA: card 2 x 2 scores, worst difference to the card | 0.0024 (dot2: 0.0014) |
| Layer-0 projections against float64, BF16 out (WMMA / dot2) | 1.0000 in every class (q, k, v, o, gate, up, down) |
| `hip-kernel-test` and `hip-kernel-contract`, WMMA tile (both geometries, fp32 and BF16 out) and the fused silu product (both blocks) | gates 1 to 4 pass, fp32 worst 1.4% of its bound (depth k plus k 2^-126); the big block bit-equal to the 32-deep block it replaces, the fused product to the plain product, its split and `tf_silu_mul`; four planted controls red: each 16-deep step summed in BF16 (gate 4), the big block's loads without the swizzle and its K loop without barriers (gate 3), silu on the fp32 gate sum (gate 3); gate 5 records current, gfx1201 only |
| 100 short inputs in one request / one short query, cold | WMMA (`matrix=on`) 0.901 to 0.911 s / 38.2 to 38.4 ms; dot2 (default) 2.470 s / 75.1 ms; SGLang 0.841 s / 38.4 ms on the same card |
| Load | 4 s from the page cache |
