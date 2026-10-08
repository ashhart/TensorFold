# Ornith-1.5-9B

A 9B dense Qwen3.5 checkpoint ([ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B))
served through the [Qwen3.8-27B recipe](qwen3.8-27b.md)'s `qwen3_5` family: same Gated DeltaNet and
full-attention mix, the same packed affine decoders, and the same engine behavior. This page covers
building a checkpoint the family admits and the memory arithmetic for 16-24 GB Macs. The checkpoint is
not qualified: no receipt exists yet, the sizes below are arithmetic from `config.json`, and decode
quality at each bit rate is unmeasured.

The upstream checkpoint ships untied embeddings, a 248,320-token vocabulary and a vision tower. The
family's decoder requires the untied head (tied checkpoints are refused), reads the language model only,
and mlx-lm's Qwen3.5 loader drops the vision tensors and the `mtp.*` weights at sanitize, so neither is
resident or served.

## Build the checkpoint

The upstream checkpoint is unquantized and is refused before any weight download. Quantize it with
mlx-lm 0.32 or newer on any machine:

```bash
python -m pip install "mlx-lm>=0.32.0"
mlx_lm convert --hf-path ornith-ai/Ornith-1.5-9B -q \
  --q-bits 4 --q-group-size 64 --quant-predicate mixed_4_6
```

Any MLX affine width (2/3/4/5/6/8-bit) in groups of 32/64/128, including the per-module overrides the
built-in predicates write, passes `check_quantization` on Metal and CUDA. The LM head takes the
predicate's high width; embeddings take the global width. A custom predicate trading body bits for head
bits sizes the same way as the table below.

## Run

```bash
tensorfold serve ~/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-9B/snapshots/<rev>-4bit \
  --name ornith9b --context 131072 --no-drafts
```

No DFlash2 draft head ships for this checkpoint; MTP tensors are dropped at load. Serve with `--no-drafts`
until a drafter exists. Engine behavior (chunk boundaries, prefix reuse, exactness contract) is the
27B recipe's.

## Memory arithmetic

The language model is 8.95B quantizable parameters: 6.92B body projections (the q projection carries the
folded attention output gate, so it is 2x hidden wide), and 2.03B embeddings plus head (248,320 x 4096,
untied). mlx-lm's loader drops the 0.23B MTP layer and the roughly 0.46B-parameter vision tower, so
neither is resident or served.

Uniform bit rates over those parameters:

| Average bpw | Weights |
| --- | --- |
| 3.0 | 3.14 GiB |
| 4.0 | 4.11 GiB |
| 5.0 | 5.18 GiB |
| 6.0 | 6.22 GiB |

A body-4/head-6 mix (everything at 4-bit, head at 6-bit) is 4.40 GiB; a body-5/embed-6/head-8 mix is
5.68 GiB. The built-in `mixed_4_6` predicate lands near 4.6 GiB.

Only 8 of 32 layers carry a KV cache (head_dim 256, 4 KV heads, bf16): 32 KiB per token. The 24
DeltaNet layers hold a fixed fp32 recurrent state of 48 MiB per stream, independent of context.
Quantized KV cache is not implemented for this family: `lane_sdpa` takes bf16 keys and values only.

| Context | KV cache |
| --- | --- |
| 131,072 | 4.00 GiB |
| 262,144 | 8.00 GiB |
| 1,048,576 (YaRN 4.0) | 32.00 GiB |

Machine classes, not measured minimums (wired limit at the macOS default of 75%):

| Mac | Recipe | Context |
| --- | --- | --- |
| 16 GB | body-4/head-6 mix, 4.40 GiB weights | 131,072 (bf16 KV would allow about 224,000; `--context` reserves headroom) |
| 24 GB | body-4/head-6 mix, 4.40 GiB weights | 262,144 (13.2 GiB resident) |
| 24 GB | body-5 mix, 5.68 GiB weights | 262,144 (14.4 GiB resident) |
| 128 GB+ | body-4 or body-5 | 1M tokens needs the 32 GiB YaRN cache and a `rope_scaling` block in `config.json` |

The upstream window is 262,144 positions; `config.json` values above that are refused. YaRN scaling is a
checkpoint edit (`rope_type: "yarn"`, factor 4.0), and the context admission still checks the reported
window.
