# Hy4-flash NVFP4: the build and its quality

The first Hy4-flash checkpoint of the family (#3801, answer 2) is
[atakans/Hy4-flash-NVFP4](https://huggingface.co/atakans/Hy4-flash-NVFP4) at revision
`30f201acf0b4f9ba178817bcbc04ccfa3ec473f9`: 164.7 GB, against 440 GB for Tencent's BF16 release and 287 GB for its FP8
release. This page says how to build it again and what it scores.

## What the build holds

- NVFP4 (ModelOpt's default: E2M1 weights and activations, one FP8 E4M3 scale per 16 elements, one FP32 scale per tensor,
  static input scales) on every MLP: the 512 routed experts of each MoE layer, the shared experts and the dense MLP. The
  routed experts are stored one tensor per expert (`experts.N.gate_proj`, `up_proj`, `down_proj`).
- FP8 E4M3 rows with one BF16 scale a row on the four Engram tables (`multi_head_embedding.embedding.weight` and
  `weight_scale_inv`; value = row x scale, scale = row maximum / 448), and `engram_fp8_embedding: true` in `config.json`.
- BF16 on everything else: attention, the compressed indexer, the router, the latent down and up projections, the
  embeddings, the DFly layers and the vision tower. The head is stored F32.
- `config.json`, the tokenizer and the chat template are Tencent's own files; `hf_quant_config.json` is ModelOpt's.

## Build it again

Source: [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash) at revision
`d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959` (BF16). The run used NVIDIA ModelOpt 0.48.0, transformers 5.24.0.dev0 (a
source build with Hy4-flash) and 8 x RTX PRO 6000 Blackwell Max-Q; the model is spread over the 8 cards.

```sh
python tools/hy4_flash/nvfp4.py tencent-Hy4-flash/ Hy4-flash-NVFP4/ --samples 128
python tools/hy4_flash/engram_fp8.py Hy4-flash-NVFP4/
```

- `nvfp4.py` quantizes with `NVFP4_DEFAULT_CFG` plus rules that keep the BF16 parts above, calibrates on ModelOpt's
  `nemotron-post-training-v3` mix (128 samples of up to 2,048 tokens; the weight scales come from the weights alone) and
  exports the checkpoint. Calibration took 42 minutes and the export 5 minutes.
- `engram_fp8.py` converts the four Engram tables in place and writes their error to `engram-fp8.json`.

Weight-level checks of the result: the experts dequantized against BF16 (gate, up and down of three experts in layers 5,
20 and 40) have 9.5 % relative error each, the usual level for NVFP4; each Engram table has 2.6 % relative RMS error
against BF16.

## Quality

GPQA Diamond (198 questions) and IFEval (541 prompts) through the OpenAI chat API on two DGX Sparks (GB10) at tensor
parallel 2, thinking off (`reasoning_effort: "no_think"`), temperature 0, 16,384 tokens a reply at most.

| Engine | GPQA Diamond | IFEval | Replies cut at the limit |
|---|---|---|---|
| TensorFold, the prototype on atakann/TensorFold `hy4-flash` (`72d3869ed`), DFly on | 72.2 % (95 % CI 65.6 to 78.0) | 84.7 % (81.4 to 87.4) | 0 and 1 |
| vLLM, the `hy4-flash` image with vllm-project/vllm#68387 | 72.7 % (66.1 to 78.5) | 83.5 % (80.2 to 86.4) | 1 and 1 |

`tensorfold-gpqa-ifeval.json` and `vllm-gpqa-ifeval.json` hold each run's settings, its summary and the score of every
item by its id (the replies are left out). The official FP8 release on four GB10s is a later check (#3801, answer 2).
