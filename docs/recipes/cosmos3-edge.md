# Cosmos3-Edge reasoner

This page describes the native engine. `cosmos3` serves the understanding tower of NVIDIA's Cosmos3-Edge (4B) for
text and image chat. The generation tower, the VAE and the action heads are not part of the family. It is off unless a
checkpoint names it; no other family's code path changes.

## Checkpoint

Tested revision: `nvidia/Cosmos3-Edge` at `344d602b128d1bbdacb43b08d0a3626f46343e29` (OpenMDW 1.1). The family reads an
MLX affine conversion (groups of 64) made by `tools/cosmos3_edge_to_mlx.py`:

```bash
hf download nvidia/Cosmos3-Edge --revision 344d602b128d1bbdacb43b08d0a3626f46343e29 --local-dir cosmos3-edge \
  --exclude 'vae/*' --exclude 'assets/*' --exclude 'images/*' --exclude 'scheduler/*' --exclude 'text_tokenizer/*'
python tools/cosmos3_edge_to_mlx.py cosmos3-edge cosmos3-tf 8
```

Upstream files the conversion reads (publisher sha256):

| File | sha256 |
| --- | --- |
| `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` | `116134e5492c514df79a2f07214e577c397aa961387dbe1c5dda488e7994ff6b` |
| `transformer/diffusion_pytorch_model-00002-of-00002.safetensors` | `3b4c0aafe2705c900ac7bc743085a8e9e09c906624331bc2baa21e172e5d48e4` |
| `vision_encoder/model.safetensors` | `2180ad739ecc96b5c1e9386892d3c5c08bfa42b9cdab9aabc53b028671db89b3` |
| `tokenizer.json` | `4dc692a99dca6d92a44e9529ffb53541eb79bb260eec7281bc51268c05d7311c` |

With transformers 5.20.2, torch 2.14.1, mlx 0.32.4 and numpy 2.4.6 on macOS, the 8-bit conversion is byte-identical
from run to run (1,340 tensors, 2.41 GiB):

| Output file | sha256 |
| --- | --- |
| `model.safetensors` | `e67cf5d59357a81d880ade79c0e700e9f1fb1ff090d5927eeb9ad4b385c29470` |
| `config.json` | `748d8bedc1d2f08a64d74dcb756e5a7ae5a8a9ff747f28a660ff94f18c13c8fa` |
| `chat_template.jinja` | `7120ee6666468d4e9b2dc11e133ac5c2fa765fa5907706bf0f906270aa5510c8` |
| `generation_config.json` | `b953ca6943f16fec3084c80138c8c61bd32181074b1a721afe95f43d715f5fab` |
| `preprocessor_config.json` | `111a151f82088f4e15cb4d488c5db0e9c481ae46ec8cf146d734ac62d13a57d6` |
| `processor_config.json` | `7444d3780a8ed0b11d850292fa54407cfd1f389c101d111090289da354b13333` |
| `special_tokens_map.json` | `e9435fefd6d838fd9fcbbc44b97a8e3ff322be7f6dfb7e4fd2468586574bb52b` |
| `tokenizer.json` | `4dc692a99dca6d92a44e9529ffb53541eb79bb260eec7281bc51268c05d7311c` |
| `tokenizer_config.json` | `85577f01be64ea03ee9aacc0536088e47b96971e4aff21e0aa604900b1410386` |

```bash
tensorfold-native serve cosmos3-tf --name cosmos3 --vision --context 16384 --parallel 4
```

## Model

A dense 28-layer text model (2,048 wide, GQA 16 over 8 heads of 128, relu^2 MLP, interleaved M-RoPE [24, 20, 20]); the
SigLIP2 tower (27 layers, 1,152 wide, 16 heads of 72) at the image's own size, its learned 16 x 16 position grid
resized as torch's antialiased bilinear; a projector (LayerNorm, 2 x 2 merge, 4,608 to 11,520 to 2,048). The converter
pads the tower MLP's 4,304 columns to 4,352 with zeros, which changes no output.

## Metal

`zig/kernels/metal/cosmos3/cosmos3.metal` holds the glue at MLX's bf16 rounding points: 8-bit embedding rows, RMS
norms, q/k norms with M-RoPE and the cache write, relu^2, LayerNorm, GELU, biases, and attention: a decode kernel for
calls under 33 rows, and flash attention on simdgroup matrices (32-row tiles, P rounded to bf16 as the eager weights
are) for the text model's prompt rows and the image tower. The projections
are the shared affine kernels.

Every cos3 kernel the engine launches is registered with the [kernel contract](../kernel-contract.md)
(`zig/tests/contract/metal_cosmos3*.zig`, eight planted controls). On an M5 Ultra, `tf-kernel-contract --only cosmos3`:
`SUMMARY 13 kernels: 13 pass gates 1-4, 0 fail, 0 known findings; 8 controls: 8 red as planted, 0 missed; 0 limit
problems`.

## Checks

Reference: transformers fp32 on CPU with the conversion's 8-bit weights dequantized in, so a difference measures the
engine rather than the quantization. Prompts: two text questions; images: the checkpoint's public
`assets/example_reasoning_input.png` (512 x 341), its `assets/example_i2v_input.jpg` resized to 640 x 370 as PNG, and a
synthetic 640 x 480 picture of three shapes and a word.

- First-token logits: top-1 equal on all five; top-4 equal on all five; cosine 0.9976 to 0.9999.
- 48-token greedy: replies part only at near-ties of the fp32 reference (its top-2 margins 0.03 to 0.26 at the parting
  token, where the engine takes the reference's second or third token); one of five is identical through 48 tokens.
- `tools/bench_concurrent.py --levels 1,2,4 --alone --serial`: every concurrent reply equals its solo run and every solo
  run equals `"draft": false`.
- Resumed equals fresh: turns 2 and 3 of text and image conversations, resumed from the prompt cache's kept states,
  equal the same conversations sent to a server with `--prompt-cache-gib 0`.
- The checkpoint has no drafter, so drafts are off.
