# Qwen-Image-2.1 (`qwen_image`)

Text-to-image generation, not a language model: a 7B single-stream transformer denoises 64-channel image latents
and a decoder turns them into pixels. It does not decode through lanes and has no `load`; `tensorfold serve` does not
route it. Package: `src/tensorfold/families/qwen_image/`. The int8 kernels are the ones written for MiniMax H3,
`src/tensorfold/kernels/minimax/h3/v1/`.

Measured on a Mac Studio M5 Ultra with 256 GB, macOS 27.0.1, MLX 0.32.3, with `Qwen/Qwen-Image-2.1` (bfloat16).
Qwen-Image-2.1 is under the Qwen Research License Agreement: non-commercial use only. TensorFold ships no weights.

## What is here and what is not

| Part | State |
| --- | --- |
| Transformer (32 blocks, hidden 4,096, 32 heads of 128, SwiGLU width 12,288) | `dit.py`, `weights.py` |
| Text prefix run once per prompt, its keys and values reused on every step | `dit.py` (`prefix`) |
| Schedule and Euler loop, no guidance | `schedule.py`, `sampler.py` |
| Adapters (diffusers LoRA layout), merged in float32 | `lora.py` |
| Image decoder | `vae.py`, float32 |
| Prompt encoder (Qwen3-VL language model and tokenizer) | not ported; `tools/qwen_image_generate_dev.py` borrows mflux's |
| Image editing, reference images, RGBA output, guidance with a negative prompt, the VAE encoder | not ported |
| `tensorfold generate`, checkpoint detection through `families.detect`, a resident engine | not started |

`tools/qwen_image_generate_dev.py` renders an image with this family after mflux's prompt encoder. Run it from an
environment that has mflux (revision `add5164` was used) on the path.

## Measurements

One prompt (94 text tokens), 1344x768 (4,032 image tokens), seed 42, one run each.

| Path | Steps | Per step | Denoise | Whole run |
| --- | ---: | ---: | ---: | ---: |
| mflux `add5164`, bfloat16 | 40 | 0.73 s | 29 s | 40.7 s |
| this family, bfloat16 | 40 | 0.78 s | 31.0 s | 34.3 s |
| this family, int8 MLP, QKV and attention output | 40 | 0.48 s | 19.2 s | 24.3 s |
| this family, int8, Viggle turbo adapter (rank 256, v0.3) on its six nodes | 6 | 0.50 s | 3.0 s | 9.2 s |

- The whole run includes the prompt encoder (2.1 s), loading and quantizing (2.2 s; 3.2 s with the adapter merge)
  and the decode (0.8 s). Peak memory 15 GiB; 28 GiB while an adapter is merged in float32.
- Against mflux on the same seed and prompt the bfloat16 image is 32.7 dB, the int8 image 30.4 dB: the same picture
  with different fine detail. The turbo image is the same scene with a different rendering (25.1 dB).
- On random inputs the bfloat16 transformer output has cosine 0.9995 to 0.9999 with mflux's, the int8 one 0.997 to
  0.999, and the decoder 0.99996. The schedule equals mflux's to 1e-7. `tools/qwen_image_parity_dev.py` prints these.
- The fused QKV kernel rotates channel `c` with `c + 64`; this model rotates neighbours. The q and k rows are
  reordered within each head to [even, odd] before quantizing, which the dot product between them does not see.
- Not graded: text rendering, other sizes, more than one prompt and seed.
