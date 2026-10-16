# Qwen-Image-2.1 on Metal

Qwen-Image-2.1 is a text-to-image model. A 7B single-stream transformer (32 blocks, hidden 4,096, 32 heads of 128,
SwiGLU width 12,288) denoises 64-channel latents. `zig/src/families/qwen_image` runs that transformer on the Metal
runtime from the released bf16 diffusers checkpoint:

- the prompt rows once a prompt: causal attention, every block's keys and values kept;
- the image rows at each step: attention over the kept prompt keys and the image's own.

The prompt encoder (Qwen3-VL) and the image decoder are not ported. `tools/zig/qwen_image_case.py` runs them through
mflux, a development reference.

It is a denoiser, not a token family. It does not register with the lane engine, the server or `families.detect`.

## Checkpoints

- `Qwen/Qwen-Image-2.1-Turbo`: 8 steps on the `sample_sigmas` saved in `model_index.json`.
- `Qwen/Qwen-Image-2.1`: the same transformer shape. It is untested here.

`transformer/config.json` must be `QwenImage21Transformer2DModel` at the released geometry. Any other config is
refused. The weights are under the Qwen Research License (non-commercial); read it before use.

## Precision

- bf16 activations, rounded where the reference rounds them.
- fp32 sums in every projection and in attention.
- fp32 for norms, rotary and softmax.
- No int8.

The projections are the Qwen-VL tower's `tfp::gemm_nax`. Attention is the prefill `tfp::attention_nax` template,
with full (non-causal) entry points added for the image rows.

## Run a case

```sh
python3 tools/zig/qwen_image_case.py case MODEL_DIR case/ --prompt-file prompt.txt   # needs mflux and MLX
zig build tf-qwen-image -Dcpu=apple_m1
zig-out/bin/tf-qwen-image MODEL_DIR/transformer case/case.safetensors case/
python3 tools/zig/qwen_image_case.py decode MODEL_DIR case/latents.f32 case/zig.png
python3 tools/zig/qwen_image_case.py compare case/reference.png case/zig.png case/side.png
```

The case holds:

- the prompt's encoder rows;
- the noise levels;
- the starting latents;
- mflux's input and velocity at every step;
- with `--fp32-steps N`, a float32 run of the same transformer.

`tf-qwen-image` first runs each step on mflux's own input and compares the velocities. It then runs the whole schedule
from the starting latents, times it, and writes the final latents.

## Measured

Setup:

- Mac Studio M5 Ultra 256 GB, macOS 27.
- `Qwen/Qwen-Image-2.1-Turbo` at 1344x768: 4,032 image rows, 44 prompt rows, seed 42, 8 steps.
- Reference: mflux `add5164` on MLX 0.32.3, bf16.

Results:

- **Velocity**, each step on the reference's input:
  - Against mflux bf16: cosine 0.99996 to 0.99999, max |diff| 0.11 to 0.21.
  - Against a float32 run of the same transformer: cosine 0.99997 to 0.99998.
  - mflux bf16's own cosine against that float32 run: 0.99994 to 0.99999.
- **Image:** our latents decoded through mflux's decoder are 43.35 dB PSNR from mflux's own image.
- **Repeat:** the same step run twice gives identical bytes.
- **Speed**, one session with no other GPU work, runs alternated:
  - mflux: 0.702 s a forward, before and after.
  - Ours: 0.740 s a forward (0.730 s on the GPU), and 0.23 s for the prompt rows once.
- **Memory:** 15.1 GB peak process footprint, with the transformer's bf16 weights (about 14 GB) in shared Metal buffers.
