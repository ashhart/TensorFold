# MOSS-Image on Metal

MOSS-Image is a text-to-image model. A 5B joint transformer (42 blocks, hidden 3,072, 24 heads of 128, an MLP of
8,192 with the "situ" gate) denoises 128-channel latents by flow matching. `zig/src/families/moss_image` runs that
transformer on the Metal runtime from the released bf16 diffusers checkpoint:

- a prompt's rows once a prompt: the input projection and the two refiner blocks (bidirectional attention, biased
  projections, no modulation);
- the image and prompt rows together at each step: every joint block, one modulation from the timestep shared by all of
  them, full attention over both.

The card's guidance runs two passes a step: the prompt, and the empty negative prompt with block 22 skipped.

The prompt encoder (a Qwen3.5 model) and the image decoder (FLUX.2's small decoder) are not ported.
`tools/zig/moss_image_case.py` runs them through the publisher's own `pipeline_moss_image.py` in torch, a development
reference.

It is a denoiser, not a token family. It does not register with the lane engine, the server or `families.detect`.

It runs on the M5's tensor units alone, the only chip it was checked on: `Model.load` refuses a GPU without them.

## Checkpoint

`OpenMOSS-Team/MOSS-Image-Base` at `85975793d8b3d432f1b6e27f352abaa4c2a4c0d3`, Apache 2.0. `transformer/config.json`
must be `MOSSImageTransformer` at the released geometry; any other config is refused, and so is a checkpoint with
tensors this family does not read.

## Precision

- bf16 activations, rounded where the publisher's torch code rounds them: once per elementwise op, as torch's bf16
  tensors do.
- fp32 sums in every projection and in attention. Biased projections add the bias to the fp32 sums and round once, as
  a bf16 GEMM's bias epilogue does: their sums come from the prefill split-K GEMM (one part), then `mi_bias`.
- fp32 for norms, softmax and the time sinusoid's input (the sinusoid itself in float64, as the publisher computes it).
- The rotary tables in bf16, as the pipeline casts them; its frequencies are the checkpoint's `rope.bands`.
- No int8.

The bf16 projections are the Qwen-VL tower's `tfp::gemm_nax`. Attention is the prefill `tfp::attention_nax` template's
full (non-causal) form. The time MLP and both modulation projections run on the host, once a timestep.

## Kernel contract

`zig/tests/contract/metal_moss_image*.zig` registers every kernel of `moss_image.metal`:

- `mi_norm`, plain and modulated;
- `mi_qk`, in both row layouts: q, k and v in three buffers, and packed in one row;
- `mi_situ`, in both layouts: hidden and gate halves of one row, and two buffers;
- `mi_gate_add` (in place) and `mi_bias`;
- the attention's three forms.

The steps are checked output by output at the bf16 points the publisher's torch code rounds at. Each output must
lie in the range its fp32 steps allow, with Metal's error bounds for the `precise::` calls and fast-math division.
Eight controls each drop or add one rounding point, and gate 3 must catch each. The attention is checked against
`attention_ref`'s float64 formula and bf16-P bound. Its gate 4 uses equal scores over values whose sums reach about
2^22 and cancel. Gate 5 holds the source's eight launches and the two split-K projections.

`tf-kernel-contract --only "moss image"` on the M5 Ultra:

```
SUMMARY 11 kernels: 11 pass gates 1-4, 0 fail, 0 known findings; 8 controls: 8 red as planted, 0 missed; 0 limit problems
```

## Run a case

`moss_image_case.py` imports and runs the checkpoint's own `pipeline_moss_image.py`, the publisher's Python. It does so
only when the file's sha256 is the pinned revision's (`1dac1aca…8b31`) and refuses any other file.

```sh
hf download OpenMOSS-Team/MOSS-Image-Base --revision 85975793d8b3d432f1b6e27f352abaa4c2a4c0d3 --local-dir MODEL_DIR
python3 tools/zig/moss_image_case.py case MODEL_DIR case/ --prompt-file prompt.txt   # needs torch, diffusers, transformers
zig build tf-moss-image -Dcpu=apple_m1
zig-out/bin/tf-moss-image MODEL_DIR/transformer case/case.safetensors case/
python3 tools/zig/moss_image_case.py decode MODEL_DIR case/latents.f32 case/zig.png
python3 tools/zig/moss_image_case.py compare case/reference.png case/zig.png case/side.png
python3 tools/zig/moss_image_case.py exact MODEL_DIR case/      # the same schedule with a float32 transformer
```

The case holds:

- both prompts' encoder rows;
- the timesteps and noise levels;
- the starting latents;
- the publisher's pipeline's input and both velocities at every step;
- with `--fp32-steps N`, a float32 run of the same transformer for the first N steps.

`tf-moss-image` first runs both branches of each step on the pipeline's own input and compares the velocities. It then
runs the whole guided schedule from the starting latents, times it, and writes the final latents. A case made by
`start` holds only the prompts, the schedule and the noise; `tf-moss-image` then just makes the picture.

On MPS, which has no float64, the case tool computes the time sinusoid on the host in float64 and moves its float32
result to the GPU; every other step of the publisher's code runs unchanged.

## Measured

Setup:

- Mac Studio M5 Ultra 256 GB, macOS 27; this commit's `tf-moss-image`.
- The card's defaults: 50 Euler steps, guidance 3, the empty negative prompt skipping block 22.
- Reference: the checkpoint's `pipeline_moss_image.py` on torch 2.14.1 (MPS, bf16), diffusers 0.42.0, transformers
  5.21.0.
- Two cases: a fox at 1024x1024 (4,096 image rows, 34 prompt rows, seed 42), and the card's Chinese sticky-note prompt
  at 1200x880 (an odd 55 x 75 grid, 4,125 image rows, 63 prompt rows, seed 7). The negative prompt has 10 rows.

Results:

- **Velocity**, both branches of each of the 50 steps on the pipeline's own input:
  - Against torch bf16: cosine 0.99973 to 0.99998, worst at the last steps.
  - Against a float32 run of the same transformer on the same inputs (the second case, every step): mean 1 - cosine
    1.602e-4 for the prompt branch, as for torch bf16's own (1.602e-4); 1.735e-4 against torch's 1.715e-4 for the
    negative branch. Ours is the closer of the two at 25 and 27 of the 50 steps.
- **Image:** our latents through the pipeline's decoder are 48.52 dB PSNR from the pipeline's own picture for the fox
  and 28.93 dB for the sticky note. Fifty guided steps magnify bf16 rounding into different fine detail, so the
  ruler is a float32 run of the whole schedule. Over 8 cases (both prompts, 4 seeds each), with the transformer in
  float32 on torch's bf16 encoder rows, our picture is the closer to it in 6, at a mean of 32.3 dB PSNR against torch
  bf16's 31.4 dB. With the encoder in float32 too, torch bf16 averages 32.5 dB and ours 31.2 dB: six cases are within
  0.5 dB or favor ours, and two sticky-note seeds favor torch's by 4.6 and 8.7 dB. Those 8 ran on an earlier build of
  this family (0b69e127): this commit's latents are byte for byte its on the two cases above.
- **Repeat:** the same step run twice, another between, gives identical bytes.
- **Speed**, one session with no other GPU work:
  - torch bf16 on MPS: 0.744 s a forward for the fox, 0.760 s for the sticky note (the median of its 100).
  - Ours: 0.489 s and 0.486 s a forward, 1.52x and 1.56x torch's (the median of three runs of 50 guided steps;
    0.472 to 0.544 s over all six). 0.08 s for both prompts' refiner once.
- **Memory:** 11.3 GB peak process footprint (10.6 GiB), with the transformer's bf16 weights (9.5 GB) in shared Metal
  buffers.
