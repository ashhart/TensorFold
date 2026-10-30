# MOSS-Image on Metal

MOSS-Image is a text-to-image model. A 5B joint transformer (42 blocks, hidden 3,072, 24 heads of 128, an MLP of
8,192 with the "situ" gate) denoises 128-channel latents by flow matching. `zig/src/families/moss_image` runs that
transformer on the Metal runtime from the released bf16 diffusers checkpoint:

- a prompt's rows once a prompt: the input projection and the two refiner blocks (bidirectional attention, biased
  projections, no modulation);
- the image and prompt rows together at each step: every joint block, one modulation from the timestep shared by all of
  them, full attention over both.

The card's guidance runs two passes a step: the prompt, and the empty negative prompt with block 22 skipped.

The prompt encoder, the checkpoint's Qwen3.5 text model, and the image decoder, FLUX.2's small VAE decoder, run on
Metal too (below): `tf-moss-image --prompt` makes both prompts' rows itself, and `tf-moss-image` turns the final latents
into the picture. `tools/zig/moss_image_case.py` runs the reference pipeline through the publisher's own
`pipeline_moss_image.py` in torch, a development reference.

It is a denoiser, not a token family. It does not register with the lane engine, the server or `families.detect`.

It runs on the M5 alone, the only chip it was checked on: `Model.load` and `Decoder.load` refuse a GPU without tensor
units.

## Checkpoint

`OpenMOSS-Team/MOSS-Image-Base` at `85975793d8b3d432f1b6e27f352abaa4c2a4c0d3`, Apache 2.0. `transformer/config.json`
must be `MOSSImageTransformer` at the released geometry, and `vae/config.json` `AutoencoderKLFlux2` at its released
decoder geometry. Any other config is refused, and so is a checkpoint with tensors this family does not read.

## Precision

- bf16 activations, rounded where the publisher's torch code rounds them: once per elementwise op, as torch's bf16
  tensors do.
- fp32 sums in every projection and in attention. Biased projections add the bias to the fp32 sums and round once, as
  a bf16 GEMM's bias epilogue does: their sums come from the prefill split-K GEMM (one part), then `mi_bias`.
- fp32 for norms, softmax and the time sinusoid's input (the sinusoid itself in float64, as the publisher computes it).
- The rotary tables in bf16, as the pipeline casts them; its frequencies are the checkpoint's `rope.bands`.
- No int8.

The decoder is float32 throughout, as the pipeline decodes: its weights as published (float32), fp32 sums in every
convolution and in its attention, GroupNorm's mean and variance in two fp32 passes, then the pipeline's own 8-bit
conversion (clamp, `* 0.5 + 0.5`, `* 255`, round to even). Its 3x3 convolutions run as Winograd F(2x2, 3x3)
(`md_wino`): the weights' transform G g G^T is computed in float64 at load and rounded once, the input's and the
output's transforms in fp32 in the kernel, 2.25x fewer products than a direct convolution. Its error stays the size
of a direct convolution's (see Measured). Like any Winograd convolution, an infinite input can come out NaN where a
direct one gives an infinity; the contract checks the classes against the Winograd formula's.

The bf16 projections are the Qwen-VL tower's `tfp::gemm_nax`. Attention is the prefill `tfp::attention_nax` template's
full (non-causal) form. The time MLP and both modulation projections run on the host, once a timestep.

## Prompt encoder

`encoder.zig` runs the checkpoint's `text_encoder/` (Qwen3.5's text model: 32 layers, three Gated DeltaNet layers to
each full-attention one, hidden 4,096) as the pipeline's `encode_prompt` does:

- the prompt in the pipeline's chat format (one user turn, then the assistant's turn with thinking on), cut at 4,096
  tokens;
- one causal forward;
- the final norm's rows.

The tokenizer is the native one on the checkpoint's `tokenizer/tokenizer.json`, and its ids equal the publisher's
tokenizer's on every prompt checked. The vision tower is not read.

Precision:

- bf16 weights as published and bf16 activations, rounded where the torch code rounds them;
- fp32 sums in every projection (the tower's `gemm_nax`), norm and attention;
- the DeltaNet's q, k, decay and state in fp32, its state a column a thread over the rows in order (the publisher's
  chunked algorithm computes the same recurrence);
- attention in fp32 throughout, the keys in order.

Its own kernels are in `moss_image_encoder.metal`:

- `me_norm`: the zero-centered RMSNorm;
- `me_head`: the head norms, the 64 rotated channels and the heads' layout;
- `me_attention`;
- `me_gate`;
- `me_silu_mul`;
- `me_conv`: the DeltaNet's 4-tap conv;
- `me_gdn_prep`;
- `me_gdn`: the recurrence;
- `me_gated_norm`.

The residual adds reuse `mi_gate_add`.

`tf-moss-image-encoder` checks a case from `moss_image_case.py encode`:

- the token ids;
- each step alone, every layer and the final norm, on the float32 run's input for that step rounded to bf16: its rows
  against float32 must be within twice torch bf16's distance on the same input (or plain bf16 rounding's, when
  larger) in cosine error and in its worst row;
- the whole prompt, every layer's rows against torch bf16's and the final rows against float32, reported but not
  gated: in bf16 the 32 layers magnify each implementation's rounding differently, so two correct runs part by more
  than any one step's error;
- a second encoding must repeat the first byte for byte.

It fails otherwise, and `reference.zig`'s tests check that a planted mismatch fails even when its repeat is identical.

## Kernel contract

`zig/tests/contract/metal_moss_image*.zig` registers every kernel of `moss_image.metal` and
`moss_image_encoder.metal`:

- `mi_norm`, plain and modulated, and `me_norm`;
- `mi_qk`, in both row layouts: q, k and v in three buffers, and packed in one row;
- `me_head` for q, k and v;
- `mi_situ`, in both layouts: hidden and gate halves of one row, and two buffers;
- `mi_gate_add` (in place), `mi_bias`, `me_gate`, `me_silu_mul`, `me_conv` and `me_gated_norm`;
- `me_gdn_prep` and `me_gdn`;
- the attentions: the transformer's three forms and `me_attention`.

The steps are checked output by output at the bf16 points the publisher's torch code rounds at. Each output must
lie in the range its fp32 steps allow, with Metal's error bounds for the `precise::` calls and fast-math division. Each
of 16 controls drops or adds one rounding point (the last adds bf16 rounding to the DeltaNet's state), and gate 3 must
catch each.

`me_gdn` is checked against the recurrence in float64. Each output's allowance is its two sums' rounding over the
state's magnitude, plus every earlier step's error: a decay and a projection by a unit key never grow it. The
attentions are checked against `attention_ref`'s float64 formula and bf16-P bound.

Gate 4:

- the attentions: equal scores over values whose sums reach about 2^22 and cancel;
- `me_gdn`: one-hot keys writing j, 2^20 and -2^20 to three state rows that its query sums back to j;
- `me_conv`: taps of 256, 1, 1 and -256 over rows that cancel to 2.

Gate 5 holds the 17 launches of the family's source and the two split-K projections.

`zig/tests/contract/metal_moss_image_decoder*.zig` registers every kernel of `moss_image_decoder.metal`:

- `md_unpack` (the latents' batch norm undone, the patches unpacked);
- `md_wino` (64 output channels a threadgroup) and `md_wino_32` (32, for 96 and 3 channels) in the forms the decoder
  launches: 3x3, 3x3 read nearest from half size, and 3x3 with a residual, against the float64 convolution through the
  exact transform, within a bound on the transformed terms' magnitude;
- `md_conv`, the 1x1 convolutions, with and without a residual;
- `md_norm`, with and without SiLU, and GroupNorm's statistics: `md_group_part` (plain and centered) and
  `md_group_finish`;
- `md_attention` and `md_pixels`.

Each fp32 step's output must lie in the range float32 allows about float64. Gate 4 uses integer channels, pixels or
values whose sums pass fp16's range and cancel. `md_pixels` is checked byte for byte against the same fp32 steps, with
every tie between two bytes among its inputs. Nine controls each break one kernel: a swapped patch axis, a 1x1's
inputs staged in half, a Winograd patch staged in half, a term dropped from its output transform, a SiLU in half,
both statistics kernels summing in half, the attention's rescale dropped, and the pixels' halves rounded up. Each is
caught at the gate it names. Gate 5 holds its nine launches.

`tf-kernel-contract --only "moss image"` on the M5 Ultra:

```
SUMMARY 37 kernels: 37 pass gates 1-4, 0 fail, 0 known findings; 25 controls: 25 red as planted, 0 missed; 0 limit problems
```

## Run a case

`moss_image_case.py` imports and runs the checkpoint's own `pipeline_moss_image.py`, the publisher's Python. It does so
only when the file's sha256 is the pinned revision's (`1dac1aca…8b31`) and refuses any other file.

```sh
hf download OpenMOSS-Team/MOSS-Image-Base --revision 85975793d8b3d432f1b6e27f352abaa4c2a4c0d3 --local-dir MODEL_DIR
python3 tools/zig/moss_image_case.py case MODEL_DIR case/ --prompt-file prompt.txt   # needs torch, diffusers, transformers
zig build tf-moss-image -Dcpu=apple_m1
zig-out/bin/tf-moss-image MODEL_DIR/transformer case/case.safetensors case/      # case/picture.png
python3 tools/zig/moss_image_case.py compare case/reference.png case/picture.png case/side.png
python3 tools/zig/moss_image_case.py exact MODEL_DIR case/      # the same schedule with a float32 transformer
python3 tools/zig/moss_image_case.py exact MODEL_DIR case/ --fp32-encoder prompt.txt   # and a float32 encoder

# the prompt encoder alone, and a picture with it
python3 tools/zig/moss_image_case.py encode MODEL_DIR enc/ --prompt-file prompt.txt
zig-out/bin/tf-moss-image-encoder MODEL_DIR enc/encoder.safetensors prompt.txt
python3 tools/zig/moss_image_case.py start MODEL_DIR pic/ --no-encoder --seed 42
zig-out/bin/tf-moss-image MODEL_DIR/transformer pic/case.safetensors pic/ --prompt prompt.txt
```

The case holds:

- both prompts' encoder rows;
- the timesteps and noise levels;
- the starting latents;
- the publisher's pipeline's input and both velocities at every step;
- with `--fp32-steps N`, a float32 run of the same transformer for the first N steps.

`tf-moss-image` first runs both branches of each step on the pipeline's own input and compares the velocities. It then
runs the whole guided schedule from the starting latents, times it, frees the transformer, and decodes the final
latents with the checkpoint's `vae/` decoder into `picture.png`. A case made by `start` holds only the prompts, the
schedule and the noise; `tf-moss-image` then just makes the picture. With `--prompt` (and `--negative`, else the empty
negative prompt), it encodes both prompts natively in place of the case's rows; `start --no-encoder` leaves them out of
the case, and torch never loads the encoder.

To check the decoder alone against a float64 decode of the same latents:

```sh
# float64 on the CPU, and float32 on MPS as the pipeline decodes
python3 tools/zig/moss_image_case.py vae MODEL_DIR case/latents.f32 case/decode.f64 case/torch.f32
zig build tf-moss-image-decoder -Dcpu=apple_m1
zig-out/bin/tf-moss-image-decoder MODEL_DIR case/latents.f32 1024 1024 case/decode.f64 case/torch.f32 case/decoded.png
```

Ours must be within twice torch float32's distance from float64, in cosine error and in its worst row of pixels, and a
second decode must repeat the first byte for byte; it fails otherwise.

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
  and 28.93 dB for the sticky note. Fifty guided steps magnify bf16 rounding into different fine detail, so the ruler
  is a float32 run of the whole schedule. There are two rulers, over the same 8 cases (both prompts, 4 seeds each):
  - The transformer in float32 on torch's bf16 encoder rows: our picture is the closer in 6, at a mean of 32.3 dB PSNR
    against torch bf16's 31.4 dB. Those 8 ran on an earlier build of this family (0b69e127); this commit's latents are
    byte for byte its on the two cases above.
  - Everything in float32, the encoder too (`exact --fp32-encoder`): torch bf16's picture averages 32.5 dB, ours 31.2
    dB. Six cases are within 0.5 dB of each other or favor ours (fox seed 1 by 2.9 dB). Two sticky-note seeds favor
    torch's by 4.6 and 8.7 dB, where the schedule settled on different fine detail.
- **Repeat:** the same step run twice, another between, gives identical bytes.
- **Speed**, one session with no other GPU work:
  - torch bf16 on MPS: 0.744 s a forward for the fox, 0.760 s for the sticky note (the median of its 100).
  - Ours: 0.489 s and 0.486 s a forward, 1.52x and 1.56x torch's (the median of three runs of 50 guided steps;
    0.472 to 0.544 s over all six). 0.08 s for both prompts' refiner once.
- **Memory:** 11.3 GB peak process footprint (10.6 GiB), with the transformer's bf16 weights (9.5 GB) in shared Metal
  buffers.
- **Prompt encoder**, both the prompt and the empty negative prompt (10, 34 and 63 tokens):
  - Token ids: equal to the publisher's tokenizer's.
  - Each layer's rows against torch bf16's, given the same token ids: cosine 0.99974 or more, the worst at the last
    layers.
  - The final rows against a float32 run of the encoder, ours beside torch bf16's own:
    - cosine 0.999826 and worst row 0.0261 against 0.999806 and 0.0294 (empty);
    - 0.999931 and 0.0261 against 0.999934 and 0.0259 (fox);
    - 0.999926 and 0.0454 against 0.999938 and 0.0318 (sticky note).
  - Repeat: identical bytes.
  - `tf-moss-image-encoder`'s gates, all 33 steps on these three prompts and six more (31 to 75 tokens): each step's
    distance from float32 is a mean of 0.50 of its limit, torch bf16's own, and at most 0.65 (the fox's layer 13,
    worst row). A row planted 5% off at layer 20, in the float32 rows and torch's, fails there and at layer 21, its
    repeat identical.
  - The whole prompt in bf16 spreads by more than any step: on the 75-token prompt, torch's own runs (MPS or CPU,
    chunks of 8 to 64, or the recurrent form) end 0.86 to 4.97 times its default's cosine error from float32, and
    ours 1.82 times. Against float64, which float32 matches to four places, the sticky note's row 58 is 0.0153 to
    0.0320 across eight of those runs, and 0.0454 in ours.
  - Pictures with our encoder, against the all-float32 ruler over the same 8 cases: a mean of 31.13 dB. Our
    transformer on torch's encoder rows gives 31.20 dB, and each is the closer in 4 of 8. The fox, against the
    pipeline's own picture: 47.07 dB.
  - Speed, one session with no other GPU work: 0.054 to 0.067 s a prompt against torch MPS's 0.158 to 0.165 s
    (warm). Loading takes 0.6 s against 2.60 s.
- **Decoder**, on the transformer's final latents of both cases, against the publisher's decoder in float64 on the CPU:
  - Ours: max |diff| 8.9e-6 (fox) and 5.2e-6 (sticky note) before the clamp; 162 of 3,145,728 and 45 of 3,168,000
    8-bit values differ from float64's, each by 1.
  - torch's own float32 decode on MPS, in the same session: 155 and 34 values differ from float64's, each by 1 (the
    count moves between torch runs). Ours and torch's pictures differ by at most 1 (95.4 and 97.7 dB PSNR).
  - Speed, back to back in one session with no other GPU work, the median of three: ours 0.181 s and 0.188 s on the
    GPU; torch's warm float32 decode on MPS 0.186 s and 0.191 s. Ours loads in 0.3 to 0.4 s. Its scratch is five
    float32 picture tensors at the widest stage: 4.7 GB at peak at 1024x1024, after the transformer is freed, so
    `tf-moss-image` peaks at the transformer's 11.3 GB.
