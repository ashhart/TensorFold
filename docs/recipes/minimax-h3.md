# MiniMax H3 (`h3`)

Joint video and audio generation, not a language model: H3 denoises one packed sequence of video, audio and text
rows and decodes it to an MP4. It does not decode through lanes and has no `load`; `tensorfold serve` does not
route it. Packages: `src/tensorfold/families/h3/` and `src/tensorfold/kernels/minimax/h3/v1/`.

Measured on a Mac Studio M5 Ultra with 256 GB, macOS 27.0.1, MLX 0.32.3, with `MiniMaxAI/MiniMax-H3` (the `FL2VA`
partition, bfloat16). MiniMax H3 is under the MiniMax H3 Community License, which limits the territories it may be
used in; TensorFold ships no weights.

## What is here and what is not

| Part | State |
| --- | --- |
| Diffusion transformer (33B, 50 blocks) | `dit.py`, `weights.py`; equal to minimax-h3-mlx's forward on a real step |
| Packed sequence, schedules, joint denoise loop | `packing.py`, `schedule.py`, `sampler.py` |
| Adapters | `lora.py`: runner layout and FastVideo `fastvideo-lora-v2`, merged in float32 |
| First-frame image to video | `sampler.py` holds the keyframe rows at their noise level; `packing.py` places them |
| Video decoder | `vae_video.py`; equal to minimax-h3-mlx's decode in float32, int8 by default |
| Text encoder (Qwen3-VL, with its vision tower for a first frame), the VAE encoder that turns a first frame into rows, audio decoder, MP4 writer | not ported; `tools/h3_generate_dev.py` borrows minimax-h3-mlx's |
| `tensorfold generate`, checkpoint detection through `families.detect`, a resident engine | not started |

`tools/h3_generate_dev.py` renders a clip with this family between minimax-h3-mlx's text encoder and audio decoder.
Run it from an environment that has minimax-h3-mlx and its requirements on the path. `--first-frame IMAGE` starts the
clip from an image: the image is stretched onto the canvas, encoded by the VAE to one latent frame of conditioning
rows and shown to the text encoder; those rows sit at timestep 0.999 for every step and are not denoised. Last-frame
and reference modes are not wired.

## What decides the speed

- A 5 second 864x480 clip (124 frames) is 15,918 rows: 14,985 video, 414 audio and the prompt's text rows. Hidden
  size 5,376, 56 heads of 128, SwiGLU width 14,336.
- One block in bfloat16 takes 235 ms: MLP 81, attention 93, QKV with the q/k norm and rotation 46, attention output
  12. Each matmul stage runs at about 80-95e12 operations a second; float16 weights give the same times.
- The projections run as int8 through the M5 tensor units: activations take one scale per row (per row and 1,024
  channels on the wide inputs), weights one per output channel, in 128x128x128 tiles. MLP 41 ms, QKV with the norm
  and rotation inside the kernel 22.5 ms, attention output 7 ms. A forward goes from 13.9 s to 9.0 s.
- AdaLN tables depend only on the schedule: they are projected once per run and the projection weights (24 GiB)
  released.
- The video decoder is a 36-layer ViT over 105 overlapping tiles for this clip. Its SwiGLU, QKV and output
  projections on the same int8 kernels take the decode from 17.3 s to 10.7 s at 47 dB against the float32 decode.

## Adapters

An adapter's update is about a thousandth of the weight it rides on. Rounding the merged weight back to bfloat16
keeps 50-85% of the update and adds rounding noise of the same size. Adapters are therefore added in float32 and
quantized straight to int8, which keeps the update in expectation. A projection that is not on an int8 kernel is
rounded to bfloat16 and the tool says so.

Step-distilled adapters set the number of forwards: lightx2v's Turbo adapter with 4 sigma points (3 forwards),
FastVideo's FastH3 dense adapter with 5 points (4 forwards).

## Measurements

5 second clip with audio, seed 2077, one render at a time, wall time from process start unless noted.

| 864x480, 124 frames | Forwards | Per forward | Denoise | Clip |
| --- | --- | --- | --- | --- |
| bfloat16 | 20 | 13.9 s | 277 s | about 305 s |
| int8 MLP, QKV, attention output | 20 | 9.7 s | 195 s | about 223 s |
| Turbo adapter, int8, int8 video decoder | 3 | 9.0 s | 27.6 s | about 50 s |

At 768x448 with the Turbo adapter the same configuration renders in 40-41 s (denoise 21.6-22.3 s, decode 8.2 s).

The int8 20-step render keeps the bfloat16 composition (median 18 dB between the two clips' frames, the same
shots). With 3 forwards the int8 path lands on a different composition from the bfloat16 one.

## What did not work

- **int8 attention.** A flash-style kernel (int8 scores, half probabilities, int8 values, online softmax over
  64-key tiles) takes 95.5 ms against 95.8 ms for MLX's scaled dot-product attention at 15,918 rows. The inner
  dimension of both matmuls is 64-128, so the work between them costs what int8 saves.
- **Block-sparse attention.** On a real 20-step run, keeping 95% of each 32-row query block's attention needs
  22-68% of block pairs per head and 51-95% with one mask shared by all heads.
- **Video decoder precision and batch.** float16, bfloat16 and a batch of 32 tiles decode in the same time as
  float32 with 8.
- **A matmul for the decoder's 1x1x1 convolution** differs from the convolution by 5e-4, which 36 layers grow to
  0.1 in places; it stays a convolution.

## Verification

`tools/h3_generate_dev.py --parity` runs one real first step through this transformer and minimax-h3-mlx's and
reports the difference: zero with the same bfloat16 input rounding. The float32 video decode equals minimax-h3-mlx's
on real latents. The sampler refuses a modality that does not start from unit noise: video and audio denoise in one
sequence, so a constant audio start corrupts the picture as well as the sound.

## A 2x decoder (optional)

`load_video_decoder(..., upscale_decoder=FILE)` and the tool's `--upscale-vae FILE` take a replacement ViT decoder
whose head packs 12 channels, such as `speach1sdef178/MiniMax-H3-X2-Detail-VAE` (`MiniMax-H3-X2-Detail-v1.safetensors`,
MiniMax H3 Community License). The tiles are decoded and blended as usual and each packed pixel is then spread over a
2x2 cell, so the frames come out twice as large along each side for the same latents and the same decode time. Only
that file's decoder is used; its reference-image detail branch is not ported.

| 8 s clip, Turbo adapter, int8 | Rows | Per forward | Clip |
| --- | ---: | ---: | ---: |
| generated and decoded at 1344x768 | 60,403 | 104.9 s | 353 s |
| generated at 672x384, decoded at 1344x768 | 15,800 | 8.9 s | 56 s |

One prompt and seed. The 2x decode of the small clip is softer than the native 1344x768 clip, with visible stair-steps
on high-contrast edges; it is an upscale, not the detail of a larger generation. A 1344x768 generation decodes to
2688x1536 in the same time as its normal decode.



## Audio with the Turbo adapter

With 3 forwards the released audio shift of 3 puts the audio nodes at 1, 0.857 and 0.6, so the last step jumps from
0.6 to clean. The result is thin: on one 5 s clip the 120-300 Hz band holds 19% of the energy against 22% at 20
steps, the 1-4 kHz band 27% against 17%, and the first 400 ms sit about 10 dB under the 20-step clip, a fade-in.
`denoise(..., audio_shift=1.3)` and the tool's `--audio-shift 1.3` move the nodes to 1, 0.72 and 0.39 at no cost:

| 5 s clip, 864x480, Turbo, int8 | Forwards | 120-300 Hz | 1-4 kHz | Spectral centroid | First 100-200 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 steps, no adapter | 20 | 21.9% | 17.3% | 692 Hz | -27 dB |
| Turbo, audio shift 3 (released) | 3 | 18.8% | 27.3% | 1,139 Hz | -37 dB |
| Turbo, audio shift 1.7 | 3 | 26.9% | 18.3% | 774 Hz | -36 dB |
| Turbo, audio shift 1.3 | 3 | 23.1% | 17.5% | 737 Hz | -36 dB |
| Turbo, audio shift 3 | 4 | 30.7% | 18.0% | 817 Hz | -35 dB |
| Turbo, audio shift 3 | 8 | 35.3% | 13.1% | 660 Hz | -30 dB |

The spoken line is transcribed correctly in every row. On a second prompt (text to video, a scene that opens on
sound) the first 100 ms are 37 dB under the clip's level at shift 3 and 5 dB under at 1.3. Higher shifts are worse
(6: centroid 1,668 Hz). Measured, not listened to; two prompts, one seed each.
\n