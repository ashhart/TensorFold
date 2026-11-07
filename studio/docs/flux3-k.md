# FLUX3-K picture models (FLUX 3 Image Klein)

Black Forest Labs' FLUX 3 Image Klein ("FLUX3-K") joins Qwen-Image-2.1-Turbo as a picture model: `IMAGE_MODELS` in
`app/picture_models.py`, `image_model` on scout, studio and edit jobs, `IMAGE_MODEL` to `scripts/image.sh` and
`scripts/edit.sh`. On a DGX Spark it runs inside ComfyUI's workflow (`spark/flux3_workflows.py`), on a Mac on
flux3-mlx in its own environment (`flux3_generate.py`, `flux3_edit.py`). `bash oneshot-setup.sh --flux3` installs it
(`scripts/setup-flux3.sh` on a Mac, `oneshot-setup-spark.sh --flux3` on a Spark). No weights ship with the Studio:
FLUX 3 Image Klein is under the FLUX Kommunity License (non-commercial use; commercial use of the outputs only for
Qualifying Users, under US$5M revenue), and the Hugging Face repositories are gated.

## Release notes and measurements (Studio 2.8.0, 2026-10-15)

**2.8.0 (2026-10-15): pick the picture model: Qwen 2.1 Turbo, FLUX3-K Fast or FLUX3-K Full.** Black Forest Labs'
FLUX 3 Image Klein ("FLUX3-K") joins Qwen-Image-2.1-Turbo wherever the Studio makes a picture from a prompt: *Image
only*, the scouts a clip starts from, and a last frame. **FLUX3-K Fast** is the distilled checkpoint (16 steps),
**FLUX3-K Full** the base one (50 steps at CFG 3.5); both get a full scene caption from BFL's prompt upsampler first.
On a DGX Spark they run on ComfyUI's own FLUX 3 nodes (ComfyUI moves to `5a665890`, v0.41.0 and later), upsampler
included, so the render stays under the memory guard; on a Mac on flux3-mlx (Apache-2.0, a port of BFL's reference
code) in an environment of its own. `oneshot-setup.sh --flux3` installs them (41 GB; the Hugging Face repos are gated:
accept the FLUX Kommunity License first). Measured at 1344x768 with the model loaded, bf16: Qwen Turbo 9.7 s on an
M5 Ultra; FLUX3-K Fast 10.3 to 10.7 s on a Spark (ComfyUI; diffusers took 15.5 s for the same picture) and 14.9 s on
an M5 Ultra; FLUX3-K Full 57.5 s on a Spark and 53.6 s on an M5 Ultra; the upsampler, once per prompt, about 18 s on
a Spark and 5 s on an M5 Ultra. Against Qwen on the same six prompts and two seeds, FLUX3-K spelled a poster's title and tagline
right where Qwen wrote "SOME ONE IS STOLL LISTENNING", drew whole figures for character sheets where Qwen cut them at
the waist, and kept two characters recognisable in a two-reference shot; Fast looked as good as Full. The same seed
draws a different picture on the two machines (the runtimes draw their noise differently). **Edit a picture** offers
the same choice: FLUX3-K takes the picture, and up to nine more beside it, as reference latents (BFL's edit layout:
every picture on the prompt and on the negative, the result at the first picture's size), and the upsampler sees the
pictures before it writes the caption of the result (its i2i system prompt). Image LoRAs, the camera ring and the
40-step base model stay Qwen's. Measured with the job's own scripts, seed 7, 1344x768, upsampler included: an edit
took 24 s with Qwen, 37 s with FLUX3-K Fast and 139 s with Full on a DGX Spark, 10, 37 and 240 s on an M5 Ultra; two
pictures (a person from one put in the place of another) 23, 44 and 188 s on a Spark, 13, 42 and 427 s on the Mac.
The reference rows lengthen every step, so Full is slow for edits on a Mac (about four times its text-to-image
time): Fast is the FLUX3-K for edits. FLUX3-K follows "switch the neon off" where Qwen left the signs lit on the
Spark. FLUX 3 Image Klein is under the FLUX Kommunity License: non-commercial use, and commercial use of the
outputs only for Qualifying Users (under US$5M revenue). The carrier image moves to
`ghcr.io/drowzeys/keys-tensorfold-studio:2.8.0`: the same Mac payload, and the DGX Spark's prebuilt engine is now the
LoRA build of 2.7.1 (`h3-cuda-lora` `1f7ecc75`), so a Spark install takes it instead of building it from source.
