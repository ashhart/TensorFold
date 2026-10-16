# Third-party notices: the Library's actions

"Turnaround sheet", "Upscale" and "Remove a person" (`app/actions.py`, `app/static/actions.js`,
`spark/spark_workflows.py`, `spark/setup-workflows.sh`, `scripts/turnaround.sh`, the `--sheet` mode of
`h3_generate.py`) run other people's work. None of it is in this repository: `spark/setup-workflows.sh`
(`oneshot-setup-spark.sh --ref2va` runs it) clones each node pack from its author's repository at the commit named
here into `$PREFIX/ComfyUI/custom_nodes` and downloads the two models from their publishers. The Studio then talks to
those nodes through ComfyUI's HTTP API as a separate program. The three LoRAs come from the LoRA hub
(`oneshot-essential-bundle.sh`), each from its author's page under its own terms.

## H3 Contact Sheet, by MATLOWAI

- ComfyUI nodes: <https://github.com/matlowai/ComfyUI-H3-ContactSheet>, commit
  `3c8c8c66e69362e5abf9cf4c4e693baddab90ca3`. GNU General Public License v3.0. Used unmodified, on a DGX Spark.
- The reference sampler those nodes follow: <https://github.com/matlowai/h3-contact-sheet> (GPL-3.0). Read, not used.
- The Mac's `h3_generate.py --sheet` is this pack's own code on TensorFold's H3 family: it uses the layout the
  authors describe in their README and model card (five one-step picture latents on the time axis of a 17-frame
  clip, 28 silent audio latents, each slot decoded alone as a still clip). No code of theirs was copied.
- The LoRA: `matlod/minimax-h3-turnaround` on Hugging Face, a Model Derivative of MiniMax H3 under the MiniMax H3
  Community License Agreement.

## H3 Video Upsampler, by dntpi (Sandpies)

- ComfyUI node: <https://github.com/dntpi/ComfyUI-H3-Video-Upsampler>, commit
  `36cb612ec3df30094eeb4fec66f1528925ebba73`. MIT License. Used unmodified, on a DGX Spark. Its own
  `THIRD_PARTY_NOTICES.md` credits LBH-123-AI (the latent upscaler's layout), bbaudio-2025 (the upscale recipe) and
  Alissonerdx (the guide layout and the `h3upscale-4k` training set).
- The latent upscaler weights: `LBH-123-AI/Minimax_h3_latent_Upscaler`, file
  `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`, SHA-256
  `4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`. Apache License 2.0.
- The LoRA: `lora/h3upscale_v2_x2.safetensors` from `sandpies/ComfyUI-H3-Video-Upsampler` on Hugging Face. Its
  page gives no licence of its own ("see third-party notices"); it derives from MiniMax H3, whose Community License
  Agreement applies. It is installed on the user's machine from its author's page and is in no hosted package of
  this pack.

## H3 Relay and the Person Remover, by Akatz Labs

- ComfyUI nodes: <https://github.com/akatz-ai/h3-relay>, commit `cf489b7f2472fdf33866fae36de49aea03ef464f`. GNU
  General Public License v3.0. Used unmodified, on a DGX Spark: its green-mask node and its Person Remover node,
  which runs the windows and carries 18 frames of history from one to the next.
- The Studio's graph is the one of the authors' example workflow `Person-Remover-Window-Reroll-V1.json`, with its
  settings (strength 1.0, windows of 22 frames, 12 passes, `er_sde` / `simple`, sigma shift 12 / 3, mask grown by 5
  points, seed 904234), on the Ref2VA weights in bfloat16 instead of the 8-bit file, and with this pack's own way to
  build the clean first frame when the user gives none.
- The LoRA: `akatz-ai/MiniMax-H3-Person-Remover-LoRA` on Hugging Face, under the MiniMax H3 Community License
  Agreement.

## SAM 3.1 (Meta)

- **Segment Anything Model 3.1** by Meta, in the single-file build `checkpoints/sam3.1_multiplex_fp16.safetensors`
  of `Comfy-Org/sam3.1` on Hugging Face, SHA-256
  `9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03`. SAM License. Run by ComfyUI's own SAM 3 nodes.
