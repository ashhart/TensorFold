# Third-party notices: the Motion Previs tab

The Motion Previs tab (`app/motion.py`, `app/motion_worker.py`, `app/static/motion.js`, `app/motion_templates/`,
`scripts/setup-motion.sh`) builds on the work below. None of it is bundled in this repository except where said:
packages and models are downloaded by `scripts/setup-motion.sh` (`oneshot-setup.sh --motion`) into
`$PREFIX/motion-venv` and `$PREFIX/motion-models`. The notices for the rest of the pack are in [NOTICE](NOTICE) and
[CREDITS.md](CREDITS.md).

## Motion Previs Studio, by Sam Wasserman

- Original: **Motion Previs Studio 4.1.0**, <https://github.com/wassermanproductions/motion-previs-studio>, read at
  commit `95e7d0ff1d4cc546f7eb09a74ccbd084988a19bc`.
- Copyright 2026 Sam Wasserman. Developed and created by Sam Wasserman for Wasserman Productions and Wasserman.ai.
- License: Apache License 2.0 (the text is this repository's [LICENSE](LICENSE)). Its NOTICE file is kept unchanged at
  [LICENSES/Motion-Previs-Studio-NOTICE](LICENSES/Motion-Previs-Studio-NOTICE).
- Windows support in that repository was contributed by Gumbii Digital; none of the Windows packaging is used here.

**What is taken, and that it was modified.** The analysis in this tab is a Python port of his: `app/motion_worker.py`
is a rewrite in Python of `src/lib/pose.ts`, `poseVideo.ts`, `openpose.ts`, `cameraMotion.ts`, `aiDepth.ts` and
`quality.ts`, and of the ffmpeg control passes and the production pack writer in `electron/main.cjs`. The pack's
file names, the layout of `bundle_manifest.json`, the schemas of `camera_motion.json`, `pose_landmarks.json` and
`openpose_keypoints.json`, the four reference modes, the wording of the prompt pack, shot bible and manifests, and the
three Blender import scripts in `app/motion_templates/` (copied from `electron/main.cjs` with a header added) are his.
The interface in `app/static/motion.js` follows his app's flow and shares no code with it.

**What differs from the original**

| Original | This port | Why |
|---|---|---|
| Electron + React/TypeScript desktop app | A tab of this web app; the analysis is one Python process per job | It runs beside the Studio's render queue, on a Mac or a DGX Spark, with no second app |
| MediaPipe Tasks Vision (WASM) in the renderer | The `mediapipe` Python package, same Pose Landmarker models and hashes | Same models, native speed |
| Pose model defaults to Lite | Defaults to Heavy (Lite and Full are offered) | Heavy tracks a 5 second shot in about a second here, and on the test clips Lite placed the shoulders well inside a coat |
| Camera solve written out by hand on a 192 pixel canvas: Shi-Tomasi corners, pyramidal Lucas-Kanade, seeded two-point RANSAC | OpenCV's `goodFeaturesToTrack`, `calcOpticalFlowPyrLK` (with a forward-backward check) and `estimateAffinePartial2D` (RANSAC, fixed seed) on a 384 pixel canvas | The same method on the library's tested routines; the same subject mask from the pose track, the same clamps and confidence |
| The shift of each step is read at the top-left corner of the frame | It is read at the centre of the frame | A push-in on the middle of the picture then has no pan or tilt in it |
| The roll of each step is the rotation of the similarity fit | Roll is kept only where both picture axes turn the same way | Travelling past near and far things shears the picture, and the fit read that as several degrees of roll that were not there |
| With no pose track, corners in the middle of the frame score half | Corners are taken from outside the middle first, then topped up | `goodFeaturesToTrack` takes a mask, not a weight |
| Depth-Anything-small through transformers.js (WebGPU or WASM), short side 518, at most 8 frames a second, levels set per frame | The same ONNX file (`Xenova/depth-anything-small-hf` at the pinned revision, 8-bit) through ONNX Runtime on the processor, long side 518, at the chosen sample rate, levels that follow the clip slowly | A quarter of the work for a control layer, no flicker between frames, and a layer smooth enough to hand on as a reference clip |
| A smaller source is enlarged to 1280 pixels | A smaller source keeps its size | The Studio's own clips are often 864 x 480 |
| Imports web video through yt-dlp; sends clips to Blockout; MCP control server; session restore; 3D preview | None of these | Not asked for here. No URL download exists in this tab |
| `pose_high_contrast.webm`, `openpose_pose.webm`, `ai_depth.webm` | Not written (their manifest entries are null) | They are the original's pre-v4 recorder output; the mp4 files are the same pictures |
| Any shot length | A shot is 2 to 15 seconds | The length of a reference clip MiniMax H3 Ref2VA accepts |
| - | `bundle_manifest.json` has one more key, `port`: what this is a port of, the camera move in words, per-stage timings, notes | The original's keys are unchanged and in its order |
| - | A "MiniMax H3" section at the end of `prompt_pack.md` with the camera move in words; `motion_reference()` | The hand-off this tab exists for |

## MediaPipe (Google)

- The `mediapipe` Python package 1.1.0, <https://github.com/google-ai-edge/mediapipe>. Apache License 2.0.
- Pose Landmarker models `pose_landmarker_lite.task`, `pose_landmarker_full.task`, `pose_landmarker_heavy.task`
  (float16, version 1), from `storage.googleapis.com/mediapipe-models`, with the SHA-256 hashes pinned in Motion Previs
  Studio's `ASSET_MANIFEST.json`. Apache License 2.0.

## Depth Anything

- **Depth Anything** (Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao; CVPR 2024),
  <https://github.com/LiheYoung/Depth-Anything>. Code: Apache License 2.0. The small model
  `LiheYoung/depth-anything-small-hf`: Apache License 2.0.
- The file used is its ONNX conversion `onnx/model_quantized.onnx` from `Xenova/depth-anything-small-hf` at revision
  `2e942621ab9f2371c1df9eb223291b5ac31475e6` (the repository and revision Motion Previs Studio pins), SHA-256
  `e4e2ffe9b0396ebd4a0c09b855762b186aab338e79f76531b09777aafc91db84`. That repository declares no license of its
  own; it is used under the Apache-2.0 license of the base model it names.

## OpenCV

- `opencv-contrib-python` 5.0.0.93 (the build MediaPipe requires), <https://github.com/opencv/opencv> and
  <https://github.com/opencv/opencv-python>. OpenCV: Apache License 2.0. The Python packaging: MIT. The wheels carry
  third-party libraries (among them FFmpeg libraries under LGPL-2.1-or-later) with their license files inside the wheel.
- Without the Motion Previs environment, the camera solve runs on the `opencv-python` already in the Mac Studio
  install, under the same licenses.

## ONNX Runtime (Microsoft)

- `onnxruntime` 1.31.0, <https://github.com/microsoft/onnxruntime>. MIT License.

## NumPy

- <https://numpy.org>. BSD 3-Clause License.

## FFmpeg

- The `ffmpeg` and `ffprobe` on the machine's PATH are run as separate programs (trim, control passes, encoding); none
  is bundled. FFmpeg is licensed under LGPL-2.1-or-later or GPL-2.0-or-later depending on how it was built.

This file is informational and does not replace the license texts shipped inside each package.

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
