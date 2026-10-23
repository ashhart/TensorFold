# Credits

**Thank you.** Almost everything here is other people's work. Keys / drowzeys ported pieces into two TensorFold
families, wrote the int8 tensor-unit kernels with their designs as the guide, pinned it and measured it on a Mac
Studio M5 Ultra. Cite the authors first.

## The film workflow in Production: Sam Wasserman's Slate

**Sam Wasserman (wassermanproductions.com)** created **Slate**, the prompt studio for AI filmmaking: projects, scenes
and shots, the six-section shot prompt, the Bible, Casting, the Art Department, Locations and the Lookbook, coverage
plans, setups, the brain's transforms, Director's Notes and the First AD. The *Wasserman Slate AI Studio* inside the
Production tab is a port of that workflow. The project file is Slate's own `project.json`; the brain prompts and the
First AD's action contract are Slate's; `app/static/slate/` holds Slate's coverage plans, setups and model profiles
unchanged, with its NOTICE. What this pack adds is the render half Slate leaves to a generator: a compile to MiniMax
H3's prompt form, the queue, and takes with their clips. Slate does much more than this port: sequence chunks, alt
takes, punch-ups, the tone dial, reference and audio analysis, the stills library, per-model compiles for a dozen
generators, and its MCP server. (Apache-2.0.)

The **Wasserman Sharp Spec** button and the QA gate behind it are his too: the locked recipe (1376x768, FastH3 at 8
passes, seed 42, one pass with no upscale) and the two gates, a transcript of the spoken line and a three-frame
contact sheet checked by a vision model, with one retake at the next seed. `app/qa.py` is this pack's writing of them.

The first preset of the *Keys and Wasserman LoRA Hub* is his **80s Horror LoRA** (Wasserman 80s Horror LoRA, a style
LoRA for MiniMax H3). The pack holds only its recipe; the weights are downloaded from his release when a user installs
them, as a Model Derivative of MiniMax H3 under the MiniMax H3 Community License Agreement. 80s Horror LoRA by Sam
Wasserman (wassermanproductions.com). Powered by MiniMax H3.

## The LoRA hub's community presets: their authors

From 2.6.3 the hub lists twenty-one community LoRAs in seven groups, twenty for MiniMax H3 and one for Qwen-Image-2.1. Each is its author's own work, under
that file's own licence. The pack holds only their recipes; the essential bundle fetches each file from its author's
page and keeps the author's model card and licence beside it.

- **Camera & Viewpoint:** Camera Motion by Jojocodex; Head-mounted FPV by neph1; Equirectangular 360° by shamanic;
  360° Orbit by pablodawson.
- **Transformation & Morphing:** Transformation by Ashmotv; Turnaround by matlod; Keyframe Animation by alvdansen.
- **Character Editing:** Character Swap and Person Remover by Akatz Labs (akatz-ai); Natural Face Speech by vpakarinen.
- **Motion & Action:** Motion Continuity Repair, Combat Base V2 and Weapon Combat by JOKER141; Spatial Physics by
  Jojocodex.
- **Art Style:** Anime Slider by adf99; Lineart Anime by DiffSynth-Studio; 3D Animation Style by SOLRICKS.
- **Support & Finishing:** Video Upscaler LoRA by dntpi (with the latent upscaler by LBH-123-AI it builds on).
- **Fun:** VH5 Tape (VHS) by KennethFal; Wushu Action by Jojocodex; Multiple Angles, for Qwen-Image-2.1, by akhaliq.

The MiniMax H3 ones are Model Derivatives of MiniMax H3. Powered by MiniMax H3.

## The Library's actions: the people whose workflows they run

Three of those LoRAs are not for making a clip. Each has a workflow of its own, built by its author, and the Library
runs that workflow as an action. On a DGX Spark it is the author's own ComfyUI nodes, cloned from the author's
repository at a pinned commit when the Studio is installed; none of their code is in this pack.

- **Turnaround sheet:** the Turnaround LoRA by matlod and the contact-sheet sampler and ComfyUI nodes by MATLOWAI
  (H3 Contact Sheet, GPL-3.0). The idea is theirs: five still pictures packed where a five-step video would go, so
  the model's consistency along time becomes consistency between views. On a Mac the MLX engine packs and decodes
  the five views the same way (`h3_generate.py --sheet`), written from their description of it.
- **Upscale:** the Video Upscaler LoRA and the H3 Video Upsampler node by dntpi (Sandpies), MIT; the latent upscaler
  network it enlarges with by LBH-123-AI (weights Apache-2.0); the upscale recipe it follows by bbaudio-2025; the
  guide layout and the training set of the LoRA by Alissonerdx.
- **Remove a person:** the Person Remover LoRA and H3 Relay (the windowed removal with its green mask and its relay
  of history frames, GPL-3.0) by Akatz Labs (akatz-ai); the person is found and followed by SAM 3.1 from Meta, in
  the build Comfy-Org repackaged for ComfyUI, under the SAM License.

## The image model: the Qwen team

The **Qwen team at Alibaba** released **Qwen-Image-2.1**: the 7B single-stream transformer with its causal text
prefix, the 64-channel VAE and the Qwen3-VL text encoder, with code in diffusers. The image half of this pack is
their model. The weights are under the Qwen Research License Agreement (non-commercial); this pack ships none of
them. Built with Qwen.

## The MLX port the image family is adapted from: Filip Strand and the mflux contributors

**mflux** has the MLX implementation of Qwen-Image-2.1 that TensorFold's family is adapted from: the transformer
module tree, running the text prefix once and reusing its keys and values, the rotary layout, the shifted schedule
and the VAE. It is also the reference every number here was checked against, and this pack still runs its prompt
encoder directly. mflux does much more than this port: editing, reference images, transparent output. On a Mac,
Edit a picture runs on mflux's own Qwen-Image-2.1 edit pipeline as it is. (MIT.)

## The image model since 2.3: Qwen-Image-2.1-Turbo

The **Qwen team** released **Qwen-Image-2.1-Turbo**, their own 8-step checkpoint with its sampling schedule saved
in it; the Studio's images come from it since 2.3. The DGX Spark build loads ComfyUI's one-file repack of it
(`Comfy-Org/Qwen-Image-2.1`).

## The image turbo adapter up to 2.2: Viggle

**Viggle** distilled **Qwen-Image-2.1-viggle-turbo**, the 6-step adapter behind the fast image numbers here, and
documented its nodes and schedule precisely enough to reproduce. (Qwen Research License.)

## The video model: MiniMax

**MiniMax** released **MiniMax H3** (Hailuo) as open weights with code: the 33B joint video and audio diffusion
transformer, the Qwen3-VL text encoder setup, and the video and audio VAEs. Nothing in this pack exists without
that release. The weights are under the MiniMax H3 Community License; this pack ships none of them.

## The first native Mac engine: Salvatore Sanfilippo (antirez)

**antirez** wrote **h3.c**, MiniMax H3 in C and Metal. It is the reference for what is possible on Apple
Silicon and the cleanest picture we measured. The int8 kernels here follow h3.c's quantization scheme: per-row
activation scales, per-output-channel weight scales, one scale per 1,024 channels on wide inputs, 128x128x128
int8 tiles on the M5 tensor units, and the q/k norm and rotation inside the QKV kernel. No h3.c source is
copied; the design is his. (MIT.) Its `--ref-image` was the working reference-to-video on a Mac that this pack's
Ref2VA layout was checked against, row for row; the layout itself follows **ComfyUI**'s MiniMax H3 nodes
(Comfy-Org, GPL-3.0: read for the arithmetic, no source copied). Reference videos and reference audio follow
**MiniMax**'s own diffusers pipeline for H3 (`packing_ref2va.py`, Apache-2.0, as vendored by minimax-h3-mlx), and
h3.c's `--ref-video` and `--ref-audio` were the second implementation the layout was compared with.

## H3MLX: RobZombAI

**RobZombAI** built **H3MLX** on top of h3.c: a second-order solver, int8 paths, 4K mastering and a large preset
library. Working through it on an M5 Ultra is what started this pack, and its Sol-style switches and solver
gave us measured points to compare against.

## The MLX port this family is adapted from: mrbizarro

**mrbizarro** wrote **minimax-h3-mlx**, the MLX port of H3 that runs inside **Phosphene**. TensorFold's H3
family is adapted from it: the transformer module tree, the packed sequence and rotary positions, the two sigma
schedules, the video decoder with its tiling and clip chunking, and the adapter layout handling. Its notes on
bfloat16 adapter merging, QKV row order and SwiGLU half order saved days. This pack still runs its text
encoder, audio decoder and MP4 writer directly, and downloads the runner-layout Turbo adapter asset that
Phosphene publishes. (Apache-2.0.)

## The engine: Ash Hart

**Ash Hart** (ashhart) and the TensorFold contributors — **TensorFold**: the family and kernel structure this
lives in, and the practice of reaching the M5 tensor units from `mx.fast.metal_kernel` that the int8 kernels
are built on (`kernels/qwen/dense/v1/lane_qmm.py`). This pack pins TensorFold 0.6.5 with three commits on top.
(Apache-2.0.)

## Sol-Engine, Sol-Attn and Sol-H3: NVIDIA Research

**NVIDIA Research, Efficient AI Team and Singapore Lab** (Jincheng Yu, Junsong Chen, Yitong Li, Haopeng Li,
Haocheng Xi, Song Han, Enze Xie and co-authors of the Sol-Engine and Sol-Attn papers). Sol-H3 showed which
optimizations matter for H3: few-step adapters, fused norm / RoPE / SwiGLU kernels, precomputed AdaLN, int8 QKV,
and dense attention on a single device. The cached AdaLN tables and the fused QKV kernel here follow that list.
Their runtime is CUDA; no Sol code runs here.

## The 2x video decoder: speach1sdef178

**speach1sdef178** trained **MiniMax-H3-X2-Detail-VAE**: H3's own ViT decoder with a head that packs 12 channels, so
the same latents decode to frames twice as large. It is what makes the 2K finish and the fast drafts here possible,
and its write-up is unusually honest about what the 2x output does and does not contain. This pack uses the decoder
only; the reference-detail branch in the same file is not ported. (MiniMax H3 Community License.)

## Few-step adapters

- **LightX2V** (ModelTC) — the **MiniMax H3 Turbo** 4-step adapter (`lightx2v/Minimax-h3-Turbo`, Apache-2.0),
  the adapter behind every fast video number in this pack.
- **LightX2V** (ModelTC) — the **Ref2VA Turbo** 4-step adapter from the same repository (Apache-2.0), behind
  `REF_TURBO=1` for reference pictures.
- **larryvrh** — `MiniMax-H3-Turbo-Lora`, the adapter lineage minimax-h3-mlx's loader was written for.
- **Ash Hart** and the TensorFold contributors — the **TensorFold 1.0 native runtime** (Zig driving Metal, Apache-2.0)
  that the FastH3 text-to-video passes run on since 2.0: its Metal bindings, checkpoint readers and build. The H3
  family and kernels added on top are this pack's; its LICENSE, NOTICE and THIRD_PARTY_NOTICES ship beside the
  program in `zig-engine/`.
- **FastVideo / Hao AI Lab @ UCSD** — **FastH3**: the distilled FastH3 8-Step V2 transformer that `scripts/fast.sh`
  runs, the VSA-H3 sparse-attention routing it was trained with (their MLX routing code is vendored in the engine,
  Apache-2.0), and the 4-step adapters (`fastvideo-lora-v2`) this family also reads.
- **TaoLiveAIGC** — the TaoMate-H3 3-step ladder, documented in Phosphene and supported by the sigma subset
  option.

## The workflow the web app follows: rookiestar28

**rookiestar28** built **ComfyUI-MiniMaxH3-Studio**: a prompt workbench, a production board for longer videos and a
clip editor for MiniMax H3 inside ComfyUI. The web app in this pack follows that create, produce, edit flow on top
of this pack's scripts. No code is shared; theirs is the far more complete tool. (Apache-2.0.)

## The many-step fast recipe: ddalcu

**ddalcu**'s **mlx-serve** showed how to make 30 steps cost nine: a velocity cache and attention reuse with careful
gates. The engine pinned here carries an adaptation of it; it is not yet enabled in this pack's scripts.

## Platform and tools

- **Apple / ml-explore** — **MLX**, and the Metal 4 tensor operations on M5.
- **mlx-vlm** contributors — the Qwen3-VL language model the text encoder runs on.
- **Qwen team (Alibaba)** — Qwen3-VL.
- **Kijai**, **DeepBeepMeep**, **ddalcu** — the ComfyUI conversions, pruned checkpoints and MLX quants that the
  wider H3 ecosystem (and Phosphene's install) relies on.
- **FFmpeg**.

## Test prompts

"BLACK MIRROR: FINAL REFLECTION" (`prompts/black-mirror-scene.txt`) is by **@itxabdullaa** on X (onlyprompts.ai X
collection).

Pack: drowzeys / keys.
