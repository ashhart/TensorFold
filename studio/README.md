# keys+Wasserman Slate AI Studio, powered by TensorFold, built with integrated Fish Audio Drama 3 (Mac + DGX Spark)

**An AI film studio on your own machine: plan a film the way a crew does in the Wasserman Slate AI Studio, and render
it here. Qwen-Image-2.1-Turbo images and MiniMax H3 / FastH3 video with sound, on TensorFold: Apple-silicon Macs
(Metal) and the NVIDIA DGX Spark (CUDA).**

[![Wasserman Slate AI Studio, powered by TensorFold, built with integrated Fish Audio Drama 3: an AI film studio for Mac and NVIDIA DGX Spark](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/banner.jpg?raw=true)](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3)

*Banner art made with this studio's own Qwen-Image-2.1 step (1664x928, seed 25); the lettering was set afterwards.*

*An app built on TensorFold, contributed to the TensorFold repository as `studio/`.*

## Big credit: Sam Wasserman (wassermanproductions.com)

**Sam Wasserman shaped the production side of this studio: the AI film workflow and the ideas behind it.** Slate, his
prompt studio for AI filmmaking, is the workflow inside the Wasserman Slate AI Studio tab. The Wasserman Sharp Spec,
the QA gate that follows it, and the 80s Horror LoRA that opens the Keys and Wasserman LoRA Hub are his too. The Motion Previs tab's analysis is a
Python port of his [Motion Previs Studio](https://github.com/wassermanproductions/motion-previs-studio). The aim
we share is the best AI film studio there is, and this pack is where his planning tools meet a render engine. See
[CREDITS.md](CREDITS.md).

**Thank you to everyone this stands on:** the Qwen team at Alibaba (Qwen-Image-2.1, Qwen3-VL), the MiniMax team
(MiniMax H3), Ash Hart and the TensorFold contributors, antirez (h3.c), RobZombAI (H3MLX), mrbizarro (minimax-h3-mlx /
Phosphene), Filip Strand and the mflux contributors, Viggle (the image turbo adapter up to 2.2), speach1sdef178 (the 2x video
decoder), LightX2V (the video Turbo adapter), NVIDIA Research (Sol-Engine, Sol-Attn, Sol-H3), FastVideo (FastH3), and Apple's MLX team and every MLX
contributor, and Fish Audio (Drama 3, text to speech). This pack is their work, ported, pinned and measured. See
[CREDITS.md](CREDITS.md). Built with Qwen. **Built with Fish Audio.**

**2.7.4** · Apple-silicon Macs (measured on a Mac Studio M5 Ultra 256 GB) and the NVIDIA DGX Spark ·
[TensorFold](https://github.com/ashhart/TensorFold) **0.6.5** + two families, and TensorFold **1.0**'s native Zig
runtime for FastH3 on Metal and on CUDA:
**Qwen-Image-2.1-Turbo** (text to image) and **MiniMax H3** (video with sound) · int8 kernels on the M5 tensor units
and on the GB10's tensor cores · a 2x video decoder for 2K finals on the Mac

## Speed: the first version on each machine, and 2.3

One run each unless a range is given; 5 second clips are 124 frames. Mac: Mac Studio M5 Ultra, 256 GB. Spark: one
NVIDIA DGX Spark (GB10), models already loaded.

| 5 second clip, 864x480 | Mac, first version (1.0, 2026-10-04) | Mac, 2.3 | DGX Spark, first build (ComfyUI's own blocks, bf16) | DGX Spark, 2.3 |
|---|---|---|---|---|
| What makes the video | MiniMax H3 + Turbo adapter on MLX, 3 passes | FastH3, 8 passes, TensorFold's Zig + Metal engine | FastH3, 8 passes, ComfyUI | FastH3, 8 passes, TensorFold's CUDA engine in int8 |
| Image, then video from it | 74 s (103 s with the sound made again, the later default) | **79 s** (image 6 s, video 73 s) | 155 s | **115 s** (measured before the faster decoder below, which takes about 10 s more off) |
| Text straight to video | not offered | **66 s** | 118 to 121 s | **73 s** |
| One denoising pass | 9.5 s | **5.3 s** | 10.5 s | **7.1 s** |
| Video decode | not timed apart | 13 s | 26 s | **14 s** |

| Other sizes and lengths | Mac, first version | Mac, 2.3 | DGX Spark, first build | DGX Spark, 2.3 |
|---|---|---|---|---|
| 864x480, 4 passes, text to video | | 43 s | 71 s | 45 s (from the pass time) |
| 1280x720, 5 s, text to video | | 199 to 217 s | not measured | **183 s** |
| 1344x768, 5 s, text to video | | 175 s | 289 s | **about 195 s** |
| 1344x768, 8 s, image then video | 364 s (709 s with the later default) | not re-measured | | |
| 1280x720, 10 s (243 frames), text to video | | 378 s | not measured | 444 s |
| One 15 s part of a chained 864x480 clip | | about 249 s | not measured | about 304 s |
| One image, 1344x768 | 9.2 s (Viggle adapter, 6 steps) | **8.9 s** (Qwen-Image-2.1-Turbo, 8 steps; peak 15 GiB against 28) | 18 to 21 s with the model load | **9.5 s** |

- The Mac's first version and 2.3 use different video models, so the 74 s and 79 s rows are not the same work: the
  first is 3 passes of MiniMax H3 with its Turbo adapter, whose sound was judged poor by ear; 2.3 is 8 passes of
  FastH3 with its own sound. Per pass the Mac went from 9.5 s to 5.3 s.
- On the Spark the model is the same in both columns. The gain is the engine (10.5 s a pass to 7.1 s) and the 8-bit
  video decoder (26 s to 14 s).
- Both engines are TensorFold H3 families on the TensorFold 1.0.4 base, from one commit of one branch. 1.0.3 and 1.0.4
  are language-model releases, so the move changed nothing here: on each machine the same seed gives a bit-identical
  clip to the 1.0.2-based engine, at the same speed.

## Which Mac runs what

Every row is Studio **2.3**. "Default" means images (Qwen-Image-2.1-Turbo) and FastH3 video with sound, the web app and
the one-click app. Only the M5 Ultra with 256 GB has actually run it; every other row follows from memory measured
on that machine and is marked untested.

| Chip | Memory | What runs | Install command | Video engine it uses | Peak memory | Status |
|---|---|---|---|---|---|---|
| M5 family | 64 GB | Default | `bash oneshot-setup.sh --app` | FastH3, native Zig engine | 32 GiB | Untested |
| M5 family | 96 GB | Default | `bash oneshot-setup.sh --app` | FastH3, native Zig engine | 32 GiB | Untested |
| M5 family | 128 GB | Default, and MiniMax H3 Turbo if wanted | `bash oneshot-setup.sh --turbo --app` | FastH3 native; Turbo on the MLX engine with int8 kernels | 32 GiB; 103 GiB for Turbo | Untested |
| M5 family | 256 GB or more | Everything | `bash oneshot-setup.sh --turbo --app` | The same | The same | **Tested** (M5 Ultra, 256 GB) |
| M1 to M4 | 64 GB | Default, slowly | `bash oneshot-setup.sh --app` (add `Q8_ONLY=1` in front to save 65 GB of disk) | FastH3, MLX engine, 8-bit weights | 37 GiB at 480p, 40 GiB at 720p | Untested; emulated on the M5 |
| M1 to M4 | 96 GB | Default, slowly | `bash oneshot-setup.sh --app` | FastH3, MLX engine, 8-bit weights | 37 to 40 GiB | Untested; emulated on the M5 |
| M1 to M4 | 128 GB | Default, slowly. Turbo installs but is not recommended | `bash oneshot-setup.sh --app` | FastH3, MLX engine, 8-bit weights | 37 to 40 GiB | Untested; Turbo's memory without the int8 kernels is not measured |
| M1 to M4 | 256 GB or more | Default, slowly; Turbo should fit | `bash oneshot-setup.sh --turbo --app` | FastH3 MLX 8-bit; Turbo on the MLX engine in bfloat16 | 37 to 40 GiB; Turbo not measured | Untested |
| Any | 48 GB | Images only | `bash oneshot-setup.sh --image-only` | none | 28 GiB | Untested below 256 GB |

How to configure it:

- **Nothing to set on a normal install.** The setup reads the chip and the memory and picks the engine: the native
  engine on an M5, the 8-bit weights on an M1 to M4 (it converts them once, about 37 GB). It stops with a message if
  the Mac has too little memory for what was asked.
- **Disk:** 175 GB for the default on an M5; 212 GB on an M1 to M4 (147 GB with `Q8_ONLY=1`); 64 GB more with
  `--turbo`; 33 GB for `--image-only`.
- **Speed on an M1 to M4:** the 8-bit path took 31 s a pass at 864x480 on the M5 Ultra's GPU, against 5.3 s for
  the native engine there. An older GPU is slower again. Start with 480p, 5 seconds and 4 passes
  (`STEPS=4 bash scripts/fast.sh "a prompt" out.mp4`, or the 4 passes card in the app).
- **Overrides:** `FASTH3_Q8=0` runs an M1 to M4 with the uncompressed weights (64 GiB peak: a 96 GB Mac or larger;
  no faster). `FASTH3_ENGINE=mlx` forces the MLX engine on an M5. `ENGINE=h3` asks for Turbo where it is installed.

| Studio version | Smallest Mac for video |
|---|---|
| 2.0 and earlier | 128 GB (M5 for the fast path) |
| 2.1 | 64 GB on an M5; 96 GB on an M1 to M4 |
| 2.2 to 2.5 | 64 GB on any Apple silicon |

On a **DGX Spark** the setup asks for 100 GB of memory for video (a Spark has 128 GB; about 70 GB was in use during a
clip). No NVIDIA machine with 64 GB has been tried.

Measured memory for each kind of job is under [Minimum requirements](#minimum-requirements).

## DGX Spark (CUDA) build

The same Studio (web app, scripts, one-shot install) also runs on an **NVIDIA DGX Spark** (GB10, 128 GB unified
memory, CUDA 13). FastH3's transformer runs there on **TensorFold's H3 family for CUDA**: the Mac engine's design
(int8 projections, tile-routed int8 attention with FastH3's own sparse selection) on the GB10's tensor cores, written
in Zig over the CUDA driver. **ComfyUI**, pinned, hosts the rest: the text encoder, the sampler, the decoders and
Qwen-Image-2.1. The header of the web app says which build you are in: a green `DGX SPARK · NVIDIA GB10 · H3-CUDA`
badge, or a purple `MAC · APPLE SILICON MX · H3-METAL`.

**Thank you** to Ash Hart for TensorFold, FastVideo (Hao AI Lab) for FastH3 and its ComfyUI release, MiniMax for H3,
the Qwen team for Qwen-Image-2.1-Turbo, and the ComfyUI team for the runtime this build stands on.

```bash
git clone https://github.com/ashhart/TensorFold.git && cd TensorFold/studio
sudo apt install -y git ffmpeg
bash oneshot-setup.sh            # the same command as on a Mac: on Linux it runs oneshot-setup-spark.sh
                                 # 99 GB of models, the engine build, a test image and a 5 second test clip
bash scripts/app.sh              # http://127.0.0.1:7870
```

ffmpeg is part of the install, and the installer stops without it. A Spark that was set up some other way (a copy of
another Spark's folders, say) may lack it: the app then starts, says so in its log and in the page's header, and
refuses the jobs that need it when they are queued. The 720p preset is not one of them any more: it generates
1280x736 and crops to 1280x720, and the crop is now made inside the ComfyUI workflow, before the clip is encoded
(one encode instead of two); in 2.6.5 it was an ffmpeg pass after the render, and a Spark without ffmpeg lost the
finished clip there. Without ffmpeg a clip also keeps ComfyUI's own sound track instead of the limited 256 kb/s one,
and the render says so.

#### Upgrade from 2.5 or 2.6 (DGX Spark)

```bash
cd TensorFold/studio && git pull
bash oneshot-setup.sh --upgrade  # on Linux: oneshot-setup-spark.sh --upgrade
```

`--upgrade` moves an existing install to the version of the clone: the environment's packages, ComfyUI at its pinned
commit, the engine (rebuilt only when its pinned commit changed) and the Motion Previs environment, which it adds
when it is missing. It **fetches no weights** (a model that is not on disk is named and left out; an image-only
install stays image-only), does not touch `~/TensorFoldStudio` (clips, projects, LoRAs, settings), prints the version
before and after, and ends with `--verify`. Running it a second time changes nothing. Restart the app afterwards
(`bash scripts/app.sh`) so it serves the new pages. Run it when no clip is rendering.


The installer takes the prebuilt engine from the carrier image when Docker is there, and otherwise builds it from
source (`spark/build-engine.sh`: it fetches Zig 0.17.0 and the engine's source,
[drowzeys/TensorFold `h3-1.0.4`](https://github.com/drowzeys/TensorFold/tree/h3-1.0.4), and needs the CUDA toolkit's
`nvcc`, 12.9 or newer). Without it the Studio still works on ComfyUI's own blocks, slower.

Measured on one DGX Spark (2026-10-09, one run each; 124 frames, text to video, models already loaded):

| Size | Passes | TensorFold engine (int8) | A pass | ComfyUI's own blocks (bf16) | A pass |
|---|---|---|---|---|---|
| 864x480 | 8 | **73 s** | 7.1 s | 118 to 121 s | 10.5 s |
| 864x480 | 4 | **45 s** (from the pass time) | | 71 s | |
| 864x480 | 20 | **158 s** (from the pass time) | | 244 s (from the pass time) | |
| 1280x720 (rendered 1280x736) | 8 | **183 s** | 18 s | not measured | |
| 1344x768 | 8 | **about 195 s** | 20.5 s | 289 s | 29.4 s |

The engine's column includes ComfyUI's 8-bit video decoder, the default here: it decodes 124 frames of 864x480 in
14 s where the fp16 one takes 26 s (29 s against 42 s at 1344x768), and the picture matches the fp16 decode to 42 to
44 dB. `SPARK_VAE=minimax_h3_video_vae_fp16.safetensors` selects the fp16 decoder if you download it.

| Also | Time |
|---|---|
| A clip from an image, 864x480, 8 passes | 85 s |
| First clip after a start, 864x480, 8 passes | about 115 s (ComfyUI's blocks: 144 to 156 s) |
| Qwen-Image-2.1-Turbo image, 1344x768 | 9.5 s (20 s with the model load) |
| Three scout images, 864x480 | 23 s |

Lowest free system memory seen with the engine was 49 GB of 121 (35 GB on ComfyUI's blocks): the transformer is held
in int8, 21 GB instead of 42. The first use converts the checkpoint on the GPU (about 90 s) and saves the int8 copy
beside it (`….safetensors.tf-int8`, 21 GB of disk); later starts read that in a few seconds.

How close the engine is: on the same input with dense attention, the stream after all 50 blocks has cosine 0.998
against ComfyUI's bf16 blocks. With sparse attention the same seed gives the same scene as ComfyUI's sparse path,
and a repeat of a seed gives a bit-identical clip.

What is different from the Mac build:

- **Slower than an M5 Ultra**, about 1.35x a pass at 480p (7.2 s against 5.3 s).
- **FastH3 only.** No MiniMax H3 Turbo, and no 2x decoder, so the sizes are 864x480, 1280x720 and 1344x768.
- **Sparse attention at every size** (FastH3 was trained with it: a video tile keeps the best 20% of video tiles).
  `SPARK_SPARSITY=0` renders with dense attention, slower.
- **Clips from an image want the app's prompt.** The web app writes MiniMax H3's first-frame lead line and structured
  prompt; a bare one-line prompt with `FIRST_FRAME` gave a jump cut in a test (on ComfyUI's own sparse path too).
- `SPARK_ENGINE=comfy` renders on ComfyUI's own blocks in bf16 (dense attention under about 720p, its block-sparse
  kernel above). The 8-bit checkpoint through ComfyUI is no faster there (13.6 to 14 s a pass), and FastVideo's
  4-bit Spark recipe is not used: on the test machine it gave flying debris and results that changed between runs.
- **Why int8.** The GB10's tensor cores run int8 products at 245 TOPS and bf16 ones at 122 (measured on registers).
  PyTorch's stock int8 routine reaches about 52; the engine's own kernel 150 to 175 on FastH3's shapes.
- ComfyUI keeps running with the models loaded after a render (`bash spark/comfy.sh stop` frees the memory). It runs
  at low priority with a memory guard beside it, because a Spark that runs out can become unreachable: see
  *The memory guard* below.

### More than one DGX Spark: the Sparks setting

A job can use more than one DGX Spark. **Sparks**, a number from 1 to 8, is how many machines a job may use, the one
the app runs on counted as one. It is at the top of the Queue on every tab, in the last step of Create, on the
Production board and in a Slate shot; through the API it is `"sparks": 5` beside `kind` and `params`. The app shows
how many Sparks are there and able and uses that many: choose 8 with five machines and it says *5 of 8 available*
and uses five.

- **Several clips** (jobs in the queue, a Production board's segments, a Slate scene's shots) render one clip per
  Spark, side by side. Each clip comes back into this Spark's Library, project or film at the path it would have
  had here, with its seed, its settings and the Spark that rendered it; the QA gate and the shot check run here.
- **One clip** renders on one Spark. So do a long video (a chain of clips), pictures, edits and Library actions, on
  this Spark; the page says which and why before the job is queued.
- **The Sparks tab** is the list of the other Sparks: add one by the address you would type after `ssh`, and
  **Update and start** sends it this Spark's Studio code (rsync, as you; models are not sent) and starts its app.
  Each row shows whether the Spark answers, its Studio's version, memory and disk free, what it can render
  (FastH3 clips, Ref2VA, the image model, LoRAs, ffmpeg) and whether it is busy. It needs a key login to each Spark
  and the Studio installed there once; nothing runs as root.
- **Safe by construction.** A worker's app listens only on its own `127.0.0.1`; this app reaches it through an ssh
  tunnel it opens and closes itself. One job runs on a Spark at a time. Every clip passes the worker's own memory
  guard before it is sent, and a clip with a LoRA goes only to a Spark that has the LoRA and whose guard accepts it.
  A worker that stops answering for 60 seconds is marked down and its clip is rendered elsewhere; cancel reaches
  the worker.

Measured through the app's own API on 2026-10-13, eight 56-frame 864x480 FastH3 clips, models loaded, one run each:

| Sparks | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Wall time, 8 clips | 263 s | 130 s | 98 s | 69 s | 65 s |
| Against one Spark | 1.00x | 2.03x | 2.70x | 3.81x | 4.05x |

Five were barely better than four for eight clips (two rounds of clips either way); the fifth pays from ten clips.
Six to eight are not measured: five machines exist here. One clip split across Sparks is measured outside the app
(1.40x, 1.73x and 1.77x on two, three and four) and is not built in. The whole of it, with what is not built:
[docs/multi-spark.md](docs/multi-spark.md).

### The memory guard (DGX Spark)

On a Spark the GPU and the system share one pool of memory. On 2026-10-13 one clip with a LoRA at 1376x768 took a
worker Spark down: the app accepted it, and the server's check (once a second, against 4 GB) did not act. The guard
in `spark/memory_guard.py` replaces that check, in three layers that every way of starting a render goes through
(the app, `scripts/video.sh`, `spark_generate.py`, a worker's copy of this folder):

1. **Before a clip is queued, and again before anything is loaded**, its peak is estimated from measured peaks. A
   clip that would leave the machine under its floor is refused, with the reason and the nearest size and length
   that fit. In the app the settings step shows the same line before you queue; a clip that fits shows "about N GB of
   this machine's memory".
2. **While the server runs**, the guard reads the machine's memory twenty times a second. It stops the server when
   available memory is under the floor (a tenth of the machine: 12.4 GB; `SPARK_MEMORY_FLOOR_MB`), or is falling so
   fast that it would be under a third of the floor within a second, or when the kernel reports work stalled on
   memory. The job fails with that reason. It also gives the model files' page cache back when little memory is
   free (`SPARK_MEMORY_FREE_MB`, 8.3 GB): the GPU driver takes its memory from what is free, not from what is
   available.
3. The server is marked as the first process the kernel ends if memory runs out all the same.

| Measured on a 124.5 GB Spark, nothing else running (118.7 GB available) | Rows a pass | Taken at the lowest point |
|---|---|---|
| TensorFold int8 engine, 864x480, 124 frames | 15.4K | 57.6 GB |
| TensorFold int8 engine, 1376x768, 124 frames | 39.2K | 62.2 GB (62.1 to 64.2 in a production run) |
| TensorFold int8 engine, 1376x768, 192 and 243 frames (that run, sampled every 11 s) | 59.9K, 75.3K | 67.1 GB, 69.0 to 70.9 GB |
| With a LoRA merged into the int8 engine (2.7.1), 864x480, 124 frames | 15.4K | 57.6 GB |
| With a LoRA merged into the int8 engine, 1280x736 and 1376x768, 124 frames | 35.0K, 39.2K | 61.4 GB, 62.3 GB |
| With a LoRA merged into the int8 engine, 1376x768, 192 and 243 frames | 59.9K, 75.3K | 66.5 GB, 69.7 GB |
| With a LoRA on ComfyUI's own blocks (bfloat16; before 2.7.1), 864x480, 56 and 124 frames | 7.3K, 15.4K | 75.4 GB, 75.9 GB |
| The same, 1280x736, 22, 56, 90, 107 and 124 frames | 7.4K to 35.0K | 74.8, 75.8 to 76.2, 76.2, 76.5, 76.6 GB |

The estimate is a straight line through those points for each route plus 6 GB (no point is more than 1.7 GB from its
line; most are one run). Three things it does not know, said plainly:

- **A clip with a LoRA is refused above the largest one measured on its route.** Since 2.7.1 a LoRA on FastH3 is
  merged into TensorFold's int8 engine and takes what a clip without one takes (the table: within 0.6 GB at every
  size), measured from 864x480 up to 1376x768 at 243 frames (75.3K rows), as far as the engine itself was measured:
  every size the app offers, up to 10 s at the Sharp Spec size. On ComfyUI's own bfloat16 blocks (a Spark whose engine
  predates 2.7.1, or `--engine comfy`) it stays refused above 1280x736 at 124 frames (35.0K rows), whatever the line
  says. By that line the clip that took the Spark down needs 77 GB and leaves 41, and that is what the machine showed
  for a minute before it was lost, so its end is not explained by how much memory the clip takes and the size is not
  offered there. Its logs were never read; what is known is in the next paragraph.
- Reference clips (Ref2VA), MiniMax H3's own transformer and a Spark without the engine were not measured here; they
  are estimated on the bfloat16 LoRA line with their references' rows added.
- Pictures, edits and Library actions are not estimated: the running guard covers them.

What was found about the old check: memory falls far faster than once a second can follow (7.2 GB a second while
ComfyUI loads a model from the page cache, 13 GB in half a second while TensorFold's engine loads its weights), and
"available" counts the page cache. With a LoRA, ComfyUI loads until nothing is free: free memory sat at 1.2 to 2.5 GB
through every such render while 43 GB read as available, and the kernel logged the GPU driver being refused memory
(`NVRM ... NV_ERR_NO_MEMORY`) at 2 GB free and 54 GB available. With the guard beside the server the same renders
kept 8 GB or more free, took the same time (146 s against 145 s), and the driver logged nothing. Proof of the two
refusals, without a machine near its limit: the app and `scripts/video.sh` both refuse the lost clip before anything
loads, and a server started with `SPARK_MEMORY_FLOOR_MB=70000` was stopped during a small clip's load at 69.8 GB
available (lowest point 69.77 GB while memory fell at 28 GB a second; gone in 0.5 s, the job failed with the reason).
`python3 spark/memory_guard.py check --width 1376 --height 768 --frames 124 --lora` prints a verdict.

Longer clips on the engine: 10 seconds at 1280x720 (243 frames) took 444 s, a pass 43.7 s, against 378 s on an M5 Ultra
for the same prompt; a 30 second chained clip at 864x480 (two 15 second parts) took 611 s.

Not tested: clips longer than 15 seconds in one piece, chains longer than 30 seconds, a download of the models from nothing
(the files were already on the test machine), any GPU other than the GB10, and two Sparks working on one clip.

## What changed

The release history (1.0 to 2.8.0) is kept in the [Studio repository's README](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/README.md#what-changed).

## Install, and the one-click app

### Minimum requirements

| What you want to run | Memory | Disk | Chip |
|---|---|---|---|
| Images only (`--image-only`) | **48 GB** | 33 GB | Any Apple silicon; M5 for the int8 kernels |
| The default: images and FastH3 video, the web app | **64 GB** | 175 GB | M5 family: the native engine, fastest |
| The default on an M1 to M4 | **64 GB** | 212 GB | FastH3 on the MLX engine with 8-bit weights, much slower |
| With `--turbo`: MiniMax H3 Turbo and its 20-step modes as well | **128 GB** | 240 GB | M5 family for the fast path |
| With `--ref2va`: reference pictures to video (MiniMax H3 Ref2VA) | **128 GB** | 64 GB more | M5 family for the fast path |

The setup enforces the memory figures. All of this was measured on one machine, a Mac Studio M5 Ultra with 256 GB:
the figures below are what each stage of a job used there (`/usr/bin/time -l`, peak memory footprint), not a run
on a smaller Mac.

| FastH3 job, native engine (M5 Ultra, 2026-10-08) | Prompt export | Passes | Decode | **Peak** |
|---|---|---|---|---|
| 864x480, 5 s, from text | 27.7 | 23.0 | 14.0 | **28 GiB** |
| 864x480, 5 s, from an image | 27.9 | 23.1 | 14.0 | **28 GiB** |
| 864x480, 5 s, 20 passes | 28.0 | 23.2 | 13.9 | **28 GiB** |
| 864x480, 15 s | 27.8 | 28.2 | 16.9 | **28 GiB** |
| 1280x720, 5 s | 27.8 | 26.4 | 15.8 | **28 GiB** |
| 1280x720, 10 s | 27.9 | 32.1 | 19.6 | **32 GiB** |
| 1280x720, 5 s, 2x decoder to 2560x1440 | 27.8 | 26.4 | 32.1 | **32 GiB** |

The three stages run one after another, so a job's peak is its largest stage. An image job peaks at 28 GiB
(1344x768). Before 2.1 the same FastH3 jobs peaked at 48 to 69 GiB. MiniMax H3 Turbo reaches 103 GiB while its
adapter is merged, which is why `--turbo` asks for 128 GB.

A chip without tensor units (M1 to M4) cannot use the int8 kernels or the native engine. Emulated on the M5 Ultra by
switching both off (MLX engine, FastVideo's SIMD-group routing, float decoder), 8 passes:

| FastH3 on the MLX engine, as an M1 to M4 runs it | Peak | A pass (on the M5 Ultra) |
|---|---|---|
| bfloat16 weights, 864x480, 5 s (before 2.2) | 64.0 GiB | 29.6 s |
| **8-bit weights**, 864x480, 5 s, from text | **37.1 GiB** | 31.2 s |
| **8-bit weights**, 864x480, 5 s, from an image | 34.3 GiB | 39.6 s |
| **8-bit weights**, 1280x720, 5 s | 39.6 GiB | 107 s |

The last two rows were run with a smaller buffer cache (4 GiB instead of the 8 GiB that ships), which is slower by
about a tenth and peaks about 3 GiB lower. Against the bfloat16 render of the same seed the 8-bit clip shows the
same scene, person and framing with different fine detail (18 dB between the two: few-step sampling amplifies any
small change, as it does between bfloat16 and int8 on the M5).

| Machine | Status |
|---|---|
| Mac Studio M5 Ultra, 256 GB, macOS 27 | **Tested.** Every number in this README is from this machine. |
| M5 Max or M5 Pro, 64 GB or more | Untested. Meets the memory check for the default install with half the memory to spare; less GPU, so expect longer times. |
| M5 family, 48 GB | Untested. Images only (`--image-only`); the setup refuses video under 64 GB, though the measured peak is 32 GiB. |
| M1 to M4, 64 GB or more | Untested on real hardware; emulated on the M5 Ultra. No tensor units: FastH3 runs on the MLX engine with 8-bit weights, about six times slower a pass than the native engine even on an M5 Ultra's GPU, and slower again on an older one. |
| macOS 26 | Untested. The kernels need Metal 4, so it is the likely minimum. |

Installed size: about 175 GB, nearly all model weights (FastH3 65 GB, MiniMax H3's text encoder and decoders 72 GB,
Qwen-Image-2.1-Turbo 31 GB, the 2x decoder 6 GB, Python environment 1.3 GB). `--turbo` adds 64 GB
(MiniMax H3's transformer 62 GB and the Turbo adapter 2 GB). The Studio's own code and the native engine are a few
megabytes. On an M1 to M4 the setup also writes the 8-bit FastH3 weights (37 GB); `Q8_ONLY=1` then deletes the
65 GB they were made from. If you run it on another machine, a report of the chip, memory and times is welcome as an issue.

```bash
brew install python@3.11 uv ffmpeg
git clone https://github.com/ashhart/TensorFold.git
cd TensorFold/studio
bash oneshot-setup.sh --app             # engine, image model, FastH3, 2x decoder, the app; ends with a test clip
bash oneshot-setup.sh --turbo --app     # the same plus MiniMax H3 Turbo (64 GB more on disk, 128 GB of memory)
bash oneshot-setup.sh --ref2va --app    # the same plus reference pictures to video (64 GB more on disk, 128 GB of memory)
```

#### Upgrade from 2.5 or 2.6 (Mac)

```bash
cd TensorFold/studio && git pull
bash oneshot-setup.sh --upgrade         # add --app to rebuild the double-clickable app as well
```

`--upgrade` moves an existing install to the version of the clone: it replaces the TensorFold engine in the
install's own environment when it is older than this version needs (2.6.1: the build with reference videos and
audio), copies the render scripts, and adds the Motion Previs environment when it is missing (0.5 GB). It **fetches
no weights**: a model that is not on disk is named and left out, and an image-only install stays image-only. It does
not touch `~/TensorFoldStudio` (clips, projects, LoRAs, settings), prints the version before and after, and ends with
`--verify`. Running it a second time changes nothing. Quit and reopen the app afterwards. A plain
`bash oneshot-setup.sh` after `git pull` upgrades the engine too, and also fetches any model the install lacks.


**Step-by-step tutorial with screenshots: [TUTORIAL.md](TUTORIAL.md).**

Then **double-click Wasserman Slate AI Studio** on the Desktop or in `~/Applications` (drag it to the Dock if you like). `bash scripts/make-app.sh` rebuilds the app on its own. It starts the studio
if it is not running and opens it in your browser; double-click again to come back to it. Without the app:
`bash scripts/app.sh`, then open http://127.0.0.1:7870.

**Or hand it to your coding agent.** Paste this:

> Clone https://github.com/ashhart/TensorFold and follow studio/AGENTS.md section "Install for a
> person and build the one-click app". Tell me before any large download starts and when the app is ready.

`--turbo` adds MiniMax H3's own transformer and its Turbo adapter, and with them a second engine card in the app;
`--image-only` installs the image model alone (33 GB). An install made before 2.1 keeps Turbo. The app keeps the clone where it is: rebuild it with `make-app.sh` if you move the folder. To
stop the server: `"$HOME/Applications/Wasserman Slate AI Studio.app/Contents/MacOS/TensorFoldStudio" stop`. There is **no
login**, and by default it listens on this Mac only; `HOST=0.0.0.0 bash scripts/make-app.sh` opens it to your
network, which you should do only on a network you trust.

## Using the studio

![Create: text to video with FastH3](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_create_fasth3.png?raw=true)

Step 1 of **Create** is where every choice lives. Pick one card in each row; the bar under them shows what will run
and roughly how long it takes.

| Choice | What it does |
|---|---|
| **The prompt goes to** | *Video with sound*, or *Image only*: Qwen-Image-2.1 makes the pictures and stops there. |
| **The clip starts from** | *Text only*; *a Qwen scout image* (step 2 makes several, you pick one); *your own image* (step 2 uploads it); or, with `--ref2va` installed, *reference pictures* (step 2 takes up to nine: who or what must look the same, in a scene the prompt describes; on a Mac whose engine packs them, also up to three reference videos and three audio clips: see [Reference videos and reference audio](#reference-videos-and-reference-audio-ref2va-mac-only-for-now)). |
| **Video engine** (shown only with `--turbo` installed) | *FastH3*: fastest; text to video runs on the native Zig engine, and the bar under the cards says which engine a clip will use. *MiniMax H3*: the Turbo adapter with the sound made again by the base model, or the full 20 steps. |
| **Passes** (FastH3) | 4 (fastest, a little softer), 8 (what it was trained for), 20 (for prompts with several actions; slowest). |
| **Quality** (MiniMax H3, with `--turbo`) | Turbo 5 + base sound (the standard), Turbo 3, 20 steps with the fast recipe, or plain 20 steps. |
| **Resolution** | 480p (864x480), 720p (1280x720), or the fixed outputs Draft, 2K 2048x1152 and Native 1344x768. |
| **2x upscale** | For 480p and 720p: the 2x decoder turns 480p into 1728x960 and 720p into 2560x1440, for about 10 s more. |
| **Length** | 5, 8, 10 or 15 s as one clip, or *Another length…* for any length the model takes (below); 30 s to 30 min as a chain of clips (further below). |
| **Seed** | Same seed and settings give the same clip. |

Then write the scene, what happens to the characters, and optionally the camera move. Steps 3 to 5: add spoken or sung lines and
the soundscape, review the exact prompt the model will receive (edit it freely), and queue the render. The **Queue**
on the right shows progress by pass, a time estimate, the log, and a cancel button.

### Clip lengths: any length the model takes

MiniMax H3 and FastH3 make clips of `17n + 5` frames at 24 frames a second and nothing in between. The four usual
lengths are four of those counts, and their names are rounded: "10 s" is 243 frames, 10.125 s. *Another length…*
in the Length box opens a slider over every count, and the page writes the frames and the exact seconds beside it.

| | Frames | Seconds |
|---|---|---|
| Shortest | 22 | 0.917 |
| The usual four | 124, 192, 243, 362 | 5.167, 8, 10.125, 15.083 |
| All 21 | 22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345, 362 | frames / 24 |

Through the API a job carries `frames` (one of the counts; another number is refused with the two nearest) or
`seconds` (any number from 0.9 to 15.1, brought to the nearest count: `seconds: 6` is 141 frames, 5.875 s). The end
moment in a last-frame prompt is the clip's true length. Production segments and Slate shots take any length the
same way. The longest clip stays the model's 15 seconds on every size of both builds. On a DGX Spark that was
measured at the largest canvas, 1376x768 with a first and a last frame: 362 frames rendered with 29.7 GB of the
machine's 121 GB still free at the lowest point (one run, TensorFold's int8 engine, 4 passes; the memory does not
depend on the passes). A clip with a LoRA renders on the same engine with the LoRA merged into it (2.7.1) and is
taken up to 1376x768 at 243 frames, the largest it was measured at, and refused above that with the reason and what
fits (*The memory guard*, in the DGX Spark section). On the Mac
22, 141 and 345 frames were rendered at 864x480 through the app (26 s, 72 s and 226 s a job on an M5 Ultra, native
engine, 8 passes); its limits are the ones it had.

### What the app tells you about a prompt and a clip

**Words that forbid.** The video model is run without a negative prompt, so every word in a prompt is something to
show. In the production run that led to this check, a clip told "No photographs or portraits appear anywhere" hung
the wall with photographs, twice, and came out clean once every mention of photographs was gone. Step 4 and step 5 of
*Create* (and a Slate shot's preflight) now list what a prompt forbids by name ("no X", "without X", "not X",
"never X") with the reason, in the box it was written in. Nothing is taken out or rewritten: the prompt goes as you
wrote it, and the job in the queue repeats the list. The Studio's own sentences forbid nothing by name any more: a clip without spoken
lines used to be told "Nobody speaks." and "No voices, no speech.", and a locked camera in Slate "the camera does not
move". They now read "Mouths stay closed." (only where the shot has people: Slate's cast, or a person in the words),
"Only ambient sound is heard." and "Static shot on a locked, still camera.", and a prompt the app writes passes its
own check with nothing passed over. One comparison was made, not a study: the same two-person shot at 864x480, 5 s,
seeds 42 and 43, in the old words and the new; in all four clips the speech-to-text model found no speech, and the
mouths stayed closed in the frames looked at. A prompt that still carries the old sentences is not flagged for them.
No route of the app has a negative prompt field: FastH3, MiniMax H3 Turbo and Qwen-Image-2.1-Turbo all run one forward a pass.
`POST /api/prompt/check {"prompt": "..."}` gives the same list for a prompt written by hand.

**The shot check.** After a single clip is rendered the app looks at its frames (`app/shot.py`) and writes the
verdict beside the clip in the queue:

- *First frame, last frame.* A keyframe is something the model reads, not a frame written into the clip, and
  almost always the clip opens and ends on it. In the same run 63 of 64 keyframes were held (a mean difference of
  1.4 to 3.9 grey levels of 255 between the frame and the picture); one clip opened on another picture (74.4), so its
  join with the clip before jumped. Rendered again on another Spark at the same seed it came out the same, frame for
  frame: it is what the model does at that seed. The check compares frame 0 with the first picture and the last frame
  with the last, and says "not held" over 12 grey levels.
- *A cut inside the shot.* FastH3 sometimes cuts to a close-up that is in neither keyframe: 12 cuts in 9 of that
  run's 32 clips. The check finds a frame that is unlike the one before it (a correlation under 0.70) and changes at
  least 1.75 times as much as the frames around it; it found all 12 and nothing else in the 7081 frame pairs. A
  prompt that writes a `[Shot 2]` asks for a cut and is not checked for one. A cross-fade is not a cut.
  No wording or setting we tried stops it. One of those clips (a couple dancing in a hall, small in a wide frame,
  8 seconds at 1280x736) was rendered again at its own seed eight ways: as it was; with a camera line in plain
  positive words ("one continuous wide shot from a locked-off camera…"); with the run's "No cut, no close-up" line;
  on ComfyUI's own bfloat16 blocks in place of TensorFold's int8 engine; with dense attention; at 20 passes; at 5
  seconds; and with the first frame alone. All eight left the wide shot for close-ups of the two faces. Five did it
  with a hard cut, which the check reports. Three (the "no close-up" line, the first frame alone, 20 passes) did it
  with a dissolve through a double exposure, **which the check does not see**: half a second of a dissolve measures
  like half a second of honest fast movement in the run's other clips, so there is no threshold to set. The plain
  positive camera line was also tried on the two other dining-hall clips at their seeds: both cut, twice each. So
  look at a clip yourself, whatever the check says: it reports hard cuts and nothing subtler.
- *The retake.* Tick "If the shot check fails, render it once more at the next seed" in step 5 (`retake: true` in a
  job), and a clip that fails is rendered a second time at seed + 1, like the Sharp Spec QA gate; the failed take is
  kept in the `qa` folder. Without the tick the verdict comes with a button, *Render it again at seed N*. The
  thresholds come from one run on one model at one size; `shot_check: false` in a job turns the check off.

- *Where to trim.* FastH3 also opens some clips on an establishing shot nobody asked for and then cuts to the shot (4
  of 36 clips in a later run, the cut at frames 10 to 28). The check's report carries `window`: the longest stretch of
  the clip without a cut, as frames (`start`, `end`) and as the timeline's `in` and `out` seconds. The job card says
  it and has *+ timeline, trimmed to it*; a job's result has it under `shot.takes[].window`, and
  `GET /api/shot/window?path=renders/x.mp4` gives it for any clip in the studio's folder. It knows hard cuts only.
  The other thing that run met, an action carried on past where the prompt stops, is not detected by anything here.

**The Sharp Spec's QA gate, where its tools are missing.** The gate needs ffmpeg (the contact sheet, the sound) and a
speech-to-text tool. On a machine without ffmpeg, a worker Spark as installed for one, the clip is made and the job's
result now says that the gate did not run and why (`qa.skipped`, `qa.note`); so does a job queued with `qa: false`.
`/api/state` has `qa.runs` and `qa.missing`. A machine that has the tools runs the gate on a clip from one that has
not: `POST /api/qa {"from": "http://worker:7870", "job": "<its job id>"}` (the clip, the prompt and the seed are read
from that job), or `{"from": ..., "path": "renders/x.mp4", "prompt": "..."}`, or `{"path": ...}` for a clip already
here. The answer is a job to poll, outside the render queue (no GPU); its `qa` is the gate's report, with the clip
kept under `qa/fetched`. Tried between two Sparks: the head fetched four clips from a worker's app and gave each a
sheet and an identity verdict in 8 to 28 s. The transcript half had been failing on the Spark build with every clip
(faster-whisper opening the sound through an older PyAV); it now runs, 12 s a clip on the CPU.

**Without ffmpeg.** The header shows a line when ffmpeg is missing, with what then does not work (long videos, the
timeline export, a frame taken from a clip, reference videos and audio), and such a job is refused when it is queued,
with the command that installs ffmpeg, not after its render. Single clips and pictures render without it.

### Image only

![Create: image only](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_create_image.png?raw=true)

Choose *Image only*, describe the picture, pick a size and how many takes. Each image has **Download** and
**animate this**, which carries it into a video as the first frame. Where FLUX3-K is installed (`--flux3`), *Picture
model* chooses between **Qwen 2.1 Turbo** (the quickest), **FLUX3-K Fast** and **FLUX3-K Full**; the same choice is
offered for scouts and for a last frame made from a prompt. FLUX3-K writes a full scene description from your prompt
before the first picture (5 to 20 s, once per prompt), then about 10 to 15 s a picture for Fast and a minute for Full.

### Edit a picture

![Create: Edit a picture, the camera ring](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_edit_camera.png?raw=true)

Choose *Edit a picture* and give the picture: an upload, one from the Library, or a frame of a Library clip. Then say
what changes. Qwen-Image-2.1 makes and edits pictures with one set of weights, so this step needs nothing more than
the image model the Studio already has: the picture goes to the text encoder's eyes and into the model's sequence as
latents, and the model draws the picture again as the prompt describes it.

- **Change it in words.** Describe the result ("The same statue at night under a full moon, wet paving stones, a
  lantern on the wall behind it"). What is not mentioned is meant to stay; small things still shift.
- **Move the camera.** With akhaliq's Multiple Angles LoRA (LoRA hub, *Fun*) the step shows a ring of 12 directions,
  30° apart with the front at the top, 4 heights (eye-level, elevated, high-angle, top-down) and a close-up of each.
  Pick angles one by one or take a set (*Five views around*, *Full orbit*, *Four heights*, *Wide and close*): each
  angle is one picture from one load of the model, up to 12 a job. The ring writes the author's captions, such as
  `<mva> right side view, high-angle shot close-up`.
- **Picture model.** Where FLUX3-K is installed, *Picture model* above the picture offers **Qwen 2.1 Turbo**,
  **FLUX3-K Fast** and **FLUX3-K Full**. FLUX3-K edits from words, with further pictures too; BFL's upsampler looks
  at the pictures and writes the caption of the result first, so a short request is enough. It follows "switch the
  neon off" where Qwen left the signs lit, and is slower: an edit took 24 s with Qwen, 37 s with Fast and 139 s with
  Full on a DGX Spark, and 10, 37 and 240 s on an M5 Ultra, where Full is too slow for edits (427 s with two pictures). The LoRAs, the camera ring and the 40-step choice are Qwen's and are hidden for FLUX3-K.
- **Several pictures.** The model takes several pictures at once, so two references need no side-by-side montage
  made elsewhere. Under the picture, *Further pictures* adds up to nine more. The first picture is the one that is
  edited and is image 1 to the model; the others are image 2, image 3, … in the order listed. Name them that way:
  "The man of image 2 stands beside the bed in the bedroom of image 1". Each further picture adds its rows to every
  step: at 1280x736 with Turbo's 8 steps, one picture took 9 s on the Mac (M5 Ultra) and 15 s on the DGX Spark, two
  13 and 22 s, three 15 and 30 s, ten 36 and 230 s (the Mac's peak was 41 GiB up to three pictures and 55 GiB with
  ten; the Spark had 40.7 GB of 121 free at its lowest with ten). One set of pictures and one seed each. Give the
  pictures that are needed: with ten, several of them showing the same two people, the two were drawn twice.
- **The size of the result** is the picture's own: a 1280x736 picture comes back 1280x736 (2.6.5 returned 1344x768,
  the model's own "about 1024x1024, in the picture's shape"). A side that is not a multiple of 32 is edited on the
  model's grid and brought back (1280x720 is made at 1280x704). A picture under 512x512 or over 1536x1536 pixels in
  all is edited at the nearest of the two in its own shape and comes back at that size, and the log says so
  (2560x1472 came back 2016x1152); nothing is enlarged afterwards. *Size of the result* also offers the model's own
  sizes, and a job may carry `width` and `height`.
- **The seed.** The noise an edit starts from is drawn for the seed and the pictures together. Before, it was drawn
  for the seed alone, and an edit made with the seed its picture had been made with started from that picture's own
  noise: it came back as the same picture, over-sharpened (12 of 71 edits in one production run). Any seed now
  gives an edit. The same seed, pictures and prompt still give the same result; a seed gives another picture than
  it did in 2.6.5.
- **Results** go to the Library. Under each: *first frame* opens a clip on it, *last frame* ends a clip on it, *edit
  again* feeds it back in. A front and a back view of one subject are a first and a last frame for a turn.

![One picture, five views and an edit in words, Mac](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/edit_angles_mac.jpg?raw=true)

Measured through the app's own queue, a generated picture of a statue at 1024x1024, seed 7, Turbo's 8 steps
(below). What we saw, one subject and one seed each:

| | Mac (M5 Ultra, MLX, bfloat16) | DGX Spark (ComfyUI, bfloat16) |
|---|---|---|
| One picture, model loaded | 10 s | 15 s |
| Five views around, one job | 55 s | 86 s (25 s for the first, with the load) |
| An edit in words, one job | 14 s | 16 s |
| Base model, 40 steps, one picture | not run (the base transformer is not on this Mac) | 49 to 54 s |
| Memory | 43 GiB at its peak | 82 GB of 121 left at the lowest |

![The same on a DGX Spark, with two views from the 40-step base model](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/edit_angles_spark.jpg?raw=true)


- The picture input works without any LoRA: the edit in words kept the statue and its plinth and changed the light.
- Without the LoRA the model already turns the subject for a side or a back view and keeps the scene. With it the
  height and the close-up are followed too (an *elevated* view is from above, a *close-up* fills the frame), and a
  side view often comes back on a plain studio background: its author trained it on single objects and characters.
- *Left side* and *right side* both came out facing the same way, on both machines. Check a set before you build
  on it.
- On the DGX Spark every view with the LoRA came back on the plain background, the front view too; on the Mac the
  front, front-right and back views kept the scene.

Scripts: `bash scripts/edit.sh photo.png "what changes" out.png`; `SET_FILE=edits.json` makes several from one load;
`LORA=path:0.9` adds an image LoRA. `BASE=1` runs the 40-step base model, which is what image LoRAs are trained on:
`oneshot-setup.sh --image-edit` (13 GB more on a Mac) or `oneshot-setup-spark.sh --image-edit` (14 GB) fetches it.
It is optional: the LoRA works on Turbo as shipped. Both checkpoints are under the **Qwen Research License
Agreement: non-commercial use only**; the LoRA's own file is Apache-2.0.

On a Mac the step runs mflux's Qwen-Image-2.1 edit pipeline in bfloat16 (43 GiB at its peak); under 96 GB of memory
the app asks for 8-bit weights (30 GiB at its peak in one run on a 256 GB Mac; not run on a smaller Mac). On a DGX
Spark it is one more ComfyUI graph, with the LoRA merged at run time on the bfloat16 file.

### Qwen image, then video

![Create: Qwen scout to video](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_create_qwen_h3.png?raw=true)

Choose *A Qwen scout image*. Step 2 makes as many candidates as you ask for (six take about half a minute) at the
size the video model starts from; click the one you want. Either engine animates it. With FastH3 this is an
untrained use that worked in our one test (see [FastH3](#fasth3-new-in-18)).

### Long videos, up to 30 minutes

![Create: a 30 minute chain](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_create_long.png?raw=true)

The model makes at most 15 s at a time, so anything longer is a **chain**: each 15 s clip opens on the last frame
of the one before, and the clips are joined into one file (the repeated frame at each join is dropped). Pick a
length from 30 s to 30 min. **Story beats** are optional, one line per clip, saying what happens in that part;
without them every clip gets the same prompt. Finished clips can be watched from the queue while the rest render.

- Checked: an 18 s chain of two clips (FastH3, 4 passes, 480p) in 183 s; the join is not visible in stills and the
  second clip followed its own beat. **Nothing longer has been rendered.** The times shown for long lengths are
  extrapolated from 5 s clips: about 7 hours for 30 minutes at 480p with 4 passes, about 11 hours with 8.
- Expect drift: each clip only sees one frame of the last, so faces, clothes and light can wander over many clips,
  and sound does not carry across a join. For a planned story with different shots, use Production instead.

### Production, Editor, Library, Tutorial

![Production board](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_production.png?raw=true)

**Production** is for a video planned shot by shot: a project of segments, each with its own scene, action, line and
length. A segment can open on the last frame of the one before; one button renders every segment not yet done.

![Editor](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_editor.png?raw=true)

**Editor** is a single-track timeline: order the clips, set in and out points, preview the sequence, export one file.

![Library](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_library.png?raw=true)

**Library** lists every render, scout and export; *use* makes an image the first frame of a new clip, *+ timeline*
sends a clip to the editor, and 🗑 moves a file to `~/TensorFoldStudio/trash`.

### Library actions: Turnaround sheet, Upscale, Remove a person

Three LoRAs of the hub each do one job that is not "make a clip", through a workflow their authors built. The Library
has a button for each, under the file it works on, and queues it like any render. On a DGX Spark each runs its
author's own ComfyUI nodes, cloned at a pinned commit by `spark/setup-workflows.sh` (`oneshot-setup-spark.sh --ref2va`
runs it: three node packs, SAM 3.1 and the H3 latent upscaler, 2.5 GB); all three render on MiniMax H3 Ref2VA in
bfloat16. Credit and licences: [CREDITS.md](CREDITS.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

| Action | On | Does | With | Mac | DGX Spark |
|---|---|---|---|---|---|
| **Turnaround** | a picture | Five views of the picture's subject, turning from where it faces now, made together so they agree. Each is a picture of its own: a first frame, or reference pictures of one character. | Turnaround LoRA by matlod; contact-sheet nodes by MATLOWAI | yes: the MLX engine packs and decodes the five views itself (`h3_generate.py --sheet`) | yes |
| **Upscale** | a clip | The clip made larger and sharper: a latent upscale, then one light re-draw in tiles that the clip itself guides. Motion, mouths and the clip's own sound stay. | Video Upscaler LoRA and H3 Video Upsampler by dntpi; latent upscaler by LBH-123-AI | no: the upscaler is a ComfyUI node pack in PyTorch and the MLX engine has no latent upscaler or guide layout (a Mac doubles a clip with the 2x decoder: the 2K and QHD sizes) | yes |
| **Remove a person** | a clip | One person taken out and the background drawn again. The person is found from a few words and followed through the clip; "1. Find the person" shows the mask alone first. | Person Remover LoRA and H3 Relay by Akatz Labs; SAM 3.1 by Meta | no: it needs SAM 3.1, a source clip fed in as a frame-aligned guide and windows that carry 18 frames of history, none of which the MLX engine has | yes |

**Turnaround.** The LoRA's own instruction, "the camera orbits the subject of `<Picture 1>` ninety degrees
clockwise", was read as a roll of the picture on some seeds, on both engines (the model without the LoRA always reads
it so). The Studio's default says the orbit is level and the subject stays upright, which held on every subject and
seed tried; the instruction box takes other wording. How far the subject turns varies: a toy robot went from front to
back, a person in a fighting stance turned about a quarter. Sizes 512 (what the LoRA was trained at), 1024 and 2048.

**Remove a person.** The graph and settings are those of the authors' example workflow (strength 1.0, windows of 22
frames with 18 frames of history, 12 passes, mask grown by 5 points), on the bfloat16 weights. Their workflow asks for
a first frame with the person already edited out. With "Build it for me" the Studio makes one: every masked point of
the first frame takes the median of the frames where it is not masked (when the camera is still), the rest is filled
in from its edges, and one window of the remover itself then draws that rough plate sharp. The clip is lengthened by
eight copies of its last frame and the result cut back, because the last frames of the last window kept patches of
green. The whole picture is drawn again, so details outside the person can change; the clip's own sound is put back,
with the removed person's voice still in it unless "No sound" is ticked.
After the windows, green that came back for a frame or two is painted over from the clean frames beside it (at most
four frames away), and so is a frame that is a different picture altogether; what is left is counted and said on the
job. The upscaler's and the remover's node packs each
replace the layout of H3's packed sequence when they first run and do not share a server: after one of the two
actions the other starts ComfyUI again, which costs the model load.

![A turnaround sheet in the Library](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_action_turnaround.png?raw=true)

![Remove a person in the Library](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_action_remove.png?raw=true)

What was measured (2026-10-12, one run each unless said; Mac Studio M5 Ultra 256 GB; DGX Spark GB10 with the models
already loaded unless said; the test pictures and the clip are generated ones):

| Action | Input | Mac | DGX Spark | Lowest free memory (Spark) | By eye |
|---|---|---|---|---|---|
| Turnaround, 512 | a toy robot, a person in a stance, an animated knight | 42 to 51 s (28 passes at 1.1 s) | 45 s; 77 to 81 s with the model load | 35 GB | upright in all, same subject in all five; robot front to back, person about a quarter turn |
| Turnaround, 1024 | the same | 150 to 162 s | 188 s | 36 GB | as at 512, sharper |
| Upscale to 2 megapixels | 5 s at 864x480 | not available | 303 s, to 1984x1088 | 32 GB | wood grain, rope and lettering sharper; framing, motion and sound unchanged |
| Upscale to 1 megapixel | the same | not available | 190 s through the app's queue | 31 GB | not looked at closely |
| Find the person | 5 s at 864x480, "man in a blue shirt" | not available | 40 s | 38 GB | the whole man in every frame, hands included |
| Remove a person, clean frame built | the same clip: the camera pushes in until the man fills a third of the frame | not available | 663 s in all: mask 40 s, clean frame 94 s, six windows 525 s | 28 GB | the man gone from every frame and the rooftop drawn again behind him, the push-in kept; green had flashed back on 22 of the 121 frames where his hand moved fast and was painted over from the frames beside them, which leaves a faint green tint on a few; the last three frames keep a dim green patch on the wall |
| The same through the app's queue | the same | not available | 775 s | 13 GB (another agent's render was loaded beside it; 28 to 30 GB alone) | as above; the clip's very last frame came out as another picture altogether, which the repair now replaces from the frame before (checked on the saved clip, not by a new render) |

Not tested: 2048 views, clips longer than 5 seconds, a clip with two people where one is removed, windows of 39 and
56 frames, and any of it on a machine other than these two.

### Clearing prompts and deleting clips

From 2.6. Nothing below erases a file except *Empty trash*.

| Where | Control | What it does |
|---|---|---|
| Create wizard | *Clear this step* | Empties the text fields of the step on screen. *Music* goes back to `N/A`; a line's language stays. |
| Create wizard | *Clear all* | Empties every prompt field of the clip: scene, action, camera, lines, soundscape, music, the notes on references and the composed prompt. Pictures, clips and settings stay. |
| Production board | *Clear* on a segment, *Clear every segment's text* | Empties scene, action, voice, line and soundscape. Titles, lengths, seeds and rendered clips stay. |
| Slate | *Clear* above a shot's prompt | Empties that shot's prompt. |
| Queue | 🗑 on a finished job | The job leaves the list; its clip or stills, its QA sheets and takes, and uploads no other job in the list uses go to the trash. |
| Queue | *Delete selected*, *Clear failed*, *Delete all finished* | The same for the ticked jobs, the failed ones, or every finished and cancelled one. A clip that is in a film (a project's timeline, a Slate take) stays. |
| Library | 🗑, or tick files and *Delete selected* | The files go to the trash. |
| Library | *Restore last deletion* | The files of the last deletion return to where they were, and its job cards to the queue. |
| Library | *Empty trash* | Shows the number of files and their size, asks, then erases them. This cannot be undone. |

A cleared form is put back by *Undo* in the message or by *Undo clear* beside the button, until the page is closed
or reloaded. A queued or running job cannot be deleted (cancel it first), and a file such a job reads or writes
stays. Deleting never touches a project file, a LoRA or a model: only pictures, clips and sound in `renders`,
`exports`, `scouts`, `uploads` and `qa`, and clips in `projects` and `slate`, can be moved. The trash is
`~/TensorFoldStudio/trash`; a LoRA removed in the hub goes there too, so *Empty trash* erases that as well.
`tests/test_clear.py` holds the path checks. Driven in a headless Chromium on both machines' builds, against a
studio folder of test clips: each Clear and Undo, one job deleted and put back, two ticked jobs, the failed ones,
all finished, restore, two ticked Library files, and the trash emptied (25 checks). Not driven in a browser: the
*Clear* above a Slate shot's prompt (it is the same control as a segment's).

![Tutorial](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_tutorial.png?raw=true)

#### Wasserman Sharp Spec

A **Wasserman Sharp Spec** button sits in Create (step 1), on the Production board and in the Slate film settings. One
click sets Sam Wasserman's locked recipe, so every clip is made the same way: 1376x768 generated in one pass with no
upscale and no crop, FastH3 at 8 passes (sparse attention 0.8), seed 42, and 8 or 10 seconds (192 or 243 frames at
24 fps). The file is H.264 High yuv420p with 32 kHz stereo AAC, as for every clip here. A retake is the next seed. The
button shows a tick while every setting still matches; changing one of them turns it off.

Each clip made to the spec then goes through his **QA gate** (`app/qa.py`), about a quarter of a minute:

- **Speech.** The clip's sound is transcribed (faster-whisper if it is installed, else the `whisper` command) and has
  to hold the line the prompt asked for. A prompt with no spoken line passes.
- **Identity.** Three frames (1 s, the middle, 1 s before the end) go side by side on one contact sheet, and a vision
  model checks that face, hair and wardrobe are the same in all three with no smeared faces. The model is the film's
  brain, or Claude Code, Codex or a local vision model, under your own sign-in: the sheet is sent to that model.
- **Retake.** A clip that fails either gate is rendered once more at the next seed; the failed take and every contact
  sheet are kept in `~/TensorFoldStudio/qa/`. The job card shows what each gate found, and the sharpness figure of
  the three frames as a diagnosis.

A gate whose tool is missing on the machine is shown as skipped and does not fail the clip.

#### Keys and Wasserman LoRA Hub

A library of LoRAs for MiniMax H3 that a clip can be rendered with: Sam Wasserman's 80s Horror LoRA and, from 2.6.3,
twenty-one community LoRAs in seven groups (six by kind, and Fun). It is in the video settings of Create (step 1), on the Production board and in
the Slate film settings. Each LoRA is its author's own work under its own licence; the page on its card is the
author's, for credit.

- **Browse.** "▸ Browse the hub" opens the library: a tab for each group, a search box, "Installed only", and a card
  for each LoRA with what it does, its author, what it works from, its licence, its size and its trigger word.
- **Install.** After the essential bundle (below) every usable LoRA is already installed. Otherwise "⤓ Install" on a
  card, or "Install all in this category", puts the file in `~/TensorFoldStudio/loras/`, checked against its sha256.
- **Use.** "Use on this clip" adds the LoRA to the clip being made. Each picked LoRA has a strength (its author's range
  is in the tooltip; a slider LoRA also takes minus values) and, when its recipe has them, a look. The trigger word and
  look phrase are put at the start of the shot for you. Up to four can be stacked.
- **Add.** "+ Add your own LoRA" takes a recipe: a name, the address of the `.safetensors` file or an uploaded file,
  the trigger word, the strength, and optional looks. Presets are in `app/lora-presets.json`.

**Wasserman**

| LoRA | By | What it does | Works from | Licence as stated | Rendered with it on |
|---|---|---|---|---|---|
| Wasserman-80's Horror LORA | Sam Wasserman (wassermanproductions.com) | The look of a 1970s-80s horror movie: VHS or 16mm film. | text or a first frame | H3 Community | not tested |

**01 Camera & Viewpoint**

| LoRA | By | What it does | Works from | Licence as stated | Rendered with it on |
|---|---|---|---|---|---|
| Camera Motion | Jojocodex | Camera moves the base model tends to miss: push-in, pull-back, orbit, tracking, aerial, crane, pan, macro. | text or a first frame | Apache-2.0 | Mac (84 s), DGX Spark (150 s) |
| Head-mounted FPV | neph1 | A first-person view from a camera on the head, with the bob of walking or running. | text or a first frame | H3 Community | Mac (84 s), DGX Spark (150 s) |
| Equirectangular 360° | shamanic | A full-sphere 360° panorama video with sound, for VR players. | text or a first frame | H3 Community | Mac (108 s), DGX Spark (180 s) |
| 360° Orbit | pablodawson | One photo becomes a frozen-moment 360° orbit that ends on the frame it started from. | first and last frame | H3 Community | Mac (140 s), DGX Spark (230 s), 768x768 |

**02 Transformation & Morphing**

| LoRA | By | What it does | Works from | Licence as stated | Rendered with it on |
|---|---|---|---|---|---|
| Transformation | Ashmotv | Morphs one picture into another, element by element. | first and last frame | H3 Community | Mac (108 s), DGX Spark (158 s on FastH3) |
| Turnaround | matlod | A five-view character sheet (front to back) from one reference picture. | the Library's *Turnaround* action | H3 Community | Mac (45 s), DGX Spark (45 to 80 s) at 512 |
| Keyframe Animation | alvdansen | Hand-drawn animation on held frames: the next key drawing, an inbetween, or a short held sequence from one or two drawings. | reference pictures | Custom, gated | Mac (260 s), DGX Spark (315 s), 20 passes |

**03 Character Editing**

| LoRA | By | What it does | Works from | Licence as stated | Rendered with it on |
|---|---|---|---|---|---|
| Character Swap | Akatz Labs (akatz-ai) | Replaces a person in a source clip with the person in a reference picture, keeping the scene. | a source video | H3 Community | Mac (285 s), DGX Spark (215 s) |
| Person Remover | Akatz Labs (akatz-ai) | Removes a chosen person from a video and rebuilds the background. | the Library's *Remove a person* action | H3 Community | DGX Spark (about 11 min for 5 s) |
| Natural Face Speech | vpakarinen | More natural face muscles and clearer English speech for a person talking to the camera. | text or a first frame | Apache-2.0 | Mac (84 s), DGX Spark (150 s) |

**04 Motion & Action**

| LoRA | By | What it does | Works from | Licence as stated | Rendered with it on |
|---|---|---|---|---|---|
| Motion Continuity Repair | JOKER141 | Keeps fast movement connected: fewer broken transitions, less unwanted slow motion, closer to the prompt. | text or a first frame | None stated | Mac (84 s), DGX Spark (150 s) |
| Combat Base V2 | JOKER141 | Faster, harder-hitting fight movement, and livelier body movement and sound in dialogue scenes. | text or a first frame | None stated | Mac (84 s), DGX Spark (144 s) |
| Weapon Combat | JOKER141 | Sword, staff and other weapon fights that hold together better. | text or a first frame | None stated | Mac (90 s), DGX Spark (150 s) |
| Spatial Physics | Jojocodex | Objects that collide, stack, fall and block each other more believably; also firmer contact in fights. | text or a first frame | Apache-2.0 | Mac (90 s), DGX Spark (150 s) |

**05 Art Style**

| LoRA | By | What it does | Works from | Licence as stated | Rendered with it on |
|---|---|---|---|---|---|
| Anime Slider | adf99 | A slider between flat 2D anime (plus) and photorealism (minus). | text or a first frame | None stated | Mac (90 s), DGX Spark (150 s) |
| Lineart Anime | DiffSynth-Studio | Colours a line-art video into finished anime. | a source video | Apache-2.0 | Mac (280 s), DGX Spark (205 s) |
| 3D Animation Style | SOLRICKS | The look of a 3D animated feature. Start the prompt with `3dsrx`. | text or a first frame | H3 Community, gated | DGX Spark (Mac: rewritten for the engine, no clip rendered yet) |

**06 Support & Finishing**

| LoRA | By | What it does | Works from | Licence as stated | Rendered with it on |
|---|---|---|---|---|---|
| Natural Face Speech | vpakarinen | More natural face muscles and clearer English speech for a person talking to the camera. | text or a first frame | Apache-2.0 | Mac (84 s), DGX Spark (150 s) |
| Motion Continuity Repair | JOKER141 | Keeps fast movement connected: fewer broken transitions, less unwanted slow motion, closer to the prompt. | text or a first frame | None stated | Mac (84 s), DGX Spark (150 s) |
| Video Upscaler LoRA | dntpi | Doubles the size of a finished clip: a latent upscale, then one guided re-draw in tiles. | the Library's *Upscale* action | MIT code, weights other | DGX Spark (5 min for 5 s to 2 MP) |

**07 Fun**

| LoRA | By | What it does | Works from | Licence as stated | Rendered with it on |
|---|---|---|---|---|---|
| Person Remover | Akatz Labs (akatz-ai) | Removes a chosen person from a video and rebuilds the background. | the Library's *Remove a person* action | H3 Community | DGX Spark (about 11 min for 5 s) |
| VH5 Tape (VHS) | KennethFal | A 1980s home-video look with a damage dial: light wear, visible tape damage, or heavy tracking errors. | text or a first frame | H3 Community | Mac (138 s), DGX Spark (150 s) |
| Wushu Action | Jojocodex | Kung fu and wushu moves with weight behind them: punches, kicks, spins, staff work. | text or a first frame | Apache-2.0 | Mac (84 s), DGX Spark (150 s) |
| Multiple Angles (image) | akhaliq | Shows the subject of a picture from another camera angle: 12 directions at 4 heights, each also as a close-up. Used in Create, *Edit a picture*. | a picture to edit | Apache-2.0 | Mac (10 s a picture), DGX Spark (15 s a picture) |

"Not usable yet" means the hub lists the LoRA and the bundle holds its file, but the LoRA needs something a clip does
not have in this version. Its card says which. Three LoRAs are not for a clip at all and have a job of their own,
which the Library runs as an action (see *Library actions*): Turnaround, Person Remover and Video Upscaler LoRA.
Multiple Angles is a LoRA for the image model: it is used in *Edit a picture*, not on a clip. 3D Animation Style is approved download by download by its author; its file is in kohya's own layout (`lora_unet_blocks_0_attn_qkv_proj`), which ComfyUI reads as it is and the Mac converter renames.

**What a picked LoRA asks for.** A LoRA that needs more than a prompt says so in its own row and again in the Render
step, each with a button to the step that takes it, and "Queue this clip" waits until nothing is missing: a last frame
for 360° Orbit and Transformation (see *A last frame: the clip ends on a picture*), a source video, a picture or 20
passes for the LoRAs made for a clip from references (below). **Camera Motion** has a *Camera move* choice with its
author's own phrases (push-in, pull-back, handheld follow, orbit, crane reveal, tilt, pan; the author's note on how
steady each is sits beside it), and **Head-mounted FPV** a *View* choice in the Studio's wording, since its author
gives none. A **slider LoRA** has a track centred on 0 with its two ends named (for Anime Slider: photographic at -5,
2D anime at +5), the number beside it, and *Off (0)*.

What was tested (2026-10-12, one seed each, 5 seconds at 864x480, FastH3 8 passes, text to video; 1120x480 for the 360°
LoRA): each usable LoRA was installed through the hub and rendered twice at the same seed, at 0.01 (the same engine
path with no visible effect) and at the strength in its recipe. "Change" is the mean absolute difference of the two
clips' frames on a 0 to 255 scale; the contact sheets were looked at.

| LoRA | Strength | Mac: time, change | DGX Spark: time, change |
|---|---|---|---|
| Camera Motion | 0.9 | 84 s, 52.1 | 150 s, 50.9 |
| Head-mounted FPV | 1 | 84 s, 62.6 | 150 s, 41.4 |
| Equirectangular 360° | 1 | 108 s, 63.8 | 180 s, 26.8 |
| Natural Face Speech | 0.7 | 84 s, 55.0 | 150 s, 29.9 |
| Motion Continuity Repair | 0.9 | 84 s, 41.0 | 150 s, 22.4 |
| Combat Base V2 | 1 | 84 s, 57.8 | 144 s, 46.6 |
| Weapon Combat | 0.6 | 90 s, 30.3 | 150 s, 14.6 |
| Spatial Physics | 0.4 | 90 s, 41.6 | 150 s, 25.8 |
| Anime Slider | 3 | 90 s, 69.3 | 150 s, 67.5 |
| VH5 Tape (VHS) | 1 | 138 s, 56.8 | 150 s, 45.4 |
| Wushu Action | 0.5 | 84 s, 34.9 | 150 s, 33.6 |

Seen on the sheets: the Anime Slider turns a 3D-leaning picture into flat 2D anime; Equirectangular 360° gives a
level panorama where the base model drew a "little planet"; Head-mounted FPV widens the view and puts it at head
height; Camera Motion pulls back from a close-up to a wide shot where the base clip only followed the fox; Combat Base
V2 gives larger, faster strikes; Natural Face Speech moves in on the speaker at the microphone; VH5 Tape turns a clean
studio shot into a worn 4:3 tape recording. The numbers say the clip changed, not that it improved: judge the others by eye for your own prompt.

How it renders: on a Mac the file is rewritten once (`scripts/convert-lora.py`) and merged at load by the MLX engine,
for MiniMax H3 at any quality and for FastH3 (which then runs on the MLX engine, not the native one). On a DGX Spark
(2.7.1) TensorFold's ComfyUI node merges it into FastH3's full-precision projections the way ComfyUI would (its LoRA
arithmetic, then its seeded rounding to bfloat16) and the engine quantizes them to int8, so the clip renders on the
engine at its usual speed and memory: 864x480, 124 frames took 114 s against 140 s on ComfyUI's own blocks, plus
12.5 s for the merge the first time a LoRA is used (a later clip without it puts the checkpoint's weights back the
same way). Same seed, The Wasserman 80s Horror at 1.0: the engine's clip and ComfyUI's bfloat16 one show the same shot
(pixel correlation 0.80, the sound 0.89), closer than the two engines are without a LoRA (0.55), and both far from
the clip without the LoRA (0.43 to 0.49). The LoRA's keys outside the blocks (the text refiner, the modulation, the
final layer) are merged by ComfyUI as before. A LoRA still has no time estimate.

**LoRAs on a clip from references.** The hub is also offered on a clip made from references (MiniMax H3 Ref2VA), and
the LoRA is merged into that transformer: by the MLX engine on a Mac, and on a DGX Spark by ComfyUI into the
full-precision (bfloat16) Ref2VA file, with or without the Turbo adapter. A LoRA merged into the Spark's 8-bit file
would corrupt the clip without an error (its weights are stored rotated), so a reference clip with a LoRA never runs
on it; it starts from a ComfyUI server with nothing loaded. Three presets were made for one job on the reference
model, and the row of a picked one says what the clip needs, whether it has it, and writes the Review prompt the way
its author trained it (`app/lora_flows.py`, each from the LoRA's own model card):

| LoRA | The clip holds | Prompt | Passes |
|---|---|---|---|
| **Character Swap** (Akatz Labs) | your clip as `<Video 1>` (4 to 5 s held best for the author), a picture of the new character as `<Picture 1>` | `Replace only <who> in <Video 1> with the character in <Picture 1>. ...`, with who is replaced taken from "Who or what it shows" of Video 1 | Turbo 4 or 20 |
| **Lineart Anime** (DiffSynth-Studio) | one video, line art (white lines on black) at the clip's own size, and no picture | Scene and Action as plain sentences, sent as written | Turbo 4 or 20 (the author: 20) |
| **Keyframe Animation** (Alvdansen Labs) | one drawing (the current key) or two (first drawing and end extreme) as pictures | `Create the next keyframe in the sequence. SCENE: ... MOVEMENT: ... STATIC: ...` | 20 (the author: not on a Turbo adapter) |
| **Anime Slider** (adf99) | anything: text, a first frame or references | yours | any |

*Make line art of Video 1* (in Lineart Anime's row) draws the line art of any clip with ffmpeg alone (`app/lineart.py`:
an edge-preserving blur, then each pixel's difference to its brightest neighbour; fine, medium or outlines only). No
model is involved. The author publishes no extractor; this one was matched by eye to the author's example clip.

**Layouts.** Most H3 LoRAs are saved in ComfyUI's layout. DiffSynth-Studio's are raw PEFT saves
(`blocks.N....lora_A.default.weight`, no prefix), which ComfyUI loads without matching a single key and without a
warning. `app/lora_layout.py` tells the layouts apart and rewrites such a file at install, on a Mac through
`scripts/convert-lora.py` and on a DGX Spark as `<id>.comfy.safetensors`. One kind of pair needs more than a new name
on a Spark. ComfyUI's `pruned` H3 checkpoints drop the time embedder: each adaln projection reads 8 numbers a
timestep from a table where MiniMax's own weights read 2,688. A LoRA pair made on the full adaln weights is carried
over with the affine map between the two (solved from the table and MiniMax's time embedder; fit error 1.4e-5 on the
real weights, and the update of a real pair agrees to 2.9e-5), `A' = A @ basis.T` with the rest in a bias term.
The basis is worked out once, from the model on the machine, a LoRA bundle that holds `ref2va-adaln-basis.npy`, or
the first 64 MB of MiniMax's own Ref2VA file (a ranged read of the public page; `LORA_OFFLINE=1` forbids it).

**Measured** (5 s at 864x480, seed 7, the same request without and with the LoRA; a generated clip of a man at a
wooden dummy as the source, a generated picture of a woman as the new character, two line drawings made from the
clip as the keys; one seed each, looked at on contact sheets):

| LoRA | Passes | Mac M5 Ultra: without / with | By eye | DGX Spark: without / with | By eye |
|---|---|---|---|---|---|
| Character Swap, 1.0 | Turbo 4 | 281 s / 287 s | the swap holds either way; with the LoRA her pose sits a little closer to the source's | 210 s / 516 s | the same: both swap her in on the same rooftop, small differences in hands |
| Lineart Anime, 1.0 | Turbo 4 | 271 s / 278 s | both colour the line art as anime on the lines; with the LoRA flatter cel shading | 541 s / 331 s | the same |
| Lineart Anime, 1.0 | 20 | - / 706 s | on the lines, but darker and closer to 3D than at Turbo 4 | - / 605 s | anime on the lines, as clean as at Turbo 4 |
| Keyframe Animation (sequence), 1.0 | 20 | 258 s / 302 s | without: a photographed man, the drawings ignored; with: the line drawing itself moves, held on white | 315 s / 306 s | a moving line drawing on white both times (without the adapter the Spark ran its 8-bit file); small differences in line weight |
| Anime Slider, +3 / -3 | Turbo 4 | 91 s / 92 s / 92 s | +3 smoother and flatter, -3 more photographic; mild next to the Spark | 235 s / 110 s / 110 s | +3 flat 2D anime with a drawn background, -3 photographic |

The Spark's times include starting ComfyUI and loading the 40 GB bfloat16 Ref2VA file, and a render that follows
another kind of render loads it again; its lowest free memory in these runs was 33 GB of 121. On this clip the base
model already swaps a person and already colours line art from the prompt, so Character Swap and Lineart Anime change
the result less than Keyframe Animation and the slider do. Why the slider's +3 is milder on the Mac than on the
Spark was not found (same file, same strength; the Mac runs these passes on its int8 kernels).

**Through the apps (2026-10-13, 2.6.3 on both machines, one seed).** Each clip below was set up in the create wizard
of the running app in a browser (headless Chromium), queued with "Queue this clip" and rendered by the app's own
queue, one at a time; every result was looked at on a contact sheet. 5 seconds at 864x480 unless noted; FastH3 8
passes for the clips from text or frames, the Ref2VA Turbo adapter (4 passes) for the clips from references.

| Clip | Mac (M5 Ultra) | DGX Spark | What came out |
|---|---|---|---|
| First and last frame (a teapot, a bonsai) | 100 s | 130 s | ends on the bonsai picture |
| 360° Orbit, first = last, 768x768, the author's prompt | 140 s | 230 s | a full orbit of the statue, back to the first view |
| Transformation, the author's prompt | 100 s | 165 s | the teapot breaks into painted ribbons that form the tree |
| Character Swap (a clip and a picture of a woman) | 285 s | 215 s | she is at the dummy in his place, the rooftop and the move kept |
| Lineart Anime, after *Make line art of Video 1* | 280 s | 205 s | the clip redrawn as anime on its own outlines |
| Keyframe Animation, two drawings, 20 passes | 260 s | 315 s | a moving line drawing |
| Anime Slider on a clip from text: -3 / 0 / +3 | 85 / 70 / 90 s | 150 / 125 / 150 s | photographic / the base look / flat 2D anime |
| A last frame made from a prompt (step 2) | 12 s | 23 s | the picture lands in the Last frame slot |

Lowest free memory on the DGX Spark across its ten jobs: 29 GB of 121.

**Known difference: the slider on a Mac clip from references.** On a clip from text the slider pulls the same on both
machines. On a clip made *from references* (Ref2VA), the same file at +3 gives full 2D anime on the DGX Spark and only
a painterly lean on the Mac; the Mac at +6 looks like the Spark at +3. It is not the int8 kernels (`INT8=0`, the same
weak pull), not the Turbo adapter (20 passes without it, the same) and not the file's scale (the text path agrees).
The cause is not found: on a Mac, use about twice the strength for a slider on a clip from references.

#### Essential LoRA bundle

```bash
bash oneshot-essential-bundle.sh          # both installers run it; --no-loras on an installer skips it
```

It fetches every preset of the hub once: fourteen in one essential package per platform (2.1 GB, from
`huggingface.co/drowzeys/keys-studio-lora-essential`, the LoRAs whose stated licences allow passing them on, each with its
author's card and licence; `LORA_PACKAGE=none` skips it) and the rest each from its author's own page (about 6.5 GB in
all and 4 minutes measured on a fast line;
the three Keyframe Animation adapters, 5.6 GB of it, need your Hugging Face account's acceptance of their licence),
into `~/TensorFoldStudio/lora-bundle/`, laid out by the hub's groups (`01-camera-viewpoint` ... `06-support-finishing`, `07-fun`,
with `bundle.json` as the menu), checks each sha256, keeps the author's model card and licence beside each file, and
installs the set into the hub (on a Mac, rewritten for the MLX engine). After that the Studio needs none of those
links: copy the folder to another machine or drive and run `bash oneshot-essential-bundle.sh --source /path/to/lora-bundle`
(or an installer with `--lora-bundle=/path`) for an install with no network. A second run fetches nothing. A dead or
moved author link is reported and skipped, not fatal. `--categories camera,style`, `--only ID,ID`, `--dest DIR`,
`--dry-run`, `--verify`, `--json`, `--no-install`. `--package FILE` unpacks an essential package (one `.tar` per
platform with the LoRAs whose licences allow passing them on, checked against the `.sha256` beside it) before anything
else; `--make-package DIR` writes those archives from a bundle. No LoRA file is in this repository.

**For AI agents** (no prompts anywhere; run in this order on a new machine):

```bash
git clone https://github.com/ashhart/TensorFold.git && cd TensorFold/studio
bash oneshot-setup.sh --no-render --no-loras        # the Studio (oneshot-setup-spark.sh on a DGX Spark); model weights as listed under One-shot
bash oneshot-essential-bundle.sh --json > lora-bundle.json; echo "exit $?"   # 0 all present, 1 some unavailable (listed), 2 usage or disk
bash oneshot-essential-bundle.sh --verify --json    # fetches nothing: "present", "unavailable", "gated", and "menu" (group -> LoRAs -> state)
```

Expect 4.8 GB and 20 of 22 present without a Hugging Face token (`gated` lists the two an author gates), in 90 to 120 s
at 40 MB/s; measured 119 s on the DGX Spark and 86 s on the Mac, a second run 2 s, an offline run from a copied bundle
3 s. Set `HF_TOKEN` (or sign in with `hf auth login`) for the gated ones once their authors have approved the account.

#### Wasserman Slate AI Studio

At the top of **Production** there are two ways to plan a film: the board of segments above, or **Wasserman Slate AI
Studio**. The second is a port of the workflow of Slate, the prompt studio for AI filmmaking by Sam Wasserman
(wassermanproductions.com), wired to this Studio's render queue, so a shot goes from its prompt to a clip without
copying anything anywhere.

| Part | What it does |
|---|---|
| **First AD** | Say what you are making. It creates scenes and shots, writes their prompts, casts characters, scouts locations, sets spoken lines and sound, and lists what it did. It changes or replaces a character, place, prop, cue or voice that is already there (the sheet keeps its id, so every shot that casts it follows), sets the song, a shot's slot in bars and a shot's LoRA. |
| **Bible** | Logline, world and tone, defaults, the song (tempo, beats in a bar, first beat), how the film renders (resolution tier, quality), and which brain to use. |
| **Studios** | Casting, Art Department, Locations, Lookbook: sheets filled from one line by ✦ Fill. A still pinned to a character is its reference picture. |
| **Shots** | Specs (seconds, aspect, size, angle, lens, movement) and the six-section prompt; ✦ Structure, Tighten, Enrich, Distill, Shot, Angle, Beats, Variants; Director's notes; Slate's setups; picture-lock and mute; history. ✦ Coverage lays out a scene from one of Slate's twelve coverage plans. |
| **Roll camera** | What the shot opens on (text, the last frame of the shot before, an image, the cast's stills), who is in it and where, spoken lines, soundscape, cue. The shot compiles to MiniMax H3's prompt form with a preflight list; one button queues it. Clips come back as takes: circle the one that works. A LoRA from the hub can be put on one shot in place of the film's. *Import a take rendered on another machine* logs a clip another app made into the shot, with its seed and settings. |
| **Sound** | Voice sheets (the voice of a spoken line) and music cues (the score line of a shot). **Cast voice** speaks a sheet's sample line with Fish Audio Drama 3 in the voice it describes; ticked, the take goes into shots that open on cast stills as that speaker's reference audio. |
| **Deliver** | Render a whole scene in order, a continuity check, the shot list as Markdown or CSV, and the film's takes to the Editor's timeline. |

The brain is Slate's: your own Claude Code or Codex sign-in on the machine that runs the app, or a local
OpenAI-compatible server (Ollama, LM Studio, vLLM, llama.cpp). Everything without a ✦ works with no brain. Films are
Slate's own `project.json` under `~/TensorFoldStudio/slate/`: a project made in Slate opens here, and one made here
downloads from the Bible and opens in Slate (what this port adds sits under `tf` keys). Shots render at any length
the model takes (their seconds, or their slot in bars) and the aspect ratios 16:9, 9:16, 1:1, 4:3, 3:4 and 21:9.

**What a shot sends to the model.** A shot's *intent* is a note for people and stays out of the prompt (it used to
stand in as the action when a shot had no beat sheet, which put a run's timecodes into its prompts): what happens is
the Subject section's, or the beat sheet's.

**A music video.** Give the song's tempo in the Bible (the Studio does not listen to the song: the tempo and the
first beat are yours, from the score or a beat tracker) and each shot a slot in bars. The shots are laid end to end
from bar 1, and a shot's clip is rendered at the shortest length the model takes that covers its slot, so the cut
can sit on the bar line: at 117.45 BPM a 2-bar slot is 98 frames and renders as 107, a 4-bar slot is 196 and renders
as 209 (with the four fixed lengths it needed a 10 s clip, 243). Roll camera shows the slot, where it starts in the
song and how many frames there are to trim. The cut itself is still made in the Editor or outside: nothing here lays
the song under the timeline.

**A LoRA on one shot.** *LoRA for this shot* in Roll camera puts one installed LoRA on that shot in place of the
film's. It goes through the hub (strength, look, what it needs) and, on a DGX Spark, through the memory guard when
the shot is queued; the shot's preflight says beforehand when it will be refused and what fits. On a Spark a clip
with a LoRA is taken up to 1376x768 at 243 frames (2.7.1; before that 1280x736 at 124, and the Sharp Spec tier took
none).

**Takes from other machines.** A film's project is on one machine and its clips are often rendered by others, each
by its own app. *Import a take* takes that app's address and one of its job ids (the clip, prompt, seed, settings and
what its checks said are read from the job), or a path in its folder, or an uploaded file, and logs the take in the
shot. `POST /api/slate/takes {"projectId", "shot", "from", "job"}` does the same for a script (`"log": true` writes
it into the project file). An app forgets its jobs when it restarts; the clip's path still works.

Not ported from Slate: sequence chunks, alt takes, punch-ups, the tone dial, pickups on a selected span, reference
and audio analysis, the stills library, compiles for other generators, and its MCP server. The app has no sign-in of
its own: anyone who can open it can use the brain, so keep it on a network you trust.

**Tutorial** teaches the other four: eleven short lessons, one step on screen at a time, with an example and a tip
where they help. *Show me in the app* opens the tab a step is about and outlines the control; a bar at the bottom
then carries the lesson forward from inside the app. Finished lessons are remembered in the browser.

The app drives the same scripts as the command line, one job at a time, and keeps projects, uploads and renders
under `~/TensorFoldStudio`. What it is not: the editor has one video track with trims and ordering only (no
transforms, text, transitions or audio mixing), and there is no assisted prompt writing. The workflow is modelled
on rookiestar28's ComfyUI-MiniMaxH3-Studio, which does far more; no code is shared with it.

## Voice (Fish Audio Drama 3)

See [docs/voice.md](docs/voice.md).

## Motion Previs

See [docs/motion-previs.md](docs/motion-previs.md).

## FLUX3-K picture models

See [docs/flux3-k.md](docs/flux3-k.md).

## FastH3 (new in 1.8)

**Credit first: FastVideo / Hao AI Lab** made FastH3 (the distilled weights and the sparse-attention routing they
were trained with), on **MiniMax's** H3. What this pack adds is the attention kernel that makes it fast on a Mac.

```bash
bash scripts/fast.sh "a prompt" out.mp4                      # 480p (864x480), 8 passes
STEPS=4 bash scripts/fast.sh "a prompt" out.mp4              # 4, 8 or 20 passes
RES=720p bash scripts/fast.sh "a prompt" out.mp4             # 1280x720
UPSCALE=1 bash scripts/fast.sh "a prompt" out.mp4            # 480p generated, 2x decoder: 1728x960
RES=720p UPSCALE=1 bash scripts/fast.sh "a prompt" out.mp4   # 720p generated, 2x decoder: 2560x1440 (2K)
# Qwen-Image-2.1 makes the first frame, FastH3 animates it (all the studio.sh presets work: X2=1, TWOK=1, QHD=1)
ENGINE=fasth3 STEPS=8 WIDTH=864 HEIGHT=480 bash scripts/studio.sh "the picture" "what happens" out.mp4
```

In the web app FastH3 is the first card under **Video engine**, with Passes, Resolution and 2x upscale beside it.

Measured on a Mac Studio M5 Ultra (64-core GPU, 256 GB), 5 second clips of 124 frames with sound, one run each,
whole command including model load and decode:

| Output | Generated at | 4 passes | 8 passes | 20 passes |
|---|---|---:|---:|---:|
| 480p, 864x480 | 864x480 | 51 s | 78 s | 174 s |
| 480p + 2x decoder, 1728x960 | 864x480 | | 88 s | |
| 720p, 1280x720 | 1280x736, cropped | 112 s | 192 s | 434 s |
| 720p + 2x decoder, 2560x1440 | 1280x736, cropped | | 203 s | |

A pass takes 7.1 s at 480p and 19.7 s at 720p; the 2x decoder adds about 10 s. A 10 second 1312x736 clip (243
frames) takes 478 s at 8 passes, against 2,183 s with FastVideo's own Metal kernel in the same engine.

- **8 passes** is what the checkpoint was trained for. **4** and **20** spread that many rungs evenly over the same
  schedule: by eye on one prompt, 4 is a little softer and 20 has more texture. Neither is a setting FastVideo
  trained or tested. FastVideo's dedicated 4-step preview checkpoint was tried too and looked no better than V2 at
  4 passes for the same time, so the pack uses one checkpoint for all three.
- **2x of 480p is 1728x960**, which is above 720p, not exactly 720p. For a 1344x768 clip generated at half size,
  the Turbo path's `X2=1` draft preset does that.
- **Why it is fast.** FastH3 lets each 64-row tile of the video attend to the text and audio rows and to the best
  fifth of the video tiles. The kernel gives one threadgroup one query tile of one head and walks only the chosen
  key tiles, read in place, with int8 scores and values. At 69K rows a block takes 0.52 s against 1.93 s for dense
  attention. The int8 scores pick a few different tiles than FastVideo's float reference would, so details differ
  from its output (cosine 0.9999 on random inputs; not a bit-exact port).
- **Image to video works, though FastVideo trained FastH3 on text to video only.** One test (the baker prompts,
  864x480, 8 passes): the clip opens on the Qwen image, keeps the face and follows the prompt; image 9 s, video
  94 s. The first-frame rows join the rows every video tile always attends to. Not tested widely.
- The sound is FastH3's own; the base-model audio step is not applied here.
- FastH3 weights are a derivative of MiniMax H3 under the MiniMax H3 Community License.

## The native engine (new in 2.0)

**Credit first: Ash Hart and the TensorFold contributors** wrote the TensorFold 1.0 runtime this uses (Zig driving
Metal directly, Apache-2.0), and **FastVideo / Hao AI Lab** made FastH3. What this pack adds is an H3 family for that
runtime: FastH3's block stack, its tile-routed attention and its int8 kernels for the M5 tensor units.

| What | Where it runs |
|---|---|
| FastH3, from text or from a first frame (a Qwen scout, your own image, every part of a long chain): all transformer passes (rows in, 50 blocks, routing, attention, both output heads) | `tf-h3-dit`, the native Zig + Metal program |
| Prompt encoding, the first frame's encoding, the checkpoint's small projections and timestep tables | Python (MLX), once per clip |
| Video decoder, audio decoder, MP4 | Python (MLX), as before |
| Qwen-Image-2.1, the Turbo adapter path, the 20-step path, the base-model audio step | MLX engine, unchanged |

`scripts/video.sh` picks the native engine for a FastH3 clip when it is installed and the chip has tensor units (M5
or later); it prints `[tensorfold] engine: zig` or `engine: mlx (why)`, and the web app
shows the same on the job. `FASTH3_ENGINE=mlx` or `zig` forces one.

Measured on a Mac Studio M5 Ultra (64-core GPU, 256 GB), FastH3 with sound, text to video unless a row says from
an image, one run each, whole command including export and decode unless a row says per pass:

| Clip | Native engine | MLX engine |
|---|---:|---:|
| 864x480, 124 frames, per pass | 5.33 s | 7.12 s |
| 864x480, 5 s, 4 passes | 43 s | 51 s |
| 864x480, 5 s, 8 passes | 66 s | 78 s |
| 864x480, 5 s, 20 passes | 129 s | 174 s |
| 864x480, 124 frames, from a Qwen-Image first frame, per pass | 6.14 s | 8.08 s |
| 864x480, 5 s, 8 passes, from a first frame (image already made) | 72 s | 90 s |
| 1728x960 through the 2x decoder, 5 s, 8 passes, from a first frame | 77 s | not run |
| 18 s chain of two clips, 864x480, 4 passes | 156 s | 183 s |
| 1280x720, 5 s, 8 passes | 199 s | 259 s |
| 1280x720, 10 s (243 frames), 8 passes | 388 s | about 480 s (53 s a pass; not re-run) |
| 1280x720, 10 s (243 frames), 20 passes | about 950 s (45 s a pass plus export and decode) | not run |

The two 1280x720, 5 s figures come from one sitting with another job using the machine on and off, so both are
slower than a quiet run would be (the MLX figure was 192 s when first measured); read them as a ratio.

- **Quality.** The native engine and the MLX engine both use int8 arithmetic and agree with a float reference to the
  same degree on the first pass (video cosine 0.961 and 0.966). The same seed gives the same shot with different
  details, not the same pixels. The clips were judged by eye by the pack's owner; the native engine's attention
  weights are 8-bit.
- **20 passes** is the setting to reach for when a prompt has several distinct actions: in two tests it brought in
  action the 8-pass clip had skipped and removed an object glitch. It is not a setting FastVideo trained.
- **Sizes.** The routing cuts the video into tiles of 4x4x4 tokens (128 pixels a side, 17 frames deep). A size that
  is not whole tiles is padded inside the attention. Rounding the size up instead is not faster at the same length
  (896x512 took 5.75 s a pass against 5.33 s for 864x480 at 124 frames), so the studio generates the size you ask
  for. Lengths of 107, 175, 243 and 311 frames are whole tiles in time; 243 is the 10 second choice.
- **From an image.** A first frame's rows sit between the text and the audio, enter on every pass at the keyframe
  noise level and are never stepped, as in the MLX engine. On the baker test the clip opens on the given image
  (frame 0 against the image: 33.7 dB on both engines) and the first pass agrees with the float reference as closely
  as the MLX engine's does (video cosine 0.995 and 0.997). One image and prompt at 864x480, plus the app tests.
- **Not in the native engine yet:** the prompt encoder, the first-frame encoder, the decoders, Qwen-Image. A Qwen-Image
  transformer for the same runtime exists in the fork and is level with the MLX one (0.483 s against 0.477 s a pass
  at 1344x768), so the studio keeps the MLX one.
- **Install.** `oneshot-setup.sh` copies the engine from the carrier image's payload (a 0.7 MB program, its Metal
  kernels inside it, compiled when it starts). Without Docker it builds from source if `zig` 0.17 and Xcode's Metal
  toolchain are present (`scripts/build-zig-engine.sh`); with neither, FastH3 stays on the MLX engine and everything
  still works.

## How it works

1. **You write two prompts**: what the picture shows, and what happens in the clip (action, spoken or sung words,
   sound).
2. **Qwen-Image-2.1-Turbo makes the first frame** from the first prompt, in about 9 seconds (8 steps). Or it
   makes several scouts for you to choose from.
3. **MiniMax H3 animates that frame** from the second prompt, with sound, in one of two modes:
   - **Standard (Turbo 5).** The Turbo adapter denoises picture and sound together in 5 passes. The adapter is then
     dropped and the base model denoises the sound again from scratch, 20 steps, with the finished picture held
     fixed. Each of those steps recomputes only the audio rows (about 4% of the sequence) against the picture's
     stored attention keys and values, so 20 steps cost less than one full pass.
   - **High quality (`QUALITY=high`).** The base model denoises picture and sound together for 20 steps, no adapter.
     About three times slower; calmer motion, softer picture, and the sound the standard mode's audio step is
     borrowed from.
4. **The decoders finish it.** The video decoder turns latents into frames, optionally at twice the size (the 2x
   decoder, for 2K). The audio decoder's output is kept clear of clipping, and a take that comes out thin gets its
   bass lifted.

Both transformers, the samplers, the adapter merges, the audio step, the int8 kernels and the image and video
decoders run in TensorFold on MLX.

**Scout, pick, animate, finish in 2K.** Six scout images from one prompt in 27 seconds; pick one; MiniMax H3 animates
it with sound and a 2x decoder finishes it in 2K: an 8 second clip at 2048x1152 in five minutes, or at 2560x1440 in
under eleven. A 1344x768 draft of the same 8 seconds takes 108 seconds.

![Six scouts from one prompt](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/scout_sheet.jpg?raw=true)

![Frames of the 2560x1440 clip, and a full-resolution crop](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/qhd_frames.jpg?raw=true)

The newest sample, made with the standard setting: [`samples/singer_standard_8s.mp4`](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/singer_standard_8s.mp4?raw=true)
(prompts `singer-image.txt` and `singer-video-8s.txt`). The baker clips below predate 1.5.

Prompts: [`prompts/baker-image.txt`](prompts/baker-image.txt) for the picture,
[`prompts/baker-video.txt`](prompts/baker-video.txt) for what happens. Clips: [`samples/` in the Studio repository](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/tree/16c8972/samples).

## The base-model audio step

*This section and the Turbo rows in "How it works" and "Measured" describe MiniMax H3 Turbo, which since 2.1 is
installed only with `oneshot-setup.sh --turbo`. FastH3, the default, makes its picture and sound together.*

The Turbo adapter is what makes the picture fast, and it is also what spoils the sound. So the sound is made twice:
once with the picture, by the adapter, and thrown away; then again by the base model, which is the one that sounds
right.

1. After the Turbo passes, the adapter-merged model is unloaded and the released weights are loaded (4 s).
2. The finished picture, the first frame and the prompt go through the model once, held still, and every block's
   attention keys and values over them are stored.
3. The audio starts again from noise and is denoised in 20 steps. Each step runs only the audio rows (1,206 of 29,009
   on a 15 second clip) against the stored keys and values: about 1.3 s a step instead of 24 s.

It is on by default whenever the adapter is used. `REVOICE=0` keeps the adapter's own sound; `REVOICE=<steps>` sets the
number of audio steps (20 is the only value tried). It was offered upstream (ashhart/TensorFold#405) and closed with the H3 family; it lives in the fork.

## The workflow

```bash
# 1. Scout: several first frames from one prompt, one model load (27 s for six), with a contact sheet
bash scripts/scout.sh "A lighthouse keeper in a yellow raincoat on a cliff at dusk, storm clouds behind him" 6 scouts/keeper
open scouts/keeper/sheet.jpg

# 2. Draft the motion from the one you like: 8 s at 1344x768 in under two minutes
IMAGE_FILE=scouts/keeper/scout_s3.png X2=1 FRAMES=192 bash scripts/studio.sh "" \
    "He raises a lantern, wind pulls at his coat, waves crash below. He says: Storm's coming." draft.mp4

# 3. Finish in 2K: same image, same prompt. TWOK=1 is 2048x1152 (5 min), QHD=1 is 2560x1440 (11 min)
IMAGE_FILE=scouts/keeper/scout_s3.png QHD=1 FRAMES=192 bash scripts/studio.sh "" \
    "He raises a lantern, wind pulls at his coat, waves crash below. He says: Storm's coming." final.mp4
```

Or all at once, with no picking: `TWOK=1 bash scripts/studio.sh "the picture" "what happens" out.mp4`.

- **Scouts are made at 1280x736**, which is exactly the frame the video model starts from for a 2560x1440 clip, so
  the scout you pick is the frame the clip opens on, with no resize in between.
- **Why 736 and not 720.** The video model needs sides in multiples of 32, and 720 is not one. The 2K preset
  generates at 1280x736, decodes at 2x to 2560x1472 and centre-crops 16 rows top and bottom to 2560x1440.
- **The draft and the final are different renders.** The draft generates at 672x384 and the final at 1280x736, so the
  motion is similar in kind, not frame for frame.

## Measured (2026-10-04, Mac Studio M5 Ultra 256 GB, macOS 27.0.1, one run each)


**Notes and Regenerate.** Under every take, in Shots and in Deliver, there is a box for notes or feedback and a
"↻ Regenerate" button. With notes, the brain revises the shot's prompt to answer them (the old prompt goes to History)
and a new take is queued; with the box empty, the same prompt is rendered at the next seed. Earlier takes are kept.

### Text to image to video (`scripts/studio.sh`), 8 second clips unless noted

| Output | Video generated at | Standard: Turbo 5 + base-model sound | Earlier: Turbo 3, adapter sound |
|---|---|---:|---:|
| Six scout images, 1280x736 (`scripts/scout.sh`) | | 27 s | 27 s |
| 1344x768, 2x decoder (`X2=1`) | 672x384 | **108 s** | 67 s |
| **2048x1152, 2x decoder (`TWOK=1`)** | 1024x576 | **299 s** | 151 s |
| **2560x1440, 2x decoder (`QHD=1`)** | 1280x736 | **635 s** | 352 s |
| 1344x768, native decode | 1344x768 | 709 s | 364 s |
| 864x480, native decode | 864x480 | 174 s | 98 s |
| 864x480, 5 s (124 frames), native decode | 864x480 | 103 s | 74 s |

Totals include the image (9-12 s). Where the standard time goes, 2048x1152: five passes 185 s (37 s each), the sound
made again 60 s, decode 31 s, loading and text 12 s. At 2560x1440: passes 436 s, sound 119 s, decode 55 s. The audio
step costs about one and a third passes at every size: one full pass to store the picture's keys and values, then 20
light steps.

`QUALITY=high` (20 steps, no adapter) took 606 s for a 15 second clip generated at 672x384, against 219 s at the
standard setting; it has not been timed at the sizes above.

The earlier column was measured with `POINTS=4 REVOICE=0`, which still works.

**Resolution costs far more than length.** The video model works on one row per 32x32 pixels of every latent frame,
and attention compares every row with every other, so its cost grows with the square of the row count. 5 s at 864x480
is 16,500 rows (9.5 s per pass); 8 s at 864x480 is 24,826 rows (18.3 s per pass); 8 s at 1344x768 is 60,403 rows
(104.6 s per pass). Going from 864x480 to 1344x768 is 2.5 times the pixels and 4 times the time for the same length.

**What the 2x decoder is.** A replacement head for H3's own video decoder (speach1sdef178's MiniMax-H3-X2-Detail-VAE)
that turns the same latents into frames twice as large along each side, in the normal decode time. It is an upscale: a
1344x768 clip decoded from a 672x384 generation is softer than one generated at 1344x768, with stair-steps on
high-contrast edges ([comparison](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/x2_compare.jpg?raw=true): native on top, 2x below). It earns its place twice: as a
6 times faster draft, and as the last step to 2K from a full-size generation, where the 2560x1440 clip costs no more
than the 1344x768 one. With `QHD=1` or `TWOK=1` the first frame is made at the size the video model starts from
(1280x736 or 1024x576), since a larger picture would only be shrunk again.

Wall time from the command to the finished files, both models loaded from disk each time, int8 kernels on. In every clip the baker lifts the loaf, speaks the scripted line
(checked by speech recognition) and smiles. The 864x480 run was the first after install, so its 14 s includes a cold
read of the text encoder.

### Qwen-Image-2.1 alone, 1344x768, same prompt and seed

| Path | Steps | Per step | Denoise | Whole run |
|---|---:|---:|---:|---:|
| mflux `add5164`, bfloat16 (the reference this port is checked against) | 40 | 0.73 s | 29 s | 40.7 s |
| TensorFold, bfloat16 | 40 | 0.78 s | 31.0 s | 34.3 s |
| TensorFold, int8 kernels | 40 | 0.48 s | 19.2 s | 24.3 s |
| TensorFold, int8 kernels + Viggle turbo adapter (the default up to 2.2) | 6 | 0.50 s | 3.0 s | 9.2 s |
| **TensorFold, int8 kernels, Qwen-Image-2.1-Turbo (2.3, measured 2026-10-09; peak 15 GiB against 28)** | **8** | 0.49 s | **3.9 s** | **8.9 s** |

![mflux bf16 40 steps / TensorFold bf16 40 steps / TensorFold int8 40 steps / TensorFold int8 turbo 6 steps](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/qwen_image_compare.jpg?raw=true)

Top: mflux, TensorFold bfloat16. Bottom: TensorFold int8, TensorFold int8 turbo. Against the mflux image the
TensorFold bfloat16 image is 32.7 dB and the int8 image 30.4 dB: the same picture with different fine detail. Turbo
gives the same scene with a different rendering.

- **Per step in bfloat16, mflux is slightly faster than this port** (0.73 s against 0.78 s). The gain comes from the
  int8 kernels (1.6x per step) and the 6-step adapter.
- **MiniMax H3 numbers**, the shootout against h3.c, H3MLX and minimax-h3-mlx, and what makes H3 fast are in the
  video-only pack: [keys-Mac-TensorFold-MiniMax-H3-MLX](https://github.com/drowzeys/keys-Mac-TensorFold-MiniMax-H3-MLX).
  At 20 steps the C engines are faster and sharper than TensorFold; TensorFold leads only with the Turbo adapter.

## One-shot

```bash
git clone https://github.com/ashhart/TensorFold.git
cd TensorFold/studio
brew install python@3.11 uv ffmpeg
bash oneshot-setup.sh               # both models and the 2x decoder: 182 GB of weights
bash oneshot-setup.sh --image-only  # or just Qwen-Image-2.1: 33 GB
```

`oneshot-setup.sh` does the following:

1. Gets TensorFold 0.6.5 with the H3 and Qwen-Image families (`drowzeys/TensorFold` at `a2068c03`), from the **GHCR
   prebuilt carrier** when Docker is available (checksums verified), otherwise from git at the same commit.
   The native FastH3 engine (`tf-h3-dit`) comes from the same carrier, or without Docker as a checksummed download
   from the [v2.0 release](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/releases/tag/v2.0); no Zig toolchain is needed.
2. Installs it with mflux at `add5164e` and the dependency lock ([`requirements.lock`](requirements.lock): mlx 0.32.3,
   mlx-lm 0.32.0, mlx-vlm 0.7.4, …) into its own venv at `~/.local/opt/tensorfold-studio`.
3. Downloads `Qwen/Qwen-Image-2.1-Turbo` (33 GB) to `~/qwen-models/Qwen-Image-2.1-Turbo`. Where an earlier install's
   `Qwen-Image-2.1` sits beside it, only the transformer (14 GB) is fetched and the rest is linked.
4. Unless `--image-only`: clones minimax-h3-mlx at `79190205`, downloads the FastH3 transformer (65 GB), the MiniMax
   H3 `FL2VA` partition without its transformer (72 GB: the text encoder and the two decoders) to
   `~/h3-models/MiniMax-H3` and the 2x video decoder (5.2 GB, checksum verified). With `--turbo`, the whole
   partition (134 GB) and the video Turbo adapter (1.96 GB, checksum verified).
5. Renders a test: a 5 second clip from a generated image (`outputs/test.png`, `outputs/test.mp4`), or a test image
   with `--image-only`.

Override `PREFIX`, `QWEN_MODEL_DIR` or `H3_MODEL_DIR` through the environment. `--verify` checks an existing install;
`--no-render` skips the test.

`studio.sh` and `video.sh` use MiniMax H3 Turbo when it is installed (`--turbo`) and FastH3 otherwise; `ENGINE=fasth3`
or `ENGINE=h3` picks one. `QUALITY=high` and `QUALITY=full` need `--turbo`. For FastH3 the sizes measured here are 864x480,
1280x736, 1344x768, and 672x384 and 1024x576 with the 2x decoder (the app's Draft and 2K cards: 44 s and 93 s for 5 seconds).

### Text to image to video

```bash
bash scripts/studio.sh "A lighthouse keeper in a yellow raincoat on a cliff at dusk, storm clouds behind him" \
                       "He raises a lantern, wind pulls at his coat, waves crash below. He says: Storm's coming." \
                       out.mp4
```

The image is written beside the clip (`out.png`). Defaults: 1344x768, 124 frames (5 s), native decode. `QHD=1` makes
a 2560x1440 clip; `X2=1` generates the video at half of `WIDTH` x `HEIGHT` and decodes at 2x (multiples of 64, up to
2688x1536); `IMAGE_FILE=picture.png` animates an image you already have. Set `FRAMES=192` for 8 seconds, `WIDTH`/`HEIGHT` for another canvas (multiples of 32, at most 768x1344 pixels in total, the video model's
released limit), `SEED`, or `IMAGE_SEED` and `VIDEO_SEED` separately. The script opens the video prompt with the line
MiniMax's own image-to-video prompts use; `RAW_PROMPT=1` passes yours untouched.

### Image only

```bash
bash scripts/image.sh "A neon shop sign that reads OPEN LATE, rainy night, reflections on wet pavement" out.png
TURBO=0 bash scripts/image.sh "..." out.png                 # 40 steps, base model
WIDTH=2048 HEIGHT=2048 bash scripts/image.sh "..." out.png  # sizes in multiples of 16
IMAGE_MODEL=flux3k-fast bash scripts/image.sh "..." out.png # FLUX3-K Fast, 16 steps (oneshot-setup.sh --flux3)
IMAGE_MODEL=flux3k-full bash scripts/image.sh "..." out.png # FLUX3-K Full, 50 steps at CFG 3.5
IMAGE_MODEL=flux3k-fast UPSAMPLE=0 bash scripts/image.sh "a full scene caption ..." out.png   # no prompt upsampler
```

### Edit a picture

```bash
bash scripts/edit.sh photo.png "The same statue at night under a full moon" night.png
LORA=~/TensorFoldStudio/loras/qwen-multiple-angles.safetensors:0.9 \
  bash scripts/edit.sh photo.png "<mva> right side view, high-angle shot" side.png
SET_FILE=edits.json bash scripts/edit.sh photo.png          # [["prompt", "out.png"], ...] from one load
BASE=1 bash scripts/edit.sh photo.png "..." out.png         # 40 steps, base model (--image-edit)
IMAGES="person.png" bash scripts/edit.sh room.png "The person of image 2 sits on the sofa of image 1" out.png   # up to nine more, one a line
SIZE=model RESOLUTION=1024 bash scripts/edit.sh photo.png "..." out.png   # the model's own size; the default is the picture's own
IMAGE_MODEL=flux3k-fast IMAGES="person.png" bash scripts/edit.sh room.png "..." out.png   # FLUX3-K Fast (flux3k-full: Full), --flux3
```

The result has the size of the first picture (`WIDTH` and `HEIGHT` ask for another). The noise is drawn for `SEED`
and the pictures together, so the seed a picture was made with is as good as any other for editing it.

### Video only, or from your own image

```bash
bash scripts/video.sh "A hummingbird hovering over red flowers, soft wing hum" out.mp4
FIRST_FRAME=photo.jpg WIDTH=1344 HEIGHT=768 bash scripts/video.sh "She turns to the camera and smiles" out.mp4
QUALITY=high bash scripts/video.sh "..." out.mp4            # 20 steps, no adapter
```

`FRAMES` must be `17n + 5` (56, 73, 90, 124, 192, 243, 362). `FIRST_FRAME` is stretched onto the canvas, so match its
aspect ratio.

### A last frame: the clip ends on a picture

MiniMax H3's first-and-last-frame mode holds a picture at the end of the clip the way it holds one at the start: the
end keyframe is encoded as a condition latent at the clip's last latent frame and kept at every pass. `LAST_FRAME`
works with `FIRST_FRAME`, or alone (the clip then arrives at the picture). Beside a first frame the last one is fitted
by a centre crop, so give both the clip's aspect ratio. A keyframe is read by the model, not written into the clip:
in one production run 63 of 64 were held and one first frame was not, and FastH3 cut to a close-up inside 9 of 32
such clips. The app checks both after the render (see [the shot check](#what-the-app-tells-you-about-a-prompt-and-a-clip)).

```bash
FIRST_FRAME=teapot.png LAST_FRAME=bonsai.png bash scripts/video.sh "The teapot opens and a bonsai grows out of it" out.mp4
LAST_FRAME=bonsai.png bash scripts/video.sh "A potter's table; a bonsai is set down" out.mp4      # arrives at the picture
FIRST_FRAME=statue.png LAST_FRAME=statue.png WIDTH=768 HEIGHT=768 FRAMES=73 LORA="orbit.safetensors:1.0" \
  bash scripts/video.sh "One frozen instant. Only the camera moves. In a continuous 360 orbit. ..." out.mp4
```

In the app, step 2 of **Create** has a **Last frame** slot under the first frame. Ways to fill it: upload a picture;
*Pick from the Library*; *Same as the first frame* (an orbit or a loop); *From a clip in the Library* (its last
frame, its first frame, or the frame at a time you type: `POST /api/frame` saves the still to the Library at the
clip's own resolution); or *make one from a prompt* with Qwen-Image-2.1. Beside *Upload an image*, *Or continue from a
clip in the Library* puts a clip's last frame in as the new first frame; in the **Library**, *continue* on a video
and *last frame* on a picture do the same in one press. Step 4 opens the prompt with a line that places `<Picture 1>`
at 0.00 seconds and `<Picture 2>` at the clip's end (MiniMax documents that line for a first frame only; the two-picture
form here follows its wording and is our reading, and the keyframes are held whatever the line says). A LoRA from the
hub that needs both ends asks for the last frame in step 2, and *Queue this clip* waits for it. A job takes
`last_image` (a picture in the studio folder), `last_same` (the first frame again) or `last_frame_from` (a clip: its
first frame). Not for reference clips (a reference is not a frame of the clip) and not for videos longer than 15 s.

![The Last frame slot in step 2](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_last_frame.png?raw=true)

Measured 2026-10-12, one seed each, 5 s at 864x480 unless noted; SSIM is the clip's final frame against the picture
it was given (0.3 to 0.5 is what an unrelated frame of the same scene scores):

| Machine | Engine | Request | Time | Final frame vs the picture | By eye |
|---|---|---|---|---|---|
| Mac | MiniMax H3, Turbo 5 passes | first + last | 105 s | 0.86 | the teapot becomes the bonsai and the clip ends on the picture |
| Mac | the same | first only (same seed) | 101 s | 0.46 | a tree grows, in another pot |
| Mac | the same | last only | 105 s | 0.86 | another teapot, arriving at the picture |
| Mac | FastH3 8 passes (MLX engine) | first + last | 103 s | 0.86 | as the first row |
| DGX Spark | FastH3 8 passes, TensorFold int8 blocks | first + last | 130 s | 0.90 | ends on the picture |
| DGX Spark | the same | first only (same seed) | 86 s | 0.32 | a tree grows, in the teapot |
| DGX Spark | the same | last only | 85 s | 0.90 | arrives at the picture |
| DGX Spark | the same, queued through the app's job API (`image` + `last_image`, prompt from `/api/compose`) | first + last | 126 s | 0.90 | ends on the picture |

![First and last frame, first only, last only (DGX Spark), last only and FastH3 first and last (Mac)](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/last_frame_plain.jpg?raw=true)

Two LoRAs from the hub that need both ends, same seed with and without, the authors' prompts and strength 1.0:

| LoRA | Machine | Engine | Time | With the LoRA | Without |
|---|---|---|---|---|---|
| 360° Orbit (first = last, 768x768, 73 frames) | Mac | MiniMax H3, Turbo 5 passes | 86 s without; with it the wall time was not clean (the run queued behind other renders) | a full orbit of the statue, back to the start | frozen |
| the same | DGX Spark | MiniMax H3 bf16, 25 passes (`SPARK_BASE=h3`) | 324 s | a full orbit | frozen |
| the same | DGX Spark | FastH3 bf16, 8 passes | 147 s | a full orbit | not run |
| Transformation (teapot to bonsai) | Mac | MiniMax H3, Turbo 5 passes | 108 s | the teapot dissolves into painted ribbons that form the tree | a cross-fade through a plant pot |
| the same | DGX Spark | MiniMax H3 bf16, 25 passes | 362 s | the same ribbons | a cut through another object |
| the same | DGX Spark | FastH3 bf16, 8 passes | 158 s | the same ribbons | not run |

![360° Orbit: Spark MiniMax H3 with and without the LoRA, Spark FastH3 with it, Mac with and without](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/last_frame_orbit.jpg?raw=true)
![Transformation: the same five rows](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/last_frame_transformation.jpg?raw=true)

On a Mac the zig engine does not hold an end keyframe, so a FastH3 clip with a last frame runs on the MLX engine
(the script says so). On a DGX Spark the last frame works on both block engines. Both LoRAs were trained on MiniMax
H3's own first-and-last-frame transformer, which a Spark install does not have by default (it renders on FastH3, where
both worked here): `bash oneshot-setup-spark.sh --h3-base` fetches it (40 GB) and `SPARK_BASE=h3` renders a clip on
it, 25 passes by default, on ComfyUI's own blocks with dense attention. Not tested: lengths other than 3 and 5 s,
sizes above 864x480, a last frame with the Qwen scout route in one job, and *make one from a prompt* in a browser.

### Another text encoder, your own size and passes

From the scripts, `TEXT_ENCODER` may be the path of another MiniMax H3 text encoder file (on a DGX Spark, the name
of a file in ComfyUI's `text_encoders` folder); unset, it is MiniMax's own. On the Mac FastH3 then runs on the MLX
engine, not the native one.
In the app, *Text encoder* is *Regular* (MiniMax H3's own), any file found in the `text_encoders` folder beside the
model (by its file name), or *Your own*: the path of a MiniMax H3 text encoder file on your machine (on a DGX Spark,
the name of a file in ComfyUI's `text_encoders` folder). The Studio ships MiniMax H3's own text encoder and no other;
it does not provide, check or recommend the file you choose. That file, its license and what you make with it are
your own choice and responsibility.

In the app, *Resolution → Custom* takes a width and a height (multiples of 32, each side 256 to 1536, at most
1344x768 pixels in all). A number of passes of your own, from 1 to 50, is the *Custom* card beside *20 passes* in each
passes choice (FastH3's *Passes*, the reference choice, and MiniMax H3's *Quality*, where it follows the two
20-pass recipes and keeps the one that was chosen); a number box opens under the cards. No time is estimated for either. From
the scripts these are `WIDTH`, `HEIGHT` and `STEPS` (FastH3, and everything on a Spark) or `POINTS` (MiniMax H3 on
the Mac: one more than the passes).

### Reference pictures to video (Ref2VA)

A first frame is the clip's opening picture. A reference picture is not in the clip at all: it fixes how a person or
a thing looks, and the prompt puts them in a new scene. This is MiniMax H3's second transformer, Ref2VA, installed
with `bash oneshot-setup.sh --ref2va` (62 GB; the text encoder and the two decoders are the ones already installed,
linked into `Ref2VA/`).

```bash
REF_IMAGES="face.jpg" bash scripts/video.sh \
  "Use <Picture 1> as the person. The person walks through a rainy neon street, turns to the camera and says hello." out.mp4
REF_IMAGES="face.jpg bike.png" bash scripts/video.sh "Use <Picture 1> as the person and <Picture 2> as the bicycle. ..." out.mp4
REF_TURBO=1 REF_IMAGES="face.jpg" bash scripts/video.sh "Use <Picture 1> as the person. ..." out.mp4   # 4 passes
```

- Up to nine pictures, named `<Picture 1>`, `<Picture 2>`, ... in the order given. Each is used at the clip's pixel
  area with its own aspect ratio (a 768x1024 portrait becomes 544x736 for an 864x480 clip), so nothing is stretched.
- 20 plain passes with the model's own sound by default: 3 min 43 s for 5 s at 864x480 on an M5 Ultra (9.7 s a pass
  over 16.2K rows, 64 GiB peak). `REF_TURBO=1` uses LightX2V's 4-step Ref2VA adapter and then makes the sound again
  without it: 1 min 28 s, 103 GiB peak while the adapter is merged. `QUALITY=high` runs the 20 passes with the fast
  recipe: 1 min 58 s. Two pictures (a person and a bicycle) took 4 min 5 s at 20 passes and 1 min 29 s with the adapter.
- Not with `FIRST_FRAME`, and not with FastH3, which has no reference-trained weights.
- **Your own number of passes:** `STEPS=12` (or `POINTS=13`, one more, which wins when both are set), and the
  *Custom* card of the passes choice in the app. It applies to every kind of reference, and with `REF_TURBO=1` too: the adapter then runs the
  number asked for instead of the 4 it was trained for, and the render says so. A reference render starts by naming
  the passes it will run, and the app's job card shows them. Before this, `STEPS` was read for FastH3 only and a
  reference render from the scripts ignored it; the app already converted its field.
- In the app: **The clip starts from → Reference pictures**. Step 4 writes the prompt in the model's own reference
  form (`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, and from 2.6 the sound under
  its own `overall_soundscape` heading); edit it freely.
- **On a DGX Spark** the same `REF_IMAGES` and `REF_TURBO=1` work after `bash oneshot-setup-spark.sh --ref2va` (39 GB:
  the Ref2VA transformer for ComfyUI in bf16 and LightX2V's adapter). Measured on one Spark, 5 s at 864x480, one
  portrait: 20 steps 288 s, `REF_TURBO=1` (4 passes, the adapter's own sound) 107 s; the app's *Reference pictures*
  start works there too (101 s). The passes run on ComfyUI's own blocks, not on TensorFold's CUDA engine. Free memory
  fell to 7 GB of 128 during these renders, with a browser and the app open: close what you can first. One picture
  was tested there, not several.
- Checked on one fictional portrait: the same face, hair and glasses in the clip, against a different person from the
  same prompt and seed without the reference, and the same identity as h3.c's `--ref-image` gives. The token layout
  matches h3.c's to the last float32 bit. Not checked: sound by ear, lengths over 5 s. Reference videos and
  reference audio are the next section.
- **What the text encoder sees of a picture (corrected after 2.5).** A first frame or a reference picture also goes
  to the text encoder's vision tower. Up to 2.5 it was normalised there with the mean and deviation of the earlier
  Qwen-VL models, which minimax-h3-mlx builds its image processor with; MiniMax H3's own processor
  (`processor/preprocessor_config.json` in the checkpoint) uses 0.5 and 0.5, pixels to -1..1. The renderer now reads
  that file (0.5 and 0.5 when it is absent). For one 864x480 picture the values handed to the tower were up to 1.12
  away from MiniMax's processor before (0.39 on average, range -1.79..2.09) and are equal to it now, and to h3.c's
  `x * 2 - 1`. The video VAE's view of the picture, which carries most of the likeness, was never affected. Same
  prompt and seed, before and after, one first-frame clip (Turbo, 5 passes) and one reference-picture clip (20
  passes), 5 s at 864x480: a different take each time, as any change to the conditioning gives, with the same
  person, colours and sharpness; we could not see a gain or a loss. `EXTRA="--vision-stats mlx"` renders with the
  old values. This covers MiniMax H3 and FastH3 on the Mac, both engines; the DGX Spark build encodes text in
  ComfyUI, which already used 0.5 and 0.5. The sample clips and timings in this README from first frames and
  reference pictures were made with the old values: the timings stand, the clips would come out as other takes.

### Reference videos and reference audio (Ref2VA)

The same Ref2VA transformer takes two more kinds of reference, as MiniMax's model card describes: videos, which
carry a subject and how it moves, and audio clips, which carry a voice or music. **On a Mac** the passes need TensorFold's H3 family to pack these references and to hold reference audio rows, which is on
the engine branch `ref-video-audio` of the fork this pack installs from (pinned to `a2b592f` since 2.6.1). An install
made with 2.6 or earlier has the older engine, with which the renderer stops with a message saying so: run
`oneshot-setup.sh` again and it replaces the engine. Reference pictures alone are unaffected. **On a DGX Spark** they run now,
through ComfyUI's node: see [below](#reference-videos-audio-and-motion-on-a-dgx-spark).

```bash
REF_VIDEOS="baker.mp4" bash scripts/video.sh \
  "<Subject 1> is the man in the green apron in <Video 1>. ... the bakery is replaced by a market stall ..." out.mp4
REF_IMAGES="singer.png" REF_VIDEOS="baker.mp4" bash scripts/video.sh \
  "<Subject 1> is the woman shown in <Picture 1>. <Video 1> is the motion reference. ..." out.mp4
REF_IMAGES="singer.png" REF_AUDIOS="voice.wav" bash scripts/video.sh \
  "<Subject 1> is the woman shown in <Picture 1>. <Audio 1> is the voice timbre reference for her voice. ..." out.mp4
REF_VIDEOS="baker.mp4" REF_VIDEO_SOUND=1 bash scripts/video.sh "..." out.mp4   # the video's own sound is <Audio 1>
```

- **Limits (MiniMax's):** up to 9 pictures, 3 videos and 3 audio clips, 12 files in all; each video and each audio
  clip 2 to 15 seconds, and 15 seconds in all per kind; audio never alone (it needs a picture or a video beside it).
  The scripts and the app refuse a request outside these and say which file or count is wrong.
- **Names in the prompt:** `<Picture N>`, `<Video N>`, `<Audio N>`, each numbered in the order given. The model is
  always handed pictures first, then videos, then audio clips. With `REF_VIDEO_SOUND=1` every video's own soundtrack
  is a reference as well and those sounds are numbered first: with one video, its sound is `<Audio 1>` and the first
  `REF_AUDIOS` clip is `<Audio 2>`. Without it a video's sound is ignored. The render prints the name of every file.
- **What is read:** a video at 24 fps up to the clip's own length, cut down to `17 n + 5` frames, at its own aspect
  ratio and brought down to the clip's pixel area (`REF_VIDEO_SIZE=full` keeps the model's 768-pixel canvas; not
  tested). Audio at 32 kHz stereo, cut to the clip's length. `ffmpeg` and `ffprobe` do the reading.
- **Cost:** every reference latent rides through every pass. Measured on an M5 Ultra, 5 s at 864x480, 20 passes:
  one 5 s reference video 12 min 32 s (34.9 s a pass over 33.2K rows), a picture and a 5 s video 13 min 36 s, a
  picture and a 5 s audio clip 4 min 10 s (11.3 s a pass over 17.0K rows), against 4 min 6 s with that picture alone;
  64 GiB peak in each. `REF_TURBO=1` with a picture and an audio clip: 1 min 35 s, 103 GiB peak. A shorter reference
  video costs less; none was timed.
- **In the app:** *The clip starts from → References*; step 2 takes pictures, videos and audio, shows each one's name,
  and step 4 writes the prompt in the model's own form (`<Video 1> is the reference video for ...`,
  `<Audio 1> is the voice timbre reference for ...`). No time is estimated when a video or an audio clip is used.
- **On a DGX Spark** the same variables, limits and messages: see the next section.
- **How it was checked.** The token layout (row order, positions, which rows are held) is equal to the last float32
  bit to MiniMax's own `build_ref2va_packed_sequence` and to h3.c's `h3_layout_build`, on six mixed requests. The
  prompt's token sequence equals MiniMax's `build_ref2va_presentation`; the frame sampling for the text encoder,
  the frames kept for the video VAE and the frame size rule equal MiniMax's functions. Reference rows decode back to
  what went in (video 38.7 dB PSNR, audio loudness envelope correlation 0.999). Two differences from MiniMax's
  pipeline remain, both shared with the picture path: the video latent is the middle of the encoder's distribution,
  not a draw from it, and a reference is not enlarged to the 768-pixel canvas.
- **What the renders showed** (fictional people from earlier renders of this pack; 5 s, 864x480, 20 passes, one seed):
  - *A reference video* (a baker lifting a loaf in a bakery), with a prompt that moves him to a market stall: the
    same man, clothes and motion, second for second, in front of a market. The same prompt and seed without the
    video gave another man holding the loaf up throughout.
  - *A picture and a video* (a woman's still, the baker clip as motion reference): the woman, in the baker's
    bakery, doing his motion second for second. Her mouth also moves as his does, though the prompt said nobody
    speaks.
  - *A video with its own sound* (`REF_VIDEO_SOUND=1`): the clip's sound follows the reference closely (pitch 120 Hz
    against 122, spectrogram correlation 0.86 moment by moment, against 0.43 for the run without the sound). With
    the app's prompt and a picture beside the video: 0.99 at 20 passes. **With `REF_TURBO=1` this was broken until
    the engine's re-voice was corrected** (branch `ref-video-audio`, `a2b592f`): the sound made again after the
    adapter's passes came out near silent, correlation 0.22, where the adapter's own sound had 0.98. The audio-only
    re-voice kept the held rows beside the clip's finished audio; with reference audio in the sequence they are now
    kept beside the starting noise: 0.98, same time. A voice clip as `<Audio 1>` through Turbo was not affected
    (139 Hz and distance 6.7 after, 140 Hz and 6.6 before; 193 Hz and 13.8 without the clip).
  - *A picture and an audio clip of the same kind of voice* (a woman's still, 6 s of a woman singing, the prompt
    asks her to sing): median pitch 281 Hz, the reference's own 281, against 314 without the clip; closer in
    timbre statistics as well (4.9 against 6.7).
  - *A picture and a voice that does not fit it* (the woman's still, a man's voice as `<Audio 1>`): **it carries
    when the prompt describes the voice, and not when the prompt is silent about it.** Measured against a clean
    5.8 s male voice at 117 Hz (the macOS speech voice "Daniel"; distance = MFCC statistics to the reference, lower
    is closer; 20 passes, one seed):

    | prompt | with the clip | same prompt without it |
    |---|---|---|
    | says only "the voice timbre references `<Audio 1>`" | 225 Hz, distance 14.1 (no carry-over) | 182 Hz, 11.7 |
    | also calls it a deep male voice (written by hand) | 137 Hz, 5.0 | 169 Hz, 11.6 |
    | the app's prompt with *Voice* = "a deep male voice" | 130 to 134 Hz, 4.1 to 4.8 | 141 Hz, 10.1 |
    | one plain sentence: "speaks with the voice in `<Audio 1>`, a deep male voice" | 132 Hz, 3.9 | not run |

    The other way round (a man's still, a woman's voice at 180 Hz, the prompt silent about the voice) it carried
    as well: 174 Hz and distance 4.3 with the clip, 143 Hz and 10.6 without. The first test of this pack found no
    carry-over because its prompt did not describe the voice and its reference was a noisy clip with a voice in
    a fifth of it; with the voice described, that same clip carries too (117 Hz against its own 122, distance 6.0
    against 15.9 without). So: **describe the voice in the line's *Voice* field as it sounds in the clip, and use a
    clean recording.** The pitch and the tone come close to the reference; none of this was judged by ear, and it
    is not an exact copy of a voice.
- **Not checked:** sound by ear (the figures above are measurements: pitch by autocorrelation, MFCC statistics,
  spectrogram correlation), more than one reference video or audio clip in a request, two seeds, lengths over
  5 s, references longer than the clip, `REF_VIDEO_SIZE=full`, `QUALITY=high` with a video, the app's screens by eye
  in a browser (its checks and the prompt it writes were exercised through its API, two clips were rendered
  through its queue, and the Create page's choosers and reference step were driven in a scripted DOM, not looked
  at).

### Reference videos, audio and motion on a DGX Spark

`REF_VIDEOS`, `REF_AUDIOS`, `REF_VIDEO_SOUND=1` and `STEPS` work on the Spark as on the Mac, in the
scripts and in the app, after `bash oneshot-setup-spark.sh --ref2va`: the same limits, checked before any model loads,
with the same messages. `spark/spark_generate.py` reads each reference with `ffmpeg` (a video at 24 fps, up to the
clip's own length, LANCZOS-scaled to its own aspect ratio and brought down to the clip's pixel area, stored without
loss; sound at 32 kHz stereo, cut to the clip's length) and hands it to ComfyUI's `MiniMaxH3ReferenceToVideo` node
(`ref_videos`, `ref_video_audios`, `ref_audios`), which numbers pictures, videos and audio clips as the Mac build
does. The passes run on ComfyUI's own blocks in bf16. Scaling the video before the node sees it matters: left alone,
the node takes a reference at up to 768 pixels on the short side whatever the clip being made, several times the
rows; for that reason `REF_VIDEO_SIZE=full` is refused on a Spark. The reference files are taken out of ComfyUI's
input folder again after the render. Nothing here needs the TensorFold engine change the Mac build is waiting for.

Measured on one DGX Spark (GB10, 121 GB usable), 5 s at 864x480, seed 7, fictional people from earlier renders, a
browser and the app open, each render under a watcher on free memory; "lowest free" is the lowest `MemAvailable`
seen during the render:

| Render | Passes | Time | Lowest free |
|---|---|---|---|
| One picture (as in 2.5, to see nothing changed) | Turbo 4 | 100 s | 39.8 GB |
| A picture and a 2.3 s reference video | Turbo 4 | 147 s | 39.1 GB |
| A picture and a 5 s reference video | Turbo 4 | 202 s | 38.3 GB |
| A picture and a 5 s audio clip | Turbo 4 | 98 s | 39.1 GB |
| Wizard through the API: a picture and the source clip, full scene | Turbo 4 | 208 s | 37.8 GB |
| Wizard through the API: a picture and the skeleton layer | Turbo 4 | 187 s | 37.8 GB |
| Wizard through the API: a picture and the depth layer, actor motion | Turbo 4 | 193 s | 15.4 GB (see below) |
| A 5 s reference video alone (the 8-bit transformer) | 20 | 677 s | 57.1 GB |

A 5 second reference video about doubles the time of a Turbo clip and costs under 2 GB more memory at this size; at
20 passes a pass over the clip's and the video's rows took 30.7 s (613 s of passes), the reference's encoding 46 s.
What the clips showed, besides the table: the 20-pass clip has the baker of the video, his clothes and his motion in
front of a market stall, as on the Mac. With the Turbo adapter: the woman of the picture doing the
baker's motion second for second in his bakery; the martial-arts shot repeated move for move with the woman of the
picture in it; the skeleton layer steering the scale and the camera with no skeleton look in the clip. The audio
clip (a man's spoken line beside a woman's picture) gave **no carry-over we could measure**, as on the Mac: her voice
at 192 Hz with the clip and 195 Hz without, the reference near 110 Hz.

The depth layer with a picture of the new person kept the stances and the punch toward the camera in a rooftop
drawn afresh from the prompt, with no depth-map look.

**Memory after a FastH3 clip.** The depth row, and a repeat of the source-clip row (210 s, 14.0 GB), ran right after
a FastH3 clip: the server then still holds what ComfyUI's unloading does not give back (TensorFold's engine keeps its
int8 weights), and the reference model loads on top. That is the 7 GB figure of 2.5 as well. A reference render
that follows a FastH3 clip now restarts ComfyUI first ("a reference render starts from a server with nothing
loaded"): the same picture and 5 s video then ran in 213 s, restart included, with 37.9 GB lowest free.

**What a sound reference does (measured on the Spark, 2.6).** Same picture, prompt and seed with and without the
reference, 5 s at 864x480, Turbo 4 passes unless said; pitch by autocorrelation, likeness as the correlation of the
two spectrograms moment by moment (two unrelated spoken clips measure about 0.3), words by a speech-to-text pass:

| Request | With the reference | Without it |
|---|---|---|
| A woman's picture, a recording of a woman's voice, the prompt's line is what the recording says | the recording itself: likeness 0.98, pitch 229 Hz (the recording's 229) | 188 Hz, likeness 0.28 |
| The same picture, a man's recording, the prompt's line is what he says | the recording itself: likeness 0.98, pitch 113 Hz (his 122) | not run |
| The same picture, the man's recording, another line | nothing carried: 188 Hz, likeness 0.11; the prompt's words | 188 Hz |
| The same picture, the man's recording, another line, and the voice described too ("in the deep male voice of `<Audio 1>`") | pitch 122 Hz (his 122), timbre distance 8.5; the prompt's words | 156 Hz, distance 13.4 (the description alone) |
| The same picture, the woman's recording, another line | pitch 229 Hz (the recording's), but the words came out garbled | 183 Hz, the words right |
| A man's picture, his own recording, another line | pitch 130 Hz (his 122); the prompt's words | 147 Hz |
| A video with its own sound (`REF_VIDEO_SOUND=1`) | likeness 0.81, loudness envelope 0.90, pitch 115 Hz (the video's 122); the prompt's words | likeness 0.34, envelope 0.48, 131 Hz |
| A picture, a video with its own sound, and a voice recording (`<Audio 2>`) | her voice at 216 Hz (the recording's 229), the prompt's words; neither sound reproduced | not run |

So a reference audio clip is a **performance to repeat** more than a voice to borrow: when the spoken line is what the
recording says, the clip plays the recording back with the mouth moving in time with it (mouth movement against
speech, from face landmarks: r 0.55 to 0.59, as for clips with the model's own voice), whoever is in the picture.
With other words, the voice comes across when the prompt describes it as well as naming the clip (the pitch went to
the recording's); named without a description, a voice unlike the pictured person changed nothing, and with the
Turbo adapter a voice of the same kind blurred the words. It is not voice cloning, and the app says so where the clip is added. A video's own sound carries
as on the Mac. *The videos' own sound* stays **off** by default in the app: the box swaps the sound written in step 3
for the video's own, which has to be a choice.

**A voice nobody asked for.** Up to this change the app wrote the soundscape on the end of the shot description of
a reference prompt. With some pictures the model then read it out: a clip of a woman drinking coffee, whose prompt
says that nobody speaks, had a voice talking nonsense in 41 % of its frames (63 % on a second seed, 49 % at 20
passes), not in time with her mouth. With the same words under an `overall_soundscape:` heading, as text-to-video
prompts have always had, the voice is gone (0 % on both seeds, the room tone at -40 dBFS), and a spoken line is still
spoken as written. The app now writes the heading. Guidance was tried as well (ComfyUI's CFG at 5 with an empty
negative): it also removes the voice but burns the picture with the 4-step adapter and doubles the time, and at 2 it
does nothing; the reference model is run without guidance, as MiniMax ships it.

**The sound of every clip on a Spark** is now made from a lossless copy of the decoder's output (it used to be
ComfyUI's own AAC track at about 130 kb/s): 256 kb/s, cut to the picture's length, and limited to -1 dBFS when the
decoded peak is higher (it usually is: speech comes out of the decoder touching full scale, and the lossy track
then overshot it, 31 samples over full scale in one test clip). The render says what it found
(`[tensorfold] sound: peak -0.5 dBFS as decoded, limited to -1 dBFS`); `AUDIO_EQ=off` keeps ComfyUI's track. Every
clip of this section was checked for sample rate (32 kHz stereo), length against the picture (within one frame),
start offset (0), silence, channel balance (within 0.6 dB), clicks at the ends, and full-scale samples (none after
the change).

More of the 2.6 surface, on the same machine (Turbo 4 passes, 864x480):

| Render | Time | Lowest free |
|---|---|---|
| Two 5 s reference videos (a man from one, a street and its camera from the other) | 373 s | 33.8 GB |
| An 8 s clip from a picture and an 8 s source shot, full scene, through the app | 345 s | 30.3 GB |
| A picture, a 5 s video with its own sound and an audio clip | 229 s | 33.2 GB |
| The 8 s clip again with the shot's own sound as a reference: the clip's sound is the shot's, likeness 0.96 (0.02 without) | 344 s | 33.4 GB |
| Object motion: the depth layer of a ball rolling over a table, no picture, a prompt for an apple on marble | 208 s | 38.1 GB |
| Camera only: the depth layer of a street shot, no picture, a prompt for an empty canyon | 198 s | 36.9 GB |

The object-motion clip has the apple on the ball's path over a table of the same shape in a new kitchen; the
camera-only clip has a canyon with the street's layout and no person, with a boulder where the figure stood (a depth
layer shows the figure, so "camera only" keeps its place filled). The same request with the 4-step adapter at 4, 3
and 2 passes (*Custom*; a picture and a 5 s reference video) took 219, 191 and 156 s; the three clips are the same
shot to the eye, equally sharp (median Laplacian variance 70, 73, 84) and equally close to the reference's motion
(pose distance 1.1 % of the frame width in each), so 2 or 3 passes are worth trying for drafts; one request, one seed.

**The shared motion test** (two fast martial-arts shots, each as *Full scene* with the source clip and as *Actor
motion* with the depth layer and a new setting; Turbo 4 passes, seed 7, the wizard's own prompts, the same on both
builds): on the Spark 202 to 216 s a clip and 29.9 GB lowest free. Pose distance from the source, as a share of the
frame width: 2.9 % median (7.7 % at the 90th percentile), 3.2 % (8.1 %) and 3.2 % (7.6 %) for the first three; the
fourth, two fighters in a new courtyard, follows by eye, and the checker, which tracks one person, picked the other
fighter (its 28 % is not a measure of that clip).

Not tested on the Spark: larger sizes than 864x480 with a reference video, clips over 8 s with one, three reference
videos at once, and sound by ear (the figures above are measurements).

### GHCR prebuilt carrier

```bash
docker pull ghcr.io/drowzeys/keys-tensorfold-studio:2.8.0
# index digest: sha256:44dba56cb8e316cdc35a3e227f1d5109901ad9209a0cdb408e3ef27e5a2265be (linux/arm64 + linux/amd64)
docker run --rm -v "$PWD":/out ghcr.io/drowzeys/keys-tensorfold-studio:2.8.0 cp -a /payload/. /out/payload/
```

One image for both builds. For a Mac it holds the TensorFold wheel, `requirements.lock`, the render scripts, the
prebuilt native FastH3 engine (`zig-engine/tf-h3-dit`, built for Apple Silicon at the pinned commit) and `SHA256SUMS`.
For a DGX Spark it holds the prebuilt CUDA engine (`spark-engine/libtf_h3.so`, GB10, aarch64; since 2.8.0 the
`h3-cuda-lora` build at `1f7ecc75`, with LoRA support, so a Spark install no longer builds it from source), which
`spark/build-engine.sh` takes when Docker is there and otherwise builds from source. **It is not a runtime**: neither
Metal nor the Spark's models run in a container, so the setup installs natively. Rebuild it with
`PUSH=1 bash scripts/build-carrier.sh` (on a Spark, after `spark/build-engine.sh`, to include its engine). Up to 2.2
the image was `ghcr.io/drowzeys/keys-mac-tensorfold-studio`; the Mac setup still falls back to it.

## Stack

| Piece | Value |
|---|---|
| Host | Mac Studio M5 Ultra, 256 GB, macOS 27.0.1 |
| Engine | TensorFold 0.6.5 (`609ca419`) + twelve commits, `drowzeys/TensorFold` branch `studio` @ `a2068c031e08109a0ec14c26b1ca655cf50ac34c` (Apache-2.0) |
| Native engine (FastH3) | TensorFold 1.0.4's Zig runtime + an H3 family for Metal and one for CUDA, `drowzeys/TensorFold` branch `h3-1.0.4` @ `b7e309e042e48b72e75e9fb87342df4e9bf22f37` (the DGX Spark's CUDA
library: branch `h3-cuda-lora` @ `1f7ecc75`, which adds `tf_h3_set_weight` for LoRAs); built with Zig 0.17. Same output bit for bit as the 1.0.2-based engine of 2.0 to 2.2 |
| Image model | `Qwen/Qwen-Image-2.1-Turbo`: 7B transformer (32 blocks, bfloat16), 64-channel VAE, Qwen3-VL text encoder; 8 steps on the schedule saved with the checkpoint |
| Image adapter (installs from before 2.3) | `Viggle/Qwen-Image-2.1-viggle-turbo`, v0.3, rank 256, 6 steps on its trained nodes, on `Qwen/Qwen-Image-2.1` |
| Video model | `MiniMaxAI/MiniMax-H3`, `FL2VA` partition: 33B transformer, Qwen3-VL text encoder, video and audio VAEs |
| Video adapter | lightx2v MiniMax H3 Turbo v1.0, runner layout as published by Phosphene |
| 2x video decoder | `speach1sdef178/MiniMax-H3-X2-Detail-VAE`, `MiniMax-H3-X2-Detail-v1.safetensors` (decoder only; its reference-detail branch is not used) |
| Borrowed at run time | mflux @ `add5164e`: Qwen-Image prompt encoder. minimax-h3-mlx @ `79190205`: H3 text encoder, first-frame encoder, audio decoder, MP4 writer |
| MLX | 0.32.3 |
| Peak memory | image 28 GiB at 1344x768, 47 GiB at 2560x1472; FastH3 28 to 32 GiB; MiniMax H3 Turbo 103 GiB during its adapter merge ([table](#minimum-requirements)) |

## Notes

- **Licences decide what you may do with this.** Qwen-Image-2.1-Turbo (and Qwen-Image-2.1 with the Viggle adapter) are under the **Qwen Research
  License Agreement: non-commercial research and evaluation only**; commercial use needs a licence from the Qwen
  team. MiniMax H3 and the 2x decoder derived from it are under the MiniMax H3 Community License, which excludes some
  territories. This pack ships no
  weights. Read both licences before downloading.
- **This is a development render path, not a product.** There is no `tensorfold generate` command and no server; each
  script loads its model from disk and exits. The two models run one after the other, never together.
- **What TensorFold runs and what it borrows.** Both transformers, both samplers, the adapter merges, the int8
  kernels, the image decoder and the video decoder are TensorFold's. The Qwen-Image prompt encoder is mflux's, and
  the H3 text encoder, first-frame encoder, audio decoder and MP4 writer are minimax-h3-mlx's.
- **Qwen-Image here is text to image only.** Editing, reference images, transparent output and guidance with a
  negative prompt are not ported; mflux has them.
- **Sound (1.5).** With the Turbo adapter the sound was judged poor by ear: harsh on speech, worse on singing, with an
  echo-like quality. Measurements agree on one cause: the model makes two audio channels, which agree at 20 steps
  and disagree under the adapter. More passes, a mono fold-down, level changes and a different audio schedule did
  not fix it. Making the sound again with the base model did, and is now standard (`REVOICE=0` keeps the adapter's
  sound). The fast audio step is not identical to recomputing the whole sequence each step: on one clip the two
  gave the same loudness contour and tone but not the same waveform. Lip-sync was judged by eye, not measured.
- **Levels.** The audio decoder's hard clip at full scale is lifted and an overshooting take is turned down as a
  whole. A take whose 120-300 Hz band is weak gets a low shelf of up to 6 dB (`AUDIO_EQ=off` skips it); re-voiced
  takes have needed 2.5 to 5 dB.
- **Passes.** On a 15 second (362 frame) clip, 3 passes lost a detail (headphones) that 4, 5 and 6 kept, and 5 was
  preferred by eye. On an 8 second (192 frame) clip 3 passes kept it. `POINTS` sets passes plus one.
- **Quality is not graded.** Checked: stills of every clip, the scripted line by speech recognition, one image prompt
  against the mflux reference. Not checked: motion and audio by eye and ear, lip-sync, text rendering in general
  (the chalkboard reads "FRESH TODAY" at 1344x768 and comes out garbled at 864x480), other prompts and seeds.
- **Turbo adapters are distilled models.** They trade some fidelity for speed; `TURBO=0` (image) and `QUALITY=high`
  (video) run the base models.
- **M5 only for these numbers.** The int8 kernels need Metal 4 tensor operations; elsewhere both families run
  bfloat16 and slower.
- **Memory.** `--image-only` asks for 48 GB; the default install 64 GB; `--turbo` and `--ref2va` 128 GB. Measured on 256 GB only; see [Minimum requirements](#minimum-requirements).
- **Not part of upstream TensorFold.** The H3 family, the Qwen-Image family and the audio step were offered to
  ashhart/TensorFold as draft pull requests (#384, #393, #405) and closed on 2026-10-07: the engine is built around
  token lanes with exact output, it does not trade precision as the int8 kernels do, its Python engine is frozen and
  its new Zig engine drives Metal without MLX. This code lives in the `drowzeys/TensorFold` fork, on 0.6.5.

## Credits

Cite the original authors first: the Qwen team (Qwen-Image-2.1), Black Forest Labs (FLUX 3 Image Klein, the FLUX3-K
picture models) and felix-ab (flux3-mlx, their Mac runtime), the MiniMax team (MiniMax H3), Ash Hart and the
TensorFold contributors, antirez (h3.c), RobZombAI (H3MLX), mrbizarro (minimax-h3-mlx, Phosphene), Filip Strand and
the mflux contributors, Viggle, speach1sdef178 (the 2x decoder), LightX2V, NVIDIA Research, FastVideo, and Apple MLX and its contributors. Full list:
[CREDITS.md](CREDITS.md). Pack: drowzeys / keys.
