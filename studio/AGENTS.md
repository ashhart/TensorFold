# Agent one-shot — Wasserman Slate AI Studio, powered by TensorFold: Qwen-Image-2.1 + MiniMax H3 (MLX, Apple Silicon)

**Repo:** https://github.com/ashhart/TensorFold, folder `studio/` (developed in drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3)
**Carrier image:** `ghcr.io/drowzeys/keys-tensorfold-studio:2.8.0` (one for both builds, not a runtime; for a Mac the TensorFold wheel + lock + render scripts + the native FastH3 engine, for a DGX Spark the prebuilt CUDA engine)
**Engine:** TensorFold 0.6.5 + H3 and Qwen-Image families @ `drowzeys/TensorFold` `a2068c03`, own venv, int8 tensor-unit kernels

```bash
git clone https://github.com/ashhart/TensorFold.git
cd TensorFold/studio
brew install python@3.11 uv ffmpeg
bash oneshot-setup.sh               # image model, FastH3 and its text encoder and decoders (175 GB) and a test clip; --image-only for 33 GB; --turbo adds MiniMax H3 Turbo (64 GB more, 128 GB of memory); --ref2va adds reference pictures to video (64 GB more, 128 GB)
bash scripts/studio.sh "the picture" "what happens in the clip" out.mp4   # text -> image -> video with sound
bash scripts/scout.sh "the picture" 6 scouts/x      # six candidates + scouts/x/sheet.jpg, 27 s
IMAGE_FILE=scouts/x/scout_s3.png QHD=1 FRAMES=192 bash scripts/studio.sh "" "what happens" final.mp4   # 2560x1440
bash scripts/image.sh "a prompt" out.png
FIRST_FRAME=photo.jpg bash scripts/video.sh "what happens next" out.mp4
REF_IMAGES="face.jpg" bash scripts/video.sh "Use <Picture 1> as the person. The person ..." out.mp4   # needs --ref2va
```

On a **DGX Spark** (Linux, CUDA) the same `bash oneshot-setup.sh` installs the CUDA build (it hands over to
`oneshot-setup-spark.sh`): needs `git`, `ffmpeg` and the CUDA toolkit's `nvcc` (12.9 or newer), 99 GB of models, about
100 GB of memory for video. Its options are `--image-only`, `--no-render` and `--verify`; there is no `--turbo` and no
2x decoder there. `bash spark/comfy.sh stop` frees the models' memory after rendering.

## Install for a person and build the one-click app

Do these in order on the person's Mac, and report each result plainly.

1. Check the machine: `uname -m` is `arm64`, `sysctl -n machdep.cpu.brand_string` names an M5-family chip (others run
   without the int8 kernels, far slower: say so and ask before continuing), memory is 64 GB or more (128 GB if they want `--turbo`; on an M1 to M4 say plainly that video will be much slower than on an M5, and that it is untested there), and
   `df -h ~` shows 185 GB free (250 GB with `--turbo`). Stop and report if memory or disk falls short.
2. `brew install python@3.11 uv ffmpeg` (install Homebrew first only with the person's agreement).
3. Tell the person the download sizes and licenses before starting: Qwen-Image-2.1 33 GB (Qwen Research License,
   non-commercial), MiniMax H3 text encoder and decoders 72 GB (MiniMax H3 Community License, territory limits; 134 GB with `--turbo`), FastH3 65 GB (same license).
4. `bash oneshot-setup.sh --app` (add `--turbo` only if they ask for MiniMax H3 Turbo and have 128 GB; `--app` does step 5 with a Desktop shortcut). It resumes if interrupted. It ends by
   rendering `outputs/test.mp4`; confirm that file exists and plays.
5. `bash scripts/make-app.sh` builds `~/Applications/Wasserman Slate AI Studio.app`. If the models are not in the default
   folders, export `QWEN_MODEL_DIR`, `H3_MODEL_DIR`, `FASTH3_DIR` first: the app records them. Do not set
   `HOST=0.0.0.0` unless the person asks to reach it from other machines; there is no login.
6. `open "$HOME/Applications/Wasserman Slate AI Studio.app"`, then check `curl -s http://127.0.0.1:7870/api/state` answers.
7. Tell the person: double-click Wasserman Slate AI Studio in Applications; renders land in `~/TensorFoldStudio`; the log is
   `~/Library/Logs/TensorFoldStudio.log`; to stop it run the app's executable with `stop`.

The app is a launcher around this clone, unsigned and built locally, so Gatekeeper does not block it. Do not move
the clone afterwards without rebuilding the app. Do not sign, notarize or redistribute the built app.

Native engine (2.0): FastH3 runs its passes in `$PREFIX/zig-engine/tf-h3-dit` (TensorFold 1.0 Zig +
Metal runtime, fork branch `h3-1.0.4`), driven by `zig_fasth3.py`; `scripts/video.sh` chooses it on an M5, for text
to video and for clips from an image alike, and prints `[tensorfold] engine: zig|mlx (why)`. `FASTH3_ENGINE=mlx|zig` forces one. Image
generation and the Turbo path are MLX. Do not apply the base-model audio step to FastH3 clips:
the owner compared it by ear and chose FastH3's own sound. Use 20 passes when a prompt has several actions.
`scripts/build-zig-engine.sh` builds the payload on a Mac (zig 0.17); never commit the binary.

FastH3 and long videos: `scripts/fast.sh` (STEPS=4|8|20, RES=480p|720p, UPSCALE=1); `ENGINE=fasth3` works with
`studio.sh` and `FIRST_FRAME` too, though FastVideo trained it on text to video only. Videos over 15 s are chains of
clips made by the web app's `long` job (each opens on the last frame of the one before); only an 18 s chain has been
rendered, so do not promise the look of a 30 minute one.

Web app: `bash scripts/app.sh` (http://127.0.0.1:7870; `HOST=0.0.0.0` to open it to the LAN, no login). It runs the
same scripts through a one-at-a-time queue and stores everything under `~/TensorFoldStudio`.

Rules:

- This is not a server. `tensorfold serve` routes neither model; render with the scripts.
- Qwen-Image-2.1-Turbo is non-commercial (Qwen Research License). MiniMax H3 has territory limits.
  Do not redistribute weights, and say so when asked to use the outputs commercially.
- Image sizes are multiples of 16. For the studio chain both models must accept the canvas: multiples of 32, at most
  768x1344 pixels in total. Video frames must be `17n + 5`, from 22 to 362 (`app/lengths.py`): the app takes any
  of the 21 counts (`frames`, or `seconds` brought to the nearest) and always shows the exact seconds, frames / 24.
  "10 s" is 243 frames, 10.125 s. Do not add a length the model does not take, and do not raise the 15 s limit
  without measuring the memory on a Spark (362 frames at 1376x768 left 29.7 GB free on TensorFold's int8 engine).
- Preferred flow: scout at 1280x736, let the person pick, then `IMAGE_FILE=... TWOK=1` (2048x1152, 2.5 min) or `QHD=1` (2560x1440, 6 min) for the 2K final; `X2=1` for a
  one-minute 1344x768 draft. `X2=1` needs WIDTH and HEIGHT in multiples of 64. 720-row canvases are not valid for the
  video model: use the QHD preset, which generates 736 rows and crops.
- The 2x decoder is an upscale, softer than a native generation at the same size. Do not describe it as added detail.
- Standard video = Turbo adapter, 5 passes, then the sound made again by the base model (`REVOICE=20`). Do not go back
  to the adapter's own sound (`REVOICE=0`) unless asked: it was judged poor by ear. `QUALITY=high` = 20 steps, no
  adapter, about three times slower. `POINTS` is passes plus one.
- Reference to video (`REF_IMAGES`, `oneshot-setup.sh --ref2va`) runs MiniMax H3's Ref2VA transformer on the MLX
  engine: up to nine pictures, named `<Picture 1>`, `<Picture 2>`, ... in the prompt in the order given. A reference
  is not a frame: never combine it with `FIRST_FRAME`, and never send it to FastH3. Default is 20 plain passes;
  `REF_TURBO=1` is the 4-step Ref2VA adapter (`POINTS=5`) with `REVOICE=20`, `QUALITY=high` the fast recipe. The FL2VA Turbo adapter is not for this
  transformer. `Ref2VA/` holds only the transformer; its text encoder, tokenizer, processor and decoders are links
  to `FL2VA/`: do not download them twice.
- Reference videos (`REF_VIDEOS`, up to 3) and reference audio (`REF_AUDIOS`, up to 3) are the same model's other
  references: each clip 2 to 15 s, 15 s in all per kind, 12 references of all kinds, audio never
  alone. Named `<Video N>` and `<Audio N>` in the prompt; with `REF_VIDEO_SOUND=1` each video's own sound is a
  reference numbered ahead of the `REF_AUDIOS` clips. They need a TensorFold whose H3 family has
  `packing.reference_layout` on a Mac (engine branch `ref-video-audio`); the renderer says so when it is missing. A 5 s
  reference video about triples the time of a 5 s clip. Do not promise voice cloning. Measured on the Spark: when the
  spoken line is what the recording says, the clip plays the recording back (likeness 0.98) with the mouth in time,
  whoever is pictured; with other words the voice carries when the prompt also describes it ("in the deep male voice of <Audio 1>":
  122 Hz, the recording's, against 156 without the clip), and a man's voice named without a description beside a
  woman's picture changed nothing. A video's own sound carries (likeness 0.81 against 0.34).
  On the Mac the described voice carried too (a man's voice beside a woman's picture: 137 Hz and MFCC distance 5.0 with
  the clip, 169 Hz and 11.6 with the description alone; the recording is at 117 Hz); use a clean recording.
  On a DGX Spark
  the same variables go through ComfyUI's `MiniMaxH3ReferenceToVideo` (`spark_generate.py video --ref-video /
  --ref-audio`), which needs no engine change; the video is scaled to the clip's pixel area before the node sees it,
  and must stay so (`REF_VIDEO_SIZE=full` multiplies the rows and the memory and is refused on a Spark).
  On a Spark the clip's sound is made again from a lossless copy of the decoder's output, limited to -1 dBFS
  (`finish_sound` in `spark/spark_generate.py`; `AUDIO_EQ=off` skips it): keep it, ComfyUI's own track overshoots full scale.
  A reference prompt keeps its sound under an `overall_soundscape:` heading (`compose_references`): on the end of the
  shot description the model reads it out as gibberish speech. Do not add ComfyUI guidance (CFG) to reference renders:
  at 5 it burns the picture with the 4-step adapter.
  In the app a reference clip is measured as it is added (`/api/probe`) and turned away with the reason; *Custom* passes
  is the card beside *20 passes* in every chooser.
- Motion as a reference: the wizard's *Analyse motion* (and *Use as motion reference* in the Motion Previs tab) sends
  one layer of a Motion Previs pack as `<Video 1>`. Defaults by measurement: the same place with a new person takes the
  source clip (written as "`<Video 1>` with its person replaced", never with the first-frame-anchor wording); a new
  place or the camera alone takes depth; the skeleton is offered but is the loosest. On the Mac the Ref2VA Turbo adapter with a
  reference video runs without the int8 kernels (`scripts/video.sh`): with them it lost fast motion (26 % against 3.7 %);
  do not put them back for that case. The pose pass, the skeleton and the distance checker follow one person only; the user says when
  `<Video 1>` has two or more people (`motion.people`) and the prompt then speaks of its people. A skeleton or depth layer goes
  with the control-video prompt (`motion_template` in `app/motion_prompt.py`: structural template,
  first-frame anchor, legend, "labels only", and the negative): keep that wording, and do not promise figures: one shot was
  measured. Do not promise an exact copy of a voice from reference audio either. On this machine every render runs through the
  ComfyUI server of `spark/comfy.sh`, never as standalone model code: the guard beside the server (`spark/memory_guard.py watch`, started by `comfy.sh`) is the
  only run-time memory guard.
- On a DGX Spark reference to video is `oneshot-setup-spark.sh --ref2va` and the same `REF_IMAGES` / `REF_TURBO=1`
  (`spark_generate.py video --ref`): ComfyUI's `MiniMaxH3ReferenceToVideo` on its own blocks, never
  `TensorFoldH3Blocks`. The adapter is merged at run time, so it needs the bf16 transformer: on an 8-bit
  (`int8_convrot`) file a run-time adapter silently spoils the picture. A ComfyUI that was running before the
  install must be restarted (`bash spark/comfy.sh stop`) to see the new model folders.
- A LightX2V adapter file (diffusers PEFT names) is rewritten with `scripts/convert-h3-adapter.py` before `--lora`
  can read it; the scale (alpha / rank) differs between their files and is read from the file's metadata.
- Levels are handled after the render (no hard clip; bass lifted on thin takes; `AUDIO_EQ=off` to skip). Leave
  `AUDIO_SHIFT` alone.
- The image turbo adapter runs on six fixed nodes (`--nodes 1.0,0.9375,0.875,0.75,0.5,0.25`); do not change the step
  count with it. `TURBO=0` runs the base model at 40 steps.
- Merge adapters with the tools (`--lora`), never by hand into bfloat16.
- The fast path is M5-only (Metal 4 tensor operations).
- The two models never load together; `studio.sh` runs them one after the other.
- Edit a picture (`edit_plan.py`, read by `qwen_image_edit.py` and `spark/spark_edit.py`): the noise is drawn for the
  seed and the pictures together (`noise_seed`). Never seed it with the user's seed alone: an edit that starts from
  the noise its picture was made from returns that picture, over-sharpened. The result has the first picture's own
  size; up to ten pictures go in one edit, and the model knows them as image 1, image 2, ...
- The video model has no negative prompt on any route. `app/negation.py` lists what a prompt forbids by name and the
  page shows the list before the job is queued; never rewrite the user's prompt, and do not add "no X" lines to the
  prompts the app writes, for the picture or the sound: a clip without lines is told `QUIET_PICTURE` ("Mouths stay
  closed.", only where the shot has people) and `QUIET_SOUND` in `app/prompt_text.py`, a locked camera is "a locked, still
  camera" (`MOVES` in `app/slate.py`). The tests hold a compiled prompt to the app's own check with `OWN` emptied.
- Slate (`app/slate.py`, `app/static/slate.js`): a shot's `intent` never goes into the prompt. A sheet of the bible is
  changed in place (`SL_BIBLE` and the `update_*` actions): never add a second one with the same name, the id is what
  shots cast. A slot in bars (`tf.bars`, with `tf.song` on the project) sets the clip's frames through
  `shot_frames`; the tempo is the user's. `tests/test_slate_page.py` runs the page code in node: keep it passing.
- A clip from another machine's app is read through `app/remote.py` (`/api/qa`, `/api/slate/takes`): addresses are
  `http://host:port` only and paths are media files inside that app's folder. Do not widen either.
- Nothing a request handler calls may run a tool without a time limit (`tests/test_api_answers.py`); the GPU's name
  is asked of nvidia-smi once and kept.
- `app/shot.py` checks a finished single clip: first and last frame held, no cut inside one shot. Its thresholds are
  set from one production run of 32 clips, whose figures are in `tests/test_shot.py` (`RUN_CUTS`): change them only with
  new measurements. A failed check never fails the job; the retake at the next seed is the user's choice.
- On a DGX Spark a crop (the 720p preset) is made inside the workflow (`crop_frames` in `spark/spark_generate.py`),
  not with ffmpeg afterwards. What needs ffmpeg is refused when it is queued (`without_ffmpeg` in `app/engines.py`).
- DGX Spark memory (`spark/memory_guard.py`; the README's *The memory guard*): a clip is checked before it is queued
  (`memory_of` in `app/job_build.py`) and again in `spark_generate.py` before anything loads, and the guard stands beside
  every server `comfy.sh` starts. Never start ComfyUI another way. `ESTIMATES`, `MEASURED_ROWS` and `MEASURED_PIXELS`
  are measurements, one route each. A LoRA on FastH3 is merged into TensorFold's int8 engine (`TensorFoldH3Blocks`
  takes the block projections' patches off ComfyUI and calls `tf_h3_set_weight`; route `int8-lora`, measured to
  1376x768 at 243 frames); on ComfyUI's bfloat16 blocks (an engine without that symbol, `--engine comfy`, Ref2VA,
  `--base h3`) a clip with a LoRA is refused above 1280x736 at 124 frames because a 1376x768 one took a Spark down
  there and nobody knows why from its logs. Never merge a LoRA into the saved `.tf-int8` copy. Raise them only by measuring from below on a worker Spark, in small steps,
  under a second watchdog that kills your own server at 25 GB available; never by trying the size that failed, and
  never on a machine with `vm.panic_on_oom=1`. Do not lower the floor to make a job fit.
- More than one DGX Spark (`app/sparks.py`, `app/static/sparks.js`, `docs/multi-spark.md`): a job's `sparks` (1 to
  `LIMIT`, 8) is how many machines it may use, this one counted; the app uses what is there and says so. Only single
  clips (`video` jobs) go to a worker, one per Spark at a time, through the worker's own job API over an ssh tunnel
  the app opens and closes itself: a worker's app listens on its own 127.0.0.1 and is never opened to the network, and
  nothing here runs as root, edits `/etc` or shares a folder. A worker renders only with this Spark's code (the files
  of `CODE`, compared by hash; *Update and start* sends them with rsync) and its own memory guard, which is asked
  before a clip is sent. A worker silent for `LOST` seconds is marked down and its clip goes back to the queue; do
  not make that wait longer, and never retry a worker that went down without a person's *Check*. `MEASURED` is one
  measured run: change it only with a new one. One clip split across Sparks (the engine's `h3-cuda-multispark`
  branch) is measured and is not in the app: do not wire part of it in.
- Picture models (`IMAGE_MODELS` in `app/picture_models.py`, `image_model` on scout, studio and edit jobs,
  `IMAGE_MODEL` to `scripts/image.sh` and `scripts/edit.sh`): `qwen` (Qwen-Image-2.1-Turbo) on every install. A further
  picture model registers there and in the scripts; Qwen stays the default.
- mflux and minimax-h3-mlx are pinned by commit and used at run time. Do not unpin.

**Credits:** keep CREDITS.md in sync. Qwen team (Qwen-Image-2.1), MiniMax (H3), Ash Hart (TensorFold), antirez (h3.c),
RobZombAI (H3MLX), mrbizarro (minimax-h3-mlx / Phosphene), Filip Strand and mflux contributors, Viggle, speach1sdef178, LightX2V,
NVIDIA Research, FastVideo, Sam Wasserman (Slate: the film workflow in Production; keep app/static/slate/NOTICE. Motion Previs Studio: the Motion Previs tab's analysis is a modified Python port of it; keep LICENSES/Motion-Previs-Studio-NOTICE, THIRD_PARTY_NOTICES.md and the headers of app/motion*.py, and keep the link to the original), Apple MLX. Authors only — no credit hyperlinks, except the link to Motion Previs Studio.
