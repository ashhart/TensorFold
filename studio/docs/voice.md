# Voice (Fish Audio Drama 3)

**Built with Fish Audio.** The **Voice** tab turns text into speech with Fish Audio Drama 3, run on this machine by
TensorFold's ports of it: on a Mac the MLX port (the 35B Qwen3.5 MoE backbone, its codebook tower and the Fish codec,
in MLX), on a DGX Spark the native CUDA port ([below](#voice-on-a-dgx-spark-cuda)). No Fish Audio cloud API is called
and the pack ships no Drama 3 weights.

- **Write what is said.** Text without a tag is speaker 0; `<|speaker:0|>` to `<|speaker:15|>` switch voices.
  Directions in brackets steer the delivery: `[whispering]`, `[sighs]`, `[laughs]`, or a description of the voice
  (`[a deep, gravelly man in his late fifties, slow and patient]`), which designs a voice without a recording and is
  not read out (measured: that one at 76 Hz, "a bright young woman" at 228 Hz, both transcribed without the bracket).
  Long text is split at sentence ends and keeps one voice.
- **Clone a voice (optional).** A clean clip of one speaker, about 10 to 30 seconds (2 to 60 are taken), and what it
  says; left empty, Whisper writes the transcript (on a Mac the engine's own, already loaded; on a Spark the app's). The consent box (*I have the speaker's consent to clone this
  voice*) must be ticked, and the API refuses a clip without `"consent": true`. Measured with a 3.9 s clip at 125 Hz:
  the takes came out at 130 and 133 Hz with every word right.
- **Settings.** Seed, and Fish's sampling defaults: temperature 1.0, top-p 0.9, top-k 30.
- **Each take** is a job in the queue, so it never runs beside a video render, and it holds the studio's GPU lock
  folder (`~/studio-gpu.lock.d`) while the model runs, waiting if a script outside the app holds it.
- **A turn a line.** Text with two or more speaker tags is sent with each turn after the first on a line of its own,
  on both machines (`voice.dialogue`): written on one line, Drama 3 dropped later turns far more often (TensorFold's
  measurement: 12 of 48 streamed dialogue takes short on one line, 1 of 48 a turn a line).
- **The word check.** On a Mac the engine checks itself (TensorFold `drama3-mlx` 4a8a982f or later): every part it
  speaks is heard by Whisper large-v3-turbo on MLX and aligned to the text; words after the end, or a long tail of
  sound, are cut; a part with words missing is drawn again with a new seed (twice at most, the best kept); and a
  take that passed a likely end and went on is stopped on the model's own end signal. The tab shows the check part by
  part (ok, trimmed, redrawn, words not heard) and its count as the take's score, with what was heard on hover. On a
  Spark the app runs the same check (`app/wordcheck.py`, the engine's rules) on what Whisper (the QA gate's own,
  `app/qa.py`, on the CPU) hears in the finished take, and makes a take with missing words again with another seed,
  not played, twice at most, keeping the best (2.7.8). With an older Mac engine Whisper only scores the take.
- **Plays while it is made**, on both machines: the tab plays the take from `/api/voice/takes/{id}/live` as the
  engine writes it. On a Mac a heard part cannot be drawn again, so a take whose check finds a part short while it
  plays is made once more without playing, and the take says so; untick *Play while it is made* to skip the played
  take (`"live": false` in the API).
- **A take as a MiniMax H3 voice reference.** *Use as Ref2VA voice reference* opens the create wizard on reference
  to video with the take as the next `<Audio N>` (2 to 15 s a clip, 15 s in all: a longer take is cut to its first 15
  seconds, asked first). Add a picture or a video of who speaks, and describe the same voice in the line's Voice box:
  what a reference audio clip does is in [Reference videos and reference audio](../README.md#reference-videos-and-reference-audio-ref2va).
  *Clone this voice* uses a take as the clone clip of the next, for the same speaker across lines.
- **Slate.** On the Sound tab a voice sheet has **Cast voice**: the sheet's sample line, with its description (age and
  gender, accent, timbre, pitch, pacing, energy, texture) as the direction. The take is kept on the sheet (`drama` in
  the project file); ticked, a shot that opens on cast stills sends it as the speaker's reference audio and names it
  in the prompt. A take over 15 s is cut on the sheet first; what does not fit MiniMax's limits is said in the
  preflight and left out.

| Measured, Mac Studio M5 Ultra (TensorFold `drama3-mlx` 92c09a8e, the persistent server, played while made), 2.7.7 | Audio | First sound on the page | Through the app, check included | Words heard |
|---|---|---|---|---|
| One speaker, 16 words, first take (loads the server, 5.2 s) | 8.0 s | 5.43 s (engine 0.21 s) | 8.2 s | 16 of 16 |
| Cloned voice from an MP3 clip (transcript left to Whisper on the CPU first) | 6.6 s | 10.7 s (engine 0.52 s) | 13.4 s | 16 of 16 |
| Two speakers, the tab's example, seeds 1, 2, 3 | 10.6, 9.4, 11.1 s | 0.30, 0.29, 0.23 s | 4.1, 3.6, 4.1 s | 20 of 20 each |
| Two speakers, seed 4: short while playing, made again without playing, redrawn once | 9.9 s | 0.28 s | 19.4 s | 20 of 20 |
| A 120-word read, three parts (two tails trimmed) | 46.4 s | 0.30 s | 16.8 s | 120 of 120 |
| Slate Cast voice | 3.2 s | 0.30 s | 1.5 s | 12 of 12 |
| One speaker, 9 words, loaded | 2.5 s | 0.24 s | 1.5 s | 9 of 9 |

First sound on the page is the first byte of `/api/voice/takes/{id}/live` after the take was asked for. Peak memory
48.5 to 49.9 GB (55.7 GB with a redraw); the server unloaded after 300 idle seconds and before a video job, and gave
the GPU lock back both times.

| Measured, Mac Studio M5 Ultra (TensorFold `drama3-mlx` 4a8a982f, 8-bit experts, played while made), 2.7.6 | Audio | Engine time | Through the app, check included | Words heard |
|---|---|---|---|---|
| One speaker, 16 words | 8.0 s | 2.7 s (71.9 frames/s, first sound 0.22 s; 5.3 s after the click) | 8.7 s | 16 of 16 |
| Cloned voice from an MP3 clip (3.9 s, transcript left to Whisper on the CPU first) | 6.6 s | 2.4 s | 18.4 s | 16 of 16 |
| Two speakers, three lines, seed 1: short while playing (7 of 20), made again and redrawn once | 12.5 s | 11.8 s the second run | 30.5 s | 20 of 20 |
| Two speakers, seed 2: short while playing, the last line still missing after three draws | 22.8 s | | 38.8 s | 16 of 20 |
| Slate Cast voice | 3.2 s | 1.3 s | 7.2 s | 12 of 12 |
| A 120-word read, three parts (two tails trimmed) | 46.4 s | 15.9 s (68.1 frames/s) | 21.9 s | 120 of 120 |

Peak memory 48.7 to 49.9 GB played while made, 55.7 GB with a redraw. Frames are drawn at about 70 a second while
playing (the codec decodes in windows) and 83 when not. The first sound comes 5.3 s after the click because each
take starts the engine's process and loads the model (3.7 s); the engine's own clock says 0.22 s.

| Measured, Mac Studio M5 Ultra (TensorFold `drama3-mlx` f623fb1a, 8-bit experts), 2.7.4 | Audio | Engine time | Through the app, word check included | Whisper |
|---|---|---|---|---|
| One speaker, 18 words | 8.3 s | 3.3 s (63.3 frames/s) | 16.7 s | 17 of 18 ("Phish") |
| Cloned voice, with `[sighs]` and `[laughs]` | 7.0 s | 2.8 s | 16.1 s | 12 of 12 |
| Cloned voice, transcript left to Whisper | 4.3 s | 1.8 s | 24.8 s | 15 of 15 |
| Two speakers, three lines | 11.3 s | 4.4 s | 17.7 s | 20 of 20 |
| Slate Cast voice (a sheet's description as direction) | 7.5 s | 3.0 s | 16.2 s | 15 of 15 |

Engine time is Drama 3's own (prefill, frames and codec); the process also loads the model (about 3 s) and the
word check loads Whisper on the CPU. Peak memory 48 to 51 GB.

**Mac requirements.** An Apple-silicon Mac with 64 GB of memory or more (the run peaks at about 52 GB; nothing else
of the studio runs beside it), about 40 GB of disk for the converted weights (and 39 GB for Fish's checkpoint while
converting), and TensorFold's Drama 3 port in an environment of its own. The port is on TensorFold's `drama3-mlx`
branch of [drowzeys/TensorFold](https://github.com/drowzeys/TensorFold). `bash oneshot-setup.sh --voice` (or
`bash scripts/setup-voice.sh` alone, `--verify` to check) does all of it: the clone at 8f3d4b33 in
`~/tensorfold-drama3-mlx`, its venv, Fish's checkpoint, the conversion and a test line. By hand:

```bash
git clone -b drama3-mlx https://github.com/drowzeys/TensorFold.git ~/tensorfold-drama3-mlx
python3.11 -m venv ~/drama3-venv && ~/drama3-venv/bin/pip install -e ~/tensorfold-drama3-mlx mlx mlx-lm mlx-whisper soundfile torch
hf download fishaudio/drama-3 --local-dir ~/models/drama-3        # Fish Audio Community License: read it first
~/drama3-venv/bin/python -m tensorfold.families.drama3.convert ~/models/drama-3 ~/models/drama-3-mlx   # once, about a minute
~/drama3-venv/bin/python -m tensorfold.drama3 --text "Hello there." --out hello.wav                  # check
```

From `drama3-mlx` 92c09a8e the app runs the engine as a persistent server (`python -m tensorfold.drama3 --serve`,
the same protocol as the Spark's), as described for the Spark [below](#voice-on-a-dgx-spark-cuda): loaded on the first
take, unloaded after five idle minutes (`DRAMA3_IDLE_S`) and before any other job, under the GPU lock and a memory
guard (vm_stat's free, inactive, speculative and purgeable pages; 8 GB, `DRAMA3_GUARD_GB`). An older engine runs once a
take. The app finds `~/drama3-venv` and `~/models/drama-3-mlx` itself (`DRAMA3_VENV` and `DRAMA3_MODEL` name others) and
says on the tab what is missing. `DRAMA3_BACKEND=mlx|cuda` picks the engine (`mlx` on a Mac, `cuda` on a DGX Spark by
default). The API: `GET /api/voice/state`, `POST /api/voice/takes {text, ref_audio?, ref_text?, consent?, seed?,
temperature?, top_p?, top_k?}` (a job of kind `voice`), `GET /api/voice/takes`, `GET /api/voice/takes/{id}/audio`,
`GET /api/voice/takes/{id}/live` (the take as it is made; `"live": false` in a take's settings: not played, Mac only), `POST /api/voice/sheet {sheet, project}` and
`POST /api/voice/cut {path, seconds}`. Takes are kept in `~/TensorFoldStudio/voice/` (the WAV, its settings and
measurements, and the text).

## Voice on a DGX Spark (CUDA)

On a DGX Spark the tab runs **TensorFold's native CUDA port of Drama 3**: one binary, `drama3-generate` (Zig with
CUDA kernels built for the GB10's sm_121: the backbone, the codebook tower and the Fish codec, no Python or PyTorch at
run time), with `libtensorfold_drama3.so` beside it.

- **A persistent server.** The first take starts `drama3-generate --serve 1` (one JSON request a line on its stdin,
  one JSON answer a line on its stderr) and waits for it to load: about 18 s from the checkpoint, 38.6 GB of weights
  on the device, 41.6 GB at the peak. Later takes go straight to it.
- **It gives the GPU back.** While loaded it holds `~/studio-gpu.lock.d` (a script outside the app that takes the
  lock waits for it). It is unloaded, and its memory comes back, after **five minutes without a take**
  (`DRAMA3_IDLE_S`), and **at once when any other job of this machine's queue starts** (a video, a picture, a long
  video): Drama 3 and MiniMax H3 never stand side by side. Before it loads, the studio's ComfyUI server is stopped
  if it holds the memory Drama 3 needs (about 50 GB free is wanted); the next video loads its models again.
- **Its own memory guard**: available memory is read four times a second and the engine is killed by its PID under
  8 GB (`DRAMA3_GUARD_GB`); the take fails with the reason and the lock goes back. The engine also ends with the app.
- **Plays while it is made.** The engine writes the take's samples as it makes them and the tab plays them behind an
  open WAV header (`/api/voice/takes/{id}/live`): the first sound about 0.3 s after a take starts on a loaded engine
  (18.5 s on the first take, which loads it), and the take is made faster than it plays.
- **Long text** goes in parts of about 60 words cut at sentence ends (each part opens on the speaker who was
  speaking), joined with 0.15 s between them. A one-speaker take without a clip clones its first part's voice for the
  others (measured on a 56 s read: 271, 272 and 269 Hz in its three thirds); a clip given is used for every part.
- **A voice to clone** may be any sound file: it is converted to 44.1 kHz mono WAV with ffmpeg first (the engine
  reads WAV).

| Measured, DGX Spark GB10 (TensorFold `drama3-cuda` 81173e6c), through the app | Audio | Engine time | Through the app, word check included | Whisper |
|---|---|---|---|---|
| One speaker, 16 words, first take (loads the engine, 18.3 s) | 8.2 s | 5.9 s (30.9 frames/s) | 37.7 s | 16 of 16, 100 % |
| Cloned voice from an MP3 clip (3.9 s, transcript left to Whisper), with `[sighs]` | 6.1 s | 5.0 s (28.0 frames/s, first sound 0.47 s) | 29.7 s | 16 of 16, 100 % |
| Two speakers, three lines, `[whispering]` and `[sighs]` | 11.6 s | 8.4 s (30.2 frames/s) | 21.1 s | 20 of 20, 100 % |
| Slate Cast voice (a sheet's description as direction) | 4.1 s | 3.3 s (28.3 frames/s) | 14.2 s | 11 of 12, 92 % ("harbor") |
| A 120-word read, three parts | 55.7 s | 40.8 s (30.7 frames/s) | 56.9 s | 118 of 120, 98 % |

The cloned take came out at about 130 Hz against the clip's 125 Hz. Engine time is the requests' own (prefill,
frames, codec); the app's time adds Whisper on the CPU (and, for a clip without its words, Whisper on the clip first).

**Spark requirements.** A DGX Spark (GB10, 128 GB), the CUDA toolkit 12.9 or newer and zig 0.17 to build, 39 GB
of disk for Fish's checkpoint and 1.6 GB for the converted codec, and about 50 GB of free memory when a take starts
(nothing else of the studio runs beside it). The port is on the `drama3-cuda` branch of
[drowzeys/TensorFold](https://github.com/drowzeys/TensorFold) (81173e6c). The setup is by hand: the branch's build
script carries its author's paths and its zig build embeds the kernel fatbins of TensorFold's whole CUDA engine
(prebuilt, or every one built by nvcc), so there is no installer step yet; `bash oneshot-setup-spark.sh --voice` (or
`scripts/setup-voice.sh`) checks what the app looks for and prints these steps for what is missing.

```bash
git clone -b drama3-cuda https://github.com/drowzeys/TensorFold.git ~/tf-drama3 && git -C ~/tf-drama3 checkout 81173e6c
hf download fishaudio/drama-3 --local-dir ~/models/drama-3          # Fish Audio Community License: read it first
# build as the branch's tools/drama3/build_local.sh does (it has its author's paths): nvcc makes drama3.fatbin,
# drama3_fp8.fatbin and drama3_codec.fatbin for sm_121 beside the CUDA engine's other fatbins, then
#   zig build -Dfatbins=FATBIN_DIR -Dsm=121 -Doptimize=ReleaseFast --prefix OUT drama3-generate drama3-lib
mkdir -p ~/.local/opt/tensorfold-studio/drama3/{bin,lib}
cp OUT/bin/drama3-generate ~/.local/opt/tensorfold-studio/drama3/bin/
cp OUT/lib/libtensorfold_drama3.so ~/.local/opt/tensorfold-studio/drama3/lib/
# once, offline: Fish's codec.pth into the native codec's layout (PyTorch and Fish's reference code, ORACLE_DIR)
python3 ~/tf-drama3/tools/drama3/codec_convert.py ~/models/drama-3 ~/.local/opt/tensorfold-studio/drama3/codec-v1.safetensors
~/.local/opt/tensorfold-studio/drama3/bin/drama3-generate --model ~/models/drama-3 \
  --codec ~/.local/opt/tensorfold-studio/drama3/codec-v1.safetensors --text "Hello there." --wav hello.wav   # check
```

The app looks in `$PREFIX/drama3` (`~/.local/opt/tensorfold-studio/drama3`) and `~/models/drama-3`;
`DRAMA3_CUDA_BIN`, `DRAMA3_CUDA_LIB`, `DRAMA3_CUDA_MODEL` and `DRAMA3_CUDA_CODEC` name others, and the tab says what is
missing. The worker Sparks of the Sparks setting do not make voice takes: a take runs on the Spark whose app you use.

**Licence.** Drama 3's weights and codec are Fish Audio Materials under the **Fish Audio Community License v1.0**:
research, non-commercial and small-company commercial use; a hosted service that lets others generate speech or
clone voices needs a separate licence from Fish Audio; show "Built with Fish Audio" where it is used; never clone a
real person's voice without their consent, impersonate anyone, or train other speech models on the outputs.
