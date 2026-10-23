# Wasserman Slate AI Studio, powered by TensorFold: install it, click it, make things

**Thank you first** to the Qwen team (Qwen-Image-2.1), MiniMax (H3), FastVideo / Hao AI Lab (FastH3), Ash Hart
(TensorFold), mrbizarro, LightX2V, Viggle, speach1sdef178 and Apple's MLX team. Full list in [CREDITS.md](CREDITS.md).

What you get: an app on your Mac that turns a prompt into an **image**, or into a **video with sound**, from 5
seconds to 30 minutes, at 480p up to 2K.

## 1. Check your Mac

| Need | Why |
|---|---|
| Apple silicon with an **M5-family** chip | the fast kernels use its tensor units; other chips run far slower |
| **64 GB** memory or more | a FastH3 clip peaks near 32 GB on an M5, near 40 GB on an M1 to M4 (8-bit weights, much slower); 128 GB for the optional MiniMax H3 Turbo |
| **185 GB** free disk (250 GB with Turbo) | model weights: 31 GB image, 65 GB FastH3, 72 GB text encoder and decoders |
| Homebrew | to install Python 3.11, uv and ffmpeg |

Everything here was measured on one machine: a Mac Studio M5 Ultra with 256 GB.

## 2. One-shot install

```bash
brew install python@3.11 uv ffmpeg
git clone https://github.com/ashhart/TensorFold.git
cd TensorFold/studio
bash oneshot-setup.sh --app
```

That one command installs the engine into its own folder (`~/.local/opt/tensorfold-studio`), downloads the models,
builds the app with a Desktop shortcut, and renders a 5 second test clip to prove it works. It can be stopped and
run again; downloads resume. Variants:

| Command | What it installs |
|---|---|
| `bash oneshot-setup.sh --app` | images and FastH3 video (about 175 GB); needs 64 GB of memory on an M5 |
| `bash oneshot-setup.sh --turbo --app` | also MiniMax H3 Turbo (about 240 GB); needs 128 GB of memory, and adds a second engine card |
| `bash oneshot-setup.sh --image-only --app` | image model only (33 GB) |
| `bash oneshot-setup.sh --verify` | checks an existing install, downloads nothing |
| `git pull && bash oneshot-setup.sh --upgrade` | moves an existing install to the newest version: the engine and scripts, no model downloads, your clips and projects untouched |

The models have their own licenses: Qwen-Image-2.1 is **non-commercial**; MiniMax H3 and FastH3 are under the
MiniMax H3 Community License. The script prints the links before each download.

**Prefer to let an agent do it?** Give Claude Code (or another coding agent) this:

> Clone https://github.com/ashhart/TensorFold and follow studio/AGENTS.md section "Install for a
> person and build the one-click app". Tell me before any large download starts and when the app is ready.

## 3. The clickable app

`--app` already built it. To build or rebuild it yourself:

```bash
bash scripts/make-app.sh                # ~/Applications/Wasserman Slate AI Studio.app
DESKTOP=1 bash scripts/make-app.sh      # and a shortcut on the Desktop
```

Double-click **Wasserman Slate AI Studio**. It starts the studio if needed and opens it in your browser; double-click again
any time to come back. Drag it to the Dock to keep it there.

- It is a small launcher around the folder you cloned. If you move that folder, run `make-app.sh` again.
- Stop it: `"$HOME/Applications/Wasserman Slate AI Studio.app/Contents/MacOS/TensorFoldStudio" stop`
- Log: `~/Library/Logs/TensorFoldStudio.log`. Everything you make: `~/TensorFoldStudio`.
- It listens on this Mac only, and has no login. `HOST=0.0.0.0 bash scripts/make-app.sh` opens it to your network;
  do that only on a network you trust.

## 4. Make a video from text

![Text to video with FastH3](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_create_fasth3.png?raw=true)

1. **The prompt goes to:** Video with sound. **The clip starts from:** Text only.
2. **Video engine:** FastH3 for speed. **Passes:** 8 (4 is faster and a little softer; 20 is slower and is the one to
   pick when the prompt has several distinct actions). The bar under the cards says which engine will run: FastH3
   uses the native Zig engine on an M5, from text or from an image.
3. **Resolution:** 480p or 720p. Turn on **2x upscale** for 1728x960 or 2560x1440.
4. **Length** and **Seed**, then describe the picture and what happens. *Next*.
5. Step 3: add spoken lines and the soundscape if you want them. Step 4: read the prompt. Step 5: **Queue this clip**.

On the M5 Ultra with the native engine a 5 second clip takes 43 s (4 passes), 66 s (8) or 129 s (20) at 480p, and
about 200 s at 720p with 8 passes; a 10 second 720p clip takes about 6½ minutes. 2x adds about 10 s.

If a clip leaves out part of the action, render it again at 20 passes. If the model reads something into the
prompt that you did not ask for, say so in the prompt ("unarmed hikers, no weapons").

## 5. Make an image only

![Image only](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_create_image.png?raw=true)

**The prompt goes to:** Image only. Describe it, pick a size and how many, **Make image**. A few seconds each.
*Download* saves one; *animate this* makes it the first frame of a video.

## 5.5 Edit a picture

![Edit a picture: the camera ring](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_edit_camera.png?raw=true)

**The prompt goes to:** Edit a picture. The image model re-draws a picture you give it.

1. **The picture.** Upload one, press *Pick from the Library*, or take a frame of a Library clip (first, last, or the
   frame at a time). In the Library, **edit** on a picture opens this step on it.
2. **What changes.** *Change it in words*: describe the picture you want ("The same statue at night under a full moon,
   wet paving stones, a lantern on the wall behind it"). What you leave out is meant to stay.
   *Move the camera* (with the Multiple Angles LoRA from the hub's Fun group): click directions on the ring, front at
   the top, 30° a step. Choose a height first, tick *Close-up* for a tighter frame, or take a ready set such as *Five
   views around*. Each angle is one picture.
3. **Make it.** About 10 seconds a picture on an M5 Ultra once the model is loaded. Under each result: *first frame*
   opens a clip on it, *last frame* ends a clip on it, *edit again* feeds it back in.

A side or back view often comes back on a plain studio background: the LoRA was trained on single objects and
characters. If a result copies your picture instead of moving the camera, lower the LoRA's strength towards 0.8.

## 6. Image first, then video

![Qwen scout to video](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_create_qwen_h3.png?raw=true)

**The clip starts from:** A Qwen scout image. In step 2, make several candidates and click the one you like; the
video opens on exactly that picture. *Your own image* uploads one instead. Either engine works. With MiniMax H3 the
standard quality (Turbo 5 passes, sound made again by the base model) is the one judged best by ear.

## 7. Make a long video

![A 30 minute chain](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/blob/16c8972/samples/ui_create_long.png?raw=true)

Pick a **Length** from 30 s to 30 min. The model makes 15 s at a time, so the studio renders clip after clip, each
opening on the last frame of the one before, and joins them. Write **story beats**, one line per 15 s, to say what
happens in each part.

Be realistic: 30 minutes is roughly 7 to 11 hours of rendering at 480p (an estimate; the longest chain actually
rendered is 18 s), the look drifts over many clips, and sound does not carry across joins. For a real story, plan
it in **Production** instead.

## 8. Production, Editor, Library

- **Production:** a project of shots, each with its own scene, action, line and length; a shot can continue from the
  last frame of the one before. One button renders all that are not done.
- **Editor:** put clips in order, trim them, preview, export one file.
- **Library:** everything you have made. *use* turns an image into a first frame; *+ timeline* sends a clip to the editor.
  Tick files and press **Delete selected** to delete several; **Restore last deletion** puts them back; **Empty trash**
  erases what the trash holds and frees the space.
- **Clearing:** every prompt form has **Clear** (with **Undo**). In the queue, 🗑 deletes a finished job and what it
  made, and **Clear failed** / **Delete all finished** tidy the list. Deleted files go to `~/TensorFoldStudio/trash`.

## 8.5 The LoRA hub: a trained look or move for a clip

A LoRA is a small file that teaches the video model one thing: a camera move, a fighting style, a drawn look. The
**Keys and Wasserman LoRA Hub** is in step 1 of Create (and on the Production board and in a Slate film's settings).

1. Press **▸ Browse the hub**. The LoRAs are in groups: *Wasserman*, *01 Camera & Viewpoint*, *02 Transformation &
   Morphing*, *03 Character Editing*, *04 Motion & Action*, *05 Art Style*, *06 Support & Finishing*, *07 Fun*. Pick a group or
   type in the search box.
2. Each card says what the LoRA does, who made it, what it works from, its licence and its size. After
   `bash oneshot-essential-bundle.sh` (the installer runs it) every usable LoRA is already installed; otherwise press
   **⤓ Install** on a card, or **Install all in this category**.
3. Press **Use on this clip**. A row appears with **Strength** (the author's range is in its tooltip) and, when the
   LoRA has them, a choice: a **Look**, or for *Camera Motion* a **Camera move** (push-in, pull-back, handheld
   follow, orbit, crane reveal, tilt, pan). Its trigger word is written into the prompt for you. Up to four can be
   stacked. If the LoRA needs something of the clip (a last frame, a source video, a picture), its row says
   **Needs:** with a button that takes you to the step, and step 5 will not queue the clip until it is there.
4. Write the prompt the way the card's note says, then render as usual. A clip with a LoRA takes a slower path and
   shows no time estimate.

A slider LoRA such as *Anime Slider* has a track centred on 0: drag right for flat 2D anime (3 is plenty on a clip
from text), left for a photographic look, or press *Off (0)*. On a Mac, a slider on a clip *from references* pulls
about half as hard as on a DGX Spark: use about twice the strength there.
*Equirectangular 360°* wants the Custom size 1120x480; the clip is the whole sphere, to be resized to 2:1 for a VR
player.

### LoRAs for a clip made from references

Three LoRAs in the hub do one job each on a clip made from references (step 1 of Create, *From references*). Pick
one and its row says what the clip needs and whether it has it; *Go to References* takes you there.

| LoRA | Put under References | Then |
|---|---|---|
| **Character Swap** | your clip as Video 1, a picture of the new character | under Video 1, in "who or what it shows", say who is replaced: *the man in the blue jacket* |
| **Lineart Anime** | any clip as Video 1, and no picture | press *Make line art of Video 1* in the LoRA's row (fine, medium or outlines only), then describe the finished picture in Scene and Action |
| **Keyframe Animation** | one drawing, or two (the first drawing and where the action ends) | choose 20 passes; Scene is the framing and the medium (*clean line on white*), Action is the movement |

The prompt in Review is then written the way that LoRA's author trained it; you can still edit it. The **Anime
Slider** works on any clip, references or not: plus leans to flat 2D anime, minus to a photographic look.

Worked example, Lineart Anime: add a clip of a man practising at a wooden dummy as Video 1, pick Lineart Anime,
press *Make line art of Video 1*, and write Scene "A young martial artist with short black hair in a dark blue
jacket on a rooftop at dusk, red paper lanterns", Action "He strikes a wooden training dummy". The clip comes back
drawn as anime on the same outlines and timing.

## 8.6 Library actions: turn a picture round, make a clip larger, take a person out

Three LoRAs of the hub have a job of their own, and the Library has a button for each. They are queued like any
render; the result lands in the Library.

**Turnaround** (under a picture; Mac and DGX Spark). Five views of the picture's subject, turning from where it
faces now, made together so the character is the same in all five.

1. Press **Turnaround** under a picture. A character or an object standing upright, whole in the frame, works best.
2. Choose the size of each view: *512* is the fastest, *1024* the one to keep. Choose **The turn** (goes furthest
   round) or **The look** (turns less, stays closer to your picture).
3. Press **Make the turnaround sheet**. The sheet shows the five views side by side; each view is also a picture of
   its own, with **first frame** (start a clip on it) and **reference** (add it to a reference clip's pictures).

If a view comes out tilted or seen from above, change the seed. The instruction box takes your own wording; it
names the picture as `<Picture 1>`.

**Upscale** (under a clip; DGX Spark). A finished clip made larger and sharper, with its motion, mouths and sound
left as they were.

1. Press **Upscale** under a clip and choose the size: *2 megapixels* is about 1920 x 1080.
2. Leave the re-draw on **Light**. *None* only enlarges; *Stronger* adds more detail of its own.
3. Press **Upscale this clip**. A 5 second 864x480 clip took about five minutes at 2 megapixels.

**Remove a person** (under a clip; DGX Spark). One person taken out, the background drawn again.

1. Press **Remove a person** under a clip and say who, in a few words: *man in a blue shirt*, *woman in a red coat*.
2. Press **1. Find the person**. In under a minute the clip comes back with the person painted green. Look at the
   hands, the hair and the shadow. If something is missed, or someone else is green, describe the person more
   exactly and press it again.
3. The removal needs a first frame without the person. **Build it for me** makes one from the clip. With **My own
   picture** you load your own edit of the clip's first frame (same size, the person painted out).
4. Press **2. Remove the person**. The clip is worked through in overlapping pieces of 22 frames; a 5 second clip
   took about eleven minutes.

The whole picture is drawn again, so small things elsewhere can change too. It works best on a short, continuous
shot. The clip keeps its own sound, which still holds the removed person's voice: tick **No sound** to leave it out.

## 10. Without the app

```bash
bash scripts/fast.sh "a prompt" out.mp4                          # FastH3: STEPS=4|8|20 RES=480p|720p UPSCALE=1
bash scripts/image.sh "a prompt" out.png                         # image only
bash scripts/studio.sh "the picture" "what happens" out.mp4      # Qwen image, then video (ENGINE=fasth3 for FastH3)
FIRST_FRAME=photo.jpg bash scripts/video.sh "what happens" out.mp4
```

## If something goes wrong

| What you see | What to do |
|---|---|
| "run oneshot-setup.sh first" | the install did not finish; run it again, it resumes |
| "no FastH3 checkpoint" | `bash oneshot-setup.sh --no-render` |
| The app says it did not start | read `~/Library/Logs/TensorFoldStudio.log`; often another program holds port 7870 (`PORT=7871 bash scripts/make-app.sh`) |
| A job fails | open its **Log** in the queue; the last lines say why |
| Very slow renders | check the chip is M5-family and nothing else is using the GPU |
| The job says "MLX engine" for FastH3 | the native engine is not installed or the chip is older than M5: run `bash oneshot-setup.sh --verify` and look for the `tf-h3-dit` line |
