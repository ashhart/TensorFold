# Motion Previs: a shot into control layers and a camera move

**New to it? Start with the walkthrough:** [TUTORIAL.md, section 9](../TUTORIAL.md#9-motion-previs-make-a-clip-that-moves-like-another)
(two worked examples, what each choice does, and what to do when a clip goes wrong), or **Tutorial → Motion Previs** in
the app.

The **Motion Previs** tab takes a clip (one you upload, or an earlier render or upload), a range of 2 to 15 seconds
and one of four reference modes (camera only, actor motion, object motion, full scene), and writes a production pack
under `~/TensorFoldStudio/motion/<id>/` with a ZIP beside it: the trimmed reference clip, a pose track
(`pose_landmarks.json`), an OpenPose BODY_25 skeleton video and `openpose_keypoints.json`, a subject-masked camera
solve (`camera_motion.json`), depth (Depth-Anything-small, and an ffmpeg proxy), edges, line art, a motion mask, a
contact sheet, a shot bible and a prompt pack. The tab shows the reference, skeleton, depth and edges side by side
and says the camera move in a sentence ("The camera pushes in while it slowly tilts up ..."). Stages report their own
progress and Cancel stops the job and keeps nothing. A pack ZIP exported by Motion Previs Studio can be uploaded in
place of an analysis.

**The analysis is a Python port, modified, of [Motion Previs Studio](https://github.com/wassermanproductions/motion-previs-studio)
by Sam Wasserman** (Apache-2.0, Copyright 2026 Sam Wasserman). Packs use his file names and manifest layout, so they
are interchangeable with his app's. His NOTICE is at `LICENSES/Motion-Previs-Studio-NOTICE`; what was changed and the
notices for MediaPipe, Depth Anything, OpenCV and ONNX Runtime are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

```bash
bash oneshot-setup.sh --motion      # or: bash scripts/setup-motion.sh   (0.5 GB, an environment of its own)
```

`--motion` is an option, not part of the default install: MediaPipe brings its own OpenCV build, which must not replace
the Studio's, so the analysis has its own environment (`$PREFIX/motion-venv`, 433 MB, and `$PREFIX/motion-models`,
70 MB). Without it the tab still trims the shot and makes the ffmpeg layers, and on a Mac the camera solve; pose and
AI depth say that they are not available on this machine. Measured on a Mac Studio M5 Ultra, a 5 second 864x480 shot
with every layer at 12 samples a second: 6.6 s in all (prepare 0.2, pose 1.3, camera 0.1, depth 3.4, encode 0.9,
pack 0.1), 0.5 GB of memory at its peak, on the processor. On a DGX Spark (`bash oneshot-setup-spark.sh --motion`;
MediaPipe 1.1.0, OpenCV 5.0.0 and ONNX Runtime 1.31.0 all installed from their Linux aarch64 wheels) the same kind of
shot with every layer took 12.6 to 17.6 s (prepare 0.4, pose 3.3, camera 0.2, depth 5.8, encode 1.9, pack 0.4 in the
quicker run), 0.4 to 0.5 GB at its peak, with the same camera sentence the Mac gave for that clip. On the Spark the
tab was driven through its API and a screenshot of it from a headless Firefox was looked at; it has not been used by
hand in a browser on either machine. `?motionref=1` in the app's address opens the create wizard on the stored motion
reference.

**Use as motion reference** makes a hand-off (one layer of the pack resampled to the length and size of the clip to
be made, the camera move in words, and the reference mode: `motion_reference()` in `app/motion.py`) and opens the
create wizard on its references screen with that layer as `<Video 1>` of a MiniMax H3 Ref2VA clip.

## Motion as a reference in the create wizard

*The clip starts from → References*, add a reference video, then **Analyse motion** under it (offered when the
motion environment is installed; without it the video is a plain reference). The analysis runs on the processor in
well under a minute and gives two choices:

- **What to carry over:** *Full scene* (the same place and motion with a new person: its person is replaced by
  `<Picture 1>`), *Actor motion* (the body motion and the camera, in a new place), *Object motion* (an object's path
  and the camera) or *Camera only*.
- **What goes to the model as `<Video 1>`:** the **source clip** (the default for *Full scene*), the **depth** layer
  as a control video (the default for the other three), or the **skeleton** (offered, never the default) or edges.
- **People in `<Video 1>`:** *One* or *Two or more*. With two or more the prompt speaks of its people and figures,
  one for one. You say it: the pose pass, its skeleton layer and the distance figures below follow **one person
  only** (asked for up to four people on a two-person duel at arm's length, it returned one in every frame).

**Fast motion, on the Mac** (through the app, 5 s at 864x480, seed 7; two cuts of a fictional martial-arts short: one
man drilling a wooden dummy while the camera pushes in to a close-up, and two fighters trading hands while the camera
cranes up to overhead; pose distance as in the table below, median / 90th percentile):

| shot | `<Video 1>` | passes | time | distance | what came out |
|---|---|---|---|---|---|
| dummy drill | source clip + a picture of a woman | Turbo 4, as first shipped | 236 s | 26.4 / 36.6 | the place kept; she drifts out of frame, strikes lost |
| dummy drill | the same request | Turbo 4, without the int8 kernels | 231 s | 3.7 / 11.2 | held |
| dummy drill | the same request | 20 | 741 s | 4.2 / 11.7 | held: she does the drill and the push-in on the same rooftop |
| dummy drill | depth + a picture of her in a bamboo clearing | Turbo 4 | 235 s | 3.5 / 11.7 | held, in the new place |
| dummy drill | depth + a picture of her on the same rooftop | Turbo 4 | 247 s | 16.0 / 25.3 | the drill and the push-in, with the layout shifted |
| duel | depth + a picture of the pair in a temple courtyard | Turbo 4 | 234 s | not valid | held by eye: both new fighters in time, the crane-up followed |
| duel | source clip + a picture of a new pair, *One* person | Turbo 4, without the int8 kernels | 294 s | 3.2 / 8.3 | motion and crane-up followed; only one fighter replaced |
| duel | the same with *Two or more* people | Turbo 4, without the int8 kernels | 284 s | 3.1 / 8.3 | both fighters replaced, motion and crane-up followed |

The first row was a defect of the Mac build, not of the source-clip route (the DGX Spark held the same request at
2.9 / 7.7): the Ref2VA Turbo adapter, a reference video and the three int8 kernels together lose the motion. Any one
kernel alone held (3.4 to 3.5), any two did not (14 to 23), and 20 passes without the adapter held with all three.
`scripts/video.sh` now runs that one case in bfloat16, about 13 s more a pass at 864x480 (`REF_INT8=1` puts the
kernels back). The depth duel's figures are not valid because the checker follows one person and picked different fighters in
source and result; the source-clip duel's are for the one fighter it followed. The adapter at 3 passes held the
dummy drill as well (3.2 / 10.5, 257 s). One seed each.

The camera move in words goes into the prompt in every case (a switch; text of your own in the Camera field wins).
With the source clip and a picture, step 4 writes "The target video is `<Video 1>` with its person replaced by
`<Subject 1>`"; calling the picture a first-frame anchor there made the picture's layout override the video. For a
control layer it writes the control-video form: `<Video 1>` is declared the structural template (layout, silhouettes,
positions, movement and camera path frame by frame), `<Picture 1>` the first-frame anchor, with a legend of the layer
(a skeleton's coloured lines are joints and limbs on black; in a depth map brighter is nearer), "These colours are
labels only, not the look of the target video", and the closing "no skeleton lines, no flat-coloured, depth-map or
segmentation-map look". Give a picture of the opening frame as `<Picture 1>` with a control layer: the look and the
setting come from it.

Where the defaults come from: one shot measured on the Mac (a martial-arts sequence with a kick and a punch toward
the camera, fictional people, 5 s at 864x480; the distance of the body's keypoints from the source's, in percent of
the frame width, lower is closer):

| `<Video 1>` | Distance | What it kept |
|---|---|---|
| Source clip, with a picture of another person | 0.6 to 2.2 | the shot move for move, in its own place, with the new person |
| Depth, with a first-frame picture | 1.3 to 2.9 | the kick and the punch toward the camera; the setting is the picture's |
| Depth, with a picture of another setting | 2.7 to 5.5 | the new place with no leak of the source's; the punch, not the kick |
| Skeleton, with a first-frame picture | 2.0 to 4.5 | the scale and the camera; the kick was lost |
| The picture alone | 5.0 to 9.8 | |

No control layer left a trace of its look in a clip. One shot: read the table as an order, not as figures to promise.
On the DGX Spark the same routes were rendered through the wizard's own API calls with the Turbo adapter and looked
at, not measured: see [the Spark's table](../README.md#reference-videos-audio-and-motion-on-a-dgx-spark).

## Sound and movement of a motion reference: Impacts or Flow

A clip's sound and its movement have to agree. The first duel renders above did not: the reference is two people
doing slow sticky hands (forearms in contact, circling), the soundscape asked for "quick slaps of forearms and hands",
and the model did both: it followed the slow hands and laid 1.7 to 1.9 slaps a second over them, on the Mac and on
the DGX Spark alike. Three things were measured before anything was changed:

- **The slaps came from the Soundscape field**, not from the source clip (its own sound is at -54 dBFS and was not
  sent).
- **The slow hands came from the reference.** The source clip's quickest movement is 14 percent of the frame width a
  second; the renders made from it are at 14 to 17. A reference video is followed whatever the prompt says about
  speed: four wordings were tried on the Spark (fast words in the action, a template that says the slow hands are
  not followed, *Camera only*, and the clip played at twice its speed and back), and the arms stayed at 15 to 21.
- **The music of the Mac's fourth test** was the model's own: nothing in that prompt asked for music.

**Sound and movement**, under *Analyse motion* on the references screen, is the choice that follows from this:

| | What the prompt says | What is sent | Sound |
|---|---|---|---|
| **Impacts** | both fight at full speed: sharp strikes and hard blocks, attacking and blocking, every strike heard as it lands | a reference that moves fast goes in as before. One that moves slowly is **left out** (a switch, on by default, when there is a picture of who is in the clip): the picture and the prompt carry the fight, and the camera move of the video stays, in words | the hits of the soundscape |
| **Flow** | the movement is slow, smooth and continuous, and no strike lands | the reference, as before | the hits are taken out of the soundscape and a video's own hits are not reused. Without music the clip is **delivered silent** (its track is muted after the render); with music named in step 3, under that music |
| **Automatic** (the default) | as you wrote it, except that a soundscape that names hits over a reference that moves slowly becomes Flow | the reference | as Flow in that one case |

The analysis measures the reference for this (`app/pace.py`, with ffmpeg and NumPy alone): **movement** is how fast
the quickest parts of the picture move against the rest of it, in percent of the frame width a second (frames lined
up on the camera's shift, each pixel's change over its edge strength, the three quickest blocks less the median
block, 90th percentile over the clip), read also against the size of the action in the frame, so that a fight seen
from across a roof is not called slow for being small; **hits** are sharp onsets a second that fill the band above
600 Hz and die away within 50 ms, which a plucked note and a spoken consonant do not. The same two figures are taken
of every finished reference clip and shown under it ("Sound and movement: agree", or "impact sound over slow motion"
in red): a flag beside the clip, like the QA gate's lines, never a failed job. `pace`, `silent` and
`motion.measured` / `motion.free` are the job's settings for scripts that drive `/api/jobs`.

Measured on the duel, through the wizard's own API calls, Turbo 4 passes, 5 s at 864x480, seed 7 (movement in percent
of the frame width a second, then the same against the size of the action; fast is 24 or 44):

| machine | Sound and movement | time | movement | hits a second | checker | by eye |
|---|---|---|---|---|---|---|
| Mac | before: source clip, slaps in the soundscape | 284 s | 14.2 / 34.8 slow | 1.7 | impact sound over slow motion | slow sticky hands under slaps |
| Mac | Impacts (video left out) | 89 s | 25.0 / 62.4 fast | 2.5 | agree | both throw straight punches and block, in time with the slaps; a new rooftop from the Scene text, the push-in and tilt followed |
| Mac | Flow, no music | 262 s | 14.5 / 35.5 slow | 0 | agree | the slow hands of the reference, silent |
| Mac | Flow, with music | 272 s | 15.6 / 40.4 slow | 0 | agree | the slow hands under a slow plucked melody |
| DGX Spark | before: source clip, slaps in the soundscape | 216 s | 16.8 / 40.6 slow | 1.9 | impact sound over slow motion | slow sticky hands under slaps |
| DGX Spark | Impacts (video left out) | 101 s | 17.0 / 59.0 fast | 2.5 | agree | both strike and block at speed, seen from further off than on the Mac; a new rooftop, the push-in followed |
| DGX Spark | Flow, no music | 206 s | 14.9 / 37.8 slow | 0 | agree | the slow hands of the reference, silent |
| DGX Spark | Flow, with music | 207 s | 15.3 / 37.6 slow | 0 | agree | the slow hands under a slow plucked melody |

What this does not do. **Impacts cannot make a slow reference fast**: with the video left out the clip keeps who is
in it (from the picture) and the camera move (in words), and loses the reference's own place and blocking, which
then come from the Scene text. A picture of the reference's first frame as a second picture was tried for the
place and brought the reference's people back with it after the first second, so it is not used. Before the track
is muted, a Flow clip asked for near silence was near silent on the Spark (-45 dBFS) and under music of the model's
own on the Mac in two of three renders, which is why the track is muted and not left to the prompt. The thresholds
were set on thirteen clips of two shots; they sort the clear cases and call the rest "moderate", which is not
flagged. One seed each.

**The mouth of a reference video.** A picture with a source clip of someone talking, in a clip where nobody speaks:
the new face mouths the reference's words (a woman asked only to smile opens and closes her mouth through the last
second, on both machines). The model copies the lip movement whatever is asked: the same clip came out with a
prompt that says her lips stay together and that the mouth of `<Video 1>` is not followed, and at 20 passes without
the Turbo adapter (761 s on the Spark). What removes it is taking the mouth out of the reference: **Keep the mouth
of `<Video 1>` out of the clip** (on by default when the source clip itself goes in, the clip has no spoken lines and
does not reuse the video's sound) smooths an ellipse over the mouth and jaw of the pose track's person before the
video goes to the model (`soft_mouth()` in `app/motion.py`, ffmpeg and NumPy; `mouth: "soft"` on
`/api/motion/reference`). With it she holds the smile to the end, on the Spark (202 s) and on the Mac (275 s); body,
head and camera follow the reference as before. It needs *Analyse motion* (the pose track) and follows one person.

**The woman who left the frame** (the first dummy-drill row above) was checked again by eye on the Mac through the
app's own job path, the way the wizard runs it, at this release's code: 259 s, she stays in frame for the whole clip
and strikes the dummy as the reference does (movement 28.8 fast, 2.3 hits a second, "agree"). The app goes through
`scripts/video.sh`, so it picks up the fix (a reference video with the Turbo adapter runs without the int8
kernels) with nothing to set.
