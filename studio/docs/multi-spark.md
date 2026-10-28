# More than one DGX Spark: the Sparks setting

A job in the Studio can use more than one DGX Spark. You choose a number from 1 to 8, **Sparks**: how many machines
the job may use, the one the app runs on counted as one. The app shows how many are there and able, uses that many,
and says what the number will do before you queue anything.

On a Mac the setting shows 1: multi-Spark rendering is for DGX Sparks.

## What you need

- The Studio installed on this Spark as usual, and on each other Spark once (`bash oneshot-setup-spark.sh`, or a copy
  of the install and of the FastH3 models). A worker needs the FastH3 models to take clips; the image model, Ref2VA,
  LoRAs and ffmpeg are optional there, and the app sends a worker only what it can render.
- A key login from this Spark to each other Spark as the user who runs the app (`ssh 192.0.2.10` must get in
  without a password). Nothing is installed as root, no system file is edited and no folder is shared.

## Set it up: the Sparks tab

1. Open **Sparks** (the tab beside Library).
2. Under *Add a Spark*, type the other machine's address as you would after `ssh` (`192.0.2.10`, or
   `user@spark-5`) and press **Add**. If its Studio is not in the same folder as on this Spark, give the folder.
3. The app looks at it and shows one of: *idle*, *rendering*, *its Studio app is not running*, *its Studio code
   differs from this Spark's*, *no Studio installed there*, *does not answer over ssh*.
4. Press **Update and start**. This sends this Spark's Studio code to the worker (rsync over ssh, as you: code only,
   never models), starts the worker's app on the worker's own `127.0.0.1:7870`, and checks it again. A worker renders
   only when its code is this Spark's, which is also how it gets the memory guard.
5. Each Spark's row then says: its Studio's version, memory available, disk free, and what it renders (FastH3 clips
   from text, a first frame or a last frame; Ref2VA; the image model; how many LoRAs; ffmpeg).

**Check** looks again and changes nothing. **Stop** stops the worker's app and its ComfyUI server, which gives its
memory back. **Remove** takes it off the list. Untick **use** to keep a Spark in the list and give it nothing.

A worker's app is never opened to the network. This app reaches it through an ssh tunnel that it opens itself and
closes when you remove the worker or stop the app. The list is kept in `sparks/sparks.json` in the studio folder
(`~/TensorFoldStudio`), not in the code.

## Choose the number

The chooser, **Sparks 1 2 3 4 5 6 7 8**, is in four places and is one setting:

- at the top of the **Queue**, on every tab;
- in the last step of **Create**, above *Queue this clip*;
- on the **Production** board, above *Generate every segment not yet rendered*;
- in a **Slate** shot, under *Render this shot*.

Under it the app says what will happen, for example *5 of 8 available. One clip per Spark, 5 at a time: 8 clips on
5 Sparks.* with a time estimate, and which Sparks are left out and why (*spark-b (the LoRA camera-motion is not
installed there)*). Choosing more than there are is allowed: the app uses what is there. Numbers above the ones
available are shown dimmed.

Through the API the number is `sparks` beside `kind` and `params`:

```bash
curl -s -X POST http://127.0.0.1:7870/api/jobs -H 'Content-Type: application/json' \
  -d '{"kind": "video", "sparks": 5, "params": {"prompt": "…", "preset": "small", "quality": "fh8", "frames": 56}}'
curl -s -X POST http://127.0.0.1:7870/api/sparks/plan -H 'Content-Type: application/json' \
  -d '{"sparks": 5, "kind": "video", "clips": 8, "params": {"preset": "small", "quality": "fh8", "frames": 56}}'
```

A job queued without `sparks` takes the number set on the Sparks tab. `GET /api/sparks` is the list and its state.

## What the number does

| The job | With Sparks = N |
|---|---|
| Several clips: jobs in the queue, the segments of a Production board, the shots of a Slate scene | One clip per Spark, side by side, on this Spark and the first N - 1 workers that are up and can render the clip. |
| One clip | It renders on one Spark, the first that is free. One clip is not split across Sparks in the app (see the end of this page). |
| A segment or shot that opens on the last frame of the one before | It waits for that one, then renders on any free Spark. A board that is one chain renders one clip at a time. |
| A long video (over 15 s) | One Spark, this one: it is a chain, each piece opens on the last frame of the one before. |
| A clip with a LoRA, or from references (Ref2VA) | Only on Sparks that have that LoRA, or Ref2VA, installed, and only when that Spark's memory guard accepts the clip. With none: this Spark. |
| Pictures, edits, Library actions (Turnaround, Upscale, Remove a person), a timeline export, a clip that starts from a new picture | This Spark. |

A clip rendered on a worker comes back into this Spark's studio folder at the path it would have had here (the
Library, a project's segment, a Slate take), and the QA gate, the shot check and the silenced sound track run here,
on the fetched clip, where their tools are. The job's card says *Rendered on spark-a*, the Library shows the
Spark's name and the seed under the clip, and `sparks/takes.jsonl` in the studio folder keeps one line per clip:
path, seed, settings, the Spark, and the time it took.

## What keeps it safe

- **One job on a Spark at a time.** A worker that is rendering anything, for this Spark or for someone else, is
  given nothing until it is idle.
- **Every clip passes the worker's own memory guard** (*The memory guard* in the README): this app asks it before
  anything is sent, the worker's app asks it again when it queues the job, and the guard stands beside the worker's
  ComfyUI server while it renders. A clip a worker refuses is rendered on another machine and the worker stays in
  use for other clips.
- **A worker that stops answering is not waited on.** After 60 seconds of silence while it renders it is marked
  *stopped answering*, its tunnel is closed, and its clip goes back to the queue for another machine. It is given
  nothing more until you press **Check**. A clip is taken back from a worker at most twice; after that it renders
  on this Spark.
- **Cancel** cancels the clip on the worker too.
- **Stopping the app** closes its tunnels. The workers' apps keep running, idle, until you press **Stop** on each;
  their models stay in memory until then.

## Measured

Eight plain FastH3 clips (56 frames, 864x480, 8 passes, TensorFold's int8 engine, seeds 101 to 108), queued through
the app's own API with `sparks` 1 to 5 on 2026-10-13: this Spark and four workers on the 200G fabric, models loaded
on all five beforehand, one run each. Wall time is from the first job queued to the last clip finished and checked.

| Sparks | Wall time, 8 clips | Against one Spark | The most there is to have (8 clips in rounds) | Clips on each Spark |
|---|---|---|---|---|
| 1 | 263.2 s | 1.00x | 1.00x | 8 |
| 2 | 129.8 s | 2.03x | 2.00x | 4 + 4 |
| 3 | 97.5 s | 2.70x | 2.67x | 3 + 3 + 2 |
| 4 | 69.0 s | 3.81x | 4.00x | 2 + 2 + 2 + 2 |
| 5 | 65.0 s | 4.05x | 4.00x | 2 + 2 + 2 + 1 + 1 |

A clip took 32 to 35 s on every Spark, the hand-over to a worker and the fetch included (0 to 2 s more than a clip
made here). **Five Sparks were not much better than four here** (65 s against 69 s): eight clips go in two rounds either
way. The fifth Spark pays when the count of clips makes a round less, ten clips or more. All 40 clips landed in this
Spark's Library with the right seed, settings and Spark in the takes list; available memory, read every 5 s, stayed above 52 GB on
this Spark and 61 GB on the workers.

Earlier the same day, without this setting (a hand-written driver over the same job API), a production's 22 clips
on five Sparks came to 4.09 times one Spark.

The time the app shows at a number is worked out from this table: the clips' rounds at that number, less what the
measured run lost at four. It is an estimate. Six to eight Sparks are not measured: five machines exist here.

## What is not built

- **One clip split across Sparks.** It is measured, outside the app, on the engine's `h3-cuda-multispark` branch
  with ranks on the other Sparks: a 5 s 864x480 FastH3 clip took 72.2 s on one Spark and 51.4, 41.8 and 40.9 s on
  two, three and four (1.40x, 1.73x, 1.77x), with the same bits in every frame; a 10 s 1280x736 clip's pass was 1.78x
  faster on two and 2.81x on four. Only plain FastH3 clips on the int8 engine, no LoRA and no Ref2VA. What is
  missing before it can be in the app: the installer does not build that engine branch; each rank needs the 21 GB
  int8 weights and the engine library on its Spark, started, stopped and shown by the app; a rank that dies fails
  the clip, with no fall back to one Spark; a rank has only its own 8 GB check, not the memory guard; this Spark as
  a rank, and so five on one clip, was never run. Until then one clip renders on one Spark and the page says so.
- **A long video's pieces side by side.** The pieces are a chain. Rendering them at once needs keyframes between
  the pieces, which is another way of making a long video and is not built.
- **Pictures and Library actions on a worker.** Only single clips are handed over.
- **The queue across a restart.** Jobs are kept in memory: if this app stops while a worker renders, the worker
  finishes its clip in its own folder and this app, started again, does not fetch it.
- **Starting a worker's install.** The app sends code, not models or the environment.
- **Six to eight Sparks.** The limit is 8 (`LIMIT` in `app/sparks.py`); nothing above five has run.
