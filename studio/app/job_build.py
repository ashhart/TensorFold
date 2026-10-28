"""A queued job's command: which script runs with which environment and writes which files, checked when queued."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

import actions
import lengths
import lora_flows
import memory_guard
import picture_models
import voice
from engines import custom, media_seconds, ref2va_ready, ref2va_turbo_ready
from motion_prompt import freed
from settings import (AUDIO_SUFFIXES, DURATIONS, FPS, HOME, LONGEST, PACK, PRESETS, QUALITIES, REFERENCE_AUDIO_LIMIT,
                      REFERENCE_CLIP_SECONDS, REFERENCE_KIND_SECONDS, REFERENCE_LIMIT, REFERENCE_TOTAL,
                      REFERENCE_VIDEO_LIMIT, SCOUT_SIZES, SPARK, SPARK_ENGINE_LIB, VIDEO_SUFFIXES, inside, relative)

# Set by server.py once the routers exist: the LoRA hub's resolver and lead, the actions' and edits' builders.
lora_resolve = lora_lead = action_build = edit_build = None


def references(params: dict) -> dict:
    """The job's references under the studio folder: {} or ``pictures``, ``videos``, ``audios`` and ``video_sound``."""

    found = {kind: [str(inside(path)) for path in (params.get(key) or []) if path]
             for kind, key in (("pictures", "references"), ("videos", "reference_videos"), ("audios", "reference_audios"))}
    if len(found["videos"]) == 1 and freed(params.get("pace"), params.get("motion") if isinstance(params.get("motion"), dict) else None, len(found["pictures"])):
        found["videos"] = []  # Impacts over a slow reference: the prompt carries the fight (see freed)
    pictures, videos, audios = found["pictures"], found["videos"], found["audios"]
    if not (pictures or videos or audios):
        return {}
    for kind, limit, files in (("reference pictures", REFERENCE_LIMIT, pictures), ("reference videos", REFERENCE_VIDEO_LIMIT, videos),
                               ("reference audio clips", REFERENCE_AUDIO_LIMIT, audios)):
        if len(files) > limit:
            raise HTTPException(400, f"at most {limit} {kind}")
    if len(pictures) + len(videos) + len(audios) > REFERENCE_TOTAL:
        raise HTTPException(400, f"at most {REFERENCE_TOTAL} references in all")
    if audios and not (pictures or videos):
        raise HTTPException(400, "reference audio needs a reference picture or video beside it: the model does not take sound alone")
    if params.get("image") or params.get("first_frame_from"):
        raise HTTPException(400, "references do not combine with a first frame")
    if not ref2va_ready():
        raise HTTPException(400, "MiniMax H3 Ref2VA is not installed here (oneshot-setup.sh --ref2va adds it)")
    if params.get("ref_turbo") and not ref2va_turbo_ready():
        raise HTTPException(400, "the Ref2VA Turbo adapter is not installed here (oneshot-setup.sh --ref2va adds it)")
    missing = [path for path in pictures + videos + audios if not Path(path).is_file()]
    if missing:
        raise HTTPException(400, f"no reference file at {relative(Path(missing[0]))}")
    wrong = [path for path in videos if Path(path).suffix.lower() not in VIDEO_SUFFIXES]
    if wrong:
        raise HTTPException(400, f"a reference video is an mp4 or mov file, and {relative(Path(wrong[0]))} is not")
    wrong = [path for path in audios if Path(path).suffix.lower() not in AUDIO_SUFFIXES + VIDEO_SUFFIXES]
    if wrong:
        raise HTTPException(400, f"a reference audio clip is a sound file, and {relative(Path(wrong[0]))} is not")
    video_sound = bool(videos and params.get("reference_video_sound"))
    measured = (params.get("motion") or {}).get("measured") if isinstance(params.get("motion"), dict) else None
    if video_sound and isinstance(measured, dict) and measured.get("sound") == "impacts" and params.get("pace") == "flow":
        video_sound = False  # Flow by choice: the reference's own hits are not reused (as the prompt says)
    low, high = REFERENCE_CLIP_SECONDS
    for kind, files in (("video", videos), ("audio clip", audios)):
        total = 0.0
        for path in files:
            seconds, sound = media_seconds(path)
            if seconds is None:
                continue
            if not low - 0.05 <= seconds <= high + 0.05:
                raise HTTPException(400, f"a reference {kind} is {low:g} to {high:g} seconds long, and "
                                         f"{relative(Path(path))} is {seconds:.1f} s")
            if not sound and (kind == "audio clip" or video_sound):
                raise HTTPException(400, f"{relative(Path(path))} has no sound" + (": untick the videos' own sound"
                                                                                  if kind == "video" else ""))
            total += seconds
        if total > REFERENCE_KIND_SECONDS + 0.05:
            raise HTTPException(400, f"the reference {kind}s are at most {REFERENCE_KIND_SECONDS:g} seconds in all, got {total:.1f} s")
    return {**found, "video_sound": video_sound}


# The longest clip in frames by the canvas's pixels: measured on a DGX Spark (lengths.py); a Mac keeps the model's 15 s.
SPARK_LONGEST = ((1376 * 768, lengths.LONGEST),)


def longest_frames(env: dict) -> int:
    """The longest clip this machine renders at the size in ``env``."""

    if not SPARK:
        return lengths.LONGEST
    try:
        pixels = int(env.get("WIDTH") or 864) * int(env.get("HEIGHT") or 480)
    except (TypeError, ValueError):
        return lengths.LONGEST
    return next((frames for most, frames in SPARK_LONGEST if pixels <= most), SPARK_LONGEST[-1][1])


def clip_frames(params: dict, env: dict | None = None) -> int:
    """The frame count of a single clip, or 400 with the reason."""

    try:
        return lengths.of(params, longest_frames(env or {}))
    except ValueError as error:
        raise HTTPException(400, str(error))


def clip_seconds(params: dict) -> float:
    """A job's length for the time estimate: a long video's total, else the clip's own seconds."""

    if params.get("total_seconds"):
        return int(params["total_seconds"])
    try:
        frames = lengths.of(params)
    except ValueError:
        return 5.0
    # the measured times are for the four classic lengths by their round names: 124 frames is "5 seconds"
    return {count: named for named, count in DURATIONS.items()}.get(frames) or min(frames / FPS * 120 / 124, float(LONGEST))


def memory_of(params: dict, env: dict, frames: int, lora: bool, found: dict | None = None) -> dict:
    """Whether a clip fits this DGX Spark's memory (spark/memory_guard.py); everything fits on a Mac."""

    if not SPARK:
        return {"fits": True, "reason": ""}
    try:
        width, height = int(env.get("WIDTH") or 864), int(env.get("HEIGHT") or 480)
    except (TypeError, ValueError):
        width, height = 864, 480
    found = found or {}
    videos = found.get("videos") or []
    lora = lora or bool(found and params.get("ref_turbo"))
    engine = SPARK_ENGINE_LIB.exists() and (not lora or memory_guard.engine_takes_lora(SPARK_ENGINE_LIB))
    # a Drama 3 server kept loaded is unloaded before this job starts (voice.release): its memory counts as free
    drama3 = voice.room_mb()
    room = max(memory_guard.room_mb() or 0, drama3) if drama3 else None
    return memory_guard.check(width, height, frames, lora=lora, references=bool(found), available_mb=room,
                              base=env.get("SPARK_BASE") == "h3", engine=engine,
                              pictures=len(found.get("pictures") or []),
                              reference_frames=sum(min(frames, int(round((media_seconds(path)[0] or frames / FPS) * FPS))) for path in videos))


def build(kind: str, params: dict, job_id: str, room: bool = True) -> tuple[list[str], dict, list[str]]:
    """(command, environment, output paths) for one job; ``room`` False skips this machine's memory (another Spark)."""

    scripts = PACK / "scripts"
    if kind in actions.KINDS:  # a Library action (actions.py): turnaround, upscale, mask, remove
        return action_build(kind, params, job_id)
    seed = str(int(params.get("seed", 0)))
    if kind == "scout":
        folder = HOME / "scouts" / job_id
        prompt = params["prompt"]
        count = max(1, min(int(params.get("count", 4)), 12))
        # a scout is made at the size the video model will start from (the draft keeps a full 1344x768 still)
        width, height = SCOUT_SIZES.get(params.get("preset", "2k"), (1024, 576))
        if params.get("width") and params.get("height"):  # an image for its own sake, at the size asked for
            width, height = int(params["width"]), int(params["height"])
            if width % 16 or height % 16 or not (256 <= width <= 2048 and 256 <= height <= 2048):
                raise HTTPException(400, "image sides are multiples of 16 between 256 and 2048")
        env = {"SEED": seed, "WIDTH": width, "HEIGHT": height, **picture_models.image_model_env(params)}
        first = int(seed)
        outputs = [str(folder / f"scout_s{first + i}.png") for i in range(count)] + [str(folder / "sheet.jpg")]
        return ["bash", str(scripts / "scout.sh"), prompt, str(count), str(folder)], env, outputs
    if kind == "export":
        return ["python3", "-c", "pass"], {}, []
    if kind == "edit":  # a picture re-drawn by the image model (picture_edit.py)
        return edit_build(params, job_id)
    preset = params.get("preset", "small" if SPARK else "draft")
    quality = params.get("quality", "standard")
    chosen, _ = custom(params)
    if (preset not in PRESETS and preset != "custom") or quality not in QUALITIES:
        raise HTTPException(400, "unknown preset or quality")
    sized = {**(PRESETS[preset][2] if preset in PRESETS else {}), **chosen}
    # any length the model takes: "frames" (17n + 5), or "seconds" brought to the nearest such count
    frames = clip_frames(params, sized)
    target = Path(params["output"]) if params.get("output") else HOME / "renders" / f"{job_id}.mp4"
    target = inside(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    env = {"SEED": seed, "FRAMES": frames, **(PRESETS[preset][2] if preset in PRESETS else {}), **QUALITIES[quality][1]}
    fast_h3 = SPARK or env.get("ENGINE") == "fasth3"
    prompt = params["prompt"]
    image = params.get("image")
    styles = lora_resolve(params.get("loras"))  # the LoRA hub (loras.py): files to merge, and their trigger words
    if styles:
        # merged at load into the weights the clip runs on (Ref2VA's in bfloat16 on a Spark, see spark_generate.py)
        env["LORA"] = "\n".join(f"{one['path']}:{one['strength']:g}" for one in styles) + "\n"
        prompt = lora_lead(prompt, styles)
    # the end keyframe (MiniMax H3's first-and-last-frame mode): a picture of its own, or the first frame again
    ending = params.get("last_image") or (image if params.get("last_same") else None)
    if ending:
        if kind not in ("video", "studio"):
            raise HTTPException(400, "a last frame is for single clips")
        if kind == "video" and references(params):
            raise HTTPException(400, "references do not combine with a last frame: a reference is not a frame of the clip")
        if not inside(ending).is_file():
            raise HTTPException(400, "the last frame's picture is not in the studio's folder")
        env["LAST_FRAME"] = str(inside(ending))
    if kind == "video":  # text to video, or from an image the user already has
        pictures = references(params)
        if styles:
            # a LoRA made for one job on the reference model says what the clip must hold (lora_flows.py)
            missing = lora_flows.fits(params.get("loras"), len(pictures["pictures"]) if pictures else 0,
                                      len(pictures["videos"]) if pictures else 0, bool(pictures and params.get("ref_turbo")))
            if missing:
                raise HTTPException(400, missing)
        if pictures:
            if styles and lora_flows.plain(params.get("loras")):
                env["REF_PLAIN"] = "1"  # the prompt as written: no `Use <Video 1> as the subject.` ahead of it
            # Ref2VA takes neither the Turbo adapter nor FastH3: the quality setting does not apply
            env = {k: v for k, v in env.items() if k not in QUALITIES[quality][1]}
            # one per line, so a path may hold spaces
            for name, kind in (("REF_IMAGES", "pictures"), ("REF_VIDEOS", "videos"), ("REF_AUDIOS", "audios")):
                if pictures[kind]:
                    env[name] = "\n".join(pictures[kind]) + "\n"
            if pictures["video_sound"]:
                env["REF_VIDEO_SOUND"] = "1"
            if params.get("ref_turbo"):
                env["REF_TURBO"] = "1"
        env.update(passes(chosen, SPARK or (fast_h3 and not pictures)))  # Ref2VA runs on MiniMax H3
        fits = memory_of(params, env, frames, bool(styles), pictures) if room else {"fits": True}
        if not fits["fits"]:
            raise HTTPException(400, fits["reason"])
        if image:
            return (["bash", str(scripts / "studio.sh"), "", prompt, str(target)],
                    {**env, "IMAGE_FILE": str(inside(image)), "RAW_PROMPT": "1"}, [str(target)])
        size = PRESETS[preset][1] if preset in PRESETS else (int(env["WIDTH"]), int(env["HEIGHT"]))
        plain = {k: v for k, v in env.items() if k not in ("TWOK", "QHD")}
        if "TWOK" in env:
            plain.update(X2="1", WIDTH="2048", HEIGHT="1152")
        if "QHD" in env:
            plain.update(X2="1", WIDTH="2560", HEIGHT="1472", CROP="2560x1440")
        if "X2" in env and "WIDTH" not in plain:
            plain.update(WIDTH=str(size[0]), HEIGHT=str(size[1]))
        return ["bash", str(scripts / "video.sh"), prompt, str(target)], plain, [str(target)]
    if kind == "studio":  # text to image to video in one go
        env.update(passes(chosen, fast_h3))
        fits = memory_of(params, env, frames, bool(styles)) if room else {"fits": True}
        if not fits["fits"]:
            raise HTTPException(400, fits["reason"])
        return (["bash", str(scripts / "studio.sh"), params["image_prompt"], prompt, str(target)],
                {**env, **picture_models.image_model_env(params), "RAW_PROMPT": "1"}, [str(target), str(target.with_suffix(".png"))])
    raise HTTPException(400, f"unknown job kind {kind}")


def passes(chosen: dict, steps: bool) -> dict:
    """Hand-set passes as the engine takes them: STEPS for FastH3 and on a Spark, else sigma points (passes + 1)."""

    out = dict(chosen)
    if "STEPS" in out and not steps:
        out["POINTS"] = str(int(out.pop("STEPS")) + 1)
    return out
