"""Voice: Fish Audio Drama 3 text to speech on this machine (TensorFold's MLX or CUDA engine), never Fish's API."""
# Fish Audio Community License v1.0 weights: a hosted TTS or cloning service needs Fish's licence; clones need consent

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from voice_engines import Cuda, Mlx
from voice_machine import SPARK, drop_lock, lock_owner, take_lock  # noqa: F401  (lock_owner: the tests')
from voice_served import LIVE, live_wav
from voice_text import (SAMPLE_LINE, SPEAKER, checked_by_engine, dialogue, engine_score, expected_seconds,  # noqa: F401
                        match, one_speaker, pieces, sheet_text, spoken)

router = APIRouter()
HOME: Path = Path.home() / "TensorFoldStudio"
FRAME_RATE = 21.5  # the Fish codec's frames per second of audio
SAMPLING = {"temperature": (1.0, 0.1, 2.0), "top_p": (0.9, 0.1, 1.0), "top_k": (30, 1, 200)}  # Fish's defaults, then the range
TEXT_LIMIT = 6000  # characters in one take (long text is split at sentence ends by the engine)
REFERENCE_SECONDS = (2.0, 60.0)  # a voice to clone: Fish suggests 10 to 30 seconds of clean speech
CREDIT = "Built with Fish Audio"
LICENCE = ("Fish Audio Drama 3 runs on this machine under the Fish Audio Community License v1.0 (research, non-commercial "
           "and small-company commercial use; a hosted text-to-speech or voice-cloning service needs a licence from "
           "Fish Audio). Clone a real person's voice only with their consent.")


def tool(name: str) -> str | None:
    return shutil.which(name) or shutil.which(name, path="/opt/homebrew/bin:/usr/local/bin")


BACKENDS = {"mlx": Mlx, "cuda": Cuda}
_backend = None


def backend():
    global _backend
    if _backend is None:
        chosen = os.environ.get("DRAMA3_BACKEND") or ("cuda" if SPARK else "mlx")
        _backend = BACKENDS.get(chosen, Cuda)()
    return _backend


def state() -> dict:
    one = backend()
    why = one.check()
    out = {"available": not why, "why": why, "backend": one.name, "label": one.label, "credit": CREDIT, "licence": LICENCE,
           "defaults": {k: v[0] for k, v in SAMPLING.items()}, "ranges": {k: v[1:] for k, v in SAMPLING.items()},
           "text_limit": TEXT_LIMIT, "reference_seconds": REFERENCE_SECONDS, "speakers": 16,
           "live": bool(getattr(one, "streams", False)) or (hasattr(one, "serving") and one.serving()), "checks": bool(getattr(one, "streams", False)),
           "serves": bool(hasattr(one, "serving") and one.serving())}
    if hasattr(one, "status"):
        out["server"] = one.status()
    return out


def room_mb() -> int | None:
    """MB free before a loaded Drama 3 server loaded (other jobs get it back: it unloads first), else None."""

    one = _backend
    if one is None or not hasattr(one, "loaded") or not one.loaded() or getattr(one, "before_gb", -1) < 0:
        return None
    return int(one.before_gb * 1024)


def release(why: str) -> bool:
    """Unload a loaded Drama 3 server before any other job of the queue, giving back memory and lock; True if loaded."""

    one = _backend
    return bool(one is not None and hasattr(one, "release") and one.release(why))


# ---------- text ----------

def estimate(text: str, referenced: bool, asr: bool) -> int:
    seconds, one = expected_seconds(text), backend()
    if getattr(one, "streams", False):  # the MLX engine checks its words itself (~0.2 s a part): no Whisper after
        asr, seconds = False, seconds * 1.05
    # load (unless loaded), frames at the engine's rate, a reference encoded first, Whisper on the CPU after
    fps, load = float(getattr(one, "fps", 60.0)), float(getattr(one, "load_s", 3.0))
    if hasattr(one, "loaded") and one.loaded():
        load = 0.0
    return int(1 + load + seconds * (FRAME_RATE / fps + 0.05) + (3 if referenced else 0) + (6 + seconds * 0.4 if asr else 0))


# ---------- jobs ----------

def checked(body: dict) -> dict:
    """A take's settings, checked before it is queued: the params of a ``voice`` job."""

    text = str(body.get("text") or "").strip()
    if not spoken(text):
        raise HTTPException(400, "the text has no words to say")
    if len(text) > TEXT_LIMIT:
        raise HTTPException(400, f"at most {TEXT_LIMIT} characters in one take")
    bad = [n for n in SPEAKER.findall(text) if int(n) > 15]
    if bad:
        raise HTTPException(400, f"speakers are <|speaker:0|> to <|speaker:15|>, not <|speaker:{bad[0]}|>")
    why = backend().check()
    if why:
        raise HTTPException(400, why)
    params = {"text": text, "title": str(body.get("title") or "").strip()[:120]}
    try:
        params["seed"] = int(body.get("seed") if body.get("seed") not in (None, "") else random.randrange(1, 100000))
        for key, (default, low, high) in SAMPLING.items():
            value = body.get(key)
            value = default if value in (None, "") else (int(value) if key == "top_k" else float(value))
            if not low <= value <= high:
                raise HTTPException(400, f"{key} is between {low:g} and {high:g}")
            params[key] = value
    except (TypeError, ValueError):
        raise HTTPException(400, "seed, temperature, top_p and top_k are numbers")
    if body.get("ref_audio"):
        clip = inside(str(body["ref_audio"]))
        if not clip.is_file():
            raise HTTPException(400, "no reference clip at " + str(body["ref_audio"]))
        if body.get("consent") is not True:
            raise HTTPException(400, "cloning a voice needs the speaker's consent: tick “I have the speaker's consent to clone this voice”")
        seconds = media_seconds(str(clip))[0]
        low, high = REFERENCE_SECONDS
        if seconds is not None and not low - 0.05 <= seconds <= high + 0.05:
            raise HTTPException(400, f"a voice to clone is {low:g} to {high:g} seconds of speech, and this clip is {seconds:.1f} s")
        ref_text = " ".join(str(body.get("ref_text") or "").split())
        if not ref_text and not qa.ready()["transcript"] and not hears_references(backend()):
            raise HTTPException(400, "write what the reference clip says (this machine has no speech-to-text tool to hear it)")
        params.update(ref_audio=relative(clip), ref_text=ref_text, consent=True)
    params["asr"] = body.get("asr", True) is not False
    # play while made: sent audio cannot be redrawn, so "live": false lets the engine redraw a short part
    params["live"] = body.get("live", True) is not False
    if isinstance(body.get("slate"), dict):  # a Slate voice sheet's cast voice: which film and sheet it belongs to
        params["slate"] = {k: str(v)[:80] for k, v in body["slate"].items() if k in ("project", "voice")}
    return params


def perform(job: dict) -> bool:
    """One voice take to its end; False when it was cancelled meanwhile."""

    params, folder = job["params"], HOME / "voice"
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / job["id"]
    wav, stats, text_file = stem.with_suffix(".wav"), stem.with_suffix(".stats.json"), stem.with_suffix(".txt")
    text_file.write_text(dialogue(params["text"]) + "\n")  # what the engine reads: a turn a line
    reference = None
    if params.get("ref_audio"):
        job["progress"] = {"stage": "reading the voice reference", "at": 0, "of": 1}
        source, clean = inside(params["ref_audio"]), stem.with_suffix(".ref.wav")
        ffmpeg = tool("ffmpeg")
        if ffmpeg:  # any sound file or video, as 44.1 kHz mono 16-bit WAV (the CUDA engine reads WAV only)
            done = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "44100",
                                   "-c:a", "pcm_s16le", str(clean)], capture_output=True, text=True, timeout=120)
            if done.returncode:
                raise RuntimeError("the reference clip could not be read: " + done.stderr.strip()[-200:])
        elif backend().name == "cuda" and source.suffix.lower() != ".wav":
            raise RuntimeError("the CUDA engine reads a WAV reference only, and this machine has no ffmpeg to convert it")
        else:
            shutil.copy(source, clean)
        if not params.get("ref_text") and hears_references(backend()):
            pass  # the Mac engine's server hears the clip itself with its loaded Whisper (ref_texts in its answer)
        elif not params.get("ref_text"):
            heard = qa.transcribe(clean)
            if not heard:
                raise RuntimeError("no transcript of the reference clip: write what it says")
            params["ref_text"] = heard.strip()
        reference = (clean, params.get("ref_text") or "")
    one = backend()
    job["engine"] = one.name
    if hasattr(one, "make") and one.serving():  # a persistent server, which holds the GPU lock itself while loaded
        measured = one.make(job, params["text"], wav, reference, int(params["seed"]), {k: params[k] for k in SAMPLING},
                            live=params.get("live", True) is not False)
        if measured is None:
            return False
        if params.get("ref_audio") and not params.get("ref_text") and measured.get("ref_texts"):
            params["ref_text"] = str(measured["ref_texts"][0]).strip()  # what the engine heard in the clip
        if not getattr(one, "checks", False) and params.get("asr") and qa.ready()["transcript"]:
            measured = redrawn(job, one, params, wav, reference, measured)
            if measured is None:
                return False
        stats.write_text(json.dumps(measured, indent=1) + "\n")
    else:
        live = bool(getattr(one, "streams", False) and params.get("live", True))
        retaken = None
        while True:
            outcome = mlx_take(job, one, text_file, wav, stats, reference, params, live)
            if outcome is None:
                return False
            measured = outcome
            short = [part for part in (measured.get("verify") or []) if isinstance(part, dict) and part.get("action") == "redraw"]
            if not (live and short):
                break
            # a played part came out short and cannot be redrawn: made once more unplayed, where the engine redraws
            retaken = {"played": checked_by_engine(measured), "seconds": measured.get("audio_s")}
            job["log"] += "[tensorfold] a part was short while playing: the take is made again without playing, so it can be redrawn\n"
            live = False
        if retaken:
            measured["retaken"] = retaken
    if not wav.is_file():
        raise RuntimeError("Drama 3 finished without writing its take")
    take = {"id": job["id"], "path": relative(wav), "title": params.get("title") or job.get("title"), "text": params["text"],
            "seed": params["seed"], **{k: params[k] for k in SAMPLING}, "backend": one.name, "created": time.time(),
            "duration": measured.get("audio_s"), "frames": measured.get("frames"), "frames_per_s": measured.get("frames_per_s"),
            "rtf": measured.get("rtf"), "total_s": measured.get("total_s"), "peak_memory_gb": measured.get("peak_memory_gb"),
            "device_gb": measured.get("device_gb"),
            "chunks": measured.get("chunks"), "finish": measured.get("finish"), "first_audio_ms": measured.get("first_audio_ms"),
            "load_s": measured.get("load_s"), "engine_s": measured.get("engine_s"), "verify": checked_by_engine(measured),
            "streamed": bool(measured.get("stream")), "retaken": measured.get("retaken"),
            "reference": {"path": params["ref_audio"], "text": params["ref_text"], "consent": True} if params.get("ref_audio") else None,
            "slate": params.get("slate"), "credit": CREDIT}
    if params.get("asr") and measured.get("app_check"):
        # the app's own end-of-text check (``redrawn``: the CUDA engine has none) heard the kept take already
        take["asr"] = {**engine_score(take["verify"]), "by": "whisper"}
    elif params.get("asr") and take["verify"]:
        # the engine's own Whisper check is the take's score: it is not transcribed a second time
        take["asr"] = engine_score(take["verify"])
    elif params.get("asr") and qa.ready()["transcript"]:
        job["progress"] = {"stage": "Whisper word check", "at": 0, "of": 1}
        try:
            heard = qa.transcribe(wav)
            take["asr"] = match(params["text"], heard) if heard is not None else {"skipped": "no speech-to-text tool"}
        except Exception as error:  # noqa: BLE001 - a take that was made is kept whatever the check does
            take["asr"] = {"skipped": str(error)[-200:]}
    elif params.get("asr"):
        take["asr"] = {"skipped": "no speech-to-text tool on this machine (pip install faster-whisper)"}
    stem.with_suffix(".take.json").write_text(json.dumps(take, indent=1) + "\n")
    job["outputs"] = [take["path"]]
    job["voice"] = take
    job["progress"] = None
    return True


def hears_references(one) -> bool:
    """Whether the engine hears a reference clip left without its text (Mac server, 8f3d4b33 on): no app Whisper."""

    return bool(getattr(one, "hears", False) and hasattr(one, "serving") and one.serving())


RETRIES = 2  # draws again of a take with missing words, as the Mac engine's --verify-retries


def redrawn(job: dict, one, params: dict, wav: Path, reference, measured: dict) -> dict | None:
    """The CUDA engine's end-of-text check (Whisper + wordcheck): a short take redrawn unplayed, ``RETRIES`` at most."""

    import wordcheck

    def heard(path: Path) -> dict:
        job["progress"] = {"stage": "Whisper word check", "at": 0, "of": 1}
        return wordcheck.check_text(params["text"], qa.transcribe(path) or "")

    def score(check: dict) -> tuple:
        return (check["action"] != "redraw", round(check["coverage"], 3), -check["longest_missing_run"], -len(check["tail_words"]))

    first = heard(wav)
    best, best_measured, attempts = first, measured, 1
    while best["action"] == "redraw" and attempts <= RETRIES:
        seed = int(params["seed"]) + 1000 * attempts
        job["log"] += f"[tensorfold] words missing ({best['reason']}): drawn again, seed {seed}, not played\n"
        attempt = wav.with_name(f"{wav.stem}.draw{attempts}.wav")
        again = one.make(job, params["text"], attempt, reference, seed, {k: params[k] for k in SAMPLING}, live=False)
        if again is None:
            attempt.unlink(missing_ok=True)
            return None
        attempts += 1
        check = heard(attempt)
        if score(check) > score(best):
            attempt.replace(wav)
            best, best_measured = check, {**again, "redraw_seed": seed}
        else:
            attempt.unlink(missing_ok=True)
    keep = ("words", "matched", "missing", "tail_words", "action", "reason", "heard", "coverage", "longest_missing_run", "ending_missing")
    out = dict(best_measured)
    out["verify"] = [{**{k: best.get(k) for k in keep}, "attempts": attempts}]
    out["app_check"] = True
    if attempts > 1:
        out["retaken"] = {"played": [{k: first.get(k) for k in keep}], "seconds": measured.get("audio_s"), "attempts": attempts}
        out["load_s"] = measured.get("load_s")
        out["first_audio_ms"] = measured.get("first_audio_ms")
    return out


def mlx_take(job: dict, one, text_file: Path, wav: Path, stats: Path, reference, params: dict, live: bool) -> dict | None:
    """One run of the MLX engine's command line under the GPU lock: its stats, or None when cancelled."""

    sampling = {k: params[k] for k in SAMPLING}
    if live:
        command = one.command(text_file, wav, stats, reference, int(params["seed"]), sampling, live=True)
    else:
        command = one.command(text_file, wav, stats, reference, int(params["seed"]), sampling)
    if not take_lock(job):
        return None
    if live:
        wav.unlink(missing_ok=True)
        LIVE[job["id"]] = {"parts": [wav], "wav": True, "done": False, "finished": 0, "gap_s": 0}
    stats.unlink(missing_ok=True)
    began = len(job.get("log") or "")
    try:
        job["progress"] = {"stage": "Drama 3 voice" + ("" if live or not params.get("live", True) else ", again without playing"), "at": 0, "of": 1}
        if not run(job, command, {}):
            return None
    finally:
        drop_lock(job)
        if live:
            LIVE[job["id"]].update(finished=1, done=True)
            threading.Timer(60, LIVE.pop, args=(job["id"], None)).start()
    try:
        measured = json.loads(stats.read_text())
    except (OSError, ValueError):
        measured = {}
    if measured.get("first_audio_s") is not None:
        measured["first_audio_ms"] = round(1000 * float(measured["first_audio_s"]), 1)
    loaded = re.search(r"\[drama3\] loaded .* in ([\d.]+)s", (job.get("log") or "")[began:])
    measured.setdefault("load_s", float(loaded.group(1)) if loaded else None)
    return measured


def takes() -> list[dict]:
    out = []
    for file in (HOME / "voice").glob("*.take.json"):
        try:
            take = json.loads(file.read_text())
        except (OSError, ValueError):
            continue
        if (HOME / take.get("path", "")).is_file():  # a take whose job was deleted is in the trash folder
            out.append(take)
    return sorted(out, key=lambda t: -float(t.get("created") or 0))


def one_take(take_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", take_id):
        raise HTTPException(400, "bad take id")
    file = HOME / "voice" / f"{take_id}.take.json"
    if not file.is_file():
        raise HTTPException(404, "no such take")
    return json.loads(file.read_text())


# ---------- the API ----------

@router.get("/api/voice/state")
def voice_state() -> dict:
    return state()


@router.post("/api/voice/takes")
def render(body: dict) -> dict:
    """Queue a take: {text, ref_audio?, ref_text?, consent?, seed?, temperature?, top_p?, top_k?, title?, asr?}."""

    params = checked(body)
    title = params["title"] or "Voice: " + spoken(params["text"])[:48]
    return enqueue("voice", title, params, estimate(params["text"], bool(params.get("ref_audio")), params["asr"] and qa.ready()["transcript"]))


@router.post("/api/voice/sheet")
def cast_voice(body: dict) -> dict:
    """A Slate voice sheet's cast voice: its sample line, its description as the direction (sheet_text)."""

    sheet = body.get("sheet") if isinstance(body.get("sheet"), dict) else {}
    return render({"text": sheet_text(sheet), "seed": body.get("seed"), "title": "Cast voice: " + str(sheet.get("name") or "voice")[:60],
                   "slate": {"project": body.get("project") or "", "voice": sheet.get("id") or ""}})


@router.get("/api/voice/takes")
def list_takes() -> list[dict]:
    return takes()


@router.get("/api/voice/takes/{take_id}")
def get_take(take_id: str) -> dict:
    return one_take(take_id)


@router.get("/api/voice/takes/{take_id}/audio")
def take_audio(take_id: str):
    take = one_take(take_id)
    target = inside(take["path"])
    if not target.is_file():
        raise HTTPException(404, "the take's file is gone (deleted to the trash folder?)")
    return FileResponse(target, media_type="audio/wav", filename=f"drama3-{take_id}.wav")


@router.get("/api/voice/takes/{take_id}/live")
def take_live(take_id: str):
    """The take that is being made, as a WAV that plays while it is made (the CUDA engine's raw PCM as it comes)."""

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", take_id):
        raise HTTPException(400, "bad take id")
    entry, began = LIVE.get(take_id), time.time()
    # a take that has started (its text is written first) and is not at the engine yet: its clip is being read
    while entry is None and (HOME / "voice" / f"{take_id}.txt").is_file() and not (HOME / "voice" / f"{take_id}.take.json").is_file() \
            and time.time() - began < 300:
        time.sleep(0.2)
        entry = LIVE.get(take_id)
    if entry is None:
        raise HTTPException(404, "no take is being made under that id")
    from fastapi.responses import StreamingResponse

    return StreamingResponse(live_wav(entry), media_type="audio/wav", headers={"Cache-Control": "no-store"})


@router.post("/api/voice/cut")
def cut(body: dict) -> dict:
    """``seconds`` (15, a reference audio clip's limit) of a sound file from ``start``, as a WAV: {path, duration}."""

    source = inside(str(body.get("path") or ""))
    if not source.is_file():
        raise HTTPException(400, "no such file")
    try:
        start, seconds = max(0.0, float(body.get("start") or 0)), float(body.get("seconds") or 15)
    except (TypeError, ValueError):
        raise HTTPException(400, "start and seconds are numbers")
    if not 2 <= seconds <= 60:
        raise HTTPException(400, "a cut is 2 to 60 seconds")
    ffmpeg = tool("ffmpeg")
    if not ffmpeg:
        raise HTTPException(400, "cutting a clip needs ffmpeg, which is not installed on this machine")
    (HOME / "voice").mkdir(parents=True, exist_ok=True)
    target = HOME / "voice" / f"{source.stem}-cut{start:g}-{seconds:g}s.wav"
    try:
        done = subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(source), "-vn",
                               "-af", "afade=t=out:st=" + f"{max(seconds - 0.08, 0):.3f}" + ":d=0.08", "-ac", "1", "-ar", "44100", str(target)],
                              capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise HTTPException(400, "the cut took longer than a minute")
    if done.returncode or not target.is_file():
        raise HTTPException(400, "the cut failed: " + (done.stderr or "").strip()[-200:])
    return {"path": relative(target), "duration": media_seconds(str(target))[0]}


def attach(home: Path, runner, queue, within, short, measure, gate) -> tuple[APIRouter, callable]:
    """Wire in server.py's run, enqueue, inside, relative, media_seconds and QA module; returns (router, perform)."""

    global HOME, run, enqueue, inside, relative, media_seconds, qa
    HOME, run, enqueue, inside, relative, media_seconds, qa = home, runner, queue, within, short, measure, gate
    (HOME / "voice").mkdir(parents=True, exist_ok=True)
    if not SPARK:
        threading.Thread(target=backend().check, daemon=True).start()  # the first look runs a process: not in a request
    return router, perform
