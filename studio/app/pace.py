"""Whether a clip's sound and motion agree: hits are heard only where arms and hands move fast enough to make them."""
# limbs: quickest blocks' normal-flow speed less the median, % of frame width a second; hits: broadband sharp onsets

from __future__ import annotations

import os
import subprocess
from pathlib import Path

DIRS = ("/opt/homebrew/bin", "/usr/local/bin")
ENV = {**os.environ, "PATH": os.pathsep.join([os.environ.get("PATH", ""), *DIRS])}
SIZE = (432, 240)
FAST, SLOW = 24.0, 18.0  # limbs: at or over FAST carries hits, under SLOW does not
FAST_SCALED, SLOW_SCALED = 44.0, 42.0  # the same against the size of the action in the frame (limbs / sqrt(area))
HITS = 1.0  # sharp onsets a second from which a clip's sound counts as impacts
FADE = 1.35  # how much louder a hit is in its first 30 ms than 50 ms later
BAND = 0.55  # the share of the band above 600 Hz that rises at an onset, from which it is a hit and not a note
PACES = ("auto", "impacts", "flow")
# words of a soundscape that are a hit; a phrase holding one is taken out of a Flow clip's sound
IMPACT_WORDS = ("slap", "smack", "strike", "hit", "punch", "thud", "knock", "clap", "crack", "impact", "blow", "whack", "bang",
                "thwack", "pound", "slam", "kick", "stomp", "clash", "snap")
QUIET = "Near silence: a faint room tone and the soft rustle of cloth."
VIGOROUS = ("{who} fight at full speed and with force: both throw sharp, snapping hand strikes and hard forearm blocks in quick "
            "succession, attacking and blocking, their arms whipping out and back, and every strike that lands is heard as it lands.")
VIGOROUS_ONE = ("{who} moves at full speed and with force: sharp, snapping strikes and hard blocks in quick succession, the "
                "arms whipping out and back, and every strike that lands is heard as it lands.")
SMOOTH = ("The movement is slow, smooth and continuous throughout: hands and forearms stay in soft contact and circle, and "
          "no strike lands.")


def limbs(clip: Path) -> dict | None:
    """``{"limbs", "median", "camera"}`` in percent of the frame width a second, or None when it cannot be measured."""

    try:
        import numpy as np
    except ImportError:
        return None
    width, height = SIZE
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf", f"fps=24,scale={width}:{height},format=gray,gblur=sigma=1.2",
                          "-f", "rawvideo", "-"], capture_output=True, env=ENV).stdout
    count = len(raw) // (width * height)
    if count < 12:
        return None
    frames = np.frombuffer(raw[:count * width * height], np.uint8).reshape(count, height, width).astype(np.float32)
    side, camera, quick, active = 24, [], [], []
    for before, after in zip(frames[:-1], frames[1:]):
        cross = np.fft.rfft2(after) * np.conj(np.fft.rfft2(before))
        peak = np.fft.irfft2(cross / (np.abs(cross) + 1e-6), s=before.shape)
        y, x = np.unravel_index(int(np.argmax(peak)), peak.shape)
        moved = np.roll(before, (y - height if y > height // 2 else y, x - width if x > width // 2 else x), (0, 1))
        down, across = np.gradient((moved + after) / 2)
        edge = np.hypot(across, down)
        strong = edge > 6
        strong[:10] = strong[-10:] = False
        strong[:, :10] = strong[:, -10:] = False
        speed = np.where(strong, np.abs(after - moved) / np.maximum(edge, 1e-3), np.nan)
        blocks = (speed[:height // side * side, :width // side * side].reshape(height // side, side, width // side, side)
                  .transpose(0, 2, 1, 3).reshape(-1, side * side))
        filled = np.sum(~np.isnan(blocks), 1) > 60
        if filled.sum() < 10:
            continue
        each = np.nanmedian(blocks[filled], 1)
        camera.append(float(np.median(each)))
        quick.append(max(float(np.sort(each)[-3:].mean()) - camera[-1], 0.0))
        moving = np.zeros(len(blocks), bool)
        moving[filled] = each - camera[-1] > max(1.5 * camera[-1], 0.25)
        active.append(moving)
    if len(quick) < 10:
        return None
    scale = 24 / width * 100
    speed = float(np.percentile(quick, 90)) * scale
    # share of the frame moving in over one frame in seven: the size of the action (far-off fighters move fewer pixels)
    area = float((np.mean(active, 0) > 0.15).mean())
    return {"limbs": round(speed, 1), "median": round(float(np.median(quick)) * scale, 1), "camera": round(float(np.median(camera)) * scale, 1),
            "area": round(area, 2), "scaled": round(speed / max(area, 0.01) ** 0.5, 1)}


def hits(clip: Path) -> dict | None:
    """``{"hits": onsets a second, "at", "level": dBFS}``, ``silent`` with no sound; None when it cannot be measured."""

    try:
        import numpy as np
    except ImportError:
        return None
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, env=ENV).stdout
    sound = np.frombuffer(raw[:len(raw) // 4 * 4], np.float32)
    window, hop = 512, 160
    if len(sound) < 16000:
        return {"hits": 0.0, "at": [], "level": None, "silent": True}
    level = float(20 * np.log10(np.sqrt(float(np.mean(sound ** 2))) + 1e-9))
    if level < -50:  # nothing to hear: its onsets are noise
        return {"hits": 0.0, "at": [], "level": round(level, 1), "silent": True}
    spectrum = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(sound, window)[::hop] * np.hanning(window), axis=1))
    loud = np.log1p(100 * spectrum)
    bins = np.maximum(loud[1:] - loud[:-1], 0)[:, 20:]  # above about 600 Hz
    rise = bins.sum(1)
    floor = np.array([np.median(rise[max(0, i - 20):i + 20]) for i in range(len(rise))]) * 1.6 + 0.15 * float(rise.max()) + 1e-3
    found, last = [], -99
    for i in range(2, len(rise) - 2):
        if rise[i] > floor[i] and rise[i] == rise[i - 2:i + 3].max() and i - last >= 6:
            last = i
            # a hit rises over the whole band (a note only in its partials) and dies away at once (speech lasts)
            wide = float((bins[i] > 0.25 * bins[i].max()).mean()) >= BAND
            short = float(loud[i + 1:i + 4, 20:].sum()) >= FADE * float(loud[i + 6:i + 9, 20:].sum()) if i + 9 <= len(loud) else True
            if wide and short:
                found.append(i)
    seconds = len(sound) / 16000
    return {"hits": round(len(found) / seconds, 2), "at": [round(i * hop / 16000, 2) for i in found], "level": round(level, 1), "silent": False}


def music(clip: Path) -> float | None:
    """Share of the 60 Hz to 4 kHz energy in steady partials (near 1 for music), or None with nothing to hear."""

    try:
        import numpy as np
    except ImportError:
        return None
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, env=ENV).stdout
    sound = np.frombuffer(raw[:len(raw) // 4 * 4], np.float32)
    if len(sound) < 16000 or float(np.mean(sound ** 2)) < 1e-5:  # under -50 dBFS
        return None
    spectrum = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(sound, 2048)[::512] * np.hanning(2048), axis=1))[:, 8:512] + 1e-9
    around = np.array([np.median(spectrum[:, max(0, k - 12):k + 13], axis=1) for k in range(spectrum.shape[1])]).T
    power = spectrum ** 2
    return round(float(np.median((power * (spectrum > 4 * around)).sum(1) / power.sum(1))), 2)


def kind(speed: float | None, scaled: float | None = None) -> str:
    """fast, moderate, slow or unknown; slow is slow both in the frame and against the size of the action in it."""

    if speed is None:
        return "unknown"
    if speed >= FAST or (scaled is not None and scaled >= FAST_SCALED):
        return "fast"
    return "slow" if speed < SLOW and (scaled is None or scaled < SLOW_SCALED) else "moderate"


def measure(clip: Path) -> dict:
    """Both measurements of a clip with a ``verdict``: "agree", "impact sound over slow motion" or "not measured"."""

    moving, heard = limbs(clip), hits(clip)
    speed = moving["limbs"] if moving else None
    tune = music(clip) if heard is not None and not heard.get("silent") else None
    # hits that can be heard: ticks of cloth and feet 45 dB down are near silence, not impacts
    sound = ("unknown" if heard is None else "silent" if heard.get("silent")
             else "impacts" if heard["hits"] >= HITS and (heard.get("level") or -99) > -40
             else "music" if tune is not None and tune >= 0.5 and (heard.get("level") or -99) > -35 else "quiet")
    motion = kind(speed, moving["scaled"] if moving else None)
    if moving is None or heard is None:
        verdict, note = "not measured", "this clip's motion or sound could not be measured on this machine"
    elif sound == "impacts" and motion == "slow":
        verdict = "impact sound over slow motion"
        note = (f"{heard['hits']:g} sharp hits a second are heard, and the quickest movement is {speed:g} percent of the frame "
                f"width a second (slow): choose Flow for this movement, or Impacts with the video left out or a faster reference")
    else:
        verdict = "agree"
        note = (f"movement {motion} ({speed:g} percent of the frame width a second), sound "
                + ("silent" if sound == "silent" else f"{sound} ({heard['hits']:g} sharp hits a second)"))
    return {"limbs": speed, "motion": motion, "median": moving and moving["median"], "camera": moving and moving["camera"],
            "area": moving and moving["area"], "scaled": moving and moving["scaled"],
            "hits": heard and heard["hits"], "hits_at": heard and heard.get("at"), "level": heard and heard.get("level"),
            "music": tune, "sound": sound, "verdict": verdict, "note": note}


def choose(asked: str | None, measured: dict | None) -> str:
    """"impacts" or "flow", or "" when nothing says; a choice wins, "auto" is Impacts only for a fast reference."""

    asked = asked if asked in PACES else "auto"
    if asked != "auto":
        return asked
    if not measured or measured.get("motion") in (None, "unknown"):
        return ""
    return "impacts" if measured["motion"] == "fast" else "flow"


def has_impacts(text: str) -> bool:
    low = (text or "").lower()
    return any(word in low for word in IMPACT_WORDS)


def calm(soundscape: str) -> str:
    """A soundscape without the phrases that name a hit; nothing left becomes near silence."""

    import re

    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", (soundscape or "").strip()):
        parts = [part.strip() for part in re.split(r",|;", sentence.rstrip(".!?")) if part.strip()]
        quiet = [part for part in parts if not has_impacts(part)]
        if quiet:
            text = ", ".join(quiet)
            kept.append(text[0].upper() + text[1:] + ".")
    return " ".join(kept) or QUIET
