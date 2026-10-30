"""How long a clip can be: MiniMax H3 and FastH3 take 17n + 5 frames (22 to 362) at 24 frames a second."""

from __future__ import annotations

FPS = 24
STEP, BASE = 17, 5
SHORTEST, LONGEST = STEP + BASE, 21 * STEP + BASE  # 22 and 362 frames
# 5, 8, 10 and 15 seconds as the app has always rendered them
CLASSIC = {5: 124, 8: 192, 10: 243, 15: 362}


def valid(frames: int) -> bool:
    return SHORTEST <= frames <= LONGEST and (frames - BASE) % STEP == 0


def counts(longest: int = LONGEST) -> list[int]:
    """Every frame count the model takes, shortest first."""

    return list(range(SHORTEST, min(longest, LONGEST) + 1, STEP))


def nearest(seconds: float, longest: int = LONGEST) -> int:
    """The frame count closest to ``seconds`` of video, within what the model (and ``longest``) takes."""

    return min(counts(longest), key=lambda frames: (abs(frames - seconds * FPS), frames))


def seconds_of(frames: int) -> float:
    return frames / FPS


def says(frames: int) -> str:
    """'243 frames, 10.125 s': the exact length, as the page writes it."""

    return f"{frames} frames, {frames / FPS:.3f}".rstrip("0").rstrip(".") + " s"


def of(params: dict, longest: int = LONGEST) -> int:
    """The frame count a job asks for: ``frames`` (a valid count, else ValueError) or ``seconds`` (the nearest)."""

    if params.get("frames") not in (None, ""):
        try:
            frames = int(params["frames"])
        except (TypeError, ValueError):
            raise ValueError("frames is a whole number")
        if not valid(frames):
            below, above = nearest((frames - STEP / 2) / FPS), nearest((frames + STEP / 2) / FPS)
            near = " or ".join(str(n) for n in sorted({below, above}))
            raise ValueError(f"a clip is 17n + 5 frames long, from {SHORTEST} to {LONGEST} ({SHORTEST / FPS:.2f} to {LONGEST / FPS:.2f} s): "
                             f"{frames} is not one of them, the nearest is {near}")
    else:
        try:
            seconds = float(params.get("seconds", 5))
        except (TypeError, ValueError):
            raise ValueError("seconds is a number")
        if not SHORTEST / FPS - 0.5 <= seconds <= LONGEST / FPS + 0.5:
            raise ValueError(f"a clip is {SHORTEST / FPS:.2f} to {LONGEST / FPS:.2f} seconds long ({SHORTEST} to {LONGEST} frames), got {seconds:g} s")
        frames = nearest(seconds)
    if frames > longest:
        raise ValueError(f"at this size a clip is at most {longest} frames ({longest / FPS:.2f} s) on this machine, got {frames}")
    return frames
