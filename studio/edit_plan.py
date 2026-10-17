"""What both builds of Edit a picture agree on before the model runs: the noise seed, the size made at and returned."""
# Never seed an edit with the user's seed alone: from its own source noise the model returns the source, oversharpened.

from __future__ import annotations

import hashlib
import math
from pathlib import Path

MULTIPLE = 32  # the model's grid: a vision slot covers 2x2 latents of 16 pixels
# the pixel counts an edit is made at here (512x512 to 1536x1536): what the app has always accepted as its size
AREA = (512 * 512, 1536 * 1536)
SIDES = (256, 4096)  # a size asked for by hand
PICTURES = 10  # pictures in one edit: the edited one and nine more (mflux's pipeline takes 10, ComfyUI's node 16)


def noise_seed(seed: int, pictures) -> int:
    """The noise seed: the user's seed and every picture's bytes, in order; 48 bits, whole in JSON and both RNGs."""

    digest = hashlib.sha256(f"studio-edit:{int(seed)}".encode())
    for picture in pictures:
        with open(picture, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"|")
    return int.from_bytes(digest.digest()[:6], "big")


def grid(value: float) -> int:
    return max(MULTIPLE, round(value / MULTIPLE) * MULTIPLE)


def fit(width: int, height: int, resolution: int) -> tuple[int, int]:
    """The model's own rule: about resolution x resolution pixels on the grid, the shape kept."""

    ratio = width / height
    return grid(math.sqrt(resolution * resolution * ratio)), grid(math.sqrt(resolution * resolution / ratio))


def canvas(width: int, height: int, mode: str = "source", resolution: int = 1024, asked: tuple[int, int] | None = None,
           most: int = 0) -> dict:
    """Where an edit of a width x height picture is made and what comes back: {made, size, source, note}."""

    if asked:
        wide, tall = int(asked[0]), int(asked[1])
        made = (grid(wide), grid(tall))
        side = grid(math.sqrt(made[0] * made[1]))
        same = abs(wide / tall - width / height) < 0.02
        return {"made": made, "size": (wide, tall), "source": made if same else fit(width, height, side), "note": ""}
    if mode == "model":
        made = fit(width, height, resolution)
        return {"made": made, "size": made, "source": made, "note": ""}
    low, high = AREA[0], min(AREA[1], most) if most else AREA[1]
    if low <= width * height <= high:
        made = (grid(width), grid(height))
        return {"made": made, "size": (width, height), "source": made, "note": ""}
    bound = int(math.sqrt(low if width * height < low else high))
    made = fit(width, height, bound)
    return {"made": made, "size": made, "source": made,
            "note": f"the picture is {width}x{height}, {'smaller' if width * height < low else 'larger'} than the model edits at: "
                    f"the result is {made[0]}x{made[1]}"}


def refused(width: int, height: int) -> str:
    """Why a size asked for by hand is not one an edit is made at ("" when it is)."""

    low, high = SIDES
    if not (low <= width <= high and low <= height <= high):
        return f"width and height are between {low} and {high}"
    if not AREA[0] <= width * height <= AREA[1]:
        return f"an edit is made at 512x512 to 1536x1536 pixels in all, and {width}x{height} is outside that"
    return ""


def beside(width: int, height: int, made: tuple[int, int]) -> tuple[int, int]:
    """The size a further picture is brought to: about as many pixels as the edit itself, in its own shape."""

    return fit(width, height, grid(math.sqrt(made[0] * made[1])))


def picture_size(path) -> tuple[int, int]:
    """(width, height) of a picture as it is shown: a phone photo's rotation tag is applied."""

    from PIL import Image, ImageOps

    with Image.open(Path(path)) as image:
        return ImageOps.exif_transpose(image).size
