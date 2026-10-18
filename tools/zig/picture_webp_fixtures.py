"""WebP pictures and the RGB bytes Pillow 13.0.0 gives them: zig/tests/fixtures/picture/webp.json (--out)."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json

import numpy as np
from PIL import Image, ImageOps, features

ANIMATED = "animated and multipage images are unsupported; send a single frame"
INVALID = "image bytes are invalid or unsupported; use JPEG, PNG or WebP"


def picture(w: int, h: int, alpha: bool) -> Image.Image:
    y, x = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(w * 1000 + h)
    a = np.empty((h, w, 4), np.uint8)
    a[..., 0] = x * 255 // max(w - 1, 1)
    a[..., 1] = y * 255 // max(h - 1, 1)
    a[..., 2] = rng.integers(0, 256, (h, w))
    a[..., 3] = (x + y) * 255 // max(w + h - 2, 1)
    return Image.fromarray(a if alpha else a[..., :3])


def loaded(data: bytes) -> tuple[int, int, bytes]:
    """As 0.6.6's images.py loaded a picture: open, ImageOps.exif_transpose, any alpha pasted over white."""
    im = Image.open(io.BytesIO(data))
    if getattr(im, "n_frames", 1) != 1:
        raise ValueError("animated")
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        im = im.convert("RGBA")
        out = Image.new("RGB", im.size, (255, 255, 255))
        out.paste(im, mask=im.getchannel("A"))
        im = out
    im = im.convert("RGB")
    return im.width, im.height, im.tobytes()


def encoded(im: Image.Image, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "WEBP", **kw)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert features.version("webp") == "1.6.0", features.version("webp")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to view
    cases = [
        ("lossy_q80", encoded(picture(157, 93, False), quality=80)),
        ("lossless", encoded(picture(64, 48, False), lossless=True)),
        ("lossy_alpha", encoded(picture(57, 41, True), quality=70)),
        ("lossless_alpha", encoded(picture(33, 27, True), lossless=True)),
        ("odd_1x1", encoded(picture(1, 1, False), quality=90)),
        ("exif_orientation_6", encoded(picture(40, 24, False), lossless=True, exif=exif.tobytes())),
        ("animated_two", encoded(picture(16, 16, False), save_all=True, append_images=[picture(16, 16, True)], duration=100)),
    ]
    whole = encoded(picture(64, 48, False), quality=80)
    cases.append(("cut_half", whole[: len(whole) // 2]))
    out = []
    for name, data in cases:
        case = {"name": name, "b64": base64.b64encode(data).decode()}
        try:
            w, h, rgb = loaded(data)
            case.update(width=w, height=h, rgb=hashlib.sha256(rgb).hexdigest())
        except ValueError as e:
            case["problem"] = ANIMATED if str(e) == "animated" else INVALID
        except OSError:
            case["problem"] = INVALID
        out.append(case)
    with open(args.out, "w") as f:
        json.dump({"source": "Pillow 13.0.0 (libwebp 1.6.0): open, ImageOps.exif_transpose, alpha pasted over white", "cases": out}, f)
        f.write("\n")


if __name__ == "__main__":
    main()
