"""LFM2-VL picture goldens on the CPU, TensorFold 0.6.6's loader then Lfm2VlImageProcessor: images.py MODEL_DIR OUT."""

import base64
import hashlib
import io
import json
import math
import struct
import sys
import zlib

import numpy as np
from PIL import Image
from transformers import AutoProcessor

from tensorfold.vision.images import ImageInputError, ImageSource, load_images

RNG = np.random.default_rng(7)
CAP = 1024 * 1024


def cap(image):
    """A picture over CAP pixels shrunk to fit with Pillow's bicubic, as d1-tensorfold bounds pictures."""
    w, h = image.size
    if w * h <= CAP:
        return image
    s = math.sqrt(CAP / (w * h))
    return image.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.BICUBIC)


def scene(w, h, channels=3, noise=6.0):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.stack([x / w * 255, y / h * 255, (x + y) / (w + h) * 255, 255 - x / w * 255][:channels], -1)
    for _ in range(12):
        x0, y0 = RNG.integers(0, w), RNG.integers(0, h)
        out[y0:y0 + RNG.integers(2, h // 3 + 3), x0:x0 + RNG.integers(2, w // 3 + 3)] = RNG.integers(0, 256, channels)
    return np.clip(out + RNG.normal(0, noise, out.shape), 0, 255).astype(np.uint8)


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png(samples, depth, color, *, interlace=False, palette=None, trns=None, extra=b""):
    """A PNG written here, for the sample layouts Pillow does not save; rows cycle through the five filters."""
    h, w = samples.shape[:2]
    flat = samples.reshape(h, w, -1).astype(">u2" if depth == 16 else np.uint8)
    bits = depth * flat.shape[2]
    passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
    raw = bytearray()
    for x0, y0, dx, dy in passes if interlace else [(0, 0, 1, 1)]:
        sub = flat[y0::dy, x0::dx]
        if sub.size == 0:
            continue
        stride, bpp = (sub.shape[1] * bits + 7) // 8, max(1, bits // 8)
        prev = bytearray(stride)
        for r, row in enumerate(sub):
            line = row.tobytes() if depth >= 8 else np.packbits(np.unpackbits(row.astype(np.uint8), axis=1)[:, -depth:]).tobytes()
            line, kind = bytearray(line.ljust(stride, b"\0")), r % 5
            out = bytearray(stride)
            for i in range(stride):
                a, b, c = line[i - bpp] if i >= bpp else 0, prev[i], prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pred = [0, a, b, (a + b) // 2, a if abs(p - a) <= min(abs(p - b), abs(p - c)) else b if abs(p - b) <= abs(p - c) else c][kind]
                out[i] = (line[i] - pred) & 255
            raw += bytes([kind]) + out
            prev = line
    head = struct.pack(">IIBBBBB", w, h, depth, color, 0, 0, int(interlace))
    body = chunk(b"IHDR", head) + extra
    body += chunk(b"PLTE", palette) if palette is not None else b""
    body += chunk(b"tRNS", trns) if trns is not None else b""
    return b"\x89PNG\r\n\x1a\n" + body + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")


def saved(image, fmt, **kw):
    buf = io.BytesIO()
    image.save(buf, fmt, **kw)
    return buf.getvalue()


def exif(orientation):
    ifd = Image.Exif()
    ifd[0x0112] = orientation
    return ifd.tobytes()


def resampled(jpeg, sampling, width, height):
    """A JPEG with its SOF0 sampling factors and size rewritten; the MCU count and block order are unchanged."""
    at = jpeg.index(b"\xff\xc0")
    sof = bytearray(jpeg[at:at + 19])
    sof[5:9] = struct.pack(">HH", height, width)
    sof[11] = sampling
    return jpeg[:at] + bytes(sof) + jpeg[at + 19:]


def url(data, mime):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def fixtures():
    gray = scene(13, 7, 1)[..., 0]
    yield "gray1", url(png(gray >> 7, 1, 0, trns=struct.pack(">H", 1)), "image/png")
    yield "gray2", url(png(gray >> 6, 2, 0), "image/png")
    yield "gray4_key", url(png(gray >> 4, 4, 0, trns=struct.pack(">H", 5)), "image/png")
    yield "gray8_key", url(png(gray, 8, 0, trns=struct.pack(">H", int(gray[3, 4]))), "image/png")
    wide = (gray.astype(np.uint16) * 4)
    yield "gray16", url(png(wide, 16, 0), "image/png")
    yield "gray16_key", url(png(wide, 16, 0, trns=struct.pack(">H", int(wide[0, 0]))), "image/png")
    yield "graya8", url(png(scene(13, 7, 2), 8, 4), "image/png")
    yield "graya16", url(png(scene(13, 7, 2).astype(np.uint16) * 257 + 3, 16, 4, interlace=True), "image/png")
    rgb = scene(13, 7)
    yield "rgb8_key", url(png(rgb, 8, 2, trns=struct.pack(">HHH", *rgb[2, 5])), "image/png")
    yield "rgba16", url(png(scene(13, 7, 4).astype(np.uint16) * 251, 16, 6), "image/png")
    pal = RNG.integers(0, 256, (16, 3), dtype=np.uint8).tobytes()
    yield "palette4", url(png(gray >> 4, 4, 3, palette=pal, trns=bytes([0, 128, 255, 7])), "image/png")
    yield "palette8", url(saved(Image.fromarray(rgb).quantize(40), "PNG"), "image/png")
    yield "percent", "data:image/png," + "".join(f"%{b:02X}" if not (chr(b).isascii() and chr(b).isalnum()) else chr(b)
                                               for b in saved(Image.fromarray(rgb), "PNG"))
    for o in (2, 4, 5, 7, 8):
        yield f"orient{o}", url(saved(Image.fromarray(scene(5, 3)), "PNG", exif=exif(o)), "image/png")
    yield "jpeg_gray", url(saved(Image.fromarray(scene(37, 23, 1)[..., 0]), "JPEG", quality=90), "image/jpeg")
    yield "jpeg444", url(saved(Image.fromarray(scene(37, 23)), "JPEG", quality=95, subsampling=0), "image/jpeg")
    yield "jpeg420_progressive", url(saved(Image.fromarray(scene(37, 23)), "JPEG", quality=70, subsampling=2,
                                           progressive=True), "image/jpeg")
    yield "jpeg_rgb", url(saved(Image.fromarray(scene(21, 11)), "JPEG", keep_rgb=True, subsampling=0), "image/jpeg")
    yield "jpeg440", url(resampled(saved(Image.fromarray(scene(32, 16)), "JPEG", subsampling=1), 0x12, 15, 29),
                         "image/jpeg")
    yield "jpeg411", url(resampled(saved(Image.fromarray(scene(32, 32)), "JPEG", subsampling=2), 0x41, 61, 15),
                         "image/jpeg")
    anim = [Image.fromarray(scene(8, 8)), Image.fromarray(scene(8, 8))]
    yield "animated", url(saved(anim[0], "PNG", save_all=True, append_images=anim[1:]), "image/png")
    yield "huge", url(png(np.zeros((1, 9000), np.uint8), 8, 0), "image/png")
    yield "truncated", url(saved(Image.fromarray(rgb), "PNG")[:-30], "image/png")
    yield "garbage", "data:image/png;base64,AAAA"
    yield "empty", "data:image/png;base64,"
    yield "bad_base64", "data:image/png;base64,@@@@"
    yield "bad_encoding", "data:image/png;utf8,AAAA"
    yield "two_parameters", "data:image/png;charset=x;base64,AAAA"
    yield "upper_base64", url(png(gray >> 6, 2, 0), "image/png").replace(";base64,", ";BASE64,")
    yield "no_comma", "data:image/png;base64"
    yield "https", "https://example.com/a.png"
    yield "ftp", "ftp://example.com/a.png"


PICTURES = {
    "photo": lambda: url(saved(Image.fromarray(scene(1280, 900, noise=2)), "JPEG", quality=80, restart_marker_rows=3),
                         "image/jpeg"),
    "turned": lambda: url(saved(Image.fromarray(scene(200, 120)), "JPEG", quality=90, subsampling=1,
                                progressive=True, exif=exif(6)), "image/jpeg"),
    "rgba": lambda: url(saved(Image.fromarray(scene(96, 64, 4)), "PNG"), "image/png"),
    "strip": lambda: url(saved(Image.fromarray(scene(1600, 80, 1, noise=1)[..., 0]), "JPEG", quality=40), "image/jpeg"),
    "tie": lambda: url(saved(Image.fromarray(scene(1225, 700, noise=0)), "JPEG", quality=30), "image/jpeg"),
    "banner": lambda: url(png(scene(360, 80, noise=1), 8, 2, interlace=True, extra=chunk(b"eXIf", exif(3)[6:])),
                          "image/png"),
}


def runs(ids):
    out = []
    for i in ids:
        if out and out[-1][0] == i:
            out[-1][1] += 1
        else:
            out.append([i, 1])
    return out


def main(model_dir, out_path):
    processor = AutoProcessor.from_pretrained(model_dir)
    tok = processor.tokenizer
    marks = {"image": "<image>", "start": "<|image_start|>", "end": "<|image_end|>", "thumbnail": "<|img_thumbnail|>"}
    marks = {k: tok.convert_tokens_to_ids(v) for k, v in marks.items()}
    marks["tiles"] = [[tok.convert_tokens_to_ids(f"<|img_row_{r}_col_{c}|>") for c in range(1, 11)] for r in range(1, 11)]
    cases = []
    for name, link in fixtures():
        case = {"name": name, "url": link}
        try:
            (img,) = load_images([ImageSource(link)])
            case.update(width=img.width, height=img.height, rgb=hashlib.sha256(img.pixels).hexdigest())
        except ImageInputError as exc:
            case["problem"] = str(exc)
        cases.append(case)
    pics = []
    for name, make in PICTURES.items():
        link = make()
        (img,) = load_images([ImageSource(link)])
        pic = cap(img.to_pil())
        pics.append(pic)
        out = processor(text=["<image>"], images=[[pic]], return_tensors="pt", add_special_tokens=False)
        tiles = out["spatial_shapes"].tolist()
        lengths = out["pixel_attention_mask"].sum(1).tolist()
        assert lengths == [h * w for h, w in tiles]
        pv = out["pixel_values"]
        cases.append({"name": name, "url": link, "width": img.width, "height": img.height,
                      "rgb": hashlib.sha256(img.pixels).hexdigest(), "tiles": tiles,
                      "patches": [hashlib.sha256(pv[i, :n].contiguous().numpy().astype("<f4").tobytes()).hexdigest()
                                  for i, n in enumerate(lengths)],
                      "ids": runs(out["input_ids"][0].tolist())})
    # the System One trunk: the head and the state block encode on their own, each picture's span between them
    turn = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "\0"}]}]
    markup = "<image><image>"
    assert f"<|im_start|>user\n{markup}\0" in processor.apply_chat_template(turn, tokenize=False)
    head = "<|startoftext|><|im_start|>user\n"
    body = json.dumps({"screen": "two pictures"}, indent=2) + "\n\n\nQUESTION:\n"
    whole = processor(text=[head + markup + body], images=[pics[:2]], add_special_tokens=False)["input_ids"][0]
    spans = [processor(text=["<image>"], images=[[p]], add_special_tokens=False)["input_ids"][0] for p in pics[:2]]
    ends = [tok.encode(text, add_special_tokens=False) for text in (head, body)]
    assert whole == ends[0] + spans[0] + spans[1] + ends[1]
    with open(out_path, "w") as f:
        json.dump({"marks": marks, "cases": cases}, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
