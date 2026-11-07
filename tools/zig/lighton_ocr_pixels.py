"""The LightOnOCR-3-1B processor's pixel bytes on the synthetic pages that lighton_ocr/pixels_test.zig makes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

# (h, w) of the pages, the same list as the Zig test's fixture rows; the first 18 are chosen edges, the rest random.
SIZES = (
    (28, 28), (29, 30), (56, 1540), (1540, 56), (1540, 1540), (1541, 1540), (2000, 1414), (1414, 2000), (3000, 40),
    (40, 3000), (777, 1203), (1203, 777), (100, 100), (15, 2000), (2480, 3508), (3508, 2480), (1653, 1169),
    (500, 4000), (1354, 3910), (645, 1645), (2694, 225), (324, 3391), (2222, 413), (1525, 2415), (265, 3754),
    (2106, 907), (181, 380), (1804, 1740), (314, 1013), (399, 2285), (1766, 270), (3414, 2344), (535, 3908),
    (942, 2611), (2597, 2415), (3909, 281), (2391, 2426), (1652, 231), (933, 218), (2308, 3544),
)


def page(h: int, w: int, np):
    """RGB uint8 [h, w, 3]: gradients plus hashed noise, in uint32 arithmetic that wraps as the Zig test's does."""
    y = np.arange(h, dtype=np.uint32)[:, None, None]
    x = np.arange(w, dtype=np.uint32)[None, :, None]
    ch = np.arange(3, dtype=np.uint32)[None, None, :]
    gy = (y * np.uint32(255) // np.uint32(max(h - 1, 1))).astype(np.int64)
    gx = (x * np.uint32(255) // np.uint32(max(w - 1, 1))).astype(np.int64)
    k = ((y * np.uint32(w) + x) * np.uint32(3) + ch) * np.uint32(0x9E3779B1)
    k ^= k >> np.uint32(15)
    k = k * np.uint32(0x85EBCA6B)
    k ^= k >> np.uint32(13)
    noise = (k & np.uint32(63)).astype(np.int64) - 32
    base = np.concatenate([(gy + gx) // 2, np.abs(gy - gx), 255 - (gy * 7 + gx * 3) // 10], axis=2)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="local LightOnOCR-3-1B snapshot directory")
    parser.add_argument("--output", type=Path, help="also write the rows, as lighton_ocr/fixtures/processor_pixels.txt")
    args = parser.parse_args(argv)
    # python -I keeps this folder off sys.path, so the pinned reference loads by its path.
    spec = importlib.util.spec_from_file_location("lighton_ocr_reference", Path(__file__).with_name(
        "lighton_ocr_reference.py"))
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    try:
        reference.pinned(args.model, "1B")
    except (ValueError, OSError, ImportError) as error:
        print(f"lighton pixels: {error}", file=sys.stderr)
        return 2
    import numpy as np
    import transformers
    from PIL import Image

    processor = transformers.AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    chat = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}]}], add_generation_prompt=True)
    rows = []
    for h, w in SIZES:
        out = processor(text=[chat], images=[Image.fromarray(page(h, w, np))], return_tensors="np")
        values = np.ascontiguousarray(out["pixel_values"][0], dtype=np.float32)
        rows.append(f"{h} {w} {'x'.join(map(str, values.shape))} {hashlib.sha256(values.tobytes()).hexdigest()}")
    print("\n".join(rows))
    if args.output:
        args.output.write_text("\n".join(rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
