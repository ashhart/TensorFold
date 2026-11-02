# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# SPDX-License-Identifier: OpenMDW-1.1

"""Offline Nemotron Parse references for native tests: HF example, positions, processor fingerprints, compare."""

import argparse
import hashlib
import json

REVISION = '4035ffb1d8f8bafcdb867dfaf6f7514f101464f9'


# Formulas of nvidia/Nemotron-3.5-Parse@4035ffb1d8f8bafcdb867dfaf6f7514f101464f9 processor; no native code.
PROCESSOR_CASES = [(1, 1), (17, 65), (31, 129), (113, 97), (2049, 101), (99, 2001), (2201, 1777), (2048, 1664)]


def processor_case(np, Image, height, width):
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    y, x, ch = np.indices((height, width, 3), dtype=np.uint32)
    rgb = ((y * 17 + x * 29 + ch * 83) % 256).astype(np.uint8)
    ratio = width / height
    h, w = height, width
    if h > 2048:
        h = 2048
        w = int(h * ratio)
    if w > 1664:
        w = 1664
        h = int(w / ratio)
    image = Image.fromarray(rgb)
    if (h, w) != (height, width):
        image = image.resize((w, h), Image.Resampling.BILINEAR)
    content = np.array(image)
    padded = np.pad(content, ((0, -h % 16), (0, -w % 64), (0, 0)), constant_values=255)
    # ToTensor divides float32 by 255; Normalize uses two separate float32 operations.
    normalized = (padded.astype(np.float32) / np.float32(255.0) - mean) / std
    ph, pw = padded.shape[:2]
    patches = normalized.reshape(ph // 16, 16, pw // 16, 16, 3).transpose(0, 2, 4, 1, 3)
    data = patches.astype('<f4').tobytes()
    return dict(height=height, width=width, content_h=h, content_w=w, padded_h=ph, padded_w=pw,
                tokens=(ph // 16) * (pw // 64) + 1, sha256=hashlib.sha256(data).hexdigest())


def processor(_args):
    import numpy as np
    from PIL import Image

    print(json.dumps(dict(source_revision=REVISION,
                          cases=[processor_case(np, Image, h, w) for h, w in PROCESSOR_CASES]), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('processor', help='print zig/src/families/nemotron_parse/fixtures/processor.json')
    p.set_defaults(run=processor)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
