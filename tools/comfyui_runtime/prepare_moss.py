import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct


QKV = re.compile(r"layers\.[0-9]+\.proj\.qkv\.weight")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint field: {key}")
        result[key] = value
    return result


def split_header(header, payload_size):
    result, spans, count = {}, [], 0
    if not isinstance(header, dict):
        raise ValueError("The safetensors header must be an object")
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict) or any(not isinstance(value, str) for value in tensor.values()):
                raise ValueError("Invalid checkpoint metadata")
            result[name] = tensor
            continue
        if not isinstance(tensor, dict) or tensor.get("dtype") not in ("BF16", "F32"):
            raise ValueError("MOSS preparation requires BF16 weights and F32 RoPE data")
        shape, span = tensor.get("shape"), tensor.get("data_offsets")
        if not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError(f"Invalid tensor shape: {name}")
        size = math.prod(shape) * (2 if tensor["dtype"] == "BF16" else 4)
        if not isinstance(span, list) or len(span) != 2 or any(type(n) is not int for n in span) or span[0] < 0 or span[1] - span[0] != size:
            raise ValueError(f"Invalid tensor extent: {name}")
        spans.append(span)
        if QKV.fullmatch(name):
            if tensor["dtype"] != "BF16" or len(shape) != 2 or shape[0] != 3 * shape[1]:
                raise ValueError(f"Invalid fused QKV tensor: {name}")
            width = size // 3
            for index, part in enumerate(("q", "k", "v")):
                key = name.replace(".qkv.", f".{part}.")
                if key in header:
                    raise ValueError("Checkpoint mixes fused and separate QKV tensors")
                start = span[0] + index * width
                result[key] = {"dtype": "BF16", "shape": [shape[1], shape[1]], "data_offsets": [start, start + width]}
            count += 1
        else:
            result[name] = tensor
    end = 0
    for start, stop in sorted(spans):
        if start != end:
            raise ValueError("Tensor storage overlaps or contains a gap")
        end = stop
    if end != payload_size or not count:
        raise ValueError("Checkpoint is truncated, has trailing data, or has no fused MOSS QKV weights")
    return result


def prepare(source, output):
    source, output = Path(source).resolve(strict=True), Path(output)
    with source.open("rb") as incoming:
        before = os.fstat(incoming.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("The source must be a regular safetensors file")
        length = incoming.read(8)
        if len(length) != 8:
            raise ValueError("Truncated safetensors header")
        size = struct.unpack("<Q", length)[0]
        if not 2 <= size <= min(8 << 20, before.st_size - 8):
            raise ValueError("Invalid safetensors header size")
        header = json.loads(incoming.read(size), object_pairs_hook=unique_object)
        header = split_header(header, before.st_size - size - 8)
        data = json.dumps(header, separators=(",", ":")).encode()
        data += b" " * (-len(data) % 8)
        with output.open("xb") as outgoing:
            try:
                outgoing.write(struct.pack("<Q", len(data)))
                outgoing.write(data)
                shutil.copyfileobj(incoming, outgoing, 8 << 20)
                after = source.stat()
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                    raise RuntimeError("Checkpoint changed during preparation")
            except BaseException:
                output.unlink()
                raise
    return output


def main():
    parser = argparse.ArgumentParser(description="Prepare ComfyUI's BF16 MOSS checkpoint for the pinned TensorFold runtime. Tensor payload bytes are copied unchanged.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, help="New output file; existing files are never replaced.")
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
