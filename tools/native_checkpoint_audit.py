"""Audit safetensors headers without loading model tensors into CPU/GPU memory."""
import argparse
import json
import re
import struct
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--filter", default="")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    tensors = {}
    paths = sorted(args.model.glob("*.safetensors"))
    if not paths:
        raise SystemExit("No complete safetensors files")
    for path in paths:
        with path.open("rb") as f:
            size, = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(size))
        end = path.stat().st_size - 8 - size
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            begin, stop = tensor["data_offsets"]
            if not 0 <= begin <= stop <= end:
                raise ValueError(f"Truncated tensor: {path.name}: {name}")
            full = f"mtp.{name}" if path.name == "mtp-4bit.safetensors" else name
            if full in tensors:
                raise ValueError(f"Duplicate tensor: {full}")
            tensors[full] = dict(file=path.name, shape=tensor["shape"], dtype=tensor["dtype"], bytes=stop-begin)
    result = dict(model=str(args.model), files=len(paths), tensors=len(tensors), bytes=sum(v["bytes"] for v in tensors.values()))
    if args.filter:
        result["matching"] = {k: v for k, v in tensors.items() if re.search(args.filter, k)}
    if args.output:
        args.output.write_text(json.dumps(dict(summary=result, tensors=tensors), indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
