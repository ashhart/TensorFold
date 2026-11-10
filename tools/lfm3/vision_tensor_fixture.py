"""Write or check the LFM3-VL vision tensor catalogue (name, dtype, shape) from a checkpoint's safetensors header.

The host admission tests read zig/src/families/lfm3/fixtures/vision_tensors.tsv. Only the header is read, never tensor data.
"""
import argparse
import json
import pathlib
import struct
import sys

FIXTURE = pathlib.Path(__file__).resolve().parents[2] / "zig/src/families/lfm3/fixtures/vision_tensors.tsv"
PREFIXES = ("vision_tower.", "multi_modal_projector.")


def catalogue(model: pathlib.Path) -> str:
    """The checkpoint's vision and projector tensors, one `name<TAB>dtype<TAB>d0,d1,...` line each, sorted by name."""
    with open(model / "model.safetensors", "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    rows = sorted((name, entry["dtype"], entry["shape"]) for name, entry in header.items() if name.startswith(PREFIXES))
    return "".join(f"{name}\t{dtype}\t{','.join(map(str, shape))}\n" for name, dtype, shape in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=pathlib.Path, help="an official LFM3-VL-2B MLX checkpoint folder (4, 6 or 8 bit)")
    parser.add_argument("--write", action="store_true", help="replace the committed catalogue; without it, compare and exit 1 on a difference")
    args = parser.parse_args()
    text = catalogue(args.model_dir)
    if args.write:
        FIXTURE.write_text(text)
        print(f"wrote {text.count(chr(10))} tensors to {FIXTURE}")
        return 0
    same = FIXTURE.read_text() == text
    print(f"{text.count(chr(10))} tensors; {'matches' if same else 'DIFFERS from'} {FIXTURE}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
