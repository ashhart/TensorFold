"""Export the exact tensor metadata required by the four fixed native model recipes.

Only safetensors headers are read; no tensors or MLX runtime are loaded. The generated
schemas let the native loader reject missing or incompatible tensors before GPU work.
"""
import argparse
import json
import struct
from pathlib import Path


RECIPES = (
    ("qwen", "Qwen3.8-27B-MLX-4bit", "language_model."),
    ("dflash", "Qwen3.8-27B-DFlash2", ""),
    ("nemotron", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", ""),
    ("flash", "Qwen3.8-Flash-Next-MLX-4bit-MTP", "language_model."),
)


def header(path):
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= length <= 100 * 1024 * 1024:
            raise ValueError(f"Invalid header length: {path}")
        return json.loads(stream.read(length))


def schema(kind, directory, prefix):
    index = directory / "model.safetensors.index.json"
    if index.exists():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = sorted(p.name for p in directory.glob("model*.safetensors"))
    if not files:
        raise ValueError(f"Missing model weights in {directory}")
    result = {}
    for name in files:
        if Path(name).name != name:
            raise ValueError(f"Invalid shard name: {name}")
        for key, value in header(directory / name).items():
            if key == "__metadata__" or not key.startswith(prefix):
                continue
            key = key[len(prefix):]
            if kind == "qwen" and key.startswith("mtp."):
                continue
            if key in result:
                raise ValueError(f"Duplicate tensor: {key}")
            result[key] = dict(name=key, dtype=value["dtype"], shape=value["shape"])
    if kind == "nemotron":
        for key, value in header(directory / "mtp-4bit.safetensors").items():
            if key != "__metadata__":
                name = "mtp." + key
                if name in result:
                    raise ValueError(f"Duplicate tensor: {name}")
                result[name] = dict(name=name, dtype=value["dtype"], shape=value["shape"])
    return "[\n" + ",\n".join("  " + json.dumps(result[k], separators=(",", ":")) for k in sorted(result)) + "\n]\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("build/models"))
    parser.add_argument("--output", type=Path, default=Path("native/schemas"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for kind, model, prefix in RECIPES:
        content = schema(kind, args.model_root / model, prefix)
        target = args.output / f"{kind}.json"
        if args.check:
            if not target.exists() or target.read_text() != content:
                raise SystemExit(f"Schema out of date: {target}")
        else:
            target.write_text(content)
        records = json.loads(content)
        print(f"{'Verified' if args.check else 'Exported'} {target}: {len(records)} tensors, dtypes {sorted({r['dtype'] for r in records})}")


if __name__ == "__main__":
    main()
