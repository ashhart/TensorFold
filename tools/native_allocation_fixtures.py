"""Small real checkpoint files for exhaustive native host-allocation failure checks."""
import argparse
import json
from pathlib import Path

import mlx.core as mx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    q, s, b = mx.quantize(mx.ones((32, 64), dtype=mx.bfloat16), group_size=64, bits=4)
    arrays = {"projection.weight": q, "projection.scales": s, "projection.biases": b}
    for mode in ("indexed", "unindexed"):
        directory = args.directory / mode
        directory.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(directory / "model.safetensors"), arrays)
        if mode == "indexed":
            (directory / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {name: "model.safetensors" for name in arrays}}))


if __name__ == "__main__":
    main()
