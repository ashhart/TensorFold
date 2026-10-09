"""Record generated affine q4/group64 BF16 preparation against actual installed MLX, without models."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    import numpy as np
    args.out.mkdir(parents=True, exist_ok=False)
    groups = []
    for value in (0.0, -0.0, 1.0, -1.0, 1e-20, -1e-20, 1e20, -1e20):
        groups.append(np.full(64, value, dtype=np.float32))
    for lo, hi in [(-1, 1), (-5, 1), (-1, 5), (-1.5, 1.5), (-3, 3), (1, 2), (-2, -1), (-1e-6, 1e-6)]:
        groups.append(np.linspace(lo, hi, 64, dtype=np.float32))
    for scale in (0.125, 0.25, 0.5, 1.0):
        values = np.tile(np.arange(16, dtype=np.float32) + 0.5, 4) * scale
        values[0], values[-1] = 0, 15 * scale
        groups.extend([values, -values, values - 7.5 * scale])
    rng = np.random.default_rng(777)
    groups.extend(rng.normal(size=(8192, 64)).astype(np.float32))
    groups.extend((rng.integers(-1000, 1001, size=(4096, 64)) / 128).astype(np.float32))
    groups.extend((rng.uniform(-1, 1, size=(256, 64)) * 1e-8).astype(np.float32))
    while len(groups) % 4:
        groups.append(np.zeros(64, dtype=np.float32))
    raw = np.asarray(groups, dtype=np.float32).reshape(-1, 256)
    x = mx.array(raw).astype(mx.bfloat16)
    words, scales, biases = mx.quantize(x, group_size=64, bits=4, mode="affine")
    mx.eval(x, words, scales, biases)
    files = {}
    for name, value, dtype in [("weights", x.view(mx.uint16), "<u2"), ("words", words, "<u4"), ("scales", scales.view(mx.uint16), "<u2"), ("biases", biases.view(mx.uint16), "<u2")]:
        data = np.asarray(value).astype(dtype, copy=False).tobytes()
        filename = name + ".bin"
        (args.out / filename).write_bytes(data)
        files[name] = {"name": filename, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    record = {"schema": "tf-affine4-v1", "n": int(x.shape[0]), "k": int(x.shape[1]), "groups": len(groups), "files": files}
    (args.out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"affine4: {len(groups)} generated groups, {x.shape}, no model loaded")


if __name__ == "__main__":
    main()
