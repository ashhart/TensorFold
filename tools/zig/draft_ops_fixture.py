"""Small DFlash primitive fixtures from installed MLX and the pinned Python 0.6.6 expressions, no model."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import tensorfold
    if tensorfold.__version__ != "0.6.6":
        raise RuntimeError("select the local Python 0.6.6 source")
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from tensorfold.drafters.dflash_drafter import _vendor
    conv = _vendor()._grouped_dynamic_convolve
    args.out.mkdir(parents=True, exist_ok=False)
    cases = []
    seed = 4721
    rng = np.random.default_rng(seed)

    def array(shape, scale=64):
        return mx.array(rng.integers(-127, 128, size=shape).astype(np.float32) / scale).astype(mx.bfloat16)

    def save(name, kind, params, inputs, expected):
        files = {}
        for field, value in {**inputs, "expected": expected}.items():
            mx.eval(value)
            data = np.asarray(value if value.dtype == mx.uint64 else value.view(mx.uint16)).tobytes()
            filename = f"{name}.{field}.bin"
            (args.out / filename).write_bytes(data)
            files[field] = {"name": filename, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        cases.append({"name": name, "kind": kind, "params": params, "files": files})

    for n, k in [(48, 5120), (64, 25600), (32, 17408), (17, 513)]:
        rows = 8
        x = mx.array(rng.integers(-3, 4, size=(rows, k)).astype(np.float32) / 16).astype(mx.bfloat16)
        w = mx.array(rng.integers(-3, 4, size=(n, k)).astype(np.float32) / 16).astype(mx.bfloat16)
        save(f"linear-{n}-{k}", "linear", {"rows": rows, "n": n, "k": k, "x_stride": k, "w_stride": k, "y_stride": n}, {"x": x, "weight": w}, x @ w.T)
    for heads, dim in [(1, 5120), (32, 128), (8, 128), (1, 64)]:
        rows, width = 8, heads * dim
        x, weight = array((rows, heads, dim)), array((dim,), 256) + mx.array(1, dtype=mx.bfloat16)
        expected = mx.fast.rms_norm(x, weight, 1e-6)
        save(f"norm-{heads}-{dim}", "norm", {"rows": rows, "heads": heads, "dim": dim, "x_stride": width, "y_stride": width, "round_normalized": 1, "eps": 1e-6}, {"x": x, "weight": weight}, expected)
    rows, width, group, block = 16, 64, 16, 8
    x, dynamic, base, residual = array((2, 8, width)), array((2, 8, 2, 2, width // group), 256), array((2, 2, width), 256), array((2, 8, width))
    for branch in (0, 1):
        def operation(hidden, dyn, b, r):
            y = conv(hidden, dyn[..., branch, :, :], b[branch], group)
            return y if branch == 0 else r + y
        for compiled in (False, True):
            expected = (mx.compile(operation) if compiled else operation)(x, dynamic, base, residual)
            save(f"conv-{branch}-{'compiled' if compiled else 'plain'}", "conv", {"rows": rows, "width": width, "group": group, "block": block, "branch": branch, "residual": branch, "x_stride": width, "dynamic_stride": 4 * width // group, "y_stride": width, "rounding": 0}, {"x": x, "dynamic": dynamic, "base": base, "residual": residual}, expected)
    for heads, dim, offset in [(32, 128, 0), (8, 128, 63), (2, 128, 2048), (1, 64, 262136)]:
        rows, width = 8, heads * dim
        x = array((1, heads, rows, dim))
        expected = mx.fast.rope(x, dims=dim, traditional=False, base=10000000, scale=1, offset=offset)
        positions = mx.array(np.arange(offset, offset + rows, dtype=np.uint64))
        save(f"rope-{heads}-{dim}-{offset}", "rope", {"rows": rows, "heads": heads, "dim": dim, "x_stride": width, "y_stride": width, "theta": 10000000}, {"x": x.transpose(0, 2, 1, 3), "positions": positions}, expected.transpose(0, 2, 1, 3))
    for width in (64, 17408):
        rows = 8
        gate, up = array((rows, width)), array((rows, width))
        for compiled in (False, True):
            operation = lambda g, u: nn.silu(g) * u
            expected = (mx.compile(operation) if compiled else operation)(gate, up)
            save(f"swiglu-{width}-{'compiled' if compiled else 'plain'}", "swiglu", {"rows": rows, "width": width, "gate_stride": width, "up_stride": width, "y_stride": width, "rounding": 0}, {"gate": gate, "up": up}, expected)
    (args.out / "manifest.json").write_text(json.dumps({"schema": "tf-draft-ops-v1", "cases": cases}, indent=2) + "\n")
    print(f"draft primitive fixtures: {len(cases)}, no model loaded")


if __name__ == "__main__":
    main()
