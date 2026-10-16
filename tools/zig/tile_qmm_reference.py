"""core/tile_qmm and mx.quantized_matmul on tf-tile-check's dumped matrices, both against float64."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

N, K, ROWS = 96, 1536, 70


def bf16(raw):
    return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)


def codes(raw, bits):
    stream = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little").reshape(N, K, bits)
    return (stream * (1 << np.arange(bits))).sum(-1).astype(np.float64)


def errors(y, ref, scale):
    rel = np.abs(y - ref) / np.maximum(scale, 1e-30)
    return dict(worst=float(rel.max()), mean=float(rel.mean()), worst_column=int(rel.max(0).argmax()),
                worst_column_mean=float(rel.mean(0).max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    args = p.parse_args()
    report = {}
    for bits in (4, 6, 8):
        f = lambda name: (args.dir / f"q{bits}_{name}.bin").read_bytes()
        w_raw, s, b, x = (f("w"), bf16(f("scales")).reshape(N, K // 64), bf16(f("biases")).reshape(N, K // 64),
                          bf16(f("x")).reshape(ROWS, K))
        q = codes(w_raw, bits)
        xg = x.astype(np.float64).reshape(ROWS, K // 64, 64)
        dots = np.einsum("rgi,ngi->rng", xg, q.reshape(N, K // 64, 64))
        sums = xg.sum(-1)
        terms_s = dots * s.astype(np.float64)[None]
        terms_b = sums[:, None, :] * b.astype(np.float64)[None]
        ref = (terms_s + terms_b).sum(-1)
        scale = (np.abs(terms_s) + np.abs(terms_b)).sum(-1)
        native = bf16(f("y")).reshape(ROWS, N).astype(np.float64)
        wq = mx.array(np.frombuffer(w_raw, np.uint32).reshape(N, K * bits // 32))
        y_mlx = mx.quantized_matmul(mx.array(x).astype(mx.bfloat16), wq, mx.array(s).astype(mx.bfloat16),
                                    mx.array(b).astype(mx.bfloat16), transpose=True, group_size=64, bits=bits)
        mlx_y = np.array(y_mlx.astype(mx.float32)).astype(np.float64)
        report[f"{bits}-bit"] = dict(native=errors(native, ref, scale), mlx=errors(mlx_y, ref, scale))
        print(bits, json.dumps(report[f"{bits}-bit"]))
    (args.dir / "tile_qmm_reference.json").write_text(
        json.dumps(dict(mlx=mx.__version__, report=report), indent=2) + "\n")


if __name__ == "__main__":
    main()
