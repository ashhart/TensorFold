"""Original tensor attention oracle for 128/256-wide heads and strided long caches."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from tensorfold.kernels.qwen.dense.v1.lane_attention import lane_sdpa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--simd", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(10000)
    arrays, cases = {}, []
    if args.simd:
        for start in (0, 11, 1023, 4095, 8191, 10000):
            for parents in ((-1, 0, 1, 2, 3), (-1, 0, 0, 1, 2)):
                key = f"c{len(cases)}"
                def random(shape):
                    return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
                q = random((1, 24, len(parents), 256))
                length = start + len(parents)
                k, v = random((1, 4, length + 17, 256)), random((1, 4, length + 17, 256))
                outputs = []
                for row in range(len(parents)):
                    path, node = [], row
                    while node >= 0:
                        path.append(start + node)
                        node = parents[node]
                    ids = mx.array(list(range(start)) + list(reversed(path)), dtype=mx.int32)
                    outputs.append(mx.fast.scaled_dot_product_attention(
                        q[:, :, row:row+1], mx.take(k, ids, axis=2), mx.take(v, ids, axis=2), scale=0.0625))
                arrays.update({key + ".q": q, key + ".k": k, key + ".v": v,
                               key + ".expected": mx.concatenate(outputs, axis=2)})
                cases.append(dict(key=key, length=length, scale=0.0625, parents=parents))
    for dim in (() if args.simd else (128, 256)):
        for length, rows in ((513, 1), (9999, 3), (10007, 8)):
            key = f"c{len(cases)}"
            def random(shape):
                return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
            q = random((1, 32, rows, dim))
            k, v = random((1, 2, length + 17, dim)), random((1, 2, length + 17, dim))
            scale = dim ** -.5
            arrays.update({key + ".q": q, key + ".k": k, key + ".v": v,
                           key + ".expected": lane_sdpa(q, k[:, :, :length], v[:, :, :length], scale)})
            cases.append(dict(key=key, length=length, scale=scale))
    mx.save_safetensors(str(args.output / "arrays.safetensors"), arrays)
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Saved {len(cases)} {'SIMD tree' if args.simd else 'tensor'} attention fixtures")


if __name__ == "__main__":
    main()
