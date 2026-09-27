"""PLE normalization oracles: preserve Python's fp32 square/mean reduction order."""
import argparse
import json
from pathlib import Path

import mlx.core as mx

from tensorfold.families.qwen4_exp.model import CenteredRMSNorm
from tensorfold.kernels.qwen.flash_next.v1.kernels import rms_norm_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    arrays, cases = {}, []
    fused_differences = 0
    for rows in (1, 3, 16):
        for magnitude in (0, 1e-4, 1, 1000):
            for seed in (1, 42, 432):
                key = f"case{len(cases)}"
                norm = CenteredRMSNorm(10240, 1e-6, group=2560)
                norm.weight = (mx.random.normal((10240,), key=mx.random.key(seed + 1)) * .1).astype(mx.bfloat16)
                x = (mx.random.normal((rows, 10240), key=mx.random.key(seed)) * magnitude).astype(mx.bfloat16)
                arrays[f"{key}.x"] = x
                arrays[f"{key}.weight"] = norm.weight
                arrays[f"{key}.expected"] = norm(x)
                fused = rms_norm_rows(x, 1 + norm.weight.astype(mx.float32), mx.array([1e-6]), group=2560)
                fused_differences += int(mx.sum(fused != arrays[f"{key}.expected"]).item())
                cases.append(key)
    mx.eval(*arrays.values())
    mx.save_safetensors(str(args.directory / "arrays.safetensors"), arrays)
    (args.directory / "cases.json").write_text(json.dumps(cases))
    if not fused_differences:
        raise AssertionError("Fixtures must detect the former fused-reduction substitution")
    print(f"Saved {len(cases)} PLE norm cases; former fused path differs at {fused_differences} elements")


if __name__ == "__main__":
    main()
