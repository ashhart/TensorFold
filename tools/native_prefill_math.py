"""Independent mlx-lm activation oracles for Zig's compiled prefill graphs."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.activations import swiglu
from mlx_lm.models.gated_delta import compute_g
from mlx_lm.models.qwen3_next import _precise_swiglu
from mlx_lm.models.gemma4_text import geglu, logit_softcap
from native_runtime import require_mlx
from tensorfold.families.deepseek_v4.moe import swiglu as clipped_swiglu
from tensorfold.families.deepseek_v4.model import HeadHC


def main():
    require_mlx()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    arrays, cases = {}, []

    def save(kind, inputs, expected):
        key = f"case{len(cases)}"
        cases.append(dict(key=key, kind=kind, inputs=len(inputs)))
        arrays.update({f"{key}.input{i}": x for i, x in enumerate(inputs)})
        arrays[f"{key}.expected"] = expected

    bits = np.arange(65536, dtype=np.uint16)
    bits = bits[(bits & 0x7f80) != 0x7f80]  # Every finite BF16, including signed zero.
    x = mx.array(bits).view(mx.bfloat16)
    save("silu", [x], nn.silu(x))
    save("gelu", [x], nn.gelu(x))
    save("gelu_tanh", [x], nn.gelu_approx(x))
    save("softcap", [x, mx.array(30.0)], logit_softcap(30.0, x))
    for factor in (0.25, -1.0, 3.0):
        up = mx.full(x.shape, factor, dtype=mx.bfloat16)
        save("swiglu", [x, up], swiglu(x, up))
        save("geglu", [x, up], geglu(x, up))
        save("gated", [x, up], _precise_swiglu(up, x, up))
        for limit in (3.0, 10.0):
            save("clipped_swiglu", [x, up, mx.array(limit)], mx.compile(lambda gate, value: clipped_swiglu(gate, value, limit))(x, up))
    mx.random.seed(5678)
    for rows in (1, 11, 2048):
        a = mx.random.normal((1, rows, 48)).astype(mx.bfloat16)
        alog = mx.random.uniform(-2, 2, (48,)).astype(mx.bfloat16)
        dt = mx.random.normal((48,)).astype(mx.bfloat16)
        save("decay", [alog, a, dt], compute_g(alog, a, dt))
    for dims in (128, 4096, 128):
        for dtype in (mx.bfloat16, mx.float32):
            streams = mx.random.normal((1, 4, dims)).astype(mx.bfloat16)
            fn = (mx.random.normal((4, 4 * dims)) * 0.02).astype(dtype)
            base = mx.random.normal((4,)) * 0.1
            scale = mx.random.uniform(0.8, 1.2, (1,))
            head = HeadHC(fn, base, scale, 1e-6, 1e-6)
            mx.eval(streams, head.fn, head.base, head.scale)
            save("deepseek_head", [streams, fn, base, scale, mx.array([1e-6]), mx.array([1e-6])], head(streams, True))
    mx.save_safetensors(str(args.directory / "arrays.safetensors"), arrays)
    (args.directory / "cases.json").write_text(json.dumps(cases) + "\n")
    print(f"Wrote {len(cases)} prefill fixtures, including all {x.size} finite BF16 inputs")


if __name__ == "__main__":
    main()
