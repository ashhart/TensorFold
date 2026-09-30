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


def ssm_fixtures(directory):
    from mlx_lm.models.ssm import ssm_attn
    mx.random.seed(8765)
    arrays, cases = {}, []
    for heads, groups, dims, state_dim in ((8, 2, 16, 32), (64, 8, 64, 128)):
        for dtype in (mx.bfloat16, mx.float32):
            state = None
            for rows in (1, 17, 255, 256, 257, 513, 2048):
                batch = 2 if heads == 8 else 1
                x = (mx.random.normal((batch, rows, heads, dims)) * 0.2).astype(dtype)
                a = mx.random.uniform(-2, 1, (heads,))
                b = (mx.random.normal((batch, rows, groups, state_dim)) * 0.1).astype(dtype)
                c = (mx.random.normal(b.shape) * 0.1).astype(dtype)
                d = mx.random.normal((heads,)).astype(dtype)
                dt = (mx.random.normal((batch, rows, heads)) * 2).astype(dtype)
                bias = mx.random.uniform(-4, 0, (heads,)).astype(dtype)
                limits = (0.001, 0.7) if rows in (17, 257) else (0.0, 1e6)
                out, next_state = ssm_attn(x, a, b, c, d, dt, bias, state, limits)
                key = f"case{len(cases)}"
                cases.append(dict(key=key, state=state is not None, limits=limits))
                inputs = [x, a, b, c, d, dt, bias, mx.zeros((batch, heads, dims, state_dim)) if state is None else state]
                arrays.update({f"{key}.input{i}": value for i, value in enumerate(inputs)})
                arrays[f"{key}.output"], arrays[f"{key}.state"] = out, next_state
                mx.eval(out, next_state)
                state = next_state
    mx.save_safetensors(str(directory / "ssm.safetensors"), arrays)
    (directory / "ssm.json").write_text(json.dumps(cases) + "\n")
    print(f"Wrote {len(cases)} chunked SSD fixtures, including production Nemotron dimensions and continuation")


def main():
    require_mlx()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ssm-only", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.ssm_only:
        return ssm_fixtures(args.directory)
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
