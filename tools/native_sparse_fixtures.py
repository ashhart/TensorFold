"""Exercise Flash Next sparse-attention boundaries without a 2K-token full-model prefill.

Uses the original Python fused attention host and kernels with deterministic synthetic
weights at the real model dimensions. Full-checkpoint parity is checked separately.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tensorfold.families.qwen4_exp.decode import FusedDecode, _stacked
from tensorfold.families.qwen4_exp import decode
from tensorfold.families.qwen4_exp.model import AttentionCache
from tensorfold.kernels.qwen.flash_next.v1 import attention as K


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    # Native Flash preserves the original row projection arithmetic. Upstream's
    # M5 default now uses lane projections with different rounding.
    decode.DENSE = "rows"
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1422)
    arrays = {}
    base = "model.layers.3.self_attn"

    def random(shape, scale=1.):
        return mx.array(rng.standard_normal(shape).astype(np.float32) * scale).astype(mx.bfloat16)

    def linear(name, inputs, outputs):
        layer = nn.QuantizedLinear(inputs, outputs, bias=False, group_size=32, bits=4)
        layer.weight, layer.scales, layer.biases = mx.quantize(random((outputs, inputs), .02), group_size=32, bits=4)
        for field in ("weight", "scales", "biases"):
            arrays[f"{base}.{name}.{field}"] = getattr(layer, field)
        return layer

    projections = [linear(name, 2560, width) for name, width in
                   (("q_proj", 12288), ("k_proj", 512), ("v_proj", 512), ("indexer.index_qk_proj", 640))]
    out = linear("o_proj", 6144, 2560)
    scales = []
    for name, width in (("q_norm", 256), ("k_norm", 256),
                        ("indexer.q_layernorm", 128), ("indexer.k_layernorm", 128)):
        value = (random((width,), .05).astype(mx.float32) + 1).astype(mx.bfloat16)
        arrays[f"{base}.{name}.weight"] = value
        scales.append(value.astype(mx.float32))
    fused = FusedDecode.__new__(FusedDecode)
    fused.cfg = SimpleNamespace(num_attention_heads=24, num_key_value_heads=2, head_dim=256,
                                indexer_n_heads=4, indexer_head_dim=128, rotary_dim=64,
                                rope_theta=10000000., indexer_compress_ratio=4)
    fused.eps = mx.array([1e-6], dtype=mx.float32)
    fused._pos = (None, None)
    fused.layers = [{"attn": (_stacked(projections)[0], *scales,
                            SimpleNamespace(o_proj=out, scale=.0625, indexer=SimpleNamespace(top_blocks=512)))}]
    cases = []
    for past, pooled in ((2044, 0), (2051, 0), (2063, 500)):
        key = f"c{len(cases)}"
        x = random((8, 2560))
        cache = AttentionCache()
        cache.offset = past
        cache.keys = random((1, 2, past, 256))
        cache.values = random((1, 2, past, 256))
        cache.index_keys = random((1, past, 128))
        arrays.update({key + ".x": x, key + ".a": cache.keys,
                       key + ".b": cache.values, key + ".raw": cache.index_keys[0]})
        if pooled:
            cache.pooled = K.index_pool(cache.index_keys[0], 0, pooled, scales[3], fused.eps,
                                        rotary_dim=64, base=10000000.)[None]
            arrays[key + ".pooled"] = cache.pooled[0]
        arrays[key + ".expected"] = fused._attention(0, x, cache)
        arrays[key + ".pooled_expected"] = cache.pooled[0]
        mx.eval(arrays[key + ".expected"], arrays[key + ".pooled_expected"])
        cases.append(dict(key=key, past=past, pooled=pooled))
    mx.save_safetensors(str(args.output / "arrays.safetensors"), arrays)
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Saved {len(cases)} sparse attention boundary fixtures in {args.output}")


if __name__ == "__main__":
    main()
