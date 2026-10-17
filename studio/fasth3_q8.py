"""Usage: python fasth3_q8.py convert <FastH3 folder>: FastH3 with MLX 8-bit weights for chips without tensor units."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

GROUP, BITS = 64, 8
FOLDER = "transformer-q8"


def _lazy_checkpoint(root: Path):
    """(model, gates, contract) with every weight still on disk."""

    from tensorfold.families.h3 import fasth3

    evaluate = mx.eval
    mx.eval = lambda *_: None
    try:
        return fasth3.load_fasth3(str(root))
    finally:
        mx.eval = evaluate


def _quantize(module: nn.Module) -> None:
    nn.quantize(module, group_size=GROUP, bits=BITS, class_predicate=lambda _, m: isinstance(m, nn.Linear))


def convert(checkpoint_dir: str) -> Path:
    root = Path(checkpoint_dir)
    out = root / FOLDER
    out.mkdir(exist_ok=True)
    mx.set_cache_limit(2 * 2**30)
    model, gates, _ = _lazy_checkpoint(root)
    started = time.perf_counter()
    count = len(model.blocks)
    for index in range(count):
        block = model.blocks[index]
        for linear in (block.attn.qkv_proj, block.attn.out_proj, block.mlp.fc1, block.mlp.fc2, block.adaln_proj.linear):
            linear.weight = linear.weight.astype(mx.float32)  # the quantizer wants float32; scales are stored as float16
        _quantize(block)
        tensors = {}
        for name, value in tree_flatten(block.parameters()):
            tensors[name] = value.astype(mx.float16) if name.endswith((".scales", ".biases")) else value
        mx.eval(*tensors.values())
        mx.save_safetensors(str(out / f"block-{index:02d}.safetensors"), tensors)
        model.blocks[index] = None
        del block, tensors
        mx.clear_cache()
        print(f"[tensorfold] 8-bit block {index + 1}/{count}", flush=True)
    rest = {name: value for name, value in tree_flatten(model.parameters()) if not name.startswith("blocks.")}
    rest.update({f"gates.{index}": tensor for index, tensor in gates.items()})
    mx.eval(*rest.values())
    mx.save_safetensors(str(out / "rest.safetensors"), rest)
    (out / "q8.json").write_text(json.dumps({"group_size": GROUP, "bits": BITS, "blocks": count, "format": 1}))
    size = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 2**30
    print(f"[tensorfold] wrote {out} ({size:.1f} GiB) in {time.perf_counter() - started:.0f}s", flush=True)
    return out


def _modulation(self, temb: mx.array) -> mx.array:
    # dit.Modulation casts the embedding to the layer's weight type, which for a quantized layer is packed integers
    linear = self.linear
    kind = linear.scales.dtype if isinstance(linear, nn.QuantizedLinear) else linear.weight.dtype
    return linear(nn.silu(temb).astype(kind))


def ready(checkpoint_dir: str) -> bool:
    return (Path(checkpoint_dir) / FOLDER / "q8.json").is_file()


def load(checkpoint_dir: str):
    """(transformer with 8-bit block weights, routing gates by block, contract), as `fasth3.load_fasth3` returns."""

    from tensorfold.families.h3 import fasth3
    from tensorfold.families.h3.dit import H3DiT

    from tensorfold.families.h3.dit import Modulation

    Modulation.__call__ = _modulation
    root = Path(checkpoint_dir)
    folder = root / FOLDER
    meta = json.loads((folder / "q8.json").read_text())
    model = H3DiT(fasth3.config(root))
    for block in model.blocks:
        _quantize(block)
    rest = mx.load(str(folder / "rest.safetensors"))
    gates = {int(name.split(".")[1]): rest.pop(name) for name in [n for n in rest if n.startswith("gates.")]}
    model.update(tree_unflatten(list(rest.items())))
    for index in range(meta["blocks"]):
        model.blocks[index].update(tree_unflatten(list(mx.load(str(folder / f"block-{index:02d}.safetensors")).items())))
    mx.eval(model.parameters())
    return model, gates, fasth3.contract(root)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "convert":
        raise SystemExit(__doc__)
    convert(sys.argv[2])
