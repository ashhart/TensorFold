"""Aleph Alpha's FP8 Kolibri 1 as an MLX affine checkpoint, one layer at a time (it never holds the bf16 model).

The FP8 linears (e4m3 codes, an fp32 scale per 128 x 128 block) are expanded exactly to bf16, then quantized to MLX
affine 8-bit (or 4-bit) in groups of 64. The router, norms and expert bias stay as stored; the embedding and the head
(bf16 in the checkpoint) are quantized at 8 bits, as the MLX conversions do.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

RAW = {"F8_E4M3": np.uint8, "BF16": np.uint16, "F32": np.float32}
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
# files copied beside the weights as they are
SIDE_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja", "LICENSE")


class Shards:
    """The official shards' headers, each file mapped once; tensors read by name as stored."""

    def __init__(self, model_dir: Path) -> None:
        self.dir = Path(model_dir)
        self.where: dict[str, str] = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self._open: dict[str, tuple[dict[str, Any], np.memmap]] = {}

    def _file(self, name: str) -> tuple[dict[str, Any], np.memmap]:
        if name not in self._open:
            path = self.dir / name
            with open(path, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(size))
            self._open[name] = (header, np.memmap(path, dtype=np.uint8, mode="r", offset=8 + size))
        return self._open[name]

    def __contains__(self, key: str) -> bool:
        return key in self.where

    def __getitem__(self, key: str) -> mx.array:
        header, data = self._file(self.where[key])
        info = header[key]
        lo, hi = info["data_offsets"]
        a = mx.array(np.frombuffer(data[lo:hi], dtype=RAW[info["dtype"]]).reshape(info["shape"]))
        return a.view(mx.bfloat16) if info["dtype"] == "BF16" else a


def fp8_block(codes: mx.array, scale_inv: mx.array, block: int = 128) -> mx.array:
    """An e4m3 weight [O, I] times its fp32 scale per 128 x 128 block, in fp32, then bf16 (the reference's dequant)."""

    rows, cols = codes.shape
    w = mx.from_fp8(codes, dtype=mx.float32)
    s = mx.repeat(mx.repeat(scale_inv.astype(mx.float32), block, axis=0)[:rows], block, axis=1)[:, :cols]
    return (w * s).astype(mx.bfloat16)


def _quantized(out: dict[str, mx.array], name: str, weight: mx.array, bits: int) -> None:
    out[f"{name}.weight"], out[f"{name}.scales"], out[f"{name}.biases"] = mx.quantize(weight, group_size=64, bits=bits)


def _linear(shards: Shards, key: str) -> mx.array:
    """A linear's bf16 weight: FP8 expanded with its block scales, or as stored."""

    if f"{key}.weight_scale_inv" in shards:
        return fp8_block(shards[f"{key}.weight"], shards[f"{key}.weight_scale_inv"])
    return shards[f"{key}.weight"]


def convert_layer(shards: Shards, layer: int, experts: int, bits: int) -> dict[str, mx.array]:
    """Layer ``layer``'s tensors in the MLX layout (``switch_mlp`` stacks, ``mlp.expert_bias``)."""

    p = f"model.layers.{layer}"
    out: dict[str, mx.array] = {}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        _quantized(out, f"{p}.self_attn.{name}", _linear(shards, f"{p}.self_attn.{name}"), bits)
    for name in PROJECTIONS:
        _quantized(out, f"{p}.mlp.shared_experts.{name}", _linear(shards, f"{p}.mlp.shared_experts.{name}"), bits)
        parts = [mx.quantize(_linear(shards, f"{p}.mlp.experts.{e}.{name}"), group_size=64, bits=bits)
                 for e in range(experts)]
        for i, suffix in enumerate(("weight", "scales", "biases")):
            out[f"{p}.mlp.switch_mlp.{name}.{suffix}"] = mx.stack([part[i] for part in parts])
        mx.eval(*[out[f"{p}.mlp.switch_mlp.{name}.{s}"] for s in ("weight", "scales", "biases")])
    for name in ("input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm",
                 "self_attn.q_norm", "self_attn.k_norm", "mlp.gate"):
        out[f"{p}.{name}.weight"] = shards[f"{p}.{name}.weight"]
    out[f"{p}.mlp.expert_bias"] = shards[f"{p}.moe.router.expert_bias"]
    mx.eval(list(out.values()))
    return out


def convert(model_dir: Path, out_dir: Path, bits: int = 8, shard_bytes: int = 5 << 30) -> Path:
    """Write ``out_dir``: MLX safetensors shards, their index, config.json with the MLX quantization, side files."""

    model_dir, out_dir = Path(model_dir), Path(out_dir)
    config = json.loads((model_dir / "config.json").read_text())
    if (config.get("quantization_config") or {}).get("quant_method") != "fp8":
        raise ValueError(f"{model_dir}: not Aleph Alpha's FP8 checkpoint (quantization_config.quant_method fp8)")
    shards = Shards(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    pending: dict[str, mx.array] = {}
    written = 0

    def flush() -> None:
        nonlocal pending, written
        if not pending:
            return
        written += 1
        name = f"model-{written:05d}.safetensors"
        mx.save_safetensors(str(out_dir / name), pending, metadata={"format": "mlx"})
        weight_map.update({k: name for k in pending})
        pending = {}

    def add(tensors: dict[str, mx.array]) -> None:
        pending.update(tensors)
        if sum(a.nbytes for a in pending.values()) >= shard_bytes:
            flush()

    head: dict[str, mx.array] = {"model.norm.weight": shards["model.norm.weight"]}
    for name in ("model.embed_tokens", "lm_head"):
        _quantized(head, name, shards[f"{name}.weight"], 8)
    mx.eval(list(head.values()))
    add(head)
    layers, experts = int(config["num_hidden_layers"]), int(config["num_experts"])
    for layer in range(layers):
        add(convert_layer(shards, layer, experts, bits))
        print(f"[convert] layer {layer + 1}/{layers}", flush=True)
    flush()
    # name the shards by their count, as mlx_lm and the hub do
    renamed = {}
    for i in range(1, written + 1):
        old, new = f"model-{i:05d}.safetensors", f"model-{i:05d}-of-{written:05d}.safetensors"
        (out_dir / old).rename(out_dir / new)
        renamed[old] = new
    total = sum((out_dir / n).stat().st_size for n in renamed.values())
    index = {"metadata": {"total_size": total}, "weight_map": {k: renamed[v] for k, v in sorted(weight_map.items())}}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=1) + "\n")
    config.pop("quantization_config", None)
    eight = {"group_size": 64, "bits": 8}
    config["quantization"] = {"group_size": 64, "bits": bits, "mode": "affine", "model.embed_tokens": eight,
                              "lm_head": eight}
    config["quantization_config"] = config["quantization"]
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in SIDE_FILES:
        if (model_dir / name).is_file():
            shutil.copy2(model_dir / name, out_dir / name)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    """``python -m tensorfold.families.kolibri1.convert FP8_DIR OUT_DIR [--bits 8]``."""

    import argparse

    parser = argparse.ArgumentParser(prog="python -m tensorfold.families.kolibri1.convert",
                                     description="Aleph Alpha's FP8 Kolibri 1 as an MLX affine checkpoint")
    parser.add_argument("source", type=Path, help="the official checkpoint folder (Aleph-Alpha/Kolibri-1)")
    parser.add_argument("out", type=Path, help="the folder to write")
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    args = parser.parse_args(argv)
    print(f"wrote {convert(args.source, args.out, bits=args.bits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
