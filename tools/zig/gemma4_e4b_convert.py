"""Gemma 4 E4B (BF16) to the native checkpoint: text weights MLX affine 4-bit g64, named as mlx-lm names them."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

GROUP = 64
# where the language model's tensors sit in the HF checkpoint and in the one mlx-lm writes (the family reads the latter)
HF_PREFIX = "model.language_model."
PREFIX = "language_model.model."
SIDECARS = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "processor_config.json")
TOWER_CONFIGS = ("vision_config", "audio_config")
ATTN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
# the per-layer input linears --dense-ple keeps in bf16
PLE_LINEARS = ("per_layer_input_gate", "per_layer_projection", "per_layer_model_projection")
QUANTIZED = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "embed_tokens", "embed_tokens_per_layer") + ATTN + PLE_LINEARS
BLOCK_VALUES = 1 << 25  # values per numpy block: ~130 MB of float32, a few times that in temporaries


def bf16_round(x: np.ndarray) -> np.ndarray:
    """float32 to bf16 bits, round to nearest even."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_float(r: np.ndarray) -> np.ndarray:
    return (r.astype(np.uint32) << 16).view(np.float32)


def quantize(w: np.ndarray, bits: int):
    """w [rows, K] float32 of bf16 values: packed uint32 [rows, K*bits/32], scales and biases bf16 bits, codes."""
    rows, k = w.shape
    g = w.reshape(rows, k // GROUP, GROUP)
    bins = np.float32((1 << bits) - 1)
    lo, hi = g.min(-1), g.max(-1)
    scale = np.maximum(hi - lo, np.float32(1e-7)) / bins
    from_min = np.abs(lo) > np.abs(hi)
    edge = np.where(from_min, lo, hi)
    q0 = np.rint(np.abs(edge) / scale)  # ties to even, as MLX's rint
    with np.errstate(all="ignore"):
        scale = np.where(q0 == 0, scale, np.where(from_min, np.abs(edge), -np.abs(edge)) / q0)
    bias = np.where(q0 == 0, lo, edge)
    # the codes use the unrounded scale and bias, as MLX does; the stored ones are rounded to bf16
    q = np.clip(np.rint((g - bias[..., None]) / scale[..., None]), 0, bins).astype(np.uint32)
    per_word = 32 // bits
    packed = (q.reshape(rows, -1, per_word) << (bits * np.arange(per_word, dtype=np.uint32))).sum(-1, dtype=np.uint32)
    return packed, bf16_round(scale), bf16_round(bias), q


def dequantize(q: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) * bf16_float(scales)[..., None] + bf16_float(biases)[..., None]).reshape(q.shape[0], -1)


class Source:
    """The text tensors of a BF16 checkpoint, read by name from the shard the index (or the lone shard) puts them in."""

    def __init__(self, path: Path):
        self.path = path
        index = path / "model.safetensors.index.json"
        if index.exists():
            self.files = json.loads(index.read_text())["weight_map"]
        else:
            self.files = {k: "model.safetensors" for k in self.header(path / "model.safetensors")[1]}
        self.shards: dict[str, tuple[int, dict, np.ndarray]] = {}

    @staticmethod
    def header(path: Path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        return 8 + n, header

    def bf16(self, name: str) -> np.ndarray:
        file = self.files[name]
        if file not in self.shards:
            base, header = self.header(self.path / file)
            self.shards[file] = (base, header, np.memmap(self.path / file, mode="r", dtype=np.uint8))
        base, header, mapped = self.shards[file]
        t = header[name]
        if t["dtype"] != "BF16":
            raise SystemExit(f"{name} is {t['dtype']}, expected BF16")
        a, b = t["data_offsets"]
        return mapped[base + a : base + b].view(np.uint16).reshape(t["shape"])


def module_of(name: str) -> str:
    """A tensor's module as the quantization table names it: its name in the converted checkpoint, without `.weight`."""
    return PREFIX + name.removeprefix(HF_PREFIX).removesuffix(".weight")


def layer_of(name: str) -> int | None:
    parts = name.removeprefix(HF_PREFIX).split(".")
    return int(parts[1]) if parts[0] == "layers" else None


def is_quantized(name: str, dense_ple: bool) -> bool:
    if not name.endswith(".weight"):
        return False
    stem = name.removeprefix(HF_PREFIX).removesuffix(".weight")
    if stem.startswith("layers."):
        stem = stem.split(".", 2)[2]
    return stem in QUANTIZED and not (dense_ple and stem in PLE_LINEARS)


def bits_of(name: str, attn_bits: int) -> int:
    return attn_bits if name.removeprefix(HF_PREFIX).split(".", 2)[-1].removesuffix(".weight") in ATTN else 4


def quantization(names: list[str], dense_ple: bool, attn_bits: int) -> dict:
    """The config table: 4-bit groups of 64, mlx-lm entries for the modules at other widths and for bf16 (False)."""
    table: dict = {"group_size": GROUP, "bits": 4, "mode": "affine"}
    for name in sorted(names):
        if is_quantized(name, dense_ple) and bits_of(name, attn_bits) != 4:
            table[module_of(name)] = {"group_size": GROUP, "bits": attn_bits, "mode": "affine"}
        elif dense_ple and name.endswith(tuple(m + ".weight" for m in PLE_LINEARS)):
            table[module_of(name)] = False
    return table


def write_safetensors(path: Path, items: list[tuple[str, str, list[int], bytes]], metadata: dict[str, str]) -> None:
    header: dict = {"__metadata__": metadata}
    off = 0
    for name, dtype, shape, data in items:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    blob = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, _, data in items:
            f.write(data)


def check_block(mx, w: np.ndarray, bits: int, mine: tuple[np.ndarray, ...]) -> list[str]:
    """What differs from mx.quantize on this block of rows (weights, scales, biases)."""
    qm, sm, bm = mx.quantize(mx.array(w).astype(mx.bfloat16), group_size=GROUP, bits=bits, mode="affine")
    theirs = (np.array(qm), np.array(sm.view(mx.uint16)), np.array(bm.view(mx.uint16)))
    return [what for what, a, b in zip(("weight", "scales", "biases"), mine, theirs) if not np.array_equal(a, b.astype(a.dtype))]


def convert(source: Path, output: Path, dense_ple: bool = False, attn_bits: int = 4, check_mlx: bool = False) -> int:
    """Writes the checkpoint; returns how many blocks differ from mx.quantize (0 without --check-mlx)."""
    config = json.loads((source / "config.json").read_text())
    text = config["text_config"]
    if config.get("model_type") != "gemma4" or not text.get("hidden_size_per_layer_input"):
        raise SystemExit("expected a Gemma 4 E-series checkpoint")
    owners = text["num_hidden_layers"] - text["num_kv_shared_layers"]
    src = Source(source)
    mx = None
    if check_mlx:
        import mlx.core as mx
        # MLX's Metal quantize rounds some scales differently from its CPU one; the CPU one is the reference
        mx.set_default_device(mx.cpu)
    names = sorted(n for n in src.files if n.startswith(HF_PREFIX))
    shared_kv = ("self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.k_norm.weight")
    names = [n for n in names if not (layer_of(n) is not None and layer_of(n) >= owners and n.endswith(shared_kv))]
    items: list[tuple[str, str, list[int], bytes]] = []
    diverged = 0
    for name in names:
        raw = src.bf16(name)
        out = PREFIX + name.removeprefix(HF_PREFIX)
        if not is_quantized(name, dense_ple):
            items.append((out, "BF16", list(raw.shape), raw.tobytes()))
            continue
        rows, k = raw.shape
        if k % GROUP:
            raise SystemExit(f"{name}: width {k} is not a multiple of {GROUP}")
        bits = bits_of(name, attn_bits)
        step = max(1, BLOCK_VALUES // k)
        words, scales, biases = [], [], []
        for r in range(0, rows, step):
            w = bf16_float(raw[r : r + step])
            packed, s, b, _ = quantize(w, bits)
            words.append(packed), scales.append(s), biases.append(b)
            if mx is not None:
                for what in check_block(mx, w, bits, (packed, s, b)):
                    diverged += 1
                    print(f"DIVERGES {name} {what}: rows {r} to {r + w.shape[0]}", flush=True)
        stem = out.removesuffix(".weight")
        items.append((out, "U32", [rows, k * bits // 32], np.concatenate(words).tobytes()))
        items.append((stem + ".scales", "BF16", [rows, k // GROUP], np.concatenate(scales).tobytes()))
        items.append((stem + ".biases", "BF16", [rows, k // GROUP], np.concatenate(biases).tobytes()))
        print(f"{name} {raw.shape} -> {bits}-bit", flush=True)
    items.sort(key=lambda x: x[0])
    output.mkdir(parents=True)
    write_safetensors(output / "model.safetensors", items, {"format": "mlx"})
    index = {"metadata": {"total_size": sum(len(i[3]) for i in items)}, "weight_map": {i[0]: "model.safetensors" for i in items}}
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    table = quantization(names, dense_ple, attn_bits)
    out_config = {k: v for k, v in config.items() if k not in TOWER_CONFIGS}
    out_config.update(quantization=table, quantization_config=table)
    (output / "config.json").write_text(json.dumps(out_config, indent=2) + "\n")
    for f in SIDECARS:
        if (source / f).exists():
            shutil.copy(source / f, output / f)
    print(f"wrote {output}: {len(items)} tensors, {sum(i[0].endswith('.scales') for i in items)} quantized")
    return diverged


def main() -> None:
    notes = "The towers and the k_proj, v_proj and k_norm of layers that read an earlier layer's cache are not written."
    p = argparse.ArgumentParser(description=__doc__, epilog=notes)
    p.add_argument("source", type=Path, help="the BF16 checkpoint as downloaded (config.json, the shards and their index)")
    p.add_argument("output", type=Path)
    p.add_argument("--attn-bits", type=int, choices=(4, 8), default=4, help="bits of the q, k, v and o projections")
    p.add_argument("--dense-ple", action="store_true", help="keep the per-layer input linears in bf16 (the per-layer table stays 4-bit)")
    p.add_argument("--check-mlx", action="store_true", help="compare every quantized block with mx.quantize (mlx installed), byte for byte")
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists")
    diverged = convert(args.source, args.output, args.dense_ple, args.attn_bits, args.check_mlx)
    if diverged:
        sys.exit(f"{diverged} blocks differ from mx.quantize")


if __name__ == "__main__":
    main()
