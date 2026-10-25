"""Spark-X2.5/2.6 (spark2_5) bf16 to the native checkpoint: linears and the tied embedding MLX affine, norms bf16."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np

GROUP = 64
COPIED = ("chat_template.jinja", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "vocab.json", "merges.txt", "LICENSE", "README.md")
QUANTIZED = (".self_attn.q_k_v_proj.weight", ".self_attn.g_proj.weight", ".self_attn.out_proj.weight",
             ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight")
SHARD_BYTES = 2 << 30


def read_shard(path: Path) -> dict[str, tuple[str, list[int], bytes]]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        out[k] = (v["dtype"], v["shape"], data[a:b])
    return out


def bf16_to_f32(raw: bytes, shape: list[int]) -> np.ndarray:
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round to nearest even, as an fp32 to bf16 cast does (no NaNs reach here)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def round_away(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))


def quantize(w: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MLX's affine rule in fp32 (bias at the larger edge, zero on the grid): w [n, k] to words, scales, biases."""
    n, k = w.shape
    if k % GROUP:
        raise SystemExit(f"width {k} is not a multiple of {GROUP}")
    bins = np.float32((1 << bits) - 1)
    g = w.reshape(n, k // GROUP, GROUP)
    w_max = g.max(axis=-1, keepdims=True)
    w_min = g.min(axis=-1, keepdims=True)
    mask = np.abs(w_min) > np.abs(w_max)
    scales = np.maximum((w_max - w_min) / bins, np.float32(1e-7)).astype(np.float32)
    scales = np.where(mask, scales, -scales)
    edge = np.where(mask, w_min, w_max)
    q0 = round_away(edge / scales)
    scales = np.where(q0 != 0, edge / np.where(q0 != 0, q0, 1), scales).astype(np.float32)
    biases = np.where(q0 == 0, np.float32(0), edge).astype(np.float32)
    q = np.clip(round_away((g - biases) / scales), 0, bins).astype(np.uint32).reshape(n, k)
    per = 32 // bits
    q = q.reshape(n, k // per, per)
    words = np.zeros((n, k // per), dtype=np.uint32)
    for j in range(per):
        words |= q[:, :, j] << np.uint32(bits * j)
    return words, f32_to_bf16(scales[..., 0]), f32_to_bf16(biases[..., 0])


def write_shard(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, at = {}, 0
    for k, (dt, shape, raw) in tensors.items():
        header[k] = {"dtype": dt, "shape": shape, "data_offsets": [at, at + len(raw)]}
        at += len(raw)
    header["__metadata__"] = {"format": "mlx"}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, raw in tensors.values():
            f.write(raw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="XHToken/Spark-X2.6-4B as downloaded")
    p.add_argument("output", type=Path)
    p.add_argument("--bits", type=int, default=8, choices=(4, 8))
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists")
    config = json.loads((args.source / "config.json").read_text())
    if config.get("model_type") != "spark2_5":
        raise SystemExit("expected a spark2_5 checkpoint")
    index = json.loads((args.source / "model.safetensors.index.json").read_text())["weight_map"]
    args.output.mkdir(parents=True)
    shards: list[dict] = [{}]
    size = 0
    weight_map = {}
    for shard in sorted(set(index.values())):
        for name, (dt, shape, raw) in sorted(read_shard(args.source / shard).items()):
            if dt != "BF16":
                raise SystemExit(f"{name} is {dt}, expected BF16")
            out: dict[str, tuple[str, list[int], bytes]] = {}
            if name.endswith(QUANTIZED) or name == "model.embedding.weight":
                base = name.removesuffix(".weight")
                w, s, b = quantize(bf16_to_f32(raw, shape), args.bits)
                out[base + ".weight"] = ("U32", list(w.shape), w.tobytes())
                out[base + ".scales"] = ("BF16", list(s.shape), s.tobytes())
                out[base + ".biases"] = ("BF16", list(b.shape), b.tobytes())
            else:
                out[name] = (dt, shape, raw)
            for k, v in out.items():
                if size + len(v[2]) > SHARD_BYTES and shards[-1]:
                    shards.append({})
                    size = 0
                shards[-1][k] = v
                size += len(v[2])
            print(f"{name} {shape}{' quantized' if len(out) == 3 else ''}", flush=True)
    names = [f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors" for i in range(len(shards))]
    total = 0
    for fname, tensors in zip(names, shards):
        write_shard(args.output / fname, tensors)
        for k, v in tensors.items():
            weight_map[k] = fname
            total += len(v[2])
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=4) + "\n")
    q = {"group_size": GROUP, "bits": args.bits, "mode": "affine"}
    out_config = dict(config, quantization=q, quantization_config=q)
    out_config.pop("auto_map", None)
    (args.output / "config.json").write_text(json.dumps(out_config, indent=2) + "\n")
    for name in COPIED:
        if (args.source / name).exists():
            shutil.copy(args.source / name, args.output / name)
    print(f"wrote {args.output}: spark2_5, affine {args.bits}-bit groups of {GROUP}, {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
