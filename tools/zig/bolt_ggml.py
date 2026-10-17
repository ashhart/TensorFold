"""Reference decode of Bolt's ggml tensors (IQ2_XXS, Q8_0), tables from the CUDA header: MODEL_DIR TENSOR [ROWS]."""

import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

HEADER = Path(__file__).resolve().parents[2] / "zig/kernels/cuda/ggml_iq2xxs_tables.cuh"
IQ2_BYTES, IQ2_ROW = 66, 256
Q8_BYTES, Q8_ROW = 34, 32


def _table(text, name):
    body = text.split(name, 1)[1].split("};", 1)[0]
    return [int(v, 0) for v in re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", body.split("=", 1)[1])]


def tables():
    text = HEADER.read_text()
    signs = np.array(_table(text, "tf_ksigns_iq2xs[128]"), dtype=np.uint8)
    grid = np.array(_table(text, "tf_iq2xxs_grid[256]"), dtype=np.uint64)
    assert signs.shape == (128,) and grid.shape == (256,)
    return signs, grid.view(np.uint8).reshape(256, 8)


def iq2_xxs(blocks):
    """[..., 66] uint8 -> [..., 256] float32, as ggml's dequantize_row_iq2_xxs."""
    signs, grid = tables()
    lead = blocks.shape[:-1]
    b = blocks.reshape(-1, IQ2_BYTES)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)[:, 0]
    q = b[:, 2:].copy().view(np.uint32).reshape(-1, 8, 2)  # 8 sub-blocks of 32 weights
    idx = np.ascontiguousarray(q[:, :, 0]).view(np.uint8).reshape(-1, 8, 4)
    aux = q[:, :, 1]
    db = d[:, None] * (0.5 + (aux >> 28).astype(np.float32)) * 0.25
    out = np.empty((b.shape[0], 8, 4, 8), dtype=np.float32)
    bits = (1 << np.arange(8)).astype(np.uint8)
    for l in range(4):
        s = signs[(aux >> (7 * l)) & 127]
        neg = (s[..., None] & bits) != 0
        out[:, :, l, :] = grid[idx[:, :, l]] * np.where(neg, -1.0, 1.0) * db[..., None]
    return out.reshape(*lead, IQ2_ROW)


def q8_0(blocks):
    """[..., 34] uint8 -> [..., 32] float32."""
    lead = blocks.shape[:-1]
    b = blocks.reshape(-1, Q8_BYTES)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(*lead, Q8_ROW)


def read(model, name):
    """The raw U8 array and its ggml format (from the export's quantized_layers)."""
    model = Path(model)
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    path = model / index[name]
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n))[name]
        lo, hi = meta["data_offsets"]
        fh.seek(8 + n + lo)
        raw = np.frombuffer(fh.read(hi - lo), dtype=np.uint8).reshape(meta["shape"])
    layers = json.loads((model / "config.json").read_text())["quantization_config"]["quantized_layers"]
    algo = layers.get(name.removesuffix(".weight"), {}).get("quant_algo")
    return raw, algo


def weight(model, name, rows=None):
    raw, algo = read(model, name)
    if rows is not None:
        raw = raw[:rows]
    if algo == "IQ2_XXS":
        assert raw.shape[-1] == IQ2_BYTES, raw.shape
        return iq2_xxs(raw).reshape(raw.shape[0], -1)
    if algo == "Q8_0":
        assert raw.shape[-1] == Q8_BYTES, raw.shape
        return q8_0(raw).reshape(raw.shape[0], -1)
    raise SystemExit(f"{name}: {algo} is not a ggml format")


if __name__ == "__main__":
    w = weight(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    print(w.shape, f"mean {w.mean():.6g} std {w.std():.6g} absmax {np.abs(w).max():.6g}")
