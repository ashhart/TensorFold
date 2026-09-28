"""DeepSeek's MTP layer (the official checkpoint's last shard) in the mlx-community layout: FP8 -> affine 4-bit g64."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

# a block's official tensors and their names in the mlx-community layout (FP8 linears, plain, hyper-connections)
LINEARS = {"attn.wq_a": "attn.wq_a", "attn.wq_b": "attn.wq_b", "attn.wkv": "attn.wkv", "attn.wo_a": "attn.wo_a",
           "attn.wo_b": "attn.wo_b", "ffn.shared_experts.w1": "ffn.shared_experts.gate_proj",
           "ffn.shared_experts.w2": "ffn.shared_experts.down_proj",
           "ffn.shared_experts.w3": "ffn.shared_experts.up_proj"}
PLAIN = {"attn_norm.weight": "attn_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
         "attn.q_norm.weight": "attn.q_norm.weight", "attn.kv_norm.weight": "attn.kv_norm.weight",
         "attn.attn_sink": "attn.attn_sink", "ffn.gate.weight": "ffn.gate.weight",
         "ffn.gate.bias": "ffn.gate.e_score_correction_bias"}
HC = {"hc_attn": "attn_hc", "hc_ffn": "ffn_hc", "hc_head": "hc_head"}
EXPERTS = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
# the extra tensors of the MTP layer and of DSpark's first and last blocks (FP8 linears quantized, the rest as stored)
MTP_LINEARS = ("e_proj", "h_proj")
MTP_PLAIN = ("enorm.weight", "hnorm.weight", "norm.weight")
DSPARK_LINEARS = ("main_proj",)
DSPARK_PLAIN = ("main_norm.weight", "norm.weight", "markov_head.markov_w1.weight", "markov_head.markov_w2.weight")


RAW = {"F8_E4M3": np.uint8, "F8_E8M0": np.uint8, "I8": np.uint8, "U8": np.uint8, "BF16": np.uint16,
       "F32": np.float32}


def read_raw(path: Path, prefix: str) -> dict[str, mx.array]:
    """Tensors under ``prefix`` from a safetensors file as stored (8-bit codes as uint8, bf16 as bf16)."""

    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + size)
    out = {}
    for name, info in header.items():
        if name == "__metadata__" or not name.startswith(prefix):
            continue
        lo, hi = info["data_offsets"]
        a = np.frombuffer(data[lo:hi], dtype=RAW[info["dtype"]]).reshape(info["shape"])
        out[name] = mx.array(a).view(mx.bfloat16) if info["dtype"] == "BF16" else mx.array(a)
    return out


def e8m0(scale: mx.array) -> mx.array:
    """E8M0 exponent bytes as fp32 powers of two (MLX's reading: byte 0 is 2^-127)."""

    bits = scale.astype(mx.uint32)
    return mx.where(bits == 0, mx.array(0x400000, dtype=mx.uint32), bits << 23).view(mx.float32)


def fp8_block(weight: mx.array, scale: mx.array, block: int = 128) -> mx.array:
    """An e4m3 weight [O, I] with E8M0 scales per 128 x 128 block, exactly in bf16."""

    rows, cols = weight.shape
    w = mx.from_fp8(weight, dtype=mx.float32)
    s = mx.repeat(mx.repeat(e8m0(scale), block, axis=0)[:rows], block, axis=1)[:, :cols]
    return (w * s).astype(mx.bfloat16)


def _linear(t: dict[str, mx.array], raw: dict[str, mx.array], src: str, dst: str) -> None:
    w = fp8_block(raw[f"{src}.weight"], raw[f"{src}.scale"])
    t[f"{dst}.weight"], t[f"{dst}.scales"], t[f"{dst}.biases"] = mx.quantize(w, group_size=64, bits=4)
    mx.eval(t[f"{dst}.weight"], t[f"{dst}.scales"], t[f"{dst}.biases"])


def convert_block(raw: dict[str, mx.array], src: str, dst: str, linears: tuple[str, ...] = (),
                  plain: tuple[str, ...] = ()) -> dict[str, mx.array]:
    """One block's tensors (``src``: e.g. ``mtp.0``) under ``dst`` in the mlx-community names, plus its extras."""

    t: dict[str, mx.array] = {}
    for name in (*LINEARS, *linears):
        if f"{src}.{name}.weight" in raw:
            _linear(t, raw, f"{src}.{name}", f"{dst}.{LINEARS.get(name, name)}")
    for name in (*PLAIN, *plain):
        if f"{src}.{name}" in raw:
            t[f"{dst}.{PLAIN.get(name, name)}"] = raw[f"{src}.{name}"]
    for name, new in HC.items():
        if f"{src}.{name}_fn" in raw:
            for part in ("fn", "base", "scale"):
                t[f"{dst}.{new}.{part}"] = raw[f"{src}.{name}_{part}"]
    experts = len([k for k in raw if k.startswith(f"{src}.ffn.experts.") and k.endswith(".w1.weight")])
    for name, new in EXPERTS.items():
        codes = [raw[f"{src}.ffn.experts.{e}.{name}.weight"] for e in range(experts)]
        scales = [raw[f"{src}.ffn.experts.{e}.{name}.scale"] for e in range(experts)]
        t[f"{dst}.ffn.switch_mlp.{new}.weight"] = mx.stack([c.view(mx.uint32) for c in codes])
        t[f"{dst}.ffn.switch_mlp.{new}.scales"] = mx.stack(scales)
        mx.eval(t[f"{dst}.ffn.switch_mlp.{new}.weight"], t[f"{dst}.ffn.switch_mlp.{new}.scales"])
    return t


def convert_mtp(shard: Path, out: Path, layer: int = 0) -> Path:
    """Write ``out`` (names ``mtp.*``) from the official shard holding ``mtp.<layer>.*``."""

    t = convert_block(read_raw(shard, f"mtp.{layer}."), f"mtp.{layer}", "mtp", MTP_LINEARS, MTP_PLAIN)
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), t, metadata={"format": "mlx"})
    return out


def convert_dspark(shards: list[Path], out: Path) -> Path:
    """Write ``out`` (names ``dspark.<i>.*``) from the DSpark checkpoint's shards holding its blocks ``mtp.<i>.*``."""

    t: dict[str, mx.array] = {}
    for shard in shards:
        raw = read_raw(shard, "mtp.")
        for i in sorted({int(k.split(".")[1]) for k in raw}):
            part = {k: v for k, v in raw.items() if k.startswith(f"mtp.{i}.")}
            t.update(convert_block(part, f"mtp.{i}", f"dspark.{i}", DSPARK_LINEARS, DSPARK_PLAIN))
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), t, metadata={"format": "mlx"})
    official = shards[0].parent / "config.json"
    if official.is_file():                         # the block size, noise token and target layers the runtime reads
        config = json.loads(official.read_text())
        (out.parent / "config.json").write_text(json.dumps({k: v for k, v in config.items()
                                                            if k.startswith("dspark_")}, indent=1))
    return out


def main(argv: list[str] | None = None) -> int:
    """``python -m tensorfold.families.deepseek_v4.convert mtp SHARD OUT | dspark SHARD... OUT``."""

    import argparse

    parser = argparse.ArgumentParser(prog="python -m tensorfold.families.deepseek_v4.convert",
                                     description="DeepSeek-V4-Flash draft heads from DeepSeek's checkpoints")
    parser.add_argument("kind", choices=("mtp", "dspark"))
    parser.add_argument("paths", nargs="+", type=Path, help="the official shard(s), then the output file")
    args = parser.parse_args(argv)
    *shards, out = args.paths
    if not shards:
        parser.error("give the official shard(s) and the output file")
    written = convert_mtp(shards[0], out) if args.kind == "mtp" else convert_dspark(shards, out)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
