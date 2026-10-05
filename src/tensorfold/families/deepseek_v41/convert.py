"""Stream DeepSeek-V4.1's mixed FP4/FP8 source shards into MLX affine weights.

Requires the optional ``tensorfold[convert]`` dependencies. The source is never modified; each
converted shard is atomically installed, and a runnable checkpoint index is written only after
all source shards have been converted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
E8M0_BIAS = 127
QUANT_CHUNK_ROWS = 8192


def _e8m0_to_pow2(scale: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, scale.view(torch.uint8).long().float() - E8M0_BIAS)


def dequant_fp4_e8m0(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode e2m1fn nibble weights and e8m0 scales to fp32."""
    if weight.dtype != torch.int8 or weight.ndim != 2:
        raise ValueError("FP4 weights must be a rank-2 int8 tensor")
    out_dim, in_half = weight.shape
    x = weight.view(torch.uint8)
    values = torch.stack((FP4_TABLE[(x & 15).long()], FP4_TABLE[(x >> 4).long() & 15]), dim=-1)
    values = values.flatten(2).reshape(out_dim, in_half * 2)
    scales = _e8m0_to_pow2(scale)
    if scales.ndim == 2 and scales.shape == (out_dim, in_half * 2 // 32):
        return values * scales.repeat_interleave(32, dim=1)
    if scales.ndim == 2 and scales.shape == (out_dim // 32, in_half * 2 // 32):
        return values * scales.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
    raise ValueError(f"FP4 scale shape {tuple(scale.shape)} does not fit weight {tuple(weight.shape)}")


def dequant_fp8_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode e4m3fn weights with dense, per-row, 32x32, or 1x128 scales."""
    values = weight.float()
    if tuple(scale.shape) == tuple(weight.shape):
        return values * scale.float()
    out_dim, in_dim = weight.shape
    scales = _e8m0_to_pow2(scale) if scale.dtype == torch.float8_e8m0fnu else scale.float()
    if scales.ndim == 2 and scales.shape == (out_dim, in_dim // 32):
        return values * scales.repeat_interleave(32, dim=1)
    if scales.ndim == 2 and scales.shape == (out_dim // 32, in_dim // 32):
        return values * scales.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
    if scales.ndim == 2 and scales.shape == (out_dim, in_dim // 128):
        return values * scales.repeat_interleave(128, dim=1)
    raise ValueError(f"FP8 scale shape {tuple(scale.shape)} does not fit weight {tuple(weight.shape)}")


def pack_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned codes, element 0 in the low bits of word 0, into row-major uint32."""
    if bits not in (3, 4) or codes.ndim != 2:
        raise ValueError("only rank-2 3-bit and 4-bit codes are supported")
    rows, width = codes.shape
    padded_width = (width + 31) // 32 * 32
    if padded_width != width:
        codes = torch.nn.functional.pad(codes, (0, padded_width - width))
    positions = torch.arange(padded_width, dtype=torch.int64) * bits
    word, offset = positions // 32, positions % 32
    packed = torch.zeros((rows, (padded_width * bits + 31) // 32), dtype=torch.int64)
    row_ids = torch.arange(rows, dtype=torch.int64).unsqueeze(1).expand(rows, padded_width)
    values = codes.to(torch.int64)
    whole = offset + bits <= 32
    whole_rows = torch.arange(rows, dtype=torch.int64).unsqueeze(1).expand(rows, int(whole.sum()))
    packed.index_put_((whole_rows, word[whole].expand(rows, -1)), values[:, whole] << offset[whole], accumulate=True)
    crossing = ~whole
    if crossing.any():
        spill = 32 - offset[crossing]
        packed.index_put_(
            (row_ids[:, crossing], word[crossing].expand(rows, -1)),
            ((values[:, crossing] & ((1 << spill) - 1)) << offset[crossing]),
            accumulate=True,
        )
        packed.index_put_(
            (row_ids[:, crossing], (word[crossing] + 1).expand(rows, -1)), values[:, crossing] >> spill, accumulate=True
        )
    return packed[:, : (width * bits + 31) // 32].to(torch.uint32)


def unpack_bits(packed: torch.Tensor, bits: int, width: int) -> torch.Tensor:
    """Unpack the row-major low-bit-first format (used by tests and verification)."""
    if bits not in (3, 4) or packed.ndim != 2 or width < 1:
        raise ValueError("invalid packed tensor shape or bit width")
    positions = torch.arange(width, dtype=torch.int64) * bits
    word, offset = positions // 32, positions % 32
    values = packed.to(torch.int64)[:, word] >> offset
    crossing = offset + bits > 32
    if crossing.any():
        spill = 32 - offset[crossing]
        values[:, crossing] |= packed.to(torch.int64)[:, word[crossing] + 1] << spill
    return values & ((1 << bits) - 1)


def quantize_pack(weight: torch.Tensor, bits: int, group: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Affine per-output-row quantization with BF16 scales and biases."""
    if weight.ndim != 2 or weight.shape[1] % group:
        raise ValueError(f"matrix width {weight.shape[-1]} must be divisible by group size {group}")
    rows, width = weight.shape
    levels = 1 << bits
    grouped = weight.float().reshape(rows, width // group, group)
    minimum, maximum = grouped.amin(-1), grouped.amax(-1)
    scales = ((maximum - minimum) / (levels - 1)).clamp_min(1e-12)
    codes = ((grouped - minimum.unsqueeze(-1)) / scales.unsqueeze(-1)).round().clamp_(0, levels - 1)
    packed = pack_bits(codes.reshape(rows, width).to(torch.int64), bits)
    return packed, scales.to(torch.bfloat16), minimum.to(torch.bfloat16)


def dequantize_packed(
    packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    bits: int,
    group: int = 64,
    width: int | None = None,
) -> torch.Tensor:
    """Reconstruct an affine tensor for conversion tests."""
    width = width or packed.shape[1] * 32 // bits
    codes = unpack_bits(packed, bits, width).float().reshape(packed.shape[0], width // group, group)
    return (codes * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)).reshape(packed.shape[0], width)


def _kind(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.int8 and tensor.ndim == 2:
        return "fp4"
    if tensor.dtype == torch.float8_e4m3fn and tensor.ndim == 2:
        return "fp8"
    if tensor.dtype in (torch.bfloat16, torch.float32):
        return "quant" if tensor.ndim == 2 and min(tensor.shape) > 1024 else "bf16"
    return "copy"


def _quantize_rows(
    tensor: torch.Tensor, scale: torch.Tensor | None, kind: str, bits: int, group: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed_parts, scale_parts, bias_parts = [], [], []
    for start in range(0, tensor.shape[0], QUANT_CHUNK_ROWS):
        stop = min(start + QUANT_CHUNK_ROWS, tensor.shape[0])
        chunk = tensor[start:stop].to(device)
        chunk_scale = None
        if scale is not None:
            if scale.shape[0] == tensor.shape[0]:
                chunk_scale = scale[start:stop].to(device)
            elif scale.shape[0] * 32 == tensor.shape[0] and start % 32 == 0 and stop % 32 == 0:
                chunk_scale = scale[start // 32 : stop // 32].to(device)
            else:
                raise ValueError(f"scale rows {scale.shape[0]} do not fit {tensor.shape[0]} rows for {kind}")
        if kind == "fp4":
            if chunk_scale is None:
                raise ValueError("FP4 matrix has no .scale tensor in its source shard")
            decoded = dequant_fp4_e8m0(chunk, chunk_scale)
        elif kind == "fp8":
            if chunk_scale is None:
                raise ValueError("FP8 matrix has no .scale tensor in its source shard")
            decoded = dequant_fp8_blockwise(chunk, chunk_scale)
        else:
            decoded = chunk.float()
        q, s, b = quantize_pack(decoded, bits, group)
        packed_parts.append(q.cpu())
        scale_parts.append(s.cpu())
        bias_parts.append(b.cpu())
    return torch.cat(packed_parts), torch.cat(scale_parts), torch.cat(bias_parts)


def _convert_shard(
    source: Path, output: Path, names: list[str], bits: int, group: int, device: str, force: bool
) -> dict[str, str]:
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    converted: dict[str, torch.Tensor] = {}
    output_map: dict[str, str] = {}
    with safe_open(source, framework="pt", device="cpu") as shard:
        available = set(shard.keys())
        handled_scales: set[str] = set()
        for name in names:
            if name in handled_scales:
                continue
            if name not in available:
                raise ValueError(f"{name} listed in source index but absent from {source.name}")
            tensor = shard.get_tensor(name)
            partner = name[: -len(".weight")] + ".scale" if name.endswith(".weight") else ""
            partner_tensor = shard.get_tensor(partner) if partner and partner in available else None
            kind = _kind(tensor)
            if kind in ("fp4", "fp8", "quant"):
                packed, scales, biases = _quantize_rows(tensor, partner_tensor, kind, bits, group, device)
                base = name.removesuffix(".weight")
                if kind in ("fp4", "fp8") and partner:
                    handled_scales.add(partner)
                    converted.pop(partner, None)
                    output_map.pop(partner, None)
                for suffix, value in ((".weight", packed), (".scales", scales), (".biases", biases)):
                    key = base + suffix
                    if key in converted:
                        raise ValueError(f"duplicate output tensor {key}")
                    converted[key] = value.contiguous()
                    output_map[key] = output.name
            else:
                if name in converted:
                    raise ValueError(f"duplicate output tensor {name}")
                converted[name] = tensor.contiguous()
                output_map[name] = output.name

    fd, temporary_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        save_file(converted, str(temporary))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output_map


def convert(
    source_dir: Path,
    output_dir: Path,
    *,
    bits: int = 3,
    group: int = 64,
    device: str = "cpu",
    shards: list[int] | None = None,
    force: bool = False,
) -> Path | None:
    """Convert source shards; return the final config path only after a complete conversion."""
    if bits not in (3, 4) or group != 64:
        raise ValueError("DeepSeek-V4.1 conversion supports affine 3/4-bit groups of 64")
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    if source_dir == output_dir or source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise ValueError("output and source directories must not contain one another")
    index_path = source_dir / "model.safetensors.index.json"
    config_path = source_dir / "config.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid source weight map: {index_path}")
    shard_names = sorted(set(weight_map.values()))
    numbers: dict[str, int] = {}
    for name in shard_names:
        match = re.fullmatch(r"model-(\d{5})-of-(\d{5})\.safetensors", name)
        if match is None:
            raise ValueError(f"unexpected source shard name {name!r}")
        numbers[name] = int(match.group(1))
    selected = sorted(numbers[name] for name in shard_names) if shards is None else sorted(set(shards))
    shard_by_number = {numbers[name]: name for name in shard_names}
    missing = [n for n in selected if n not in shard_by_number]
    if missing:
        raise ValueError(f"source index has no shards numbered {missing}")
    selected_names = [shard_by_number[n] for n in selected]
    absent = [name for name in selected_names if not (source_dir / name).is_file()]
    if absent:
        raise FileNotFoundError(f"missing source shard {absent[0]}")
    if not selected:
        raise ValueError("no source shards selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    converted_map: dict[str, str] = {}
    by_shard: dict[str, list[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        by_shard.setdefault(shard_name, []).append(tensor_name)
    for shard_name in selected_names:
        output_path = output_dir / shard_name
        converted_map.update(
            _convert_shard(
                source_dir / shard_name, output_path, sorted(by_shard[shard_name]), bits, group, device, force
            )
        )
        print(f"converted {shard_name}: {len(by_shard[shard_name])} source tensors")

    if selected_names != shard_names:
        manifest = {"weight_map": converted_map, "complete": False}
        (output_dir / "converted-shards.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return None

    if not any(name.endswith(".scales") for name in converted_map):
        raise ValueError("converted shards contain no quantized matrices")
    config = json.loads(config_path.read_text())
    config["quantization"] = {"bits": bits, "group_size": group, "mode": "affine"}
    staged_index = {name: converted_map[name] for name in sorted(converted_map)}
    total_size = sum((output_dir / name).stat().st_size for name in set(staged_index.values()))
    final_index = {"metadata": {"total_size": total_size}, "weight_map": staged_index}
    index_temp = output_dir / (index_path.name + ".tmp")
    config_temp = output_dir / (config_path.name + ".tmp")
    index_temp.write_text(json.dumps(final_index, indent=2) + "\n")
    config_temp.write_text(json.dumps(config, indent=2) + "\n")
    os.replace(config_temp, output_dir / config_path.name)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "special_tokens_map.json",
        "token_map.json",
    ):
        source_file = source_dir / name
        if source_file.is_file():
            (output_dir / name).write_bytes(source_file.read_bytes())
    os.replace(index_temp, output_dir / index_path.name)
    return output_dir / "config.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="official source checkpoint directory")
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    parser.add_argument("--bits", choices=(3, 4), type=int, default=3)
    parser.add_argument("--device", default="cpu", help="torch device for dequantization (default: cpu)")
    parser.add_argument("--shards", help="comma-separated 1-based shard numbers for an incomplete probe only")
    parser.add_argument("--force", action="store_true", help="replace converted shard files in the output directory")
    args = parser.parse_args(argv)
    numbers = None if args.shards is None else [int(item) for item in args.shards.split(",") if item]
    result = convert(args.src, args.out, bits=args.bits, device=args.device, shards=numbers, force=args.force)
    if result is None:
        print("partial conversion written; no runnable checkpoint index was created")
    else:
        print(f"runnable checkpoint: {result}")


if __name__ == "__main__":
    main()
