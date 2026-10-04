"""Family-agnostic lossless prepare for GGUF expert weights (SoA / tile layouts)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch

from .linear import FORMATS, Packed

LAYOUT_VERSION = 3  # IQ2 SoA + Q2_K SoA (dm/sc/qs split)
PREPARE_FORMATS = ("IQ2_XXS", "Q2_K")
DEFAULT_BN = 64


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(
        name="tensorfold_gguf_pack",
        sources=[
            str(here / "experts.cpp"),
            str(here / "experts_pack.cu"),
            str(here / "experts_iq2_prefill.cu"),
            str(here / "experts_q2_prefill.cu"),
            str(here / "experts_q8_act.cu"),
        ],
        extra_cuda_cflags=["-O3", "--fmad=false", f"-I{here.parent}"],
        verbose=False,
    )


def prepare_capable(fmt: str, shape: tuple[int, ...]) -> bool:
    return fmt in PREPARE_FORMATS and len(shape) == 3 and shape[0] % 256 == 0


def iq2_soa_bytes(nblk: int) -> tuple[int, int]:
    """Return (dq_bytes padded to 64, total payload bytes)."""

    dq = ((nblk * 2 + 63) // 64) * 64
    return dq, dq + nblk * 64


def q2_soa_bytes(nblk: int) -> tuple[int, int, int]:
    """Return (dm_bytes, sc_bytes, total) with 64B pads between sections."""

    dm = ((nblk * 4 + 63) // 64) * 64
    sc = ((nblk * 16 + 63) // 64) * 64
    return dm, sc, dm + sc + nblk * 64


def pack_iq2_soa(raw: torch.Tensor, shape: tuple[int, ...]) -> tuple[torch.Tensor, int]:
    k, n, e = shape
    nblk = e * n * (k // 256)
    dq_bytes, total = iq2_soa_bytes(nblk)
    out = torch.empty(total, dtype=torch.uint8, device=raw.device)
    _ext().pack_iq2_soa(raw.contiguous(), out, nblk, dq_bytes)
    return out, dq_bytes


def pack_q2_soa(raw: torch.Tensor, shape: tuple[int, ...]) -> tuple[torch.Tensor, int, int]:
    k, n, e = shape
    nblk = e * n * (k // 256)
    dm_bytes, sc_bytes, total = q2_soa_bytes(nblk)
    out = torch.empty(total, dtype=torch.uint8, device=raw.device)
    _ext().pack_q2_soa(raw.contiguous(), out, nblk, dm_bytes, sc_bytes)
    return out, dm_bytes, sc_bytes


def pack_tiles(raw: torch.Tensor, shape: tuple[int, ...], fmt: str, *, bn: int = DEFAULT_BN) -> torch.Tensor:
    if not prepare_capable(fmt, shape):
        raise ValueError(f"prepare does not support {fmt} {shape}")
    k, n, e = shape
    block = FORMATS[fmt][1]
    nb = (n + bn - 1) // bn
    kg = k // 256
    out = torch.empty(e * nb * kg * bn * block, dtype=torch.uint8, device=raw.device)
    _ext().pack(raw.contiguous(), out, k, n, e, block, bn)
    return out


def prepare_packed(weight: Packed, *, bn: int = DEFAULT_BN) -> Packed:
    """Replace raw GGUF bytes with the prepare layout for this format."""

    if weight.layout != "raw" or not prepare_capable(weight.format, weight.shape):
        return weight
    if weight.format == "IQ2_XXS":
        data, dq_bytes = pack_iq2_soa(weight.data, weight.shape)
        return Packed(data, weight.shape, weight.format, layout="soa", bn=dq_bytes)
    if weight.format == "Q2_K":
        data, dm_bytes, sc_bytes = pack_q2_soa(weight.data, weight.shape)
        # Encode sc_bytes in the high 32 bits of bn for the consumer.
        return Packed(data, weight.shape, weight.format, layout="soa", bn=dm_bytes | (sc_bytes << 32))
    return weight


def prepare_file(gguf_path: str | Path, out_dir: str | Path, *, bn: int = DEFAULT_BN) -> dict:
    """Offline cache: prepare every 3D IQ2/Q2 tensor from a GGUF into ``out_dir``."""

    from .weights import Weights

    gguf_path = Path(gguf_path).resolve(strict=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    with Weights(gguf_path, prepare_experts=False) as weights:
        for name, info in weights.inventory.items():
            if info.type_name not in PREPARE_FORMATS or len(info.shape) != 3:
                continue
            packed = weights.tensor(name)
            if not isinstance(packed, Packed):
                continue
            prepared = prepare_packed(packed, bn=bn) if packed.layout == "raw" else packed
            if prepared.layout == "raw":
                continue
            torch.save(
                {
                    "data": prepared.data.cpu(),
                    "shape": prepared.shape,
                    "format": prepared.format,
                    "layout": prepared.layout,
                    "bn": prepared.bn,
                },
                out_dir / f"{name.replace('/', '__')}.pt",
            )
            names.append(name)
            del prepared, packed
            weights._loaded.pop(name, None)
            torch.cuda.empty_cache()
    meta = {
        "layout_version": LAYOUT_VERSION,
        "bn": bn,
        "gguf": str(gguf_path),
        "gguf_size": gguf_path.stat().st_size,
        "gguf_mtime_ns": gguf_path.stat().st_mtime_ns,
        "tensors": names,
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_prepared(
    prepared_dir: str | Path,
    name: str,
    device: torch.device,
    *,
    gguf_path: str | Path | None = None,
) -> Packed | None:
    prepared_dir = Path(prepared_dir)
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("layout_version") != LAYOUT_VERSION or name not in manifest.get("tensors", []):
        return None
    if gguf_path is not None:
        src = Path(gguf_path)
        try:
            st = src.stat()
        except OSError:
            return None
        if manifest.get("gguf_size") != st.st_size or manifest.get("gguf_mtime_ns") != st.st_mtime_ns:
            return None
        if Path(manifest.get("gguf", "")).resolve() != src.resolve():
            return None
    path = prepared_dir / f"{name.replace('/', '__')}.pt"
    if not path.is_file():
        return None
    blob = torch.load(path, map_location=device, weights_only=True)
    layout = blob.get("layout", "tile")
    return Packed(
        blob["data"].to(device=device, dtype=torch.uint8),
        tuple(blob["shape"]),
        blob["format"],
        layout=layout,
        bn=int(blob["bn"]),
    )
