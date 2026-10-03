"""Bounded uploads from GGUF spans, preserving quantized bytes and GGML dimension order."""

from __future__ import annotations

import mmap
from pathlib import Path

import numpy as np
import torch

from tensorfold.gguf import parse_gguf_tensors

from .linear import FORMATS, Packed


class Weights:
    """One GGUF mapping. Loading is explicit, cached, and independent of model architecture.

    Chunked copies bound host staging. Closing the mapping leaves uploaded tensors valid.
    No checkpoint payload is downloaded, rewritten, or expanded by this reader.
    """

    def __init__(self, path: str | Path, device: str | torch.device = "cuda", chunk_bytes: int = 32 << 20):
        if chunk_bytes <= 0:
            raise ValueError("GGUF upload chunk must be positive")
        self.path = Path(path)
        self.device = torch.device(device)
        self.chunk_bytes = chunk_bytes
        self._file = self.path.open("rb")
        try:
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            inventory = parse_gguf_tensors(self._map)
        except BaseException:
            if hasattr(self, "_map"):
                self._map.close()
            self._file.close()
            raise
        self.metadata = inventory.info.metadata
        self.inventory = {t.name: t for t in inventory.tensors}
        self._loaded: dict[str, Packed | torch.Tensor] = {}
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if not self._closed:
            self._map.close()
            self._file.close()
            self._closed = True

    def tensor(self, name: str) -> Packed | torch.Tensor:
        if name in self._loaded:
            return self._loaded[name]
        if self._closed:
            raise RuntimeError("GGUF mapping is closed")
        info = self.inventory[name]
        if info.type_name not in {"F32", "F16", "I32", *FORMATS}:
            raise ValueError(f"unsupported GGUF CUDA format {info.type_name}: {name}")
        raw = torch.empty(info.size, dtype=torch.uint8, device=self.device)
        for offset in range(0, info.size, self.chunk_bytes):
            size = min(self.chunk_bytes, info.size - offset)
            # Owned, writable staging: no Torch view survives a closed mmap.
            stage = np.frombuffer(self._map, dtype=np.uint8, count=size, offset=info.start + offset).copy()
            raw[offset : offset + size].copy_(torch.from_numpy(stage))
        if info.type_name in ("F32", "F16", "I32"):
            dtype = {"F32": torch.float32, "F16": torch.float16, "I32": torch.int32}[info.type_name]
            value = raw.view(dtype).reshape(tuple(reversed(info.shape)))
        else:
            value = Packed(raw, info.shape, info.type_name)
        self._loaded[name] = value
        return value

    def clear(self):
        """Drop the loader's references; callers retain their tensors."""
        self._loaded.clear()
