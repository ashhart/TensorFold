"""Flash Next's checkpoint shards read a tensor at a time, and the row and group slices a rank takes."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch

_DT = {"U32": torch.int32, "I32": torch.int32, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
       "I64": torch.int64, "U8": torch.uint8, "I8": torch.int8, "U16": torch.int16, "I16": torch.int16,
       "F8_E4M3": torch.float8_e4m3fn}


class _Reader:
    """Read checkpoint shards sequentially (O_DIRECT where allowed) and release each shard's cached pages."""

    def __init__(self, model_dir: Path, device: str) -> None:
        from tensorfold.cuda.direct_read import ReadAhead, Reader

        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.where = index["weight_map"]
        self.dir = model_dir
        self.device = device
        self.headers: dict[str, tuple[int, dict]] = {}
        self.touched: set[str] = set()
        self.io = Reader()
        self.reads = ReadAhead(self.io)

    def queue(self, names) -> None:
        """Start reading ``names`` ahead (neighbours in shared reads, uploaded on a side stream) for ``get`` to take."""

        items = []
        for name in names:
            shard = self.where[name]
            base, header = self._header(shard)
            begin, end = header[name]["data_offsets"]
            items.append((name, self.dir / shard, base + begin, base + end, None))
            self.touched.add(shard)
        self.reads.queue(items, self.device)

    def drop(self, names) -> None:
        self.reads.drop(names)

    def layer_names(self, prefix: str, base: str, chosen: list[int], mtp: bool) -> list[list[str]]:
        """Each chosen layer's tensor names, then the MTP layer's; the n-gram shards stay with their memory map."""

        pattern = re.compile(re.escape(prefix) + "(" + re.escape(base) + r"layers\.\d+\.|mtp\.)")
        groups: dict[str, list[str]] = {}
        for name in self.where:
            m = pattern.match(name)
            if m and ".ngram_embedding." not in name:
                groups.setdefault(m.group(0), []).append(name)
        order = [f"{prefix}{base}layers.{i}." for i in chosen] + [f"{prefix}mtp."] * bool(mtp)
        return [groups.get(key, []) for key in order]

    def close(self) -> None:
        self.reads.close()
        self.io.close()

    def _header(self, shard: str) -> tuple[int, dict]:
        got = self.headers.get(shard)
        if got is None:
            import struct

            with open(self.dir / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                got = (8 + n, json.loads(f.read(n)))
            self.headers[shard] = got
        return got

    def get(self, name: str) -> torch.Tensor:
        shard = self.where[name]
        base, header = self._header(shard)
        entry = header[name]
        begin, end = entry["data_offsets"]
        raw = self.reads.take(name)
        if raw is None:
            raw = self.io.read(self.dir / shard, base + begin, end - begin, self.device)
        self.touched.add(shard)
        return raw.view(_DT[entry["dtype"]]).reshape(entry["shape"])

    def has(self, name: str) -> bool:
        return name in self.where

    def release(self) -> None:
        """Drop read shards' cached pages so unified memory does not retain both host-cache and GPU copies."""

        for shard in list(self.touched):
            try:
                fd = os.open(self.dir / shard, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except (OSError, AttributeError):
                pass
        self.touched.clear()


def norms_around_one(reader: _Reader, base: str, layers: list[int]) -> bool:
    """True when the norm weights are stored as scales (around 1), False when centred (around 0: add 1)."""

    means = []
    for i in layers:
        name = f"{base}layers.{i}.attn_hyper_connection.hc_norm.weight"
        if reader.has(name):
            means.append(float(reader.get(name).float().mean()))
    if not means:
        return True
    means = np.array(means)
    around_one = (means > 0.5).mean() >= 0.9 and 0.75 <= float(np.median(means)) <= 1.5
    around_zero = (means > 0.5).mean() <= 0.1 and -0.5 <= float(np.median(means)) <= 0.25
    if not (around_one or around_zero):
        raise ValueError(f"cannot tell how the norm weights are stored (median mean {np.median(means):.3f})")
    return around_one


def _rows(t3, lo: int, hi: int):
    """Output rows [lo, hi) of (words, scales, biases), stacked experts included."""

    return tuple(x[..., lo:hi, :].contiguous() for x in t3)


def _rows_at(t3, idx: torch.Tensor):
    return tuple(x.index_select(x.dim() - 2, idx).contiguous() for x in t3)


def _groups(t3, g0: int, g1: int):
    """Input groups [g0, g1) (32 inputs each) of (words, scales, biases): no repacking."""

    w, sc, b = t3
    return w[..., g0 * 4:g1 * 4].contiguous(), sc[..., g0:g1].contiguous(), b[..., g0:g1].contiguous()
