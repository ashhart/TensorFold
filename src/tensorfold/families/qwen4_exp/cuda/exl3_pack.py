"""An EXL3 Flash Next pack: its tensors by name (index shards plus the files beside them), its admission, and its n-gram row codec."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import torch

from tensorfold.cuda.exl3.format import is_exl3  # noqa: F401  (exl3.py and engine.py import it from here)

EXTRA_FILES = ("ngram_embedding.safetensors", "mtp_hyper_connection_mixer_patch.safetensors")
MOE_WINDOW = 1024        # most rows a routed-expert call takes (its grouping keeps every pick in 48 KB of shared memory)
_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64, "I32": torch.int32,
       "I16": torch.int16, "U8": torch.uint8, "I8": torch.int8, "U16": torch.int16, "U32": torch.int32}


def extra_files(model_dir: str | Path) -> tuple[Path, ...]:
    """The pack's files outside the index that the loader reads (n-gram rows, the MTP head's mixer)."""

    return tuple(Path(model_dir) / f for f in EXTRA_FILES if (Path(model_dir) / f).is_file())


def admission(geometry):
    """The engine's geometry plus the EXL3 path's fixed scratch: routed-expert windows and prompt rows."""

    from tensorfold.cuda.geometry import PREFILL_ROWS, exl3_indexed_scratch, with_fixed

    return lambda text: with_fixed(geometry(text), exl3_indexed_scratch(text, MOE_WINDOW, PREFILL_ROWS))


class Pack:
    """Tensors by name from the index's shards and the extra files, read with large sequential reads; ``release`` drops read pages."""

    def __init__(self, model_dir: str | Path) -> None:
        self.dir = Path(model_dir)
        self.where: dict[str, str] = dict(json.loads((self.dir / "model.safetensors.index.json").read_text())
                                          ["weight_map"])
        self.headers: dict[str, tuple[int, dict]] = {}
        for extra in EXTRA_FILES:
            if (self.dir / extra).is_file():
                _, header = self.header(extra)
                for name in header:
                    if name != "__metadata__":
                        self.where.setdefault(name, extra)
        self.touched: set[str] = set()

    def header(self, file: str) -> tuple[int, dict]:
        got = self.headers.get(file)
        if got is None:
            with open(self.dir / file, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                got = (8 + n, json.loads(f.read(n)))
            self.headers[file] = got
        return got

    def has(self, name: str) -> bool:
        return name in self.where

    def entry(self, name: str) -> tuple[str, int, int, str, list[int]]:
        """(file, absolute begin, absolute end, dtype, shape)."""

        file = self.where[name]
        base, header = self.header(file)
        e = header[name]
        begin, end = e["data_offsets"]
        return file, base + begin, base + end, e["dtype"], list(e["shape"])

    def read(self, file: str, begin: int, end: int) -> torch.Tensor:
        raw = torch.empty((end - begin,), dtype=torch.uint8)
        view = memoryview(raw.numpy())
        with open(self.dir / file, "rb", buffering=0) as f:
            f.seek(begin)
            at = 0
            while at < len(view):
                got = f.readinto(view[at:at + (64 << 20)])
                if not got:
                    raise IOError(f"short read of {file} at {begin + at}")
                at += got
        self.touched.add(file)
        return raw

    def get(self, name: str) -> torch.Tensor:
        file, begin, end, dtype, shape = self.entry(name)
        return self.read(file, begin, end).view(_DT[dtype]).reshape(shape)

    def release(self) -> None:
        for file in list(self.touched):
            try:
                fd = os.open(self.dir / file, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except (OSError, AttributeError):
                pass
        self.touched.clear()

    def codebook(self, prefix: str) -> str:
        return "mul1" if self.has(prefix + ".mul1") else "mcg" if self.has(prefix + ".mcg") else "3inst"

    def scales(self, prefix: str, packed: str, expanded: str) -> torch.Tensor:
        """A group's fp16 scales: ``.suh``/``.svh`` as stored, or the packed sign words ``.su``/``.sv`` expanded."""

        from tensorfold.cuda.exl3 import format as fmt

        if self.has(f"{prefix}.{expanded}"):
            return self.get(f"{prefix}.{expanded}")
        return torch.from_numpy(fmt.unpack_signs(self.get(f"{prefix}.{packed}").numpy()))


class NgramTable:
    """The n-gram embedding's shards in ExLlamaV3's row codec, memory-mapped; ``words``/``lock``/``prefetch`` as ``HostTable``."""

    def __init__(self, pk: Pack, base: str, shards: int, device) -> None:
        starts, offsets, fidx, files, words = [0], [], [], [], None
        maps: dict[str, int] = {}
        self.words: list[np.ndarray] = []
        self.scales: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(shards):
            file, begin, end, dtype, shape = pk.entry(f"{base}shard_{i}.trellis")
            if dtype != "I16" or len(shape) != 2:
                raise ValueError(f"n-gram shard {i}: expected int16 [rows, words], got {dtype} {shape}")
            if words not in (None, shape[1]):
                raise ValueError("n-gram shards of different widths")
            words = shape[1]
            if file not in maps:
                maps[file] = len(files)
                files.append(file)
            fidx.append(maps[file])
            offsets.append(begin)
            starts.append(starts[-1] + shape[0])
            self.words.append(np.memmap(pk.dir / file, dtype=np.int16, mode="r", offset=begin, shape=tuple(shape)))
        self.words_per_row = int(words)
        self.dh = 160
        self.bits = (self.words_per_row - 1) * 16 // self.dh
        if 1 + self.dh * self.bits // 16 != self.words_per_row:
            raise ValueError(f"n-gram rows of {self.words_per_row} words are not one scale plus 160 values")
        self.maps = [np.memmap(pk.dir / f, dtype=np.uint8, mode="r") for f in files]
        self.fidx = np.array(fidx, dtype=np.int64)
        self.offsets = np.array(offsets, dtype=np.int64)
        self.starts = np.array(starts, dtype=np.int64)
        self.rows = int(self.starts[-1])
        self.row_bytes = 2 * self.words_per_row
        self.head_bias = pk.get(base + "head_bias").to(torch.float16).to(device).contiguous()
        self.head_offsets = pk.get(base + "head_offsets").cpu().numpy()
        self.head_sizes = pk.get(base + "head_vocab_sizes").cpu().numpy()
        self.multipliers = pk.get(base + "layer_multipliers").cpu().numpy()

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """Rows ``ids`` (global) -> int16 [n, words]."""

        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        at = self.offsets[shard] + (flat - self.starts[shard]) * self.row_bytes
        where = self.fidx[shard]
        out = np.empty((len(flat), self.row_bytes), dtype=np.uint8)
        cols = np.arange(self.row_bytes)
        for f in np.unique(where):
            sel = np.nonzero(where == f)[0]
            out[sel] = self.maps[f][at[sel, None] + cols]
        return out.view(np.int16)

    def lock(self) -> bool:
        from ..host_table import HostTable

        return HostTable.lock(self)

    def prefetch(self, workers: int = 8) -> float:
        from ..host_table import HostTable

        return HostTable.prefetch(self, workers)


def stage_ple(table: NgramTable, sc, ids: np.ndarray, at: int = 0) -> None:
    """Copy the rows' packed n-gram entries into the shared staging buffers from row ``at``."""

    rows = table.gather(ids)
    n = rows.shape[0]
    sc.ple_host[at:at + n].numpy()[:] = rows
    sc.ple_dev[at:at + n].copy_(sc.ple_host[at:at + n], non_blocking=True)
