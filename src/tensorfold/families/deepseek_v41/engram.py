"""Engram n-gram tables of DeepSeek-V4.1 (layers 1 and 14): token map, bucket layout, hashes, row reads.

The token map and hashing are ported to NumPy from vLLM's ``models/deepseek_v4_1/common/engram.py`` (Apache-2.0,
Copyright contributors to the vLLM project; itself a port of DeepSeek's MIT reference): ``token_map`` and
``_next_prime`` follow its ``build_compressed_token_map`` and ``find_next_prime``; see THIRD_PARTY_NOTICES.md:
compressed token ids, one odd multiplier per (layer, lookback) from NumPy's PCG64, a rolling XOR, and 24 prime-sized
buckets per layer (3 n-gram orders x 8 heads). The tables stay in the original release's FP8 shards and are read by row.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import Config


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


@dataclass(frozen=True)
class Layout:
    """Per Engram layer: 24 bucket primes and offsets (column = (n - 2) * heads + head), and 4 multipliers."""

    primes: np.ndarray        # int64 [layers, cols]
    offsets: np.ndarray       # int64 [layers, cols]
    multipliers: np.ndarray   # int64 [layers, max_ngram]
    heads: int
    orders: int               # n-gram orders: 2 .. max_ngram

    @classmethod
    def from_config(cls, c: Config) -> Layout:
        seen: set[int] = set()
        primes = []
        for _ in c.engram_layer_ids:
            row = []
            for _ in range(c.engram_max_ngram_size - 1):
                current = c.engram_vocab_size - 1
                for _ in range(c.engram_n_heads):
                    current = _next_prime(current, seen)
                    seen.add(current)
                    row.append(current)
            primes.append(row)
        primes = np.array(primes, dtype=np.int64)
        offsets = np.concatenate([np.zeros((len(primes), 1), np.int64), np.cumsum(primes, 1)[:, :-1]], 1)
        bound = max(1, (np.iinfo(np.int64).max // c.engram_compressed_vocab_size) // 2)
        mults = np.stack([np.random.default_rng(10007 * layer).integers(0, bound, c.engram_max_ngram_size,
                                                                        dtype=np.int64) * 2 + 1
                          for layer in c.engram_layer_ids])
        for sizes, want in zip(primes, c.engram_num_embeddings):
            if int(sizes.sum()) != want:
                raise ValueError(f"Engram bucket primes sum to {int(sizes.sum())}, config says {want}")
        return cls(primes, offsets, mults, c.engram_n_heads, c.engram_max_ngram_size - 1)


def token_map(tokenizer_json: str | Path, expected: int) -> np.ndarray:
    """int64 [vocab]: each token id's compressed id (tokens that normalize alike share one)."""

    from tokenizers import Regex, Tokenizer, normalizers

    sentinel = ""
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    tok = Tokenizer.from_file(str(tokenizer_json))
    keys: dict[str, int] = {}
    n = tok.get_vocab_size(with_added_tokens=True)
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        text = tok.decode([i], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(i)
        else:
            key = norm.normalize_str(text) or text
        out[i] = keys.setdefault(key, len(keys))
    if len(keys) != expected:
        raise ValueError(f"Engram token map has {len(keys)} ids, config says {expected}")
    return out


def hashes(ids: np.ndarray, tmap: np.ndarray, layout: Layout, pad_token: int) -> np.ndarray:
    """int64 [T, layers, cols]: table rows of every position of one sequence (ids from position 0)."""

    T = len(ids)
    src = tmap[np.asarray(ids, dtype=np.int64)]
    pad = int(tmap[pad_token])
    L, cols = layout.primes.shape
    out = np.zeros((T, L, cols), dtype=np.int64)
    for ell in range(L):
        rolling = np.zeros(T, dtype=np.int64)
        blocked = np.zeros(T, dtype=bool)
        for s in range(layout.orders + 1):
            q = np.arange(T) - s
            blocked |= q < 0
            val = np.where(blocked, pad, src[np.maximum(q, 0)])
            rolling ^= val * layout.multipliers[ell, s]
            if s >= 1:
                for h in range(layout.heads):
                    col = (s - 1) * layout.heads + h
                    out[:, ell, col] = rolling % layout.primes[ell, col] + layout.offsets[ell, col]
    return out


class Tables:
    """Row reads from the original FP8 shards: ``layers.{L}.engram.embed.weight`` e4m3 and ``.scale`` UE8M0 per 32."""

    def __init__(self, directory: str | Path, layer_ids: tuple[int, ...], workers: int = 96) -> None:
        import os
        from concurrent.futures import ThreadPoolExecutor

        root = Path(directory)
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.maps = []
        self.spans = []                      # per layer: (fd, weight offset, scale offset, width, scale width)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        for layer in layer_ids:
            w_name, s_name = f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale"
            path = root / index[w_name]
            header = _header(path)
            w0, s0 = header[w_name]["data_offsets"][0], header[s_name]["data_offsets"][0]
            base = 8 + header["__len__"]
            rows, width = header[w_name]["shape"]
            weight = np.memmap(path, dtype=np.uint8, mode="r", offset=base + w0, shape=(rows, width))
            scale = np.memmap(path, dtype=np.uint8, mode="r", offset=base + s0, shape=(rows, width // 32))
            self.maps.append((weight, scale))
            fd = os.open(path, os.O_RDONLY)
            # rows are read at random (one 264-byte row a hashed n-gram): no readahead around a missed page
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            self.spans.append((fd, base + w0, base + s0, width, width // 32))

    def gather(self, idx: np.ndarray, out=None, threads: int = 64, layers: list[int] | None = None):
        """All rows of one position batch, every layer at once, with the native reader into ``out``.

        ``idx`` int64 [len(layers), n]: rows per Engram layer (``layers``: which ones, default all). Returns uint8 [layers, n, 256 + 8] (row bytes | scale bytes)
        in a pinned host tensor (``out`` when given).
        """

        import torch

        from .cuda.rowread import reader

        L, n = idx.shape
        layers = list(range(L)) if layers is None else layers
        spans = [self.spans[ell] for ell in layers]
        width, sw = spans[0][3], spans[0][4]
        row = width + sw
        if out is None:
            out = torch.empty((L, n, row), dtype=torch.uint8).pin_memory()
        fds = np.repeat([s[0] for s in spans], 2 * n).astype(np.int32)
        offs = np.empty((L, 2, n), dtype=np.int64)
        sizes = np.empty((L, 2, n), dtype=np.int32)
        dest = np.empty((L, 2, n), dtype=np.int64)
        slot = (np.arange(L)[:, None] * n + np.arange(n)[None, :]) * row
        for k, (_, w_off, s_off, _, _) in enumerate(spans):
            offs[k, 0], offs[k, 1] = w_off + idx[k] * width, s_off + idx[k] * sw
        sizes[:, 0], sizes[:, 1] = width, sw
        dest[:, 0], dest[:, 1] = slot, slot + width
        failed = reader().read_many(torch.from_numpy(fds), torch.from_numpy(offs.reshape(-1)),
                                    torch.from_numpy(sizes.reshape(-1)), torch.from_numpy(dest.reshape(-1)), out,
                                    threads)
        if failed:
            raise OSError(f"Engram: {failed} row reads failed")
        return out

    def raw(self, requests: list[tuple[int, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
        """For each (layer, row ids) the stored bytes (uint8 [n, 256], [n, 8]): every row read concurrently."""

        import os

        jobs = []
        for ell, idx in requests:
            fd, w_off, s_off, width, sw = self.spans[ell]
            flat = np.asarray(idx, dtype=np.int64).reshape(-1)
            jobs.append((flat, [self.pool.submit(os.pread, fd, width, w_off + int(r) * width) for r in flat],
                         [self.pool.submit(os.pread, fd, sw, s_off + int(r) * sw) for r in flat]))
        out = []
        for flat, wf, sf in jobs:
            w = np.frombuffer(b"".join(f.result() for f in wf), dtype=np.uint8).reshape(len(flat), -1)
            s = np.frombuffer(b"".join(f.result() for f in sf), dtype=np.uint8).reshape(len(flat), -1)
            out.append((w, s))
        return out

    def rows(self, ell: int, idx: np.ndarray):
        """float32 torch [*, 256] of the rows ``idx`` of Engram layer ``ell``, dequantized (fp8 * 2^(e - 127))."""

        import torch

        weight, scale = self.maps[ell]
        flat = idx.reshape(-1)
        order = np.argsort(flat)
        w = np.empty((len(flat), weight.shape[1]), np.uint8)
        s = np.empty((len(flat), scale.shape[1]), np.uint8)
        w[order] = weight[flat[order]]
        s[order] = scale[flat[order]]
        vals = torch.from_numpy(w).view(torch.float8_e4m3fn).float()
        exps = torch.from_numpy(s.astype(np.int32) << 23).view(torch.float32)
        out = vals.view(len(flat), -1, 32) * exps[:, :, None]
        return out.view(*idx.shape, weight.shape[1])


@lru_cache(maxsize=8)
def _header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header["__len__"] = n
    return header


def dequant(w, s):
    """Stored e4m3 rows and UE8M0 scales (torch uint8 [..., 256], [..., 8], any device) -> fp32 [..., 256]."""

    import torch

    vals = w.view(torch.float8_e4m3fn).float()
    exps = (s.to(torch.int32) << 23).view(torch.float32)
    return (vals.view(*w.shape[:-1], -1, 32) * exps[..., None]).view(*w.shape)
