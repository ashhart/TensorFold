"""Engram n-gram hash lookups gated into the residual streams, after oMLX's ``engram.py`` (MIT)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.families.deepseek_v41.quant import Linear


def build_compressed_token_map(tokenizer: Any) -> tuple[list[int], int]:
    """Every token id onto the compressed id space DeepSeek's training hashed over, and that space's size."""

    from tokenizers import Regex, normalizers

    sentinel = ""           # keeps a token that is exactly one space from collapsing to the empty string
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    backend = tokenizer.backend_tokenizer
    keys: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)        # a partial UTF-8 byte token: keyed by its raw form
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        lookup[token_id] = keys.setdefault(key, len(keys))
    return lookup, len(keys)


def _prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


class NgramHash:
    """Row ids of every Engram lookup for a stream's tokens, given the compressed ids before them."""

    def __init__(self, cfg: Any, token_map: Any) -> None:
        self.token_map = np.asarray(token_map, dtype=np.int64)
        vocab = int(self.token_map.max()) + 1
        if vocab != cfg.engram_compressed_vocab_size:
            raise ValueError(f"deepseek_v41: the tokenizer's Engram vocabulary is {vocab} ids, the checkpoint's "
                             f"{cfg.engram_compressed_vocab_size}")
        self.depth = cfg.engram_max_ngram_size - 1
        self.pad = int(self.token_map[cfg.engram_pad_token_id])
        primes, multipliers, seen = [], [], set()
        for layer in cfg.engram_layer_ids:
            groups = []
            for _ in range(self.depth):
                current, group = cfg.engram_vocab_size - 1, []
                for _ in range(cfg.engram_n_heads):
                    current += 1
                    while not _prime(current) or current in seen:
                        current += 1
                    seen.add(current)
                    group.append(current)
                groups.append(group)
            primes.append(groups)
            rng = np.random.default_rng(10007 * layer)
            bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
            multipliers.append(rng.integers(0, bound, cfg.engram_max_ngram_size, dtype=np.int64) * 2 + 1)
        self.primes = np.array(primes, dtype=np.int64)                    # [tables, depth, heads]
        self.multipliers = np.array(multipliers, dtype=np.int64)          # [tables, max_ngram]
        flat = self.primes.reshape(len(primes), -1)
        self.offsets = np.cumsum(np.concatenate([np.zeros((len(primes), 1), dtype=np.int64), flat[:, :-1]], -1), -1)
        if [int(v) for v in flat.sum(-1)] != [int(v) for v in cfg.engram_num_embeddings]:
            raise ValueError("deepseek_v41: Engram table rows do not match the official prime layout")

    def compress(self, ids: Any) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= len(self.token_map)):
            raise ValueError("deepseek_v41: a token id outside Engram's tokenizer vocabulary")
        return self.token_map[ids]

    def __call__(self, ids: Any, history: np.ndarray) -> np.ndarray:
        """Row ids [L, tables, depth * heads] for tokens ``ids`` [L] after compressed ``history`` [depth] (-1: none)."""

        tokens = self.compress(ids).reshape(-1)
        joined = np.concatenate([np.asarray(history, dtype=np.int64).reshape(-1), tokens])
        positions = np.arange(tokens.shape[0]) + self.depth
        blocked = np.zeros(tokens.shape, dtype=bool)
        lookback = []
        for shift in range(self.depth + 1):
            source = joined[positions - shift]
            blocked |= source == -1
            lookback.append(np.where(blocked, self.pad, source))
        product = np.stack(lookback, -1)[:, None, :] * self.multipliers    # [L, tables, max_ngram]
        rolling, hashes = product[..., 0], []
        for shift in range(1, self.depth + 1):
            rolling = np.bitwise_xor(rolling, product[..., shift])
            hashes.append(rolling[..., None] % self.primes[:, shift - 1])
        return np.concatenate(hashes, -1) + self.offsets

    def history(self, tokens: np.ndarray, start: int) -> np.ndarray:
        """The compressed ids of positions start - depth .. start - 1 from the token ids there (-1 before 0)."""

        out = np.full((self.depth,), -1, dtype=np.int64)
        have = min(self.depth, start)
        if have:
            out[self.depth - have:] = self.compress(np.asarray(tokens)[-have:])
        return out


_NP = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U8": np.uint8}


PAGE = 16384
PREFETCH_ROWS = 128          # a gather of more rows reads its pages concurrently first (oMLX's storage.py pattern)
IO_WORKERS = 48
_IO_POOL: Any = None


class TensorMap:
    """One tensor of a safetensors file, memory-mapped, whose big gathers read their cold pages on worker threads."""

    def __init__(self, path: Path, key: str) -> None:
        with open(path, "rb") as f:
            size = int(np.frombuffer(f.read(8), dtype="<u8")[0])
            header = json.loads(f.read(size))
        entry = header[key]
        start, end = entry["data_offsets"]
        dtype = np.dtype(_NP[entry["dtype"]]).newbyteorder("<")
        shape = tuple(int(s) for s in entry["shape"])
        if (end - start) != int(np.prod(shape)) * dtype.itemsize:
            raise ValueError(f"{path.name}: {key}'s bytes do not match its shape")
        self.base = 8 + size + start
        self.array = np.memmap(path, dtype=dtype, mode="r", offset=self.base, shape=shape)
        self.path = Path(path)
        self.row_bytes = (end - start) // shape[0] if shape[0] else 0
        self.seen = np.zeros((self.base + (end - start)) // PAGE + 2, dtype=np.uint8)
        self.shape, self.nbytes = shape, end - start

    def rows(self, rows: np.ndarray) -> np.ndarray:
        if rows.size > PREFETCH_ROWS and 0 < self.row_bytes <= PAGE:
            self._prefetch(rows)
        return np.ascontiguousarray(self.array[rows])

    def _prefetch(self, rows: np.ndarray) -> None:
        import os
        from concurrent.futures import ThreadPoolExecutor

        global _IO_POOL
        offsets = self.base + rows.astype(np.int64) * self.row_bytes
        pages = np.unique(np.concatenate([offsets // PAGE, (offsets + self.row_bytes - 1) // PAGE]))
        fresh = pages[self.seen[pages] == 0]
        if not fresh.size:
            return
        if _IO_POOL is None:
            _IO_POOL = ThreadPoolExecutor(max_workers=IO_WORKERS, thread_name_prefix="dsv41-engram-io")
        fd = os.open(self.path, os.O_RDONLY)

        def touch(group: np.ndarray) -> None:
            for page in group:
                os.pread(fd, PAGE, int(page) * PAGE)

        try:
            list(_IO_POOL.map(touch, np.array_split(fresh, min(IO_WORKERS, fresh.size))))
        finally:
            os.close(fd)
        self.seen[fresh] = 1


def tensor_map(path: Path, key: str) -> TensorMap:
    return TensorMap(path, key)


class EngramTable:
    """An Engram table in MLX's affine layout, read row by row from its file (only the rows a step looks up)."""

    def __init__(self, folder: Path, spec: dict[str, Any]) -> None:
        if str(spec.get("mode", "affine")) != "affine" or spec.get("bias_key") is None:
            raise ValueError(f"deepseek_v41: Engram table {spec.get('weight_key')} is not affine-quantized")
        weight_file = Path(folder) / spec["weight_file"]
        scale_file = Path(folder) / (spec.get("scale_file") or spec["weight_file"])
        self.weight = tensor_map(weight_file, spec["weight_key"])
        self.scales = tensor_map(scale_file, spec["scale_key"])
        self.biases = tensor_map(scale_file, spec["bias_key"])
        self.bits, self.group = int(spec["bits"]), int(spec.get("group_size", 32))
        self.rows = int(self.weight.shape[0])
        self.dim = int(self.weight.shape[1]) * 32 // self.bits

    def nbytes(self) -> int:
        return int(self.weight.nbytes + self.scales.nbytes + self.biases.nbytes)

    def __call__(self, rows: np.ndarray) -> mx.array:
        """The rows' dequantized values [*rows.shape, dim] in bf16."""

        flat = np.asarray(rows, dtype=np.int64).reshape(-1)
        if flat.size and (flat.min() < 0 or flat.max() >= self.rows):
            raise IndexError("deepseek_v41: an Engram row outside its table")
        w = mx.array(self.weight.rows(flat))
        s = mx.array(self.scales.rows(flat)).view(mx.bfloat16)
        b = mx.array(self.biases.rows(flat)).view(mx.bfloat16)
        values = mx.dequantize(w, s, b, group_size=self.group, bits=self.bits, mode="affine")
        return values.reshape(*rows.shape, self.dim).astype(mx.bfloat16)


class Engram:
    """One Engram layer: lookups -> wkv -> a key per stream and one value; each stream adds gate * value."""

    def __init__(self, table: EngramTable, wkv: Linear, q_weight: mx.array, k_weight: mx.array, dim: int,
                 hc: int, eps: float) -> None:
        self.table, self.wkv = table, wkv
        self.q_weight, self.k_weight = q_weight, k_weight
        self.dim, self.hc, self.eps = int(dim), int(hc), float(eps)

    def arrays(self) -> list[mx.array]:
        return [*self.wkv.arrays(), self.q_weight, self.k_weight]

    def __call__(self, h: mx.array, rows: np.ndarray) -> mx.array:
        """Streams h [R, 4, D] and their rows' lookups [R, lookups]: the new streams (oMLX's arithmetic order)."""

        R = int(h.shape[0])
        kv = self.wkv(self.table(rows).reshape(R, -1))
        key, value = kv[:, :self.dim * self.hc], kv[:, self.dim * self.hc:]
        key = key.astype(mx.float32).reshape(R, self.hc, self.dim)
        x = h.astype(mx.float32)
        inv = (mx.rsqrt(mx.mean(x * x, -1) + self.eps) * mx.rsqrt(mx.mean(key * key, -1) + self.eps))
        dot = mx.sum(x * self.q_weight * self.k_weight * key, -1) * inv * self.dim ** -0.5
        gate = mx.sigmoid(mx.sign(dot) * mx.sqrt(mx.maximum(mx.abs(dot), 1e-6)))
        gate = mx.where(dot == 0, mx.sigmoid(mx.array(0.001)), gate)      # copysign(+sqrt, 0) is positive
        return (x + gate[..., None] * value.astype(mx.float32)[:, None, :]).astype(h.dtype)
