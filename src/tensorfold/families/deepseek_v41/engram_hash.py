"""Engram's n-gram row ids from a token history, as DeepSeek's engram.py hashes them (numpy int64, exact)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isqrt
from pathlib import Path

import numpy as np

from .config import Config

# the checkpoint's compressed vocabulary: its size and the sha256 of the map as little-endian int32
TOKEN_MAP_SIZE = 99092
TOKEN_MAP_SHA256 = "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481"
# numpy default_rng(10007 * layer) multipliers for (compressed vocabulary, max n-gram) = (99092, 4), kept as
# constants because numpy does not promise its generator's stream across versions
MULTIPLIERS = {(TOKEN_MAP_SIZE, 4): {1: (76632096046245, 4839876093313, 35959672319349, 73987337458391),
                                     14: (67716810739261, 51510806800915, 30921347202721, 82619226485591)}}


@dataclass(frozen=True)
class TokenMap:
    """Token id -> compressed id, tokens that normalize alike sharing one; ``size`` counts the compressed ids."""

    table: np.ndarray           # int32 [vocabulary]
    size: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.table.astype("<i4").tobytes()).hexdigest()

    @classmethod
    def build(cls, tokenizer_json: str | Path, expect_size: int, expect_sha: str | None = None) -> TokenMap:
        """The map from tokenizer.json, refusing a size or digest other than the expected ones."""

        from tokenizers import Regex, Tokenizer, normalizers

        sentinel = ""     # keeps a lone space from stripping to the empty key
        normalizer = normalizers.Sequence([
            normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(), normalizers.Replace(sentinel, " ")])
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        keys: dict[str, int] = {}
        table = np.empty(tokenizer.get_vocab_size(with_added_tokens=True), dtype=np.int32)
        for token in range(len(table)):
            text = tokenizer.decode([token], skip_special_tokens=False)
            if "�" in text:            # part of a UTF-8 sequence: keyed by its raw token
                key = tokenizer.id_to_token(token)
            else:
                key = normalizer.normalize_str(text) or text
            table[token] = keys.setdefault(key, len(keys))
        out = cls(table, len(keys))
        if out.size != expect_size:
            raise ValueError(f"the tokenizer's Engram token map has {out.size} classes, the checkpoint was trained "
                             f"with {expect_size}: its n-gram ids would differ")
        if expect_sha is not None and out.sha256 != expect_sha:
            raise ValueError(f"the tokenizer's Engram token map has sha256 {out.sha256}, not {expect_sha}: this "
                             f"tokenizers version normalizes differently from the checkpoint's")
        return out


def _is_prime(n: int) -> bool:
    if n < 2 or n % 2 == 0:
        return n == 2
    return bool(np.all(n % np.arange(3, isqrt(n) + 1, 2, dtype=np.int64)))


def primes(layer_ids: tuple[int, ...], vocab_size: int, n_heads: int, max_ngram: int) -> np.ndarray:
    """int64 [layers, max_ngram - 1, n_heads]: per (n, head) the next unused prime above ``vocab_size - 1``."""

    seen: set[int] = set()
    out = np.empty((len(layer_ids), max_ngram - 1, n_heads), dtype=np.int64)
    for layer in range(len(layer_ids)):
        for n in range(max_ngram - 1):
            current = vocab_size - 1
            for head in range(n_heads):
                current += 1
                while current in seen or not _is_prime(current):
                    current += 1
                seen.add(current)
                out[layer, n, head] = current
    return out


def multipliers(layer_ids: tuple[int, ...], max_ngram: int, compressed: int) -> np.ndarray:
    """int64 [layers, max_ngram]: odd and below int64's max over ``compressed``, so ``id * m`` cannot overflow."""

    known = MULTIPLIERS.get((compressed, max_ngram), {})
    bound = max(1, np.iinfo(np.int64).max // compressed // 2)
    rows = [known[layer] if layer in known else
            np.random.default_rng(10007 * layer).integers(0, bound, size=(max_ngram,), dtype=np.int64) * 2 + 1
            for layer in layer_ids]
    return np.array(rows, dtype=np.int64).reshape(len(layer_ids), max_ngram)


class Hasher:
    """Row ids of the n-grams ending at each position, per Engram layer: column ``(n - 2) * heads + head``."""

    def __init__(self, cfg: Config, token_map: TokenMap) -> None:
        if token_map.size != cfg.engram_compressed_vocab_size:
            raise ValueError(f"the Engram token map has {token_map.size} classes, config.json says "
                             f"{cfg.engram_compressed_vocab_size}")
        layers = tuple(cfg.engram_layer_ids)
        self.map = token_map.table.astype(np.int64)
        self.pad = int(self.map[cfg.engram_pad_token_id])
        self.max_ngram = cfg.engram_max_ngram_size
        found = primes(layers, cfg.engram_vocab_size, cfg.engram_n_heads, self.max_ngram)
        sums = tuple(int(s) for s in found.sum(axis=(1, 2)))
        if sums != tuple(cfg.engram_num_embeddings):
            raise ValueError(f"the Engram primes give tables of {sums} rows, config.json says "
                             f"{tuple(cfg.engram_num_embeddings)}")
        self.primes = found.reshape(len(layers), -1)                        # [layers, columns]
        self.offsets = np.cumsum(self.primes, axis=1) - self.primes
        self.multipliers = multipliers(layers, self.max_ngram, token_map.size)

    def ids(self, history: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """int64 [n, layers, columns] for ``tokens``; ``history`` holds the min(position, max_ngram - 1) ids before."""

        back = self.max_ngram - 1
        history = np.asarray(history, dtype=np.int64)[-back:]
        seq = self.map[np.concatenate([history, np.asarray(tokens, dtype=np.int64)])]
        at = np.arange(len(history), len(seq))
        # the id s back, or the pad once the n-gram reaches past the sequence's start
        shifted = np.stack([np.where(at >= s, seq[np.maximum(at - s, 0)], self.pad) for s in range(self.max_ngram)],
                           axis=-1)                                      # [n, max_ngram]
        products = shifted[:, None, :] * self.multipliers[None]          # [n, layers, max_ngram]
        rolling = np.bitwise_xor.accumulate(products, axis=-1)[..., 1:]  # [n, layers, max_ngram - 1]
        rolled = np.repeat(rolling, self.primes.shape[1] // back, axis=-1)  # [n, layers, columns]
        return rolled % self.primes[None] + self.offsets[None]


def rank_columns(ids: np.ndarray, rank: int, world: int = 2) -> np.ndarray:
    """The columns rank ``rank`` of ``world`` reads: a contiguous 1/world of each layer's."""

    width = ids.shape[-1] // world
    return ids[..., rank * width:(rank + 1) * width]
