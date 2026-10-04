"""What the two ranks say to each other: startup settings, the doorbell, request headers, rank 0's ints."""

from __future__ import annotations

import struct
from datetime import timedelta
from typing import NamedTuple

from tensorfold.engine.exact_sampling import Sampling

BELL = "tf_dsv41_request_{}"
STOP = [0]                  # a header no request sends (max_tokens is at least 1): rank 1 leaves ``follow``
KV_STORAGE = 0              # caches hold the model's QDQ'd values as bf16
ENGRAM_SPLIT = 0            # each rank reads a contiguous half of every Engram table's hash columns
SETTINGS = ("start_error", "dspark_loaded", "capacity", "prefill_rows", "max_rows", "ring", "drafts",
            "confidence_ppm", "layers", "world", "kv_storage", "engram_split", "engram_digest_hi", "engram_digest_lo")
SPARE = ("cache_mib", "cache_entries")      # each rank's room for kept snapshots: both use the smaller
TOP_K_MAX = 2**31 - 1       # a header's largest top_k; any top_k past the vocabulary keeps every token


class Request(NamedTuple):
    max_tokens: int
    stop_eos: bool
    draft: bool                             # resume and keep snapshots; draft unless ``policy`` drafts 0
    cached: int                             # prompt tokens resumed from a kept snapshot
    sampling: Sampling | None               # None: greedy
    policy: tuple[int, float | None]        # (drafts a round, confidence the chain stops below)


def _f64(x: float) -> list[int]:
    return list(struct.unpack("<2i", struct.pack("<d", float(x))))


def _from_f64(lo: int, hi: int) -> float:
    return struct.unpack("<d", struct.pack("<2i", lo, hi))[0]


def encode(max_tokens: int, stop_eos: bool, draft: bool, cached: int, sampling: Sampling | None,
           policy: tuple[int, float | None]) -> list[int]:
    """A request as 16 int32 values; floats travel as an f64's two halves, the seed as 31+31+2 bits."""

    if max_tokens < 1:
        raise ValueError(f"max_tokens {max_tokens}: a request decodes at least one token")
    seed = (sampling.seed if sampling else 0) & 0xFFFFFFFFFFFFFFFF
    drafts, confidence = policy
    return [int(max_tokens), int(stop_eos), int(draft), int(cached),
            seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62,
            *_f64(sampling.temperature if sampling else 0.0), min(int(sampling.top_k), TOP_K_MAX) if sampling else 0,
            *_f64(sampling.top_p if sampling else 1.0), *_f64(sampling.min_p if sampling else 0.0),
            int(drafts), -1 if confidence is None else round(confidence * 1e6)]


def decode(header: list[int]) -> Request:
    """``encode``'s request; a temperature of 0 or less is greedy (``sampling`` None)."""

    if len(header) != 16:
        raise ValueError(f"a request header has 16 values, not {len(header)}")
    (max_tokens, stop_eos, draft, cached, s_lo, s_hi, s_top, t_lo, t_hi, top_k, p_lo, p_hi, m_lo, m_hi, drafts,
     ppm) = header
    temperature = _from_f64(t_lo, t_hi)
    seed = (s_top << 62) | (s_hi << 31) | s_lo
    sampling = (Sampling(seed, temperature, top_k, _from_f64(p_lo, p_hi), _from_f64(m_lo, m_hi))
                if temperature > 0 else None)
    return Request(max_tokens, bool(stop_eos), bool(draft), cached, sampling,
                   (drafts, None if ppm < 0 else ppm / 1e6))


class Bell:
    """The idle doorbell on the rendezvous store (no store: no doorbell, rank 1 waits in the all-gather)."""

    def __init__(self, store) -> None:
        self.store, self.rung = store, 0

    def ring(self) -> None:
        if self.store is not None:
            self.rung += 1
            self.store.set(BELL.format(self.rung), b"1")

    def wait(self) -> None:
        """Rank 1: until rank 0 rings the next request, in hour-long waits; a lost rank 0 raises."""

        if self.store is None:
            return
        key = BELL.format(self.rung + 1)
        while True:
            try:
                self.store.wait([key], timedelta(hours=1))
                break
            except Exception as exc:        # an idle hour waits again; anything else goes up
                if "timeout" not in str(exc).lower():
                    raise
        self.store.delete_key(key)
        self.rung += 1


def gather_ints(comm, values: list[int], device: str = "cuda") -> list[list[int]]:
    """Each rank's int32 ``values`` on every rank, in rank order."""

    import torch

    mine = torch.tensor(values, dtype=torch.int32, device=device)
    got = torch.empty((2 * len(values),), dtype=torch.int32, device=device)
    comm.all_gather(mine, got)
    return [got[:len(values)].tolist(), got[len(values):].tolist()]


def share(comm, rank: int, values: list[int] | None, device: str = "cuda") -> list[int]:
    """Rank 0's int32 list on both ranks (its length first, then the values); rank 1 passes None."""

    import torch

    mine = torch.tensor(values, dtype=torch.int32, device=device) if rank == 0 else None   # raises before any gather
    count = gather_ints(comm, [len(values) if rank == 0 else 0], device)[0][0]
    send = mine if rank == 0 else torch.zeros((count,), dtype=torch.int32, device=device)
    got = torch.empty((2 * count,), dtype=torch.int32, device=device)
    comm.all_gather(send, got)
    return [int(v) for v in got[:count].tolist()]


def settings(*, start_error: bool, dspark: bool, capacity: int, prefill_rows: int, max_rows: int, ring: int,
             policy: tuple[int, float | None], layers: int, world: int, engram_digest: int, cache_bytes: int,
             cache_entries: int) -> list[int]:
    """This rank's agreement vector: SETTINGS, then SPARE."""

    drafts, confidence = policy
    lo, hi = struct.unpack("<2i", struct.pack("<q", engram_digest))
    return [int(start_error), int(dspark), capacity, prefill_rows, max_rows, ring, drafts,
            -1 if confidence is None else round(confidence * 1e6), layers, world, KV_STORAGE, ENGRAM_SPLIT, hi, lo,
            cache_bytes >> 20, cache_entries]


def agree(both: list[list[int]]) -> tuple[int, int]:
    """Refuse ranks started with different settings, naming each -> (snapshot bytes, entries) both use."""

    n = len(SETTINGS)
    differ = [f"{name} rank 0 {a}, rank 1 {b}" for name, a, b in zip(SETTINGS, both[0][:n], both[1][:n]) if a != b]
    if differ:
        raise RuntimeError("the two ranks were started with different settings (" + "; ".join(differ) + "): give both "
                           "the same flags, --context and TF_DSV41_* variables, and the same checkpoint files")
    mib, entries = (min(a, b) for a, b in zip(both[0][n:], both[1][n:]))
    return mib << 20, entries
