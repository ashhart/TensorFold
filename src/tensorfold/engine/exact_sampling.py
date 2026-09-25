"""Sampling that is byte exact to serial decoding: every token a fixed function of its position.

Greedy decoding is what makes lanes simple to check, but Qwen3.8 ships with
``do_sample: true, temperature 1.0, top_k 20, top_p 0.95`` and loops under
greedy decoding in agent sessions (an agent session repeated one tool call six
times and then doubled its reply every turn to 32k tokens, 2026-09-23).

Here the token at absolute position p is

    argmax over the kept candidates i of  logit_i / T + g(seed, p, i)

where the kept candidates are the top_k logits (ties broken by token id) cut to
the smallest set holding top_p of the tempered probability, and g is Gumbel
noise from a hash of (seed, p, i). That is an exact draw from the top-k/top-p
distribution (Gumbel-max), and it depends only on the row's own logits and its
position. A verify window row gets the same token as the serial step at that
position, so drafted output stays byte identical to serial sampling with the
same seed, and a draft is accepted exactly when it equals that token.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import numpy as np

MARGIN = 8     # candidates beyond top_k read from the GPU, so tied values resolve by id on the CPU
# top_k 0 (a model that sets only top_p, e.g. Nemotron): candidates read from the GPU for the nucleus
NUCLEUS_CANDIDATES = 256


@dataclass(frozen=True)
class Sampling:
    seed: int
    temperature: float = 1.0
    top_k: int = 20
    top_p: float = 0.95


def seed_for(tokens: Sequence[int], salt: int = 0) -> int:
    """A reproducible seed from the prompt: the same conversation samples the same reply."""

    digest = hashlib.sha256((",".join(str(int(t)) for t in tokens) + f"|{salt}").encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _mix(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def uniform(seed: int, position: int, ids: np.ndarray) -> np.ndarray:
    """Uniform (0, 1) doubles from a splitmix64 hash of (seed, position, token id)."""

    with np.errstate(over="ignore"):
        x = _mix(np.uint64(seed & 0xFFFFFFFFFFFFFFFF) + np.uint64(0x9E3779B97F4A7C15))
        x = _mix(x ^ (np.uint64(position) * np.uint64(0xD1B54A32D192ED03)))
        x = _mix(x ^ ids.astype(np.uint64))
    return (x >> np.uint64(11)).astype(np.float64) * 2.0 ** -53 + 2.0 ** -54


def uniform_rows(seed: int, positions: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """``uniform`` for many positions at once: row r is ``uniform(seed, positions[r], ids[r])``, same bits."""

    with np.errstate(over="ignore"):
        x = _mix(np.uint64(seed & 0xFFFFFFFFFFFFFFFF) + np.uint64(0x9E3779B97F4A7C15))
        x = _mix(x ^ (np.asarray(positions).astype(np.uint64)[:, None] * np.uint64(0xD1B54A32D192ED03)))
        x = _mix(x ^ ids.astype(np.uint64))
    return (x >> np.uint64(11)).astype(np.float64) * 2.0 ** -53 + 2.0 ** -54


def choose(values: np.ndarray, ids: np.ndarray, position: int, s: Sampling) -> int:
    """One row: candidate logits ``values`` for token ``ids`` -> the sampled token id."""

    order = np.lexsort((ids, -values))
    k = max(1, min(int(s.top_k) if s.top_k else len(ids), len(ids)))
    ids = ids[order][:k]
    scaled = values[order][:k].astype(np.float64) / max(float(s.temperature), 1e-6)
    probs = np.exp(scaled - scaled.max())
    probs /= probs.sum()
    if 0.0 < s.top_p < 1.0:
        keep = int(np.searchsorted(np.cumsum(probs), s.top_p) + 1)
        ids, scaled = ids[:keep], scaled[:keep]
    gumbel = -np.log(-np.log(uniform(s.seed, position, ids)))
    return int(ids[int(np.argmax(scaled + gumbel))])


def choose_rows(values: np.ndarray, ids: np.ndarray, positions: Sequence[int], s: Sampling) -> list[int]:
    """``choose`` for every row at once, with the same bits: each step is elementwise or runs along
    one row in the same order (a row alone or among others gives the same token; checked against
    ``choose`` on 46,750 rows, ties included). ~0.23 -> 0.035 ms for 16 rows (2026-09-24)."""

    rows, width = ids.shape
    order = np.lexsort((ids, -values), axis=-1)
    k = max(1, min(int(s.top_k) if s.top_k else width, width))
    ids = np.take_along_axis(ids, order, axis=-1)[:, :k]
    scaled = np.take_along_axis(values, order, axis=-1)[:, :k].astype(np.float64) / max(float(s.temperature), 1e-6)
    score = scaled - np.log(-np.log(uniform_rows(s.seed, np.asarray(positions), ids)))
    if 0.0 < s.top_p < 1.0:
        probs = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        keep = (np.cumsum(probs, axis=-1) < s.top_p).sum(axis=-1) + 1    # searchsorted(cumsum, top_p) + 1
        score[np.arange(k)[None, :] >= keep[:, None]] = -np.inf
    return [int(t) for t in ids[np.arange(rows), np.argmax(score, axis=-1)]]


def top_candidates(logits: Any, s: Sampling) -> tuple[Any, Any] | None:
    """The rows' candidates ``sample_rows`` reads (lazy MLX arrays), so a caller can evaluate them with
    the forward instead of in a GPU round trip of their own; None where ``sample_rows`` takes another path."""

    import mlx.core as mx

    from tensorfold.engine.topk import MAX_K, topk_rows

    vocab = int(logits.shape[-1])
    count = min(vocab, (int(s.top_k) if s.top_k else vocab) + MARGIN)
    if count <= MAX_K and logits.dtype == mx.bfloat16:
        return topk_rows(logits.reshape(-1, vocab), count)      # radix select: the exact top by (value, id)
    return None


def sample_rows(logits: Any, positions: Sequence[int], s: Sampling, keep: dict | None = None,
                top: tuple[Any, Any] | None = None) -> list[int]:
    """Rows of ``logits`` [W, V] (MLX) at absolute ``positions`` -> one token each.

    ``keep`` (a dict) receives the rows' candidates the draw chose among: ``cand`` [W, K] ids and
    ``vals`` [W, K] logits (host arrays already read for the draw; nothing more is computed).
    ``top``: ``top_candidates(logits, s)``, already evaluated.
    """

    import mlx.core as mx

    if not s.top_k and 0.0 < s.top_p < 1.0 and top is None and keep is None:
        drawn = _nucleus_rows(logits, positions, s)
        if drawn is not None:
            return drawn
    if top is None:
        top = top_candidates(logits, s)
    if top is not None:
        cand, vals = top
    else:
        vocab = int(logits.shape[-1])
        count = min(vocab, (int(s.top_k) if s.top_k else vocab) + MARGIN)
        flat = logits.reshape(-1, vocab).astype(mx.float32)
        if count < vocab:
            cand = mx.argpartition(-flat, kth=count - 1, axis=-1)[:, :count]
        else:
            cand = mx.broadcast_to(mx.arange(vocab)[None, :], flat.shape)
        vals = mx.take_along_axis(flat, cand, axis=-1)
    cand_np, vals_np = np.array(cand), np.array(vals)
    if keep is not None:
        keep["cand"], keep["vals"] = cand_np, vals_np
    return choose_rows(vals_np, cand_np.astype(np.int64), positions, s)


def _nucleus_rows(logits: Any, positions: Sequence[int], s: Sampling) -> list[int] | None:
    """top_k 0 and 0 < top_p < 1 without sorting the vocabulary on the host.

    The GPU returns each row's top ``NUCLEUS_CANDIDATES`` logits and the log of its full
    normalizer; the nucleus is cut from the candidates in (value, id) order with those
    probabilities. Every token above the candidates' lowest value is a candidate, so a
    nucleus that ends above that value is the one the whole row gives; a row whose nucleus
    reaches it returns None and is drawn from the whole vocabulary. ~0.4 ms a row for a
    131k vocabulary instead of ~9.7 (2026-09-25).
    """

    import mlx.core as mx

    vocab = int(logits.shape[-1])
    count = min(vocab, NUCLEUS_CANDIDATES)
    if count >= vocab:
        return None
    flat = logits.reshape(-1, vocab).astype(mx.float32)
    temperature = max(float(s.temperature), 1e-6)
    cand = mx.argpartition(-flat, kth=count - 1, axis=-1)[:, :count]
    vals = mx.take_along_axis(flat, cand, axis=-1)
    norm = mx.logsumexp(flat / temperature, axis=-1)
    cand_np, vals_np, norm_np = np.array(cand).astype(np.int64), np.array(vals), np.array(norm).astype(np.float64)
    out: list[int] = []
    for row in range(cand_np.shape[0]):
        order = np.lexsort((cand_np[row], -vals_np[row]))
        ids, values = cand_np[row][order], vals_np[row][order].astype(np.float64)
        scaled = values / temperature
        kept = int((np.cumsum(np.exp(scaled - norm_np[row])) < s.top_p).sum()) + 1
        if kept >= count or values[kept - 1] <= values[-1]:
            return None
        score = scaled[:kept] - np.log(-np.log(uniform(s.seed, int(positions[row]), ids[:kept])))
        out.append(int(ids[int(np.argmax(score))]))
    return out


__all__ = ["MARGIN", "Sampling", "choose", "choose_rows", "sample_rows", "seed_for", "top_candidates", "uniform",
           "uniform_rows"]
