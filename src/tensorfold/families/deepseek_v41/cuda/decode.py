"""Decode on one rank: verify windows, serial rounds and DSpark rounds that keep drafts equal to the draws."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from tensorfold.engine.exact_sampling import Sampling

from ..engram_hash import rank_columns
from . import BLOCK, DEFAULT_DRAFTS, MAX_ROWS, PREFILL_ROWS, dspark, sample
from . import forward as F
from .buffers import Buffers, State
from .graphs import Graphs
from .weights import Weights

HOST_PARTS = ("engram", "sampling", "markov", "launch")


def _clock() -> dict[str, float]:
    """Seconds of one round by part: host work (HOST_PARTS) and ``device``, the waits for the forward and block."""

    return dict.fromkeys((*HOST_PARTS, "device"), 0.0)


class Engine:
    """One rank's weights, sequence state, buffers, DSpark scratch and Engram host side; tests replace ``propose``."""

    def __init__(self, w: Weights, capacity: int, prefill_rows: int = PREFILL_ROWS, graphs: bool = False, *,
                 hasher=None, reader=None) -> None:
        if w.engram and (hasher is None or reader is None):
            raise ValueError("a checkpoint with Engram layers needs its hasher and reader")
        cfg, dev = w.cfg, w.device
        self.w, self.hasher, self.reader = w, hasher, reader
        self.st = State(cfg, capacity, dev)
        self.pbuf = Buffers(cfg, w.world, min(prefill_rows, capacity), capacity, prefill=True, device=dev)
        self.dbuf = Buffers(cfg, w.world, MAX_ROWS, capacity, device=dev)
        self.dwork = dspark.Work(cfg, w.world, dev) if w.dspark is not None else None
        self.eos: tuple[int, ...] = (cfg.eos_token_id,)
        self.clock = _clock()                       # the current round's
        self.replays = {"graph": 0, "eager": 0}     # verify forwards by path
        self.graphs = None
        if graphs:
            self.graphs = Graphs(self)
            self.graphs.warm()
            self.reset()

    def reset(self) -> None:
        self.st.reset()

    def _timed(self, part: str, start: float) -> float:
        now = time.perf_counter()
        self.clock[part] += now - start
        return now

    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """A verify window at ``st.pos``, staged then replayed (or eager) -> logits fp32 [R, V / world]."""

        w, st, b = self.w, self.st, self.dbuf
        t = time.perf_counter()
        R = F.stage(w, st, b, tokens, self.hasher, self.reader)
        t = self._timed("engram", t)
        g = self.graphs.verify.get(R) if self.graphs is not None else None
        if g is not None:
            g.replay()
        else:
            F.compute(w, st, b, R, prompt=False, head_rows=R)
        self.replays["graph" if g is not None else "eager"] += 1
        t = self._timed("launch", t)
        torch.cuda.current_stream().synchronize()
        self._timed("device", t)
        return b.logits[:R]

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None) -> list[int]:
        tokens, seconds = sample.target_rows(self.w, logits, positions, sampling)
        self.clock["sampling"] += seconds
        return tokens

    def advise(self, context: Sequence[int], token: int) -> None:
        """Start reading the rank's Engram rows of ``token`` placed after ``context`` (the ids before it)."""

        if not self.w.engram:
            return
        t = time.perf_counter()
        back = self.w.cfg.engram_max_ngram_size - 1
        before = np.asarray(list(context)[-back:] if back else [], dtype=np.int64)
        ids = rank_columns(self.hasher.ids(before, np.asarray([token], dtype=np.int64)), self.w.rank, self.w.world)
        starts = np.asarray(self.reader.layout.starts[:ids.shape[1]], dtype=np.int64)
        self.reader.advise((ids + starts[None, :, None]).reshape(-1))
        self._timed("engram", t)

    def absorb(self, n: int) -> None:
        """The last forward's first ``n`` rows, just committed, into the DSpark stage rings."""

        t = time.perf_counter()
        g = self.graphs.absorb.get(n) if self.graphs is not None else None
        if g is not None:
            g.replay()
        else:
            dspark.absorb(self, self.dbuf, n, prompt=False)
        self._timed("launch", t)

    def _block(self, e) -> None:
        t = time.perf_counter()
        g = self.graphs.block if self.graphs is not None else None
        if g is not None:
            g.replay()
        else:
            dspark.block(e)
        t = self._timed("launch", t)
        torch.cuda.current_stream().synchronize()
        self._timed("device", t)

    def _markov_input(self, e, i: int) -> None:
        if self.graphs is not None and self.graphs.inputs:
            self.graphs.inputs[i].replay()
        else:
            dspark.markov_input(e, i)

    def _markov_step(self, e, i: int) -> None:
        if self.graphs is not None and self.graphs.steps:
            self.graphs.steps[i].replay()
        else:
            dspark.markov_step(e, i)

    def propose(self, y: int, p: int, sampling: Sampling | None, d: int,
                threshold: float | None = None) -> tuple[list[int], list[float]]:
        """``dspark.propose`` through the graphs, advising each draft's Engram rows -> (drafts, logits)."""

        start, other = time.perf_counter(), sum(self.clock.values())
        context = [*self.st.history, y]

        def landed(i: int, token: int) -> None:
            self.advise(context, token)
            context.append(token)

        out = dspark.propose(self, y, p, sampling, d, threshold,
                             steps=(self._block, self._markov_input, self._markov_step), landed=landed)
        self.clock["sampling"] += self.dwork.sampling
        self.clock["markov"] += time.perf_counter() - start - (sum(self.clock.values()) - other)
        return out


@dataclass
class DecodeResult:
    """A reply from its pending token on: the tokens (pending first), rounds, drafts and per-round seconds."""

    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    depths: list[int] = field(default_factory=list)     # drafts verified each round
    keeps: list[int] = field(default_factory=list)      # rows kept each round
    clocks: list[dict[str, float]] = field(default_factory=list)

    @property
    def tokens_per_round(self) -> float:
        return (len(self.tokens) - 1) / self.rounds if self.rounds else 0.0

    def stats(self) -> dict:
        """The request's decode stats, with each host part's and the device's mean ms a round as ``stages_ms``."""

        n = max(len(self.clocks), 1)
        return {"decode_s": self.seconds, "rounds": self.rounds, "drafted": self.drafted, "accepted": self.accepted,
                "tokens_per_round": round(self.tokens_per_round, 3),
                "stages_ms": {k: round(sum(c[k] for c in self.clocks) * 1e3 / n, 3) for k in (*HOST_PARTS, "device")}}


def accept(sampled: Sequence[int], drafts: Sequence[int], ends: Sequence[int] = ()) -> int:
    """Rows a window keeps: 1 plus the leading drafts equal to the draw before them, up to a drawn end token."""

    keep = 1
    for s, d in zip(sampled, drafts):
        if s != d or s in ends:
            break
        keep += 1
    return keep


def _emit(on_tokens: Callable | None, tokens: list[int]) -> None:
    if on_tokens is not None and tokens:
        on_tokens(tokens)


def _rounds(e: Engine, pending: int, count: int, sampling: Sampling | None, stop_eos: bool, on_tokens,
            drafter: Callable[[int, int], list[int]] | None) -> DecodeResult:
    """Verify windows until ``count`` tokens or an end token; ``on_tokens``' return is ignored, so ranks agree."""

    w, st = e.w, e.st
    ends = e.eos if stop_eos else ()
    out, res = [pending], DecodeResult([], 0.0, 0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    _emit(on_tokens, [pending])
    e.clock = _clock()
    if drafter is not None and len(out) < count and pending not in ends:
        e.advise(st.history, pending)
    keep = 0
    while len(out) < count and out[-1] not in ends:
        if drafter is not None and keep:
            e.absorb(keep)
        guess = drafter(out[-1], count - len(out)) if drafter is not None else []
        tokens = [out[-1], *guess]
        R = len(tokens)
        logits = e.forward(tokens)
        sampled = e.sample(logits, [st.pos + 1 + r for r in range(R)], sampling)
        keep = accept(sampled, guess, ends)
        F.commit(w, st, e.dbuf, R, keep)
        out.extend(sampled[:keep])
        if drafter is not None and len(out) < count and out[-1] not in ends:
            e.advise(st.history, out[-1])
        _emit(on_tokens, sampled[:keep][:max(0, count - (len(out) - keep))])
        res.rounds += 1
        res.drafted += len(guess)
        res.accepted += keep - 1
        res.depths.append(len(guess))
        res.keeps.append(keep)
        res.clocks.append(e.clock)
        e.clock = _clock()
    torch.cuda.synchronize()
    res.tokens, res.seconds = out[:count], time.perf_counter() - start
    return res


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, stop_eos: bool = True,
                  on_tokens=None) -> DecodeResult:
    """The reference: one-row windows from ``pending`` (the prompt's first token) until ``count`` tokens."""

    return _rounds(e, pending, count, sampling, stop_eos, on_tokens, None)


@torch.no_grad()
def dspark_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, drafts: int = DEFAULT_DRAFTS,
                  confidence: float | None = None, stop_eos: bool = True, on_tokens=None) -> DecodeResult:
    """``serial_decode``'s tokens in windows of the pending token and up to ``drafts`` DSpark drafts."""

    if e.w.dspark is None:
        raise ValueError("DSpark rounds need the checkpoint's DSpark stages")
    if not 1 <= drafts <= BLOCK:
        raise ValueError(f"{drafts} drafts a round: DSpark proposes 1 to {BLOCK}")

    def drafter(y: int, remaining: int) -> list[int]:
        d = min(drafts, remaining - 1)
        return list(e.propose(y, e.st.pos - 1, sampling, d, confidence)[0][:d]) if d > 0 else []

    return _rounds(e, pending, count, sampling, stop_eos, on_tokens, drafter)
