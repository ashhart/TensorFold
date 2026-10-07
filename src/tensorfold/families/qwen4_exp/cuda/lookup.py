"""Prompt-lookup drafts for Flash Next (qwen4_exp): continue the longest earlier occurrence of the text's suffix,
with no model forward. Ported from glm53-tensorfold-spark patches/0020 (``families/glm5_next/cuda/lookup.py``).

Agent traffic (file edits, tool calls, JSON, code quoting the context) repeats long spans of what is already in
the prompt or the reply. ``SuffixIndex`` keeps every ``n``-gram of the request's history (prompt, then every
committed token) and, each round, finds the earlier occurrence whose backward match with the history's end is
longest (ties to the most recent); the tokens that followed it are the drafts (``copy``: a match that overlaps the
end continues periodically, as an LZ77 copy does). Qwen's ``CopyIndex`` draws from the context the same way.

``Lookup`` decides each round whether to spend a verify window on those tokens and how many: the drafts are
verified by the same window, keyed sampler and commit as every other draft, so a draft is kept exactly when it
equals the token serial decoding produces there (greedy or sampled), and the reply is byte-identical to serial
decoding whatever is proposed. The decision only changes the speed:

- forced (env ``TF_EXL3_LOOKUP=lN[:M]``): every round with a match of at least M tokens verifies up to N of its
  tokens; the other rounds draft with the MTP head as usual.
- gated (env ``TF_EXL3_LOOKUP=1``/``auto``, default OFF): a round with a match of at least ``TF_EXL3_LOOKUP_MIN``
  tokens (default 4) verifies the k tokens (0 to 7) with the most expected committed tokens per millisecond, and
  only when that beats the tokens a millisecond this request's MTP rounds have committed (their recent rounds,
  priced as ``round_ms`` prices them). ``p``, the chance a lookup draft is kept given the one before it was, is a
  running estimate per match-length band, starting from a cautious prior.

Everything here is a function of the committed tokens, the model costs and the env, so the decision is
deterministic. Pure Python (no torch): the unit tests run anywhere.
"""

from __future__ import annotations

import os
from typing import Sequence

DEFAULT_MIN = 3                   # ``lN``'s match threshold when M is not given
MAX_DRAFTS = 7                    # a verify window holds the pending token and up to 7 drafts
CANDIDATES = 64                   # earlier occurrences of the suffix's n-gram examined a round (most recent first)
EXTEND = 64                       # a backward match is not measured past this many tokens
# (shortest match of the band, prior p): the chance a lookup draft is kept after a kept one, before this request
# has evidence. Cautious for short matches, which prose makes by chance.
BANDS = ((0, 0.45), (8, 0.7), (16, 0.8), (32, 0.9))
PRIOR_WEIGHT = 2.0                # trials the prior counts for
DECAY = 0.85                      # a band's evidence fades by this a round that updates it
ALT_WINDOW = 8                    # the other drafters' rate: their last 8 rounds
PROBE_EVERY = 8                   # a band of matches of 16+ tokens the gate skipped this many times is tried once
PROBE_MATCH = 16
_DEBUG = bool(os.environ.get("TF_EXL3_LOOKUP_DEBUG"))     # TF_EXL3_LOOKUP_DEBUG=1 prints each gate decision


# verify window wall time (ms) by row count, measured single-stream on tf-exl3 (P7.1.2 microbench, tools/p7_cost_fit.py).
# A window is dominated by a ~27 ms per-round fixed cost, so extra rows are cheap and the gate should draft deep.
_VERIFY_MS = (27.0, 30.4, 33.7, 35.9, 40.6, 41.7, 42.8, 46.9)
_MTP_MS = 1.8                      # MTP arm's fixed per-round overhead, from the same fit (mtp_ms ~ a + b*(steps-1))
_MTP_STEP_MS = 1.3                 # and per chained step (the backlog term fit to ~0, absorbed by the fixed part)


def costs_for(rows: int) -> dict:
    """The per-round model costs the gate prices rounds with, in milliseconds (relative timings only:
    ``verify[k]`` is a k+1-row window, ``mtp``/``mtp_step``/``mtp_row`` the MTP head's). ``rows`` is the engine's
    widest verify window. Calibrated P7.1.2 (tf-exl3); correctness does not depend on it."""

    rows = max(2, int(rows))
    if rows <= len(_VERIFY_MS):
        verify = list(_VERIFY_MS[:rows])
    else:
        verify = list(_VERIFY_MS)
        step = _VERIFY_MS[-1] - _VERIFY_MS[-2]
        while len(verify) < rows:
            verify.append(verify[-1] + step)
    return {"verify": [float(v) for v in verify], "mtp": _MTP_MS, "mtp_step": _MTP_STEP_MS, "mtp_row": 0.0,
            "block": 0.0, "taps_row": 0.0}


def parse_policy(spec: str) -> list[int] | None:
    """``lN`` or ``lN:M`` -> [N, M]; None when ``spec`` is not a lookup spec. N from 1 to 7, M from 1 to 64."""

    spec = str(spec).strip()
    if not spec.startswith("l"):
        return None
    parts = spec[1:].split(":")
    bad = ValueError(f"lookup policy {spec!r}: expected lN or lN:M (N from 1 to {MAX_DRAFTS}, M from 1 to 64)")
    if len(parts) not in (1, 2):
        raise bad
    try:
        most = int(parts[0])
        least = int(parts[1]) if len(parts) == 2 else DEFAULT_MIN
    except ValueError:
        raise bad from None
    if not 1 <= most <= MAX_DRAFTS or not 1 <= least <= 64:
        raise bad
    return [most, least]


def env_spec() -> str | None:
    """``TF_EXL3_LOOKUP``: None (MTP only, the default) when unset/0/off; ``auto`` for the gated mode; else a
    forced ``lN[:M]`` spec."""

    raw = os.environ.get("TF_EXL3_LOOKUP", "").strip().lower()
    if raw in ("", "0", "false", "off", "no"):
        return None
    if raw in ("1", "true", "on", "yes", "auto"):
        return "auto"
    return raw


def env_min() -> int:
    """``TF_EXL3_LOOKUP_MIN`` (default 4): the gated mode's shortest match that drafts. 1 to 64."""

    least = int(os.environ.get("TF_EXL3_LOOKUP_MIN", "4"))
    if not 1 <= least <= 64:
        raise ValueError(f"TF_EXL3_LOOKUP_MIN={least}: expected 1 to 64")
    return least


def build_lookup(prompt: Sequence[int], rows: int, spec: str | None, *, eos: Sequence[int] = (),
                 stop_eos: bool = True) -> "Lookup | None":
    """The request's ``Lookup`` for env ``spec`` (``env_spec``), or None when it drafts none."""

    if spec is None:
        return None
    costs = costs_for(rows)
    if spec == "auto":
        return Lookup(prompt, costs, most=MAX_DRAFTS, min_match=env_min(), gated=True, eos=eos, stop_eos=stop_eos)
    parsed = parse_policy(spec)
    if parsed is None:
        raise ValueError(f"TF_EXL3_LOOKUP={spec!r}: expected auto or lN[:M]")
    return Lookup(prompt, costs, most=parsed[0], min_match=parsed[1], gated=False, eos=eos, stop_eos=stop_eos)


# -- the matcher ----------------------------------------------------------------------------------------------------
class SuffixIndex:
    """Where every ``n``-gram of a growing token history ends, for the longest earlier match of its suffix.

    ``extend`` appends tokens (O(1) each); ``match`` returns (length, end): ``history[end - length:end]`` equals
    the last ``length`` tokens, ``end < len(history)``, the longest such match among the ``candidates`` most
    recent occurrences of the last ``n`` tokens (ties to the most recent), measured up to ``extend_to`` tokens;
    (0, -1) when the last n-gram never occurred before."""

    def __init__(self, n: int = 3, *, candidates: int = CANDIDATES, extend_to: int = EXTEND) -> None:
        if n < 1:
            raise ValueError("n must be positive")
        self.n = n
        self.candidates = candidates
        self.extend_to = max(extend_to, n)
        self.tokens: list[int] = []
        self.ends: dict[tuple[int, ...], list[int]] = {}
        self.indexed = 0            # n-grams ending at or before this many tokens are in ``ends``
        self.owner: object = None   # the ``Lookup`` extending it (``reuse_index`` hands it to the next request)

    def __len__(self) -> int:
        return len(self.tokens)

    def extend(self, tokens: Sequence[int]) -> None:
        self.tokens.extend(int(t) for t in tokens)

    def _index(self) -> None:
        # the n-gram ending at the history's end is the query itself: it joins once a token follows it, so a match
        # always has a continuation
        n, t = self.n, self.tokens
        first, last = max(self.indexed, n), len(t)          # new n-grams end at first .. last - 1
        if first < last:
            ends = self.ends
            grams = zip(*(t[first - n + i:last - n + i] for i in range(n)))
            for end, key in zip(range(first, last), grams):
                got = ends.get(key)
                if got is None:
                    ends[key] = [end]
                else:
                    got.append(end)
        self.indexed = max(self.indexed, last)

    def match(self) -> tuple[int, int]:
        t = self.tokens
        L = len(t)
        if L <= self.n:
            return 0, -1
        self._index()
        ends = self.ends.get(tuple(t[L - self.n:]))
        if not ends:
            return 0, -1
        best_len, best_end = 0, -1
        for end in reversed(ends[-self.candidates:]):
            length = self.n                     # the n-gram matches by construction
            limit = min(self.extend_to, end)
            while length < limit and t[end - 1 - length] == t[L - 1 - length]:
                length += 1
            if length > best_len:
                best_len, best_end = length, end
                if length >= self.extend_to:
                    break
        return best_len, best_end

    def copy(self, end: int, count: int) -> list[int]:
        """The ``count`` tokens that follow position ``end`` if the history repeats from there: past the history's
        end the copy reads its own output (an overlapping match continues with its period)."""

        t = self.tokens
        L = len(t)
        out: list[int] = []
        for i in range(count):
            j = end + i
            out.append(t[j] if j < L else out[j - L])
        return out

    def propose(self, count: int, min_match: int = 1) -> tuple[list[int], int]:
        """(up to ``count`` drafts, match length) from the longest match of at least ``min_match`` tokens."""

        length, end = self.match()
        if end < 0 or length < min_match or count <= 0:
            return [], length
        return self.copy(end, count), length


_last: list[SuffixIndex] = []         # the previous request's index (prompt and reply), for the next turn


def reuse_index(n: int, prompt: Sequence[int]) -> SuffixIndex:
    """An index of ``prompt``: the previous request's, extended, when ``prompt`` begins with its whole history
    (the next turn of a conversation: indexing a 100k-token prompt from scratch takes ~0.1 s), else a new one.
    Matches are a function of the tokens alone, so reuse never changes a proposal."""

    prompt = [int(t) for t in prompt]
    idx = _last[0] if _last else None
    if idx is None or idx.n != n or len(idx.tokens) > len(prompt) or prompt[:len(idx.tokens)] != idx.tokens:
        idx = SuffixIndex(n)
    idx.extend(prompt[len(idx.tokens):])
    _last[:] = [idx]
    return idx


# -- per-round decision ---------------------------------------------------------------------------------------------
def round_ms(costs: dict, arm: str, rows: int, steps: int, backlog: int) -> float:
    """Model ms of a round. ``arm`` "m" (MTP) prices the verify window plus the head's steps; "l" (lookup) the
    window only."""

    verify = costs["verify"][min(rows, len(costs["verify"])) - 1]
    if arm == "l":
        return verify
    if arm == "m":
        return (verify + costs.get("mtp", 0.0) + costs.get("mtp_step", 0.0) * max(steps - 1, 0)
                + costs.get("mtp_row", 0.0) * max(backlog - 1, 0))
    return verify + costs.get("block", 0.0) + costs.get("taps_row", 0.0) * backlog


def best_depth(p: float, costs: dict, most: int, row_ms: float = 0.0) -> tuple[int, float]:
    """(k, rate): the number of drafts (1 to ``most``) whose expected committed tokens, 1 + p + ... + p^k, per
    ms of a (k + 1)-row window (plus ``row_ms`` a committed row for the drafters' catch-up) is highest."""

    best, best_rate = 0, 0.0
    expected = run = 1.0
    verify = costs["verify"]
    for k in range(1, min(most, len(verify) - 1) + 1):
        run *= p
        expected += run
        rate = expected / (verify[k] + row_ms * expected)
        if rate > best_rate:
            best, best_rate = k, rate
    return best, best_rate


class Lookup:
    """A request's lookup drafter: ``plan`` before each round (the drafts, or [] to let the MTP arm draft),
    ``record`` after it."""

    def __init__(self, prompt: Sequence[int], costs: dict, *, most: int = MAX_DRAFTS, min_match: int = 4,
                 gated: bool = True, eos: Sequence[int] = (), stop_eos: bool = True) -> None:
        self.costs = costs
        self.most = max(1, min(int(most), MAX_DRAFTS, len(costs["verify"]) - 1))
        self.min_match = max(1, int(min_match))
        self.gated = gated
        self.eos = frozenset(int(t) for t in eos) if stop_eos else frozenset()
        self.prompt = [int(t) for t in prompt]
        self.index = reuse_index(min(self.min_match, 8), self.prompt)
        self.index.owner = self
        self.prompt_len = len(prompt)
        self.synced = 0                           # tokens of the decode loop's ``out`` in the index
        self.acc = [0.0] * len(BANDS)             # decayed kept drafts, trials per band
        self.tri = [0.0] * len(BANDS)
        self.alt: list[tuple[int, float]] = []    # (tokens committed, model ms) of recent non-lookup rounds
        # a committed row is later taken by the drafter that drafts next (an MTP row)
        self.row_ms = max(costs.get("mtp_row", 0.0), costs.get("taps_row", 0.0))
        self.skipped = [0] * len(BANDS)          # gated-out rounds per band since its last lookup round
        self.band = 0
        self.rounds = self.drafted = self.accepted = 0

    def _band(self, length: int) -> int:
        b = 0
        for i, (least, _) in enumerate(BANDS):
            if length >= least:
                b = i
        return b

    def p(self, band: int) -> float:
        prior = BANDS[band][1]
        return (self.acc[band] + prior * PRIOR_WEIGHT) / (self.tri[band] + PRIOR_WEIGHT)

    def alt_rate(self) -> float:
        """Tokens a model ms of this request's recent MTP rounds; before any, an MTP round keeping one draft."""

        if self.alt:
            return sum(k for k, _ in self.alt) / sum(ms for _, ms in self.alt)
        c = self.costs
        return 2.0 / (c["verify"][min(2, len(c["verify"])) - 1] + c.get("mtp", 0.0))

    def sync(self, out: Sequence[int]) -> None:
        """The history is the prompt then ``out`` (the decode loop's tokens: every committed one and the pending
        token, which is already the exact sample for its position)."""

        if self.index.owner is not self:          # a later request took the shared index: a private one
            self.index = SuffixIndex(self.index.n)
            self.index.extend(self.prompt + [int(t) for t in out[:self.synced]])
            self.index.owner = self
        if self.synced < len(out):
            self.index.extend(out[self.synced:])
            self.synced = len(out)

    def plan(self, out: Sequence[int], room: int) -> list[int]:
        """The drafts for the positions after ``out[-1]`` (at most ``room``), or [] for a round of the MTP arm."""

        self.sync(out)
        most = min(self.most, room)
        if most <= 0:
            return []
        drafts, length = self.index.propose(most, self.min_match)
        if self.eos:                       # a draft past the end of the reply is never kept
            for i, d in enumerate(drafts):
                if d in self.eos:
                    drafts = drafts[:i]
                    break
        if not drafts:
            return []
        self.band = self._band(length)
        if _DEBUG:
            print("LOOKUP band=%d len=%d p=%.3f ndraft=%d alt=%.4f" % (self.band, length, self.p(self.band),
                  len(drafts), self.alt_rate()), flush=True)
        if self.gated:
            b = self.band
            k, rate = best_depth(self.p(b), self.costs, len(drafts), self.row_ms)
            if _DEBUG:
                print("LOOKUP gate k=%d rate=%.4f alt=%.4f skip=%s" % (k, rate, self.alt_rate(),
                      k == 0 or rate <= self.alt_rate()), flush=True)
            if k == 0 or rate <= self.alt_rate():
                # a long match's acceptance is learned only by trying it: once in a while, it is
                self.skipped[b] += 1
                if k == 0 or BANDS[b][0] < PROBE_MATCH or self.skipped[b] < PROBE_EVERY:
                    return []
            self.skipped[b] = 0
            drafts = drafts[:k]
        return drafts

    def record(self, arm: str, rows: int, steps: int, backlog: int, keep: int) -> None:
        """After a round: ``arm`` "l" updates the band's acceptance; another arm's committed tokens and model ms
        join the rate lookup rounds must beat."""

        if arm == "l":
            drafts = rows - 1
            kept = keep - 1
            b = self.band
            self.acc[b] = self.acc[b] * DECAY + kept
            self.tri[b] = self.tri[b] * DECAY + kept + (1 if kept < drafts else 0)
            self.rounds += 1
            self.drafted += drafts
            self.accepted += kept
            if _DEBUG:
                print("REC l band=%d p=%.3f acc=%.2f tri=%.2f kept=%d/%d alt=%.4f" % (
                    b, self.p(b), self.acc[b], self.tri[b], kept, drafts, self.alt_rate()), flush=True)
            return
        _ms = round_ms(self.costs, arm, rows, steps, backlog)
        if _DEBUG:
            print("REC m arm=%s rows=%d steps=%d backlog=%d keep=%d ms=%.2f alt=%.4f" % (
                arm, rows, steps, backlog, keep, _ms, self.alt_rate()), flush=True)
        self.alt.append((keep, _ms))
        del self.alt[:-ALT_WINDOW]
