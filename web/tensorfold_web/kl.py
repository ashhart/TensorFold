"""KL divergence and top-token agreement between two engines, position by position, where both saw the same text.

Each side writes a greedy reply with its 20 most likely tokens at each position. Up to and including the first
position where their tokens differ, both conditioned on exactly the same text, so their distributions compare
there; after it the texts differ and the prompt stops. Making the candidate continue the reference's text after a
difference would measure every position, but TensorFold 1.0.5 neither continues a final assistant message (it
answers it as a finished turn) nor scores a prompt.

The other way, a fixed text (`score`, `pair_text`): both engines read the same text through `/v1/completions` with
vLLM's `prompt_logprobs` (TensorFold from 1.0.8 on CUDA; its Metal and Python lines refuse it with a 400), and every
position of it compares, none lost where the replies would part.

Only the top 20 log-probabilities are given, so the divergence is a lower bound of the full KL, computed on what both
lists establish (see `divergence`); whether the two distributions are the same is checked directly on the lists.
Comparing an engine with itself gives 0, which checks the method on a given engine.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Self

import httpx

from . import httpread

TOP = 20                # the most an OpenAI-compatible engine gives (TensorFold 1.0.5 on CUDA, vLLM, llama.cpp)
MAX_TOKENS = 128        # reply length compared per prompt
FLOOR = 1e-10
TEXT_MAX = 20_000       # characters of a fixed text: prompt_logprobs prefill it whole, never from the cache


@dataclass
class Step:
    token: bytes
    top: dict[bytes, float]          # token bytes (a fixed text: token ids) -> probability
    ident: str | None = None         # a fixed text: the token's id, which both engines must read alike
    said: dict[bytes, str] = field(default_factory=dict)    # a fixed text: each listed id's text


@dataclass
class Target:
    url: str                          # .../v1/chat/completions
    headers: dict
    model: str
    extra: dict = field(default_factory=dict)    # thinking off, "draft": false…


class Steps(list):
    """A reply's steps, with the engine's own block about it (``runtime``)."""

    def __init__(self, steps=(), runtime: dict | None = None) -> None:
        super().__init__(steps)
        self.runtime = runtime or {}


def _steps(body: dict) -> list[Step]:
    content = ((body.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
    out = []
    for c in content:
        tok = bytes(c.get("bytes") or c.get("token", "").encode())
        top = {bytes(t.get("bytes") or t.get("token", "").encode()): math.exp(t["logprob"])
               for t in c.get("top_logprobs") or [] if t.get("logprob") is not None}
        if tok not in top and c.get("logprob") is not None:
            top[tok] = math.exp(c["logprob"])
        out.append(Step(tok, top))
    return out


def ask(client: httpx.Client, t: Target, messages: list[dict], max_tokens: int,
        cancel: threading.Event | None = None) -> list[Step]:
    """A greedy reply with the top log-probabilities at each position; a cancel ends the wait for it."""

    body = {"model": t.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
            "stream": False, "logprobs": True, "top_logprobs": TOP, **t.extra}
    got: list[httpx.Response] = []

    def post() -> None:
        with client.stream("POST", t.url, json=body, headers=t.headers) as r:
            got.append(r)                                # known as soon as its headers are in: a cancel closes it
            r.read()
    httpread.until_cancelled(post, cancel, close=lambda: got and got[0].close(), name="kl-reply")
    r = got[0]
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    body_out = r.json()
    steps = Steps(_steps(body_out))
    # what the engine says of the reply (drafts on or off): a reference asked for drafts off is trusted on its word
    steps.runtime = {**(body_out.get("tensorfold") or {}), **(body_out.get("speculative") or {})}
    if max_tokens and not steps:
        raise RuntimeError("the engine gave no log-probabilities (it needs `logprobs` on chat completions)")
    return steps


def completions_url(chat_url: str) -> str:
    return chat_url.removesuffix("/chat/completions") + "/completions"


def _scored(body: dict) -> list[Step]:
    """vLLM's `prompt_logprobs`: one entry per prompt token, `null` for the first (nothing predicts it), then the
    prompt token itself keyed by its id, followed by the best tokens, each with `logprob`, `rank`, `decoded_token`."""

    rows = (body.get("choices") or [{}])[0].get("prompt_logprobs")
    if not isinstance(rows, list):
        raise RuntimeError("the engine gave no prompt_logprobs for the text")  # noqa: TRY004 - the run says it
    out = []
    for row in rows[1:]:
        if not isinstance(row, dict) or not row:
            raise RuntimeError("the engine gave a position of the text without its log-probabilities")
        ident = next(iter(row))                           # the prompt token comes first
        good = {str(k).encode(): v for k, v in row.items() if isinstance(v, dict) and isinstance(v.get("logprob"), int | float)}
        out.append(Step(str((row[ident] or {}).get("decoded_token") or "").encode(),
                        {k: math.exp(v["logprob"]) for k, v in good.items()}, str(ident),
                        {k: str(v.get("decoded_token") or "") for k, v in good.items()}))
    return out


def score(client: httpx.Client, t: Target, text: str, cancel: threading.Event | None = None) -> list[Step]:
    """Every position of ``text`` as the engine reads it, with the top log-probabilities there; a cancel ends the wait.
    Of the run's extra fields only `draft` goes (a raw text has no chat template to turn thinking off in)."""

    body = {"model": t.model, "prompt": text, "max_tokens": 1, "temperature": 0.0, "stream": False,
            "prompt_logprobs": TOP, **{k: v for k, v in t.extra.items() if k == "draft"}}
    got: list[httpx.Response] = []

    def post() -> None:
        with client.stream("POST", completions_url(t.url), json=body, headers=t.headers) as r:
            got.append(r)
            r.read()
    httpread.until_cancelled(post, cancel, close=lambda: got and got[0].close(), name="kl-score")
    r = got[0]
    if r.status_code == 400:
        raise RuntimeError(f"it does not score a text's positions (prompt_logprobs on /v1/completions): TensorFold "
                           f"does from 1.0.8 on CUDA, not on Metal nor its Python line; vLLM does too. It said: "
                           f"{r.text[:300]}")
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    out = r.json()
    steps = Steps(_scored(out))
    steps.runtime = {**(out.get("tensorfold") or {}), **(out.get("speculative") or {})}
    return steps


def divergence(p: dict[bytes, float], q: dict[bytes, float]) -> float:
    """A lower bound of KL(P || Q), from the tokens each side lists.

    The partition: each token both sides list, the tokens only P lists (A), and everything else (R). P is known on
    every cell. Q is known on the shared tokens and on A and R together; on A it is at most the smallest probability
    Q lists, once per token of A, since none of them made Q's list. KL on a coarser partition never exceeds the full
    KL, and Q's unknown split between A and R is taken where it gives the least divergence, so the result is a lower
    bound whatever the unlisted tokens hold. It can be 0 for two different distributions, which `same_distribution`
    tells apart.
    """

    shared = [t for t in p if t in q]
    only_p = [t for t in p if t not in q]
    total = sum(p[t] * math.log(p[t] / max(FLOOR, q[t])) for t in shared if p[t] > 0)
    q_left = max(0.0, 1.0 - sum(q[t] for t in shared))       # Q's mass on A and R together
    p_a = sum(p[t] for t in only_p)
    p_r = max(0.0, 1.0 - sum(p[t] for t in shared) - p_a)
    cap = min(q_left, len(only_p) * min(q.values())) if q else q_left
    x = q_left * p_a / (p_a + p_r) if p_a + p_r > 0 else 0.0  # the least divergence over the two cells...
    x = min(x, cap)                                           # ...within what Q's list allows on A
    if p_a > 0:
        total += p_a * math.log(p_a / max(FLOOR, x))
    if p_r > 1e-12:
        total += p_r * math.log(p_r / max(FLOOR, q_left - x))
    return max(0.0, total)                                    # only rounding can take it below zero


def usable(top: dict[bytes, float]) -> bool:
    """A position's probabilities say something: finite, and some mass (a log-probability of -9999 underflows to 0)."""

    return bool(top) and all(math.isfinite(p) for p in top.values()) and sum(top.values()) > 0


def same_distribution(p: dict[bytes, float], q: dict[bytes, float], tol: float = 1e-9) -> bool:
    """The same listed tokens with the same probabilities, checked directly rather than read from a zero KL."""

    return p.keys() == q.keys() and all(abs(p[t] - q[t]) <= tol for t in p)


def pair(reference: list[Step], theirs: list[Step]) -> dict:
    """Two replies to one prompt, compared up to and including the first position where their tokens differ."""

    kls, agree, equal, rows, widths = [], 0, 0, [], []
    differs_at, unscored = None, 0
    for k, (a, b) in enumerate(zip(reference, theirs, strict=False)):
        if not usable(a.top) or not usable(b.top):   # no probability mass at this position on one side
            unscored += 1
            if a.token != b.token:
                differs_at = k
                break
            continue
        d = divergence(a.top, b.top)
        kls.append(d)
        widths.append(min(len(a.top), len(b.top)))
        same = max(a.top, key=a.top.get, default=a.token) == max(b.top, key=b.top.get, default=b.token)
        agree += same
        equal += same_distribution(a.top, b.top)
        rows.append({"at": k, "kl": d, "ref": a.token.decode(errors="replace"), "cand": b.token.decode(errors="replace"),
                     "same_top": same})
        if a.token != b.token:
            differs_at = k
            break                     # the texts part here: what follows was not written on the same text
    identical = differs_at is None and len(reference) == len(theirs)
    rows.sort(key=lambda w: -w["kl"])
    return {"positions": len(kls), "tokens": len(reference), "kl": kls, "agree": agree, "equal": equal,
            "differs_at": differs_at, "identical": identical, "worst": rows[:5], "requests": 2, "unscored": unscored,
            "width": widths}                       # each scored position: the fewest alternatives either side gave


def pair_text(reference: list[Step], theirs: list[Step]) -> dict:
    """The same text read by both engines: every position compares, none dropped where replies would part. Both
    must split it into the same tokens, or positions would not match."""

    if [a.ident for a in reference] != [b.ident for b in theirs]:
        raise RuntimeError(f"the two engines split the text into different tokens ({len(reference) + 1} and "
                           f"{len(theirs) + 1}, or other ids): not the same tokenizer, so positions do not match")
    kls, agree, equal, rows, widths, unscored = [], 0, 0, [], [], 0
    for k, (a, b) in enumerate(zip(reference, theirs, strict=True), start=1):
        if not usable(a.top) or not usable(b.top):
            unscored += 1
            continue
        d = divergence(a.top, b.top)
        kls.append(d)
        widths.append(min(len(a.top), len(b.top)))
        best_a, best_b = max(a.top, key=a.top.get), max(b.top, key=b.top.get)
        agree += best_a == best_b
        equal += same_distribution(a.top, b.top)
        # the text's token there, and what each engine found most likely in its place
        rows.append({"at": k, "kl": d, "token": a.token.decode(errors="replace"), "ref": a.said.get(best_a, ""),
                     "cand": b.said.get(best_b, ""), "same_top": best_a == best_b})
    rows.sort(key=lambda w: -w["kl"])
    return {"positions": len(kls), "tokens": len(reference) + 1, "kl": kls, "agree": agree, "equal": equal,
            "differs_at": None, "identical": bool(kls) and equal == len(kls) and not unscored, "worst": rows[:5],
            "requests": 2, "unscored": unscored, "width": widths}


def compare(client: httpx.Client, ref: Target, cand: Target, messages: list[dict], *, max_tokens: int = MAX_TOKENS,
            cancel: threading.Event | None = None) -> dict:
    """One prompt: both replies, then :func:`pair`."""

    if cancel is not None and cancel.is_set():
        raise InterruptedError
    return pair(ask(client, ref, messages, max_tokens), ask(client, cand, messages, max_tokens))


def summary(per_prompt: list[dict]) -> dict:
    kls = [k for p in per_prompt for k in p["kl"]]
    unscored = sum(p.get("unscored", 0) for p in per_prompt)
    if not kls:
        return {"positions": 0, "unscored": unscored}
    xs = sorted(kls)

    def q(f: float) -> float:
        return xs[min(len(xs) - 1, int(f * (len(xs) - 1) + 0.5))]

    positions = len(kls)
    agree = sum(p["agree"] for p in per_prompt)
    worst = sorted((dict(w, prompt=p["prompt"]) for p in per_prompt for w in p["worst"]), key=lambda w: -w["kl"])[:10]
    parted = [p["differs_at"] for p in per_prompt if p["differs_at"] is not None]
    return {"positions": positions, "prompts": len(per_prompt), "mean_kl": statistics.fmean(kls), "median_kl": q(0.5),
            "p95_kl": q(0.95), "p99_kl": q(0.99), "max_kl": xs[-1], "same_top": agree / positions,
            "zero_kl": sum(1 for k in kls if k < 1e-6) / positions,
            "same_distribution": sum(p.get("equal", 0) for p in per_prompt) / positions,
            "identical": sum(1 for p in per_prompt if p["identical"]) / len(per_prompt),
            "differs_at_median": statistics.median(parted) if parted else None,
            "tokens": sum(p["tokens"] for p in per_prompt),
            "requests": sum(p["requests"] for p in per_prompt), "worst": worst, "top": TOP, "unscored": unscored,
            # the narrowest distribution either side gave: an engine capping top_logprobs makes "the same 20"
            # mean less, and is said so
            "width": min((w for p in per_prompt for w in p.get("width", [TOP])), default=TOP)}


class Load:
    """Requests kept going in the background while the candidate is measured, ``n`` at a time. ``ok`` counts the
    replies that ended while it was measured (one ending as it stops is ``late``), ``failed`` the refusals, and
    ``full_share`` the part of the measured time when all ``n`` were in flight: what says the load was really there."""

    def __init__(self, target: Target, n: int, messages: list[dict]) -> None:
        self.target, self.n, self.messages = target, n, messages
        self.ok = self.failed = self.late = 0
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._in_flight, self._since, self._full_s = 0, 0.0, 0.0
        self._t0 = self._t1 = 0.0

    def _flight(self, step: int) -> None:
        now = time.monotonic()
        with self._lock:
            if self._in_flight >= self.n and not self._stop.is_set():
                self._full_s += now - self._since
            self._in_flight += step
            self._since = now

    @property
    def full_share(self) -> float | None:
        span = (self._t1 or time.monotonic()) - self._t0
        return min(1.0, self._full_s / span) if self._t0 and span > 0 else None

    def _loop(self) -> None:
        with httpx.Client(timeout=httpx.Timeout(300.0, connect=5.0)) as c:
            while not self._stop.is_set():
                self._flight(+1)
                try:
                    r = c.post(self.target.url, headers=self.target.headers,
                               json={"model": self.target.model, "messages": self.messages, "max_tokens": 200,
                                     "temperature": 0.7, "stream": False, **self.target.extra})
                    good, why = r.status_code == 200, f"{r.status_code}: {r.text[:160]}"
                except httpx.HTTPError as exc:
                    good, why = False, str(exc)[:200]
                self._flight(-1)
                with self._lock:
                    if good and self._stop.is_set():
                        self.late += 1               # ended after the measurement: not evidence of load during it
                    elif good:
                        self.ok += 1
                    else:
                        self.failed += 1
                        self.last_error = why
                if not good:
                    self._stop.wait(1.0)        # a refusing server is not hammered

    def __enter__(self) -> Self:
        self._t0 = self._since = time.monotonic()
        for _ in range(self.n):
            t = threading.Thread(target=self._loop, daemon=True, name="kl-load")
            t.start()
            self._threads.append(t)
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    def stop(self, wait: float = 320.0) -> None:
        """Stops the background requests and waits for the ones in flight up to ``wait`` seconds; a cancelled run
        passes a short wait and leaves them to end on their own (nothing can cut a request the engine holds)."""

        self._flight(0)                     # count the full stretch up to now, then stop
        self._t1 = time.monotonic()
        self._stop.set()
        end = time.monotonic() + wait
        for t in self._threads:
            t.join(timeout=max(0.0, end - time.monotonic()))
