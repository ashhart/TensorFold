"""The Bench: usage suites run against an engine under a fixed protocol, stored in SQLite, exported as receipts.

Each prompt is sent twice when the exactness check is on: with drafts, then with `"draft": false`. Same `token_sha`
(or, from an engine without one, the same text) means the speed-up changed nothing in the answer. Figures come from
each reply's `tensorfold` block when the engine sends it, from this TensorFold Web's clock otherwise, and every result says
which. Gaps between stream events are what the client sees; an event may carry several tokens when drafts are kept.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import sqlite3
import statistics
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import psutil

from . import __version__, httpread, kl, power, quality, tooleval
from . import suites as custom_suites
from .engine import parse_metrics

SCHEMA = 2
# the protocol every run prints in its receipt
PROTOCOL = {"temperature": 0.0, "top_p": 1.0, "stream": True, "warmup_tokens": 32, "concurrency": 1,
            "thinking": "off, checked during the warm-up: the first way of saying it that the engine accepts and that "
                        "brings no reasoning is kept, and named in the receipt",
            "output_format": "asked in the prompt; response_format never sent"}
# Ways of turning thinking off, tried in this order when the engine refuses one (a 400) or thinks anyway. Each one
# says "off" explicitly: a request that simply leaves the fields out turns thinking on for hybrid models.
THINKING_OFF = (
    ("chat_template_kwargs: enable_thinking, thinking and thinking_mode off",
     {"chat_template_kwargs": {"enable_thinking": False, "thinking": False, "thinking_mode": "disabled"}}),
    ("chat_template_kwargs: enable_thinking off", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("chat_template_kwargs: thinking off", {"chat_template_kwargs": {"thinking": False}}),
    ("top level: enable_thinking off, thinking disabled", {"enable_thinking": False, "thinking": {"type": "disabled"}}),
    ("reasoning_effort none (Ollama and others)", {"reasoning_effort": "none"}),
    ("top level: think off (Ollama)", {"think": False}),
)
RUNTIME_KEYS = ("tokens_per_second", "seconds", "time_to_first_token", "prefill_seconds", "rounds", "drafted",
                "accepted", "acceptance_rate", "tokens_per_round", "token_sha", "drafts", "engine", "sampling")
MODES = {"quick": "Quick score: three prompts (English, code, French), about half a minute",
         "profile": "Usage profile: every suite, about two minutes",
         "prefill": "Prefill sweep: one prompt per context size, from 1k tokens up to the size chosen",
         "load": "Load sweep: waves of 1, 2, 4… requests at once, up to the number chosen, for total and per-request speed",
         "quality": "Quality: 228 items scored strictly (arithmetic, knowledge, instructions, corrected facts in long texts)",
         "tools": "Tool calls: tool-eval-bench's scripted conversations with mock tools, scored pass, partial or fail",
         "kl": "KL divergence: top-20 log-probabilities against a reference, position by position, where both wrote the "
               "same reply or over every position of a fixed text"}
TEXT_SUITE = {"id": "text", "name": "Fixed text"}    # a KL run on a text given: its one "prompt"
PARTIAL_BELOW = 0.98                      # readings covering less of a run: its energy figures are lower bounds
LEVELS = (1, 2, 4, 8, 16, 32)             # requests at once in a load sweep
LOAD_MIN_REQUESTS = 8                     # each level sends at least this many, in waves of its size
LOAD_TOKENS = 200                         # reply length in a load sweep
LOAD_LIMIT_S = 900.0                      # a load sweep stops before a level would pass this in all
LOAD_TTFT_LIMIT_S = 120.0                 # ...or once the slowest first tokens wait longer than this
# context sizes of the prefill sweep, kept below the served window
SIZES = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
MAX_GAP_S = 10.0                          # power readings further apart than this are not bridged
SWEEP_LIMIT_S = 900.0                     # a sweep stops before a size would take longer than this in all


def prefill_repeats(size: int) -> int:
    """Tries per size when the prefill is estimated: three up to 32k, two above (each one costs a long read)."""

    return 3 if size <= 32768 else 2


GROWTH = 2.2                              # each prefill size takes about this many times the previous one


def suites_digest(data: dict) -> str:
    """A short digest of the prompts and passages: two runs compare only when it is the same."""

    canon = json.dumps({"suites": data["suites"], "passages": data.get("passages", {})}, sort_keys=True,
                       ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def load_suites() -> dict:
    data = json.loads(resources.files("tensorfold_web.data").joinpath("suites.json").read_text(encoding="utf-8"))
    data["digest"] = suites_digest(data)
    passages = data.get("passages", {})
    for suite in data["suites"]:
        for p in suite["prompts"]:
            text = passages.get(p.get("passage", ""), "")
            p["messages"] = [{**m, "content": m["content"].replace("{passage}", text)} for m in p["messages"]]
    return data


def pick(suites: dict, mode: str, only: list[str] | None = None) -> list[tuple[dict, dict]]:
    """(suite, prompt) pairs for a mode: the quick prompts, or every prompt of the chosen suites."""

    out = []
    for s in suites["suites"]:
        if only and s["id"] not in only:
            continue
        for p in s["prompts"]:
            if mode == "profile" or p.get("quick"):
                out.append((s, p))
    return out


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# -- store ------------------------------------------------------------------------------------------------------------

class Store:
    """Runs and their results in one SQLite file; a run's receipt is built from both."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with self._db() as db:
            db.executescript("""
                create table if not exists runs (id text primary key, created real not null, finished real,
                    mode text not null, status text not null, model text, doc text not null);
                create table if not exists results (run_id text not null references runs(id) on delete cascade,
                    seq integer not null, suite text not null, prompt text not null, draft integer not null,
                    doc text not null, primary key (run_id, seq));
            """)
            # a TensorFold Web stopped mid-run leaves it running in the file: it is not anymore
            for rid, doc in db.execute("select id, doc from runs where status='running'").fetchall():
                run = {**json.loads(doc), "status": "interrupted", "progress": None}
                db.execute("update runs set status='interrupted', doc=? where id=?", (json.dumps(run), rid))

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("pragma foreign_keys = on")
        return db

    def save_run(self, run: dict) -> None:
        with self._lock, self._db() as db:
            db.execute("insert into runs (id, created, finished, mode, status, model, doc) values (?,?,?,?,?,?,?) "
                       "on conflict(id) do update set finished=excluded.finished, status=excluded.status, "
                       "doc=excluded.doc",
                       (run["id"], run["created"], run.get("finished"), run["mode"], run["status"], run.get("model"),
                        json.dumps(run)))

    def add_result(self, run_id: str, seq: int, r: dict) -> None:
        with self._lock, self._db() as db:
            db.execute("insert into results values (?,?,?,?,?,?)",
                       (run_id, seq, r["suite"], r["prompt"], 1 if r["draft"] else 0, json.dumps(r)))

    def run(self, run_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("select doc from runs where id=?", (run_id,)).fetchone()
            if not row:
                return None
            results = [json.loads(d) for (d,) in db.execute("select doc from results where run_id=? order by seq",
                                                             (run_id,))]
        return {**json.loads(row[0]), "results": results}

    def runs(self, limit: int = 50) -> list[dict]:
        with self._db() as db:
            rows = db.execute("select doc from runs order by created desc limit ?", (limit,)).fetchall()
        out = []
        for (d,) in rows:
            run = json.loads(d)
            out.append({k: run.get(k) for k in ("id", "created", "finished", "mode", "status", "model", "engine",
                                                "summary", "error")})
            if run.get("kl"):
                out[-1]["kl"] = {"mean_kl": run["kl"].get("mean_kl")}     # the list's figure for a KL run
        return out

    def delete(self, run_id: str) -> bool:
        with self._lock, self._db() as db:
            return db.execute("delete from runs where id=?", (run_id,)).rowcount > 0


# -- one request --------------------------------------------------------------------------------------------------------

@dataclass
class Reply:
    text: str = ""
    reasoning: str = ""
    finish: str | None = None
    usage: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)
    sent_at: float = 0.0
    first_at: float | None = None                         # first token of any kind, reasoning included
    answer_at: float | None = None                        # first token of the answer itself
    end_at: float = 0.0
    gaps: list[float] = field(default_factory=list)       # seconds between content events after the first
    events: int = 0
    error: str | None = None


def text_digest(text: str) -> str:
    """The whole reply's digest: the stored text stops at 4,000 characters, the comparison must not."""

    return hashlib.sha256(text.encode()).hexdigest()[:16]


def stream_once(client: httpx.Client, url: str, headers: dict, body: dict, cancel: threading.Event,
                on_text=None) -> Reply:
    """One streamed reply, read apart: a cancel ends it at once, even while the engine has not answered yet (a long
    prompt, a stalled engine), instead of after the read timeout with the run's gate held meanwhile."""

    rep = Reply(sent_at=time.perf_counter())
    opened: list[httpx.Response] = []
    try:
        httpread.until_cancelled(lambda: _stream(client, url, headers, body, cancel, rep, opened, on_text), cancel,
                                 close=lambda: opened and opened[0].close(), name="bench-reply")
    except InterruptedError:
        rep.error = "cancelled"
        rep.end_at = time.perf_counter()
    return rep


def _stream(client: httpx.Client, url: str, headers: dict, body: dict, cancel: threading.Event, rep: Reply,
            opened: list, on_text=None) -> None:
    last = None
    done = False
    try:
        with client.stream("POST", url, headers=headers, json=body) as r:
            opened.append(r)
            if r.status_code != 200:
                # at most 4 KB and 10 s of an error body, and none after a cancel, even while a read is blocked: an
                # engine that never ends one must not hold the run, nor the gate it keeps, forever
                text = httpread.error_text(r, cancel=cancel)
                try:
                    msg = json.loads(text).get("error", {}).get("message") or text
                except (ValueError, AttributeError):
                    msg = text
                rep.error = f"{r.status_code}: {str(msg)[:300]}"
                rep.end_at = time.perf_counter()
                return
            buf = ""
            for chunk in r.iter_text():
                if cancel.is_set():
                    rep.error = "cancelled"
                    break
                # normalised on what has gathered, not chunk by chunk: a CRLF can arrive split between two chunks
                buf = (buf + chunk).replace("\r\n", "\n")
                while "\n\n" in buf:
                    event, buf = buf.split("\n\n", 1)
                    data = "\n".join(line[5:].strip() for line in event.split("\n") if line.startswith("data:"))
                    if data == "[DONE]":
                        done = True
                        break                           # the reply is whole: a server keeping the line open is not waited for
                    if not data:
                        continue
                    try:
                        msg = json.loads(data)
                    except ValueError:
                        continue
                    if isinstance(msg.get("error"), (dict, str)):     # an engine failing mid-reply says so in the stream
                        err = msg["error"]
                        rep.error = "the engine stopped: " + str(err.get("message") if isinstance(err, dict) else err)[:280]
                        break
                    if msg.get("usage"):
                        rep.usage = msg["usage"]
                    if isinstance(msg.get("tensorfold"), dict):
                        rep.runtime = msg["tensorfold"]
                    if isinstance(msg.get("speculative"), dict):  # the native engine's draft counters sit beside it
                        rep.runtime = {**rep.runtime, **msg["speculative"]}
                    for choice in msg.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = delta.get("content") or ""
                        think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if piece or think:
                            now = time.perf_counter()
                            rep.first_at = rep.first_at or now
                            if last is not None:
                                rep.gaps.append(now - last)
                            last = now
                            rep.events += 1
                            if piece and rep.answer_at is None:
                                rep.answer_at = now
                            rep.text += piece
                            rep.reasoning += think
                            if on_text is not None:          # a page watching the reply being written
                                on_text(rep)
                        if choice.get("finish_reason"):
                            rep.finish = choice["finish_reason"]
                if rep.error or done:                       # [DONE] ends the reading: the response is closed here
                    break
    except httpx.HTTPError as exc:
        rep.error = f"{type(exc).__name__}: {exc}"[:300]
    if rep.error is None and rep.finish is None and not done:
        # a stream cut before its end: the reply is not whole, so it counts neither for speed nor for exactness
        rep.error = "the reply ended before the engine finished it (no finish_reason, no [DONE])"
    rep.end_at = time.perf_counter()


def drafts_off_confirmed(runtime: dict) -> bool:
    """Whether the engine's block says drafts were off, and says nothing against it: a token digest alone is no
    evidence, and "drafts: true" beside "drafted: 0", or "drafts: false" beside tokens drafted, is a contradiction."""

    drafts, drafted = runtime.get("drafts"), runtime.get("drafted")
    if drafts is True or (isinstance(drafted, (int, float)) and drafted > 0):
        return False
    return drafts is False or drafted == 0


def drafts_on_confirmed(runtime: dict) -> bool:
    """Whether the engine's block says drafting ran: tokens drafted when it counts them, else drafts on. Without it,
    two replies alike prove nothing about the speed-up, since neither may have used it."""

    drafts, drafted = runtime.get("drafts"), runtime.get("drafted")
    if drafts is False:
        return False
    if isinstance(drafted, (int, float)):
        return drafted > 0
    return drafts is True


def thought(rep: Reply) -> bool:
    """Whether a reply carried reasoning, as its own field or inline in a think tag."""

    return bool(rep.reasoning.strip()) or "<think>" in rep.text or "</think>" in rep.text


def measure(rep: Reply) -> dict:
    """The figures of one reply: the engine's own when its block has them, this TensorFold Web's clock otherwise."""

    rt = {k: rep.runtime[k] for k in RUNTIME_KEYS if k in rep.runtime}
    tokens = rep.usage.get("completion_tokens")
    ttft_client = (rep.first_at - rep.sent_at) if rep.first_at else None
    decode_client = (rep.end_at - rep.first_at) if rep.first_at else None
    # a stream event may carry several tokens (drafts kept together): without the engine's count there is no token
    # figure, only events, kept apart; a speed per token then comes from the engine's block or not at all
    tps_client = tokens / decode_client if tokens and decode_client and decode_client > 0 else None
    from_engine = rt.get("tokens_per_second") is not None
    return {
        "tokens": tokens,
        "tokens_counted_by": "engine" if tokens is not None else None,
        "prompt_tokens": rep.usage.get("prompt_tokens"),
        "cached_tokens": (rep.usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "reasoning_tokens": (rep.usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "tps": rt["tokens_per_second"] if from_engine else tps_client,
        "ttft": rt.get("time_to_first_token", ttft_client),
        # with reasoning, the answer starts later than the first token; TensorFold Web's clock, as the engine has no field
        "ttft_answer": (rep.answer_at - rep.sent_at) if rep.answer_at and rep.reasoning else None,
        "prefill_s": rt.get("prefill_seconds"),
        "timing": "engine" if from_engine else "studio",
        "tps_client": tps_client, "ttft_client": ttft_client,
        "gap_p50": percentile(rep.gaps, 0.5), "gap_p95": percentile(rep.gaps, 0.95), "events": rep.events,
        "tokens_per_event": (tokens / rep.events) if tokens and rep.events else None,
        "wall_s": rep.end_at - rep.sent_at, "finish": rep.finish, "sha": rt.get("token_sha"), "runtime": rt,
    }


# -- memory, sampled while a run goes ------------------------------------------------------------------------------

class Sampler:
    """Once a second: the engine's footprint, the machine's used memory, and power and temperatures where the
    platform reports them. Energy counts only the measured part of a run, from `measuring()` on."""

    def __init__(self, read_engine: Callable[[], int | None], every: float = 1.0,
                 source: power.Source | None = None) -> None:
        self.read_engine, self.every, self.source = read_engine, every, source
        self.engine_peak: int | None = None
        self.system_peak = 0
        self.samples = 0
        self.series: list[list] = []        # [seconds, watts, cpu °C, gpu °C, engine bytes, throttled]
        self.reasons: set[str] = set()       # why the machine slowed down, while measuring
        self.idle_watts: float | None = None
        self.measured_from: float | None = None
        self.t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="bench-sampler", daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.every)

    def sample(self) -> None:
        vm = psutil.virtual_memory()
        self.system_peak = max(self.system_peak, vm.total - vm.available)
        try:
            e = self.read_engine()
        except Exception:  # noqa: BLE001 - a missed reading is not a failed run
            e = None
        if e:
            self.engine_peak = max(self.engine_peak or 0, e)
        r = None
        if self.source is not None:
            try:
                r = self.source.read()
            except Exception:  # noqa: BLE001
                r = None
        t = round(time.monotonic() - self.t0, 2)
        self.series.append([t, r.watts if r else None, r.cpu_c if r else None, r.gpu_c if r else None, e,
                            r.throttled if r else None])
        if r and r.reasons and self.measured_from is not None and t >= self.measured_from:
            self.reasons.update(r.reasons)
        self.samples += 1

    def idle(self, seconds: float = 3.0) -> None:
        """The machine's power with the model loaded and nothing asked of it, as the baseline."""

        if self.source is None:
            return
        self._stop.wait(1.0)                       # the machine settles after the last request or the model load
        start = time.monotonic() - self.t0
        self._stop.wait(seconds)
        watts = [w for t, w, *_ in self.series if t >= start and w is not None]
        self.idle_watts = statistics.median(watts) if watts else None

    def measuring(self) -> None:
        self.measured_from = round(time.monotonic() - self.t0, 2)   # rounded as the readings' times are
        if self._thread.is_alive():
            self.sample()                              # a reading at the start of the measured work, not up to 1 s later

    def report(self, output_tokens: int | None) -> dict | None:
        if self.source is None:
            return None
        start = self.measured_from if self.measured_from is not None else 0.0
        live = self._thread.is_alive()
        if live and self.measured_from is not None:
            self.sample()                              # and one at its end: the energy covers the whole measured work
        rows = sorted((row for row in self.series if row[0] >= start), key=lambda row: row[0])
        pts = [(row[0], row[1]) for row in rows if row[1] is not None]
        # energy over the intervals between two readings close enough to trust; a longer gap is left out, not guessed
        steps = [(t1, w1, t2, w2) for (t1, w1), (t2, w2) in itertools.pairwise(pts) if t2 - t1 <= MAX_GAP_S]
        energy = sum((t2 - t1) * (w1 + w2) / 2 for t1, w1, t2, w2 in steps)
        span = sum(t2 - t1 for t1, _, t2, _ in steps)
        # the whole measured period, from the start of the measured work to now (the work just ended): the readings
        # cover part of it, and the energy of the part they miss is not in the figure
        end = max(rows[-1][0] if rows else start, time.monotonic() - self.t0 if live else 0.0)
        window = max(0.0, end - start) if self.measured_from is not None else \
            (rows[-1][0] - rows[0][0] if len(rows) > 1 else 0.0)
        coverage = min(1.0, span / window) if window else None
        # an idle reading above the run's average (taken while a load was settling) gives no energy below zero
        above = max(0.0, energy - (self.idle_watts or 0.0) * span) if self.idle_watts is not None else None
        flags = [row[5] for row in rows if len(row) > 5 and row[5] is not None]
        # how long the machine was slowed: each throttled reading until the next one (or the end of the measured
        # work), a reading's interval rather than the sampling period, since boundary readings come between
        stamps = [row[0] for row in rows] + [end]
        throttled_s = round(sum(max(0.0, min(stamps[i + 1], end) - row[0]) for i, row in enumerate(rows)
                                if len(row) > 5 and row[5]), 2)
        heat = rows or self.series                       # the measured part; a run shorter than a sample, the whole
        temps = [x for row in heat for x in row[2:4] if x is not None]
        cpu = [row[2] for row in heat if row[2] is not None]
        gpu = [row[3] for row in heat if row[3] is not None]
        step = max(1, len(self.series) // 300)
        return {**self.source.describe(), "estimate": not self.source.whole_machine,
                "idle_watts": self.idle_watts, "avg_watts": energy / span if span else None,
                "peak_watts": max((w for _, w in pts), default=None), "measured_s": span, "window_s": window,
                "coverage": coverage, "partial": coverage is not None and coverage < PARTIAL_BELOW,
                "throttle": {"reported": bool(flags), "throttled": any(flags) if flags else None,
                             "seconds": throttled_s if flags else None,
                             "reasons": sorted(self.reasons)},
                "energy_j": energy if span else None, "energy_above_idle_j": above if span else None,
                "output_tokens": output_tokens,
                "j_per_token": energy / output_tokens if span and output_tokens else None,
                "j_per_token_above_idle": above / output_tokens if span and output_tokens and above is not None else None,
                "peak_cpu_c": max(cpu) if cpu else None, "peak_gpu_c": max(gpu) if gpu else None,
                "peak_c": max(temps) if temps else None, "series": self.series[::step]}

    def __enter__(self) -> Sampler:  # noqa: PYI034
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=3)
        if self.source is not None:
            self.source.close()


def energy_cost(hw: dict, price_per_kwh: float | None, currency: str) -> dict | None:
    """What the measured energy cost at the price set in Settings, for the run and per million tokens written."""

    if not price_per_kwh or hw.get("energy_j") is None:
        return None
    per_j = price_per_kwh / 3.6e6
    out = {"price_per_kwh": price_per_kwh, "currency": currency, "run": hw["energy_j"] * per_j,
           "per_million_tokens": hw["j_per_token"] * 1e6 * per_j if hw.get("j_per_token") is not None else None}
    if hw.get("j_per_token_above_idle") is not None:
        out["per_million_tokens_above_idle"] = hw["j_per_token_above_idle"] * 1e6 * per_j
    return out


def engine_memory(client: httpx.Client, base: str, headers: dict) -> int | None:
    """Bytes the engine holds: `/health` memory (peak or active) when it reports it, else `/metrics`' footprint."""

    try:
        r = client.get(base + "/health", headers=headers)
        mem = (r.json() or {}).get("memory") or {} if r.status_code == 200 else {}
        for k in ("peak", "active", "footprint"):
            if isinstance(mem.get(k), (int, float)) and mem[k] > 0:
                return int(mem[k])
        r = client.get(base + "/metrics", headers=headers)
        if r.status_code == 200:
            v = parse_metrics(r.text).get("process_footprint_bytes")
            if v:
                return int(v)
        r = client.get(base + "/api/ps", headers=headers)          # Ollama: the loaded models and their size
        if r.status_code == 200:
            sizes = [m.get("size") for m in (r.json() or {}).get("models") or [] if isinstance(m.get("size"), int)]
            return sum(sizes) or None
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    return None


# -- a run ----------------------------------------------------------------------------------------------------------------

def summarize(results: list[dict]) -> dict:
    """Per suite and overall: median speed with and without drafts, gain, first token, gaps, exactness."""

    def med(xs: list) -> float | None:
        xs = [x for x in xs if x is not None]
        return statistics.median(xs) if xs else None

    by_suite: dict[str, list[dict]] = {}
    for r in results:
        by_suite.setdefault(r["suite"], []).append(r)

    def block(rows: list[dict]) -> dict:
        on = [r for r in rows if r["draft"] and not r.get("error")]
        off = [r for r in rows if not r["draft"] and not r.get("error")]
        pairs = [(a, b) for a in on for b in off if a["prompt"] == b["prompt"] and a["suite"] == b["suite"]
                 and b.get("same") is not None]          # a pair the engine could not confirm is not a checked one
        exact = [p for p in pairs if p[1].get("same") is True]    # the check sits on the drafts-off reply
        tps_on, tps_off = med([r["m"]["tps"] for r in on]), med([r["m"]["tps"] for r in off])
        # a reply the engine did not count leaves the total unknown, not short of it
        counted = all(r["m"]["tokens"] is not None for r in on)
        return {"prompts": len({(r["suite"], r["prompt"]) for r in rows}),
                "tokens": sum(r["m"]["tokens"] for r in on) if counted else None,
                "tps": tps_on, "tps_no_drafts": tps_off,
                "gain": (tps_on / tps_off - 1) if tps_on and tps_off else None,
                "ttft": med([r["m"]["ttft"] for r in on]),
                "ttft_answer": med([r["m"].get("ttft_answer") for r in on]), "prefill_s": med([r["m"]["prefill_s"] for r in on]),
                "gap_p50": med([r["m"]["gap_p50"] for r in on]), "gap_p95": med([r["m"]["gap_p95"] for r in on]),
                "acceptance": med([r["m"]["runtime"].get("acceptance_rate") for r in on]),
                "checked": len(pairs), "exact": len(exact),
                "errors": sum(1 for r in rows if r.get("error"))}

    out = {"overall": block(results), "suites": {k: block(v) for k, v in by_suite.items()}}
    graded = [r for r in results if r.get("passed") is not None]
    unscored = sum(1 for r in results if "passed" in r and r["passed"] is None)
    if graded or unscored:
        by: dict[str, list[dict]] = {}
        for r in graded:
            by.setdefault(r.get("category") or r["suite"], []).append(r)
        out["quality"] = {"overall": {"items": len(graded), "passed": sum(1 for r in graded if r["passed"]),
                                      "unscored": unscored},
                          "categories": {k: {"items": len(v), "passed": sum(1 for r in v if r["passed"])}
                                         for k, v in by.items()}}
    load = [r for r in results if "concurrency" in r]
    if load:
        levels = []
        for c in sorted({r["concurrency"] for r in load}):
            rows = [r for r in load if r["concurrency"] == c]
            ok = [r for r in rows if not r.get("error")]
            waves = {}
            for r in ok:
                waves.setdefault(r["wave"], []).append(r)
            # a wave's total only when the engine counted every reply in it: stream events are not tokens
            totals = [sum(x["m"]["tokens"] for x in w) / w[0]["wave_s"] for w in waves.values()
                      if w[0]["wave_s"] and all(x["m"]["tokens"] is not None for x in w)]
            waits = [r["m"].get("ttft_client") for r in ok if r["m"].get("ttft_client") is not None]
            levels.append({"concurrency": c, "requests": len(rows), "errors": len(rows) - len(ok),
                           "total_tps": med(totals), "per_request_tps": med([r["m"]["tps"] for r in ok]),
                           "ttft_p50": percentile(waits, 0.5), "ttft_p95": percentile(waits, 0.95)})
        out["load"] = levels
    sweep = [r for r in results if "size" in r]
    if sweep:
        out["sweep"] = [{"size": r["size"], "prompt_tokens": r["m"]["prompt_tokens"], "cached": r["m"]["cached_tokens"],
                         "prefill_s": r["m"]["prefill_s"], "ttft": r["m"]["ttft"], "prefill_tps": r["m"].get("prefill_tps"),
                         "estimated": bool(r["m"].get("prefill_estimated")), "weak": bool(r["m"].get("prefill_weak")),
                         "error": r.get("error")} for r in sweep]
    return out


# modes whose measured energy goes to the tokens they count: the others get no energy per token
# the share of a KL measurement during which all background requests must be in flight to call it "under load"
LOAD_SHARE = 0.5
PER_TOKEN_MODES = ("quick", "profile", "load", "quality")


def on_this_machine(host: str) -> bool:
    """The address is one of this machine's own: loopback, or an address it can bind (its LAN or tailnet IP)."""

    import socket

    if host in ("127.0.0.1", "localhost", "::1", ""):
        return True
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    for family, _type, _proto, _name, addr in infos:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.bind((addr[0], 0))          # only an address of this machine can be bound
            return True
        except OSError:
            continue
    return False


def machine_facts() -> dict:
    """What the receipt says of the machine; no host name, so a receipt can be shared as it is."""

    from .machine import detect

    m = detect().to_dict()
    return {k: m.get(k) for k in ("os", "os_version", "arch", "chip", "ram_bytes", "unified_memory", "gpus",
                                  "backend", "tensorfold_version", "gpu_ceiling_bytes")}


class Bench:
    """Runs one at a time in a thread; the page polls the run while it goes."""

    def __init__(self, store: Store, target: Callable[[], tuple[str, str | None, str | None]]) -> None:
        self.store = store
        self.target = target                 # () -> (engine base URL, API key, served model)
        self.current: dict | None = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._starting = threading.Lock()   # two clicks at once start one run
        self.suites = load_suites()
        self.custom = custom_suites.CustomSuites(store.path)     # the user's own suites, in the same file
        self.power = True                    # tests turn it off: the machine running them is not measured
        self.price: Callable[[], tuple[float | None, str]] = lambda: (None, "EUR")   # per kWh, from Settings

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, mode: str = "quick", *, suites: list[str] | None = None, exactness: bool = True,
              engine_name: str | None = None, max_context: int | None = None,
              target: tuple[str, str | None, str | None] | None = None, drafts: bool = True,
              tags: dict | None = None, max_concurrency: int | None = None, full: bool = False,
              reference: tuple[str, str | None, str] | None = None, kl_load: int = 0,
              kl_text: str | None = None) -> dict:
        """``target`` is another engine's (base URL, key, model, and headers of its own, such as another TensorFold Web's
        token, as a fourth part); without it, the engine this TensorFold Web serves.
        ``drafts`` False sends every request with `"draft": false` (no exactness check then: there is nothing to
        compare against); ``tags`` are kept on the run, such as the configuration comparison it belongs to.
        A KL run compares with ``reference`` (an engine's base URL, key, model), or without one with the same engine
        and drafts off; ``kl_load`` keeps that many requests going while the candidate answers. ``kl_text`` compares
        both on that text, every position of it scored (`prompt_logprobs`), instead of on their replies to the suites."""

        with self._starting:
            return self._start(mode, suites, exactness, engine_name, max_context, target, drafts, tags,
                               max_concurrency, full, kl_load, reference, kl_text)

    def _start(self, mode: str, suites: list[str] | None, exactness: bool, engine_name: str | None,
               max_context: int | None, target: tuple[str, str | None, str | None] | None = None,
               drafts: bool = True, tags: dict | None = None, max_concurrency: int | None = None,
               full: bool = False, kl_load: int = 0, reference: tuple[str, str | None, str] | None = None,
               kl_text: str | None = None) -> dict:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if self.busy():
            raise RuntimeError("a run is already going; cancel it or wait")
        if kl_text is not None and (mode != "kl" or not isinstance(kl_text, str) or not kl_text.strip()
                                    or len(kl_text) > kl.TEXT_MAX):
            raise ValueError(f"a fixed text is for a KL run, 1 to {kl.TEXT_MAX:,} characters")
        self._kl_reference = reference          # only once no run is going: a refused start leaves the running one's
        # the built-in suites, then the user's; a profile without a choice runs the built-in ones only
        data = {**self.suites, "suites": self.suites["suites"] + [custom_suites.as_bench(c) for c in self.custom.list()]}
        tools_exe = None
        if mode == "tools":
            tools_exe = tooleval.find()
            if tools_exe is None:
                raise ValueError(f"tool-eval-bench is not installed here: uv tool install tool-eval-bench=={tooleval.VERSION}")
            suites = None
        qdata: dict = {}
        if mode == "quality":
            qdata = quality.load()
            cats = list(qdata["categories"])
            if suites and not set(suites) <= set(cats):
                raise ValueError(f"unknown quality categories: {', '.join(sorted(set(suites) - set(cats)))}")
        if kl_text is not None:
            suites = None                        # the text replaces the suites
        elif mode in ("profile", "load", "kl") and not suites:
            suites = [x["id"] for x in self.suites["suites"]]
        known = {s["id"] for s in data["suites"]}
        if suites and mode != "quality" and not set(suites) <= known:
            raise ValueError(f"unknown suites: {', '.join(sorted(set(suites) - known))}")
        base, key, model, *more = target or self.target()
        self._target_headers = more[0] if more else {}       # kept here, never with the run: it may hold a token
        # the drafts-off check is TensorFold's own (`"draft": false`); another engine is compared by text instead
        # the served engine is whatever Settings point at: its name says whether it is TensorFold
        tensorfold = engine_name.startswith("TensorFold") if engine_name else target is None
        dropped = exactness and mode in ("quick", "profile") and not tensorfold
        exactness = exactness and mode in ("quick", "profile") and tensorfold and drafts
        if not model:
            raise RuntimeError("no model is being served; start one from Models")
        work = (quality.items(qdata, self.suites["passages"]["water"] + "\n\n", suites) if mode == "quality"
                else [(TEXT_SUITE, {"id": "text", "text": kl_text})] if kl_text is not None
                else [] if mode in ("prefill", "tools") else pick(data, "profile" if mode in ("load", "kl") else mode, suites))
        order = [x["id"] for x in data["suites"]] + [TEXT_SUITE["id"]]
        chosen = (sorted({w["category"] for w in work}, key=list(qdata["categories"]).index) if mode == "quality"
                  else sorted({s["id"] for s, _ in work}, key=order.index))
        version = {"version": self.suites.get("version"), "digest": self.suites["digest"]}
        custom = {s["id"]: s["digest"] for s in data["suites"] if s.get("custom") and s["id"] in chosen}
        if custom:
            version["custom"] = custom
        if mode == "quality":
            version = {"quality": qdata["version"], "digest": qdata["digest"]}
        if kl_text is not None:
            version = {"kl_text": len(kl_text), "digest": text_digest(kl_text)}
        if mode == "tools":
            tv = tooleval.version(tools_exe or "")
            self._tools_exe = tools_exe          # kept here, not in the run: its path holds the user's name
            # two tool runs compare only on the same tool version and scenario set
            version = {"tool_eval_bench": tv, "digest": f"teb-{tv}-{'standard' if full else 'short'}-s{tooleval.SEED}"}
        sizes = [n for n in SIZES if n <= (max_context or SIZES[-1])]
        levels = [n for n in LEVELS if n <= (max_concurrency or 16)]
        if mode == "prefill" and not sizes:
            raise ValueError(f"the sweep starts at {SIZES[0]} tokens")
        if mode not in ("prefill", "tools") and not work:
            raise ValueError("nothing to run: the chosen suites have no prompts for this mode")
        # an engine on another machine: this machine's chip, memory and power say nothing about it
        # judged by the address measured, whether it came from the Bench's target or from Settings
        elsewhere = not on_this_machine(urlsplit(base).hostname or "")
        run = {"schema": SCHEMA, "id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6], "created": time.time(),
               "finished": None, "mode": mode, "status": "running", "model": model, "engine": engine_name,
               "exactness": exactness, "suites": chosen,
               "target": {"kind": "address" if target else "served", "url": base}, "drafts": drafts,
               **({"tags": tags} if tags else {}),
               "suite_names": ({c: qdata["categories"][c]["name"] for c in chosen} if mode == "quality" else
                               {TEXT_SUITE["id"]: TEXT_SUITE["name"]} if kl_text is not None else
                               {s["id"]: s["name"] for s in data["suites"] if s["id"] in chosen}),
               "protocol": PROTOCOL, "suites_version": version,
               "thinking": None, "studio": __version__, "python": platform.python_version(),
               **({"machine": None, "driver": machine_facts(), "elsewhere": True} if elsewhere else
                  {"machine": machine_facts()}),
               "progress": {"done": 0, "total": len(sizes) if mode == "prefill" else len(levels) if mode == "load"
                            else len(work) if mode == "quality" else (69 if full else 15) if mode == "tools"
                            else len(work) * (2 if exactness else 1),
                            "current": None, "phase": "starting", "measuring_since": None, "eta_s": None, "eta_at": None,
                            "stopping": False},
               "max_context": max_context if mode == "prefill" else None,
               "max_concurrency": levels[-1] if mode == "load" else None,
               **({"tools_full": full} if mode == "tools" else {}),
               **({"kl_reference": ({"kind": "address", "url": self._kl_reference[0], "model": self._kl_reference[2]}
                                    if self._kl_reference else {"kind": "drafts-off"}), "kl_load": kl_load,
                   "kl_text": kl_text}
                  if mode == "kl" else {}),
               "memory": None, "hardware": None, "summary": None, "error": None,
               "window": None, "prefill_overhead_s": None, "prefill_repeated": False, "tools": None,
               "kl_load_done": None, "kl_load_verified": None, "kl": None,
               "notes": ["no drafts-off check: it is TensorFold's own; compare the replies with a TensorFold run instead"]
               if dropped else []}
        self.store.save_run(run)
        self._cancel.clear()
        self.current = run
        self._thread = threading.Thread(target=self._go, args=(run, work, base, key, levels if mode == "load" else sizes),
                                        name="bench", daemon=True)
        self._thread.start()
        return run

    def cancel(self) -> bool:
        if not self.busy():
            return False
        self._cancel.set()
        if self.current:
            self.current["progress"]["stopping"] = True
        return True

    def _go(self, run: dict, work: list[tuple[dict, dict]], base: str, key: str | None, sizes: list[int]) -> None:
        headers = {**({"Authorization": f"Bearer {key}"} if key else {}), **getattr(self, "_target_headers", {})}
        results: list[dict] = []
        try:
            try:
                source, why = (power.detect() if self.power and run["mode"] != "kl" and not run.get("elsewhere") else
                               (None, "not read for a KL run, which compares distributions" if run["mode"] == "kl" else
                                "the engine runs on another machine: its power is not read from here"
                                if run.get("elsewhere") else "power not read"))
            except Exception as exc:  # noqa: BLE001 - a power source that cannot open: the run goes on without it
                source, why = None, f"the power source failed to open ({str(exc)[:120]})"
            if source is None:
                run["notes"].append(f"no power figures: {why}")
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0)) as client, \
                    httpx.Client(timeout=3.0) as probe, \
                    Sampler(lambda: engine_memory(probe, base, headers), source=source) as sampler:
                try:
                    if source is not None:
                        run["progress"]["current"], run["progress"]["phase"] = "idle power", "idle"
                        self.store.save_run(run)
                        sampler.idle()
                    off = self._warm(run, client, base + "/v1/chat/completions", headers)
                    sampler.measuring()
                    run["progress"]["phase"], run["progress"]["measuring_since"] = "measuring", time.time()
                    if run["mode"] == "prefill":
                        self._prefill(run, sizes, client, base, headers, results, off)
                    elif run["mode"] == "tools":
                        self._tools(run, base, key, results)
                    elif run["mode"] == "kl":
                        self._kl(run, work, base, headers, results, off)
                    elif run["mode"] == "quality":
                        self._quality(run, work, client, base + "/v1/chat/completions", headers, results, off)
                    elif run["mode"] == "load":
                        self._load(run, sizes, work, client, base + "/v1/chat/completions", headers, results, off)
                    else:
                        self._sweep(run, work, client, base + "/v1/chat/completions", headers, results, off)
                    run["status"] = "done"
                finally:
                    run["memory"] = {"engine_peak_bytes": sampler.engine_peak,
                                     "system_peak_bytes": None if run.get("elsewhere") else sampler.system_peak,
                                     "samples": sampler.samples}
                    written = [r["m"]["tokens"] for r in results if not r.get("error")]
                    # energy per token only over replies the engine counted, all of them: a partial count would charge
                    # the whole run's energy to some of its tokens
                    run["hardware"] = sampler.report(sum(written) if written and None not in written else None)
                    if run["hardware"] and run["mode"] not in PER_TOKEN_MODES:
                        # the energy went to work not counted in written tokens (long prompts read, repeats, probes):
                        # a figure per token would charge it all to the few tokens written
                        run["hardware"].update(j_per_token=None, j_per_token_above_idle=None,
                                               per_token_note="not given for this mode: most of its energy goes to "
                                                              "reading prompts, not to the tokens it writes")
                    if run["hardware"] and run["mode"] == "prefill":
                        hw = run["hardware"]
                        read = sum(r["m"].get("prompt_tokens") or 0 for r in results if not r.get("error"))
                        # per prompt token read, only when every request measured is one the sweep counts
                        ok = bool(read) and not run.get("prefill_repeated")
                        hw["j_per_1k_prompt"] = hw["energy_j"] / read * 1000 if ok and hw.get("energy_j") else None
                        hw["j_per_1k_prompt_above_idle"] = (hw["energy_above_idle_j"] / read * 1000 if ok and
                                                            hw.get("energy_above_idle_j") is not None else None)
                        if run.get("prefill_repeated"):
                            hw["per_token_note"] = ("not given: this engine gives no prefill time, so each size was "
                                                    "timed several times and the energy covers requests not counted")
                    if run["hardware"]:
                        run["hardware"]["cost"] = energy_cost(run["hardware"], *self.price())
        except InterruptedError:
            run["status"] = "cancelled"
        except Exception as exc:  # noqa: BLE001 - whatever stopped it, the run is kept and says why
            run["status"], run["error"] = "failed", plain_error(str(exc))
            run["error_detail"] = str(exc)[:500]
        try:
            run["summary"] = summarize(results) if results else None
        except Exception as exc:  # noqa: BLE001 - a summary that cannot be made leaves a failed run, not a live one
            run["status"], run["error"], run["summary"] = "failed", f"the summary failed: {str(exc)[:200]}", None
        run["progress"].update(current=None, phase=run["status"], eta_s=None)
        run["finished"] = time.time()
        self.store.save_run(run)

    def _prefill(self, run: dict, sizes: list[int], client: httpx.Client, base: str, headers: dict,
                 results: list[dict], off: dict) -> None:
        """One prompt per size, each opening with its own nonce so the prompt cache cannot serve the next one."""

        url = base + "/v1/chat/completions"
        filler = self.suites["passages"]["water"] + "\n\n"
        per_filler, window = None, None
        try:
            r = client.post(base + "/tokenize", json={"model": run["model"], "prompt": filler}, headers=headers)
            if r.status_code == 200:
                body = r.json()
                per_filler = body.get("count") or None
                window = body.get("max_model_len") if isinstance(body.get("max_model_len"), int) else None
        except (httpx.HTTPError, ValueError):
            pass
        if not per_filler:
            per_filler = max(1, len(filler) // 4)
            run["notes"].append("the engine has no /tokenize; prompt sizes estimated at four characters a token")
        if window:
            run["window"] = window
            kept = [n for n in sizes if n <= window - 256]
            if len(kept) < len(sizes):
                run["notes"].append(f"sizes above the served window ({window:,} tokens) left out")
            sizes = kept
        run["progress"]["total"] = len(sizes)
        question = "\n\nIn one word: which river does the town draw its water from?"
        spent = 0.0
        overhead: float | None = None
        for size in sizes:
            if self._cancel.is_set():
                raise InterruptedError
            if results and spent + GROWTH * results[-1]["m"]["wall_s"] > SWEEP_LIMIT_S:
                run["notes"].append(f"stopped before {size:,} tokens: the sweep would pass {int(SWEEP_LIMIT_S)} s")
                break
            run["progress"]["current"] = f"{size:,} tokens"
            self.store.save_run(run)

            def once(size: int = size) -> tuple[Reply, dict]:
                nonce = uuid.uuid4().hex             # a new opening each time, so no repeat is served from the cache
                reps = max(1, (size - 64) // per_filler)
                content = (f"Reference {nonce}. Read this report, then answer the question after it.\n\n"
                           + filler * reps + question)
                body = {"model": run["model"], "messages": [{"role": "user", "content": content}], "max_tokens": 8,
                        "temperature": 0.0, "top_p": 1.0, "stream": True, "stream_options": {"include_usage": True},
                        **off, **({} if run.get("drafts", True) else {"draft": False})}
                got = stream_once(client, url, headers, body, self._cancel)
                if got.error == "cancelled":
                    raise InterruptedError
                return got, measure(got)

            rep, m = once()
            if not rep.error and m["prefill_s"] is None:
                # no prefill time from this engine: the first token less a measured overhead, median of a few tries
                if overhead is None:
                    overhead = self._overhead(run, client, url, headers, off)
                tries = [(rep, m)] + [once() for _ in range(prefill_repeats(size) - 1)]
                extra_wall = sum(t[1]["wall_s"] for t in tries)
                tries = [t for t in tries if not t[0].error and t[1]["ttft"]] or tries
                rep, m = sorted(tries, key=lambda t: t[1]["ttft"] or 0)[len(tries) // 2]
                m["repeats"] = len(tries)
                m["tries_wall_s"] = extra_wall                  # what the size really took, every try counted
                run["prefill_repeated"] = True       # energy then went to repeats and probes the sweep does not count
            if m["prefill_s"]:
                took = m["prefill_s"]
            elif m["ttft"] and overhead is not None:
                took = max(m["ttft"] - overhead, 1e-3)
                m["prefill_estimated"] = True
                m["prefill_weak"] = overhead > 0.3 * m["ttft"]       # the overhead is most of what was timed
            else:
                took = m["ttft"]
            m["prefill_tps"] = (m["prompt_tokens"] - (m["cached_tokens"] or 0)) / took if m["prompt_tokens"] and took else None
            if m["prompt_tokens"] and m["prompt_tokens"] < 0.8 * (size - 64) and not rep.error:
                run["notes"].append(f"{size:,} tokens: the engine read only {m['prompt_tokens']:,}; its context is "
                                    "smaller than the prompt and it cut it (Ollama: OLLAMA_CONTEXT_LENGTH)")
            r = {"suite": "prefill", "prompt": str(size), "size": size, "draft": True, "m": m, "error": rep.error,
                 "text": rep.text[:200]}
            results.append(r)
            self.store.add_result(run["id"], len(results) - 1, r)
            run["progress"]["done"] = len(results)
            run["summary"] = summarize(results)
            took_all = m.get("tries_wall_s") or m["wall_s"]
            spent += took_all
            run["progress"]["eta_s"] = sweep_eta(took_all, len(sizes) - len(results), spent)
            run["progress"]["eta_at"] = time.time()
            if rep.error:
                run["notes"].append(f"stopped at {size:,} tokens: {rep.error}")
                break
            if m["cached_tokens"]:
                run["notes"].append(f"{size:,} tokens: {m['cached_tokens']:,} came from the prompt cache")

    def _tools(self, run: dict, base: str, key: str | None, results: list[dict]) -> None:
        """tool-eval-bench on the same engine; its progress events drive the run's, its scenarios become results."""

        def on_event(e: dict) -> None:
            if e.get("event") == "scenario_start":
                run["progress"]["current"] = f"{e.get('scenario_id')}: {e.get('title', '')}"
                if isinstance(e.get("total"), int):
                    run["progress"]["total"] = e["total"]
            elif e.get("event") == "scenario_result":
                # no time left is given: the scenarios grow from one call to long multi-turn chains, so the first
                # ones say nothing of the last (on the M6 the estimate said 7 minutes for a 10-minute run)
                run["progress"]["done"] = (e.get("index") or 0) + 1

        t = tooleval.Run(self._tools_exe, base, key, run["model"], full=run.get("tools_full", False),
                         on_event=on_event)
        try:
            doc = t.wait(self._cancel)
        finally:
            t.close()
        run["tools"] = tooleval.summary(doc)
        run["thinking"] = {"sent": "--no-think (tool-eval-bench)", "off": True, "refused": []}
        for r in tooleval.results(doc, t.categories):
            results.append(r)
            self.store.add_result(run["id"], len(results) - 1, r)
        run["progress"]["done"] = len(results)
        run["summary"] = summarize(results)

    def _kl(self, run: dict, work: list[tuple[dict, dict]], base: str, headers: dict, results: list[dict],
            off: dict) -> None:
        """The reference's replies first, then the candidate's (under load if asked), compared prompt by prompt."""

        url = base + "/v1/chat/completions"
        cand = kl.Target(url, headers, run["model"], dict(off))
        ref_at = getattr(self, "_kl_reference", None)
        if ref_at:
            r_base, r_key, r_model, *more = ref_at       # a fourth part: headers, such as another TensorFold Web's token
            ref = kl.Target(r_base.rstrip("/") + "/v1/chat/completions",
                            {**({"Authorization": f"Bearer {r_key}"} if r_key else {}), **(more[0] if more else {})},
                            r_model, dict(off))
        else:
            ref = kl.Target(url, headers, run["model"], {**off, "draft": False})
        run["progress"]["total"] = 2 * len(work)
        replies: list[list[kl.Step]] = []
        with httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0)) as c:
            for side, target in (("reference", ref), ("candidate", cand)):
                load = (kl.Load(cand, run.get("kl_load") or 0, [{"role": "user", "content": "Write a long story about "
                        "a lighthouse keeper."}]) if side == "candidate" and run.get("kl_load") else None)
                if load:
                    load.__enter__()
                try:
                    for i, (suite, prompt) in enumerate(work):
                        if self._cancel.is_set():
                            raise InterruptedError
                        run["progress"]["current"] = f"{side}: {suite['name']}: {prompt['id']}"
                        self.store.save_run(run)
                        try:
                            steps = (kl.score(c, target, prompt["text"], cancel=self._cancel) if "text" in prompt else
                                     kl.ask(c, target, prompt["messages"], min(prompt.get("max_tokens", 300), kl.MAX_TOKENS),
                                            cancel=self._cancel))
                        except RuntimeError as exc:
                            raise RuntimeError(f"the {side} engine: {exc}") from exc
                        if side == "reference":
                            replies.append(steps)
                            if ref_at is None:            # the same engine asked for drafts off: did it say so?
                                confirmed = drafts_off_confirmed(getattr(steps, "runtime", {}) or {})
                                ref = run["kl_reference"]
                                ref["drafts_off"] = confirmed if ref.get("drafts_off") is None else \
                                    ref["drafts_off"] and confirmed
                        else:
                            p = kl.pair_text(replies[i], steps) if "text" in prompt else kl.pair(replies[i], steps)
                            p["prompt"] = f"{suite['id']}/{prompt['id']}"
                            r = {"suite": suite["id"], "prompt": prompt["id"], "draft": True, "error": None,
                                 "m": {"tokens": p["tokens"], "tps": None, "ttft": None, "ttft_answer": None,
                                       "prefill_s": None, "gap_p50": None, "gap_p95": None, "runtime": {}},
                                 "kl": {k: p[k] for k in ("positions", "agree", "equal", "differs_at", "identical", "worst",
                                                          "width", "unscored")},
                                 "kl_values": p["kl"]}
                            results.append(r)
                            self.store.add_result(run["id"], len(results) - 1, r)
                        run["progress"]["done"] += 1
                finally:
                    if load:
                        load.stop(wait=2.0 if self._cancel.is_set() else 320.0)
                        run["kl_load_done"] = {"ok": load.ok, "failed": load.failed, "late": load.late,
                                               "full_share": load.full_share, "last_error": load.last_error}
                if load and not (load.ok or load.late):         # refused throughout
                    raise RuntimeError(f"the background load never got through ({load.failed} refused: "
                                       f"{load.last_error}); the run would not say what it claims")
                if load:
                    # shown as "under load" only when replies came back during the measurement and all the
                    # requests were in flight for most of it
                    share = load.full_share or 0.0
                    run["kl_load_verified"] = bool(load.ok) and share >= LOAD_SHARE
                    if not run["kl_load_verified"]:
                        run["notes"].append(f"the background load is not established: {load.ok} replies during the "
                                            f"measurement, all {run['kl_load']} requests in flight {share * 100:.0f}% "
                                            "of the time; this run does not show the candidate under load")
        run["kl"] = kl.summary([{**r["kl"], "kl": r["kl_values"], "prompt": f"{r['suite']}/{r['prompt']}",
                                 "tokens": r["m"]["tokens"], "requests": 2} for r in results])

    def _quality(self, run: dict, work: list[dict], client: httpx.Client, url: str, headers: dict,
                 results: list[dict], off: dict) -> None:
        """Each item once, scored strictly; the reply and what was read from it kept with the result."""

        for it in work:
            if self._cancel.is_set():
                raise InterruptedError
            run["progress"]["current"] = f"{run['suite_names'].get(it['category'], it['category'])}: {it['id']}"
            self.store.save_run(run)
            body = {"model": run["model"], "messages": it["messages"], "max_tokens": it["max_tokens"],
                    "temperature": 0.0, "top_p": 1.0, "stream": True, "stream_options": {"include_usage": True}, **off,
                    **({} if run.get("drafts", True) else {"draft": False})}
            rep = stream_once(client, url, headers, body, self._cancel)
            if rep.error == "cancelled":
                raise InterruptedError
            # a request that failed got no answer to grade: unscored, kept out of the share right, counted apart
            passed, read = (None, rep.error) if rep.error else quality.score(it["category"], it["item"], rep.text)
            if not passed and rep.finish == "length":       # the reply did not end: said apart from a wrong answer
                read = f"cut at {it['max_tokens']} tokens ({read})"
            r = {"suite": it["category"], "prompt": it["id"], "draft": True, "m": measure(rep), "error": rep.error,
                 "text": rep.text[:4000], "text_sha": text_digest(rep.text), "passed": passed, "read": read,
                 "expected": it["item"].get("answer")}
            results.append(r)
            self.store.add_result(run["id"], len(results) - 1, r)
            run["progress"]["done"] = len(results)
            run["summary"] = summarize(results)
            run["progress"]["eta_s"] = statistics.mean(x["m"]["wall_s"] for x in results) * (len(work) - len(results))
            run["progress"]["eta_at"] = time.time()

    def _load(self, run: dict, levels: list[int], work: list[tuple[dict, dict]], client: httpx.Client, url: str,
              headers: dict, results: list[dict], off: dict) -> None:
        """Waves of requests sent together, each level at least LOAD_MIN_REQUESTS of them, the prompts taken in turn
        from the usage suites. A level that fails half its requests, or keeps the slowest waiting too long, ends it."""

        turn = itertools.cycle(work)
        spent = 0.0
        for c in levels:
            if self._cancel.is_set():
                raise InterruptedError
            last = [r for r in results if r.get("concurrency") == c // 2]
            if last and spent + 2 * max(r["wave_s"] for r in last) * math.ceil(LOAD_MIN_REQUESTS / c) > LOAD_LIMIT_S:
                run["notes"].append(f"stopped before {c} at once: the sweep would pass {int(LOAD_LIMIT_S)} s")
                break
            run["progress"]["current"] = f"{c} at once"
            self.store.save_run(run)
            level: list[dict] = []
            for wave in range(math.ceil(LOAD_MIN_REQUESTS / c)):
                if self._cancel.is_set():
                    raise InterruptedError
                bodies, asked = [], []
                for _ in range(c):
                    suite, prompt = next(turn)
                    asked.append(f"{suite['id']}/{prompt['id']}")
                    bodies.append({"model": run["model"], "messages": prompt["messages"], "max_tokens": LOAD_TOKENS,
                                   "temperature": 0.0, "top_p": 1.0, "stream": True,
                                   "stream_options": {"include_usage": True}, **off,
                                   **({} if run.get("drafts", True) else {"draft": False})})
                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=c) as pool:
                    replies = list(pool.map(lambda b: stream_once(client, url, headers, b, self._cancel), bodies))
                wall = time.perf_counter() - t0
                spent += wall
                if any(r.error == "cancelled" for r in replies):
                    raise InterruptedError
                for i, rep in enumerate(replies):
                    r = {"suite": "load", "prompt": f"{c}x{wave}.{i}", "asked": asked[i], "concurrency": c, "wave": wave,
                         "wave_s": wall,
                         "draft": True, "m": measure(rep), "error": rep.error, "text": rep.text[:200]}
                    results.append(r)
                    level.append(r)
                    self.store.add_result(run["id"], len(results) - 1, r)
            run["progress"]["done"] += 1
            run["summary"] = summarize(results)
            failed = sum(1 for r in level if r["error"])
            if failed * 2 >= len(level):
                run["notes"].append(f"stopped at {c} at once: {failed} of {len(level)} requests failed "
                                    f"({next(r['error'] for r in level if r['error'])})")
                break
            waits = [r["m"]["ttft_client"] for r in level if r["m"].get("ttft_client")]
            if waits and max(waits) > LOAD_TTFT_LIMIT_S:
                run["notes"].append(f"stopped at {c} at once: the slowest request waited {max(waits):.0f} s for its "
                                    "first token")
                break

    def _overhead(self, run: dict, client: httpx.Client, url: str, headers: dict, off: dict) -> float:
        """The first-token time of a nearly empty prompt: what an engine spends whatever the prompt, median of three."""

        times = []
        for _ in range(3):
            body = {"model": run["model"], "messages": [{"role": "user", "content": f"Ref {uuid.uuid4().hex[:8]}. Say ok."}],
                    "max_tokens": 1, "temperature": 0.0, "stream": True, **off}
            got = stream_once(client, url, headers, body, self._cancel)
            if got.error == "cancelled":
                raise InterruptedError
            if got.first_at:
                times.append(got.first_at - got.sent_at)
        value = statistics.median(times) if times else 0.0
        run["prefill_overhead_s"] = value
        run["notes"].append(f"the engine gives no prefill time: prefill is estimated as the first token less "
                            f"{value * 1000:.0f} ms of overhead (three near-empty prompts), median of a few tries per size")
        return value

    def _warm(self, run: dict, client: httpx.Client, url: str, headers: dict) -> dict:
        """One short request first, so loading and compiling are not measured; it also finds how to turn thinking off.

        Returns the fields to add to every request. A way the engine refuses (400), or accepts while still reasoning,
        is passed over; when every accepted way still reasons, the run goes on with the first one and says so.
        """

        run["progress"]["current"], run["progress"]["phase"] = "warm-up", "warm-up"
        warm = {"model": run["model"], "messages": [{"role": "user", "content": "Count from 1 to 40."}],
                "max_tokens": PROTOCOL["warmup_tokens"], "temperature": 0.0, "top_p": 1.0, "stream": True}
        accepted: tuple[str, dict] | None = None
        refused: list[str] = []
        for name, fields in THINKING_OFF:
            rep = stream_once(client, url, headers, {**warm, **fields}, self._cancel)
            if rep.error == "cancelled":
                raise InterruptedError
            if rep.error and rep.error.startswith("400"):
                refused.append(name)
                continue
            if rep.error:
                raise RuntimeError(f"warm-up failed: {rep.error}")
            if not thought(rep):
                run["thinking"] = {"sent": name, "off": True, "refused": refused}
                return fields
            accepted = accepted or (name, fields)
        if accepted is not None:
            run["thinking"] = {"sent": accepted[0], "off": False, "refused": refused}
            run["notes"].append("thinking stayed on whichever way it was turned off; the figures include reasoning tokens")
            return accepted[1]
        # every way refused: the request goes without the fields, and the receipt says thinking was not controlled
        rep = stream_once(client, url, headers, warm, self._cancel)
        if rep.error == "cancelled":
            raise InterruptedError
        if rep.error:
            raise RuntimeError(f"warm-up failed: {rep.error}")
        run["thinking"] = {"sent": None, "off": not thought(rep), "refused": refused}
        run["notes"].append("the engine refused every way of turning thinking off; requests go without them")
        return {}

    def _sweep(self, run: dict, work: list[tuple[dict, dict]], client: httpx.Client, url: str, headers: dict,
               results: list[dict], off: dict) -> None:
        for suite, prompt in work:
            first: Reply | None = None
            for draft in ((True, False) if run["exactness"] else (True,)):
                if self._cancel.is_set():
                    raise InterruptedError
                run["progress"]["current"] = f"{suite['name']}: {prompt['id']}" + ("" if draft else ", drafts off")
                self.store.save_run(run)
                body = {"model": run["model"], "messages": prompt["messages"], "max_tokens": prompt.get("max_tokens", 300),
                        "temperature": 0.0, "top_p": 1.0, "stream": True, "stream_options": {"include_usage": True},
                        **off}
                if not draft or not run.get("drafts", True):
                    body["draft"] = False
                rep = stream_once(client, url, headers, body, self._cancel)
                if rep.error == "cancelled":
                    raise InterruptedError
                r = {"suite": suite["id"], "prompt": prompt["id"], "draft": draft, "m": measure(rep),
                     "error": rep.error, "text": rep.text[:4000], "text_sha": text_digest(rep.text)}
                if thought(rep) and (run.get("thinking") or {}).get("off"):
                    run["notes"].append(f"{prompt['id']}: the model reasoned although thinking was turned off")
                if draft:
                    first = rep
                cached = (rep.usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                note = ("some drafts-off replies read their prompt from the cache the drafts-on request left: their "
                        "first token and prefill are not comparable with drafts on (speeds still are)")
                if not draft and cached > 0 and note not in run["notes"]:
                    run["notes"].append(note)
                if not draft and first is not None and not first.error and not rep.error:
                    # a reply counts as "the same without drafts" only when the engine says drafts were off (or
                    # drafted nothing); a reply with no runtime block at all may have ignored `"draft": false`
                    # and the first reply ran with drafts: two replies written without them prove nothing of drafts
                    off_known = drafts_off_confirmed(rep.runtime) and drafts_on_confirmed(first.runtime)
                    if rep.runtime.get("drafts") is True:
                        run["notes"].append(f"{prompt['id']}: the engine reported drafts on although draft=false was sent")
                    if not drafts_on_confirmed(first.runtime):
                        note = "the engine does not say drafts ran on the first reply: the exactness check is not counted"
                        if note not in run["notes"]:
                            run["notes"].append(note)
                    if off_known:
                        sha_a, sha_b = first.runtime.get("token_sha"), rep.runtime.get("token_sha")
                        r["same"] = (sha_a == sha_b) if sha_a and sha_b else (first.text == rep.text and bool(rep.text))
                        r["compared_by"] = "token_sha" if sha_a and sha_b else "text"
                    else:
                        r["same"], r["compared_by"] = None, "unverified"
                        note = "the engine does not say whether drafts were off: the exactness check is not counted"
                        if note not in run["notes"]:
                            run["notes"].append(note)
                results.append(r)
                self.store.add_result(run["id"], len(results) - 1, r)
                run["progress"]["done"] = len(results)
                run["summary"] = summarize(results)
                run["progress"]["eta_s"] = suites_eta(results, work, run["exactness"])
                run["progress"]["eta_at"] = time.time()


def suites_eta(results: list[dict], work: list, exactness: bool) -> float:
    """Seconds left in a suite run: replies with drafts and without are timed apart, the second being much slower;
    a kind not timed yet is guessed from the other."""

    walls = {d: [r["m"]["wall_s"] for r in results if r["draft"] is d] for d in (True, False)}
    mean = {d: statistics.mean(w) if w else None for d, w in walls.items()}
    left_on = len(work) - len(walls[True])
    left_off = (len(work) - len(walls[False])) if exactness else 0
    on = mean[True] or mean[False] or 0.0
    off = mean[False] or 1.4 * on                       # without drafts: the speed-up is unknown until measured
    return left_on * on + left_off * off


def sweep_eta(last_wall: float, left: int, spent: float) -> float:
    """Seconds left in a prefill sweep: each size about doubles the last one's time, within the sweep's limit."""

    est = sum(last_wall * GROWTH ** k for k in range(1, left + 1))
    return max(0.0, min(est, SWEEP_LIMIT_S - spent))


# errors as the page says them; the engine's own words stay in the receipt
PLAIN_ERRORS = (
    ("ConnectTimeout", "The engine does not answer: it may have stopped or run out of memory. See Monitor and its log."),
    ("ConnectError", "The engine does not answer: it may have stopped or run out of memory. See Monitor and its log."),
    ("ReadTimeout", "The engine took longer than ten minutes on one reply."),
    ("RemoteProtocolError", "The engine closed the connection in the middle of a reply."),
    ("401", "The engine refused TensorFold Web's key. Check the key in Settings."),
    ("warm-up failed: 400", "The engine refused the warm-up request; its message is in the receipt."),
)


def plain_error(text: str) -> str:
    for needle, said in PLAIN_ERRORS:
        if needle in text:
            return said
    return text[:500]


def receipt(run: dict) -> dict:
    """The run as a document to keep or share: protocol, machine, engine, every result and the summary."""

    out = {k: v for k, v in run.items() if k != "progress"}
    out["kind"] = "tensorfold-studio-bench-receipt"
    return out
