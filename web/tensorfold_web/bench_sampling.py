"""Bench sampling."""

from __future__ import annotations

import itertools
import statistics
import threading
import time
from collections.abc import Callable

import httpx
import psutil

from . import bench, power
from .engine import parse_metrics


class Sampler:

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
            # a reading at the start of the measured work, not up to 1 s later
            self.sample()

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
        steps = [(t1, w1, t2, w2) for (t1, w1), (t2, w2) in itertools.pairwise(pts) if t2 - t1 <= bench.MAX_GAP_S]
        energy = sum((t2 - t1) * (w1 + w2) / 2 for t1, w1, t2, w2 in steps)
        span = sum(t2 - t1 for t1, _, t2, _ in steps)
        # Energy covers the sampled part of the full work interval; unsampled work remains uncounted.
        end = max(rows[-1][0] if rows else start, time.monotonic() - self.t0 if live else 0.0)
        window = max(0.0, end - start) if self.measured_from is not None else \
            (rows[-1][0] - rows[0][0] if len(rows) > 1 else 0.0)
        coverage = min(1.0, span / window) if window else None
        # an idle reading above the run's average (taken while a load was settling) gives no energy below zero
        above = max(0.0, energy - (self.idle_watts or 0.0) * span) if self.idle_watts is not None else None
        flags = [row[5] for row in rows if len(row) > 5 and row[5] is not None]
        # Throttle duration uses each reading's actual interval, including boundary samples.
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
                "coverage": coverage, "partial": coverage is not None and coverage < bench.PARTIAL_BELOW,
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
                           "ttft_p50": bench.percentile(waits, 0.5), "ttft_p95": bench.percentile(waits, 0.95)})
        out["load"] = levels
    sweep = [r for r in results if "size" in r]
    if sweep:
        out["sweep"] = [{"size": r["size"], "prompt_tokens": r["m"]["prompt_tokens"], "cached": r["m"]["cached_tokens"],
                         "prefill_s": r["m"]["prefill_s"], "ttft": r["m"]["ttft"], "prefill_tps": r["m"].get("prefill_tps"),
                         "estimated": bool(r["m"].get("prefill_estimated")), "weak": bool(r["m"].get("prefill_weak")),
                         "error": r.get("error")} for r in sweep]
    return out


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
