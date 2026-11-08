

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import threading
import time
import uuid
from collections.abc import Callable
from importlib import resources
from urllib.parse import urlsplit

import httpx

from . import __version__, kl, power, quality, tooleval
from . import suites as custom_suites
from .bench_modes import BenchModes
from .bench_sampling import (
    Sampler,
    energy_cost,
    engine_memory,
    machine_facts,
    on_this_machine,
    summarize,
)
from .bench_store import (
    Store,
)
from .bench_stream import (
    Reply,
    _stream,
    drafts_off_confirmed,
    drafts_on_confirmed,
    measure,
    stream_once,
    text_digest,
    thought,
)

SCHEMA = 2

PROTOCOL = {"temperature": 0.0, "top_p": 1.0, "stream": True, "warmup_tokens": 32, "concurrency": 1,
            "thinking": "off, checked during the warm-up: the first way of saying it that the engine accepts and that "
                        "brings no reasoning is kept, and named in the receipt",
            "output_format": "asked in the prompt; response_format never sent"}

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

LOAD_SHARE = 0.5

PER_TOKEN_MODES = ("quick", "profile", "load", "quality")

class Bench(BenchModes):
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
        # Drafts-off verification belongs to TensorFold; other engines are compared by reply text.
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
        # Remote engines have no local hardware or power measurements, regardless of how their address was selected.
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
            with httpx.Client(trust_env=False, timeout=httpx.Timeout(600.0, connect=5.0)) as client, \
                    httpx.Client(trust_env=False, timeout=3.0) as probe, \
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
                    # Per-token energy needs every reply counted; partial counts cannot absorb the whole run's energy.
                    run["hardware"] = sampler.report(sum(written) if written and None not in written else None)
                    if run["hardware"] and run["mode"] not in PER_TOKEN_MODES:
                        # Do not charge prefill work, repeats and probes to its few output tokens.
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









def suites_eta(results: list[dict], work: list, exactness: bool) -> float:

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


__all__ = ['GROWTH', 'LEVELS', 'LOAD_LIMIT_S', 'LOAD_MIN_REQUESTS', 'LOAD_SHARE', 'LOAD_TOKENS', 'LOAD_TTFT_LIMIT_S', 'MAX_GAP_S', 'MODES', 'PARTIAL_BELOW', 'PER_TOKEN_MODES', 'PLAIN_ERRORS', 'PROTOCOL', 'RUNTIME_KEYS', 'SCHEMA', 'SIZES', 'SWEEP_LIMIT_S', 'TEXT_SUITE', 'THINKING_OFF', 'Bench', 'Reply', 'Sampler', 'Store', '_stream', 'drafts_off_confirmed', 'drafts_on_confirmed', 'energy_cost', 'engine_memory', 'load_suites', 'machine_facts', 'measure', 'on_this_machine', 'percentile', 'pick', 'plain_error', 'prefill_repeats', 'receipt', 'stream_once', 'suites_digest', 'suites_eta', 'summarize', 'sweep_eta', 'text_digest', 'thought']
