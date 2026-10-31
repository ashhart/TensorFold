"""Compare mode: two to four configurations of the served engine, measured one after the other.

A configuration is a model (any downloaded checkpoint), a context, a memory budget and drafts on or off. Between two
configurations the engine is restarted when the model, context or budget changes; drafts only change the requests.
Each configuration is an ordinary Bench run, tagged with the comparison it belongs to, so it also shows in the runs
list and compares two by two. At the end the engine is put back as it was, whatever happened.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from . import compare
from .gate import Gate

MAX_CONFIGS = 4
RESTART_KEYS = ("model", "context", "budget_fraction")


def validate(configs: object) -> list[dict]:
    if not isinstance(configs, list) or not 2 <= len(configs) <= MAX_CONFIGS:
        raise ValueError(f"a comparison has 2 to {MAX_CONFIGS} configurations")
    out = []
    for i, c in enumerate(configs, 1):
        if not isinstance(c, dict) or not isinstance(c.get("model"), str) or "/" not in c["model"]:
            raise ValueError(f"configuration {i} needs a model, as owner/name")
        ctx = c.get("context")
        if ctx is not None and (isinstance(ctx, bool) or not isinstance(ctx, int) or ctx < 1024):
            raise ValueError(f"configuration {i}: the context must be at least 1,024 tokens, or empty for automatic")
        frac = c.get("budget_fraction")
        if frac is not None and (isinstance(frac, bool) or not isinstance(frac, (int, float)) or not 0.3 <= frac <= 0.95):
            raise ValueError(f"configuration {i}: the memory budget must be between 0.3 and 0.95, or empty")
        out.append({"model": c["model"], "context": ctx, "budget_fraction": None if frac is None else float(frac),
                    "drafts": c.get("drafts", True) is not False, "label": str(c.get("label") or "")[:60]})
    if len({json.dumps(c, sort_keys=True) for c in out}) < len(out):
        raise ValueError("two configurations are the same")
    return out


def label(c: dict) -> str:
    """How a configuration reads in the matrix: the model's short name and what sets it apart."""

    if c.get("label"):
        return c["label"]
    parts = [c["model"].split("/")[-1]]
    if c.get("context"):
        parts.append(f"{c['context'] // 1024}k context")
    if c.get("budget_fraction"):
        parts.append(f"{round(c['budget_fraction'] * 100)}% memory")
    if not c.get("drafts", True):
        parts.append("drafts off")
    return ", ".join(parts)


class Series:
    """Runs a comparison in a thread; ``switch`` puts the engine in a configuration (blocking until it serves or
    raising; a configuration without a model stops it), ``current`` says how it is now, ``bench`` runs one
    configuration's measurement."""

    def __init__(self, path: Path, bench, switch: Callable[..., dict], current: Callable[[], dict],
                 engine_name: Callable[[], str | None], gate: Gate | None = None,
                 on_end: Callable[[], None] | None = None) -> None:
        self.path, self.bench, self.switch, self.current, self.engine_name = path, bench, switch, current, engine_name
        self.gate = gate or Gate()                 # held from the start to the engine being put back
        self.on_end = on_end                       # the settings it changed for a while go back to the user's
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._starting = threading.Lock()          # two requests at once start one comparison, not two
        self.doc: dict | None = None
        with self._db() as db:
            db.execute("create table if not exists series (id text primary key, created real not null, doc text not null)")
            # a comparison left going by a stopped TensorFold Web is not going anymore
            for sid, d in db.execute("select id, doc from series").fetchall():
                doc = json.loads(d)
                if doc.get("status") == "running":
                    doc["status"] = "interrupted"
                    doc["restore_error"] = "TensorFold Web stopped during it, so the engine was not put back as it was"
                    db.execute("update series set doc=? where id=?", (json.dumps(doc), sid))

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def _save(self) -> None:
        if self.doc is not None:
            with self._db() as db:
                db.execute("insert or replace into series (id, created, doc) values (?,?,?)",
                           (self.doc["id"], self.doc["created"], json.dumps(self.doc)))

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get(self, sid: str) -> dict | None:
        with self._db() as db:
            row = db.execute("select doc from series where id=?", (sid,)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, limit: int = 20) -> list[dict]:
        with self._db() as db:
            rows = db.execute("select doc from series order by created desc limit ?", (limit,)).fetchall()
        return [json.loads(d) for (d,) in rows]

    def start(self, configs: object, *, mode: str = "profile", suites: list[str] | None = None,
              exactness: bool = True) -> dict:
        with self._starting:
            self.gate.take("a comparison of configurations")
            try:
                return self._start(configs, mode, suites, exactness)
            except BaseException:
                self.gate.give()
                raise

    def _start(self, configs: object, mode: str, suites: list[str] | None, exactness: bool) -> dict:
        if self.busy() or self.bench.busy():
            raise RuntimeError("a run is already going; cancel it or wait")
        if mode not in ("quick", "profile", "quality"):
            raise ValueError("a comparison measures a quick score, a usage profile or the quality suite")
        cs = validate(configs)
        self.doc = {"id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6], "created": time.time(),
                    "status": "running", "mode": mode, "suites": suites, "exactness": exactness, "configs": cs,
                    "runs": [None] * len(cs), "served": [None] * len(cs), "errors": [None] * len(cs),
                    "progress": {"index": 0, "phase": "starting", "run": None}, "before": None, "restored": None}
        self._save()
        self._cancel.clear()
        self._thread = threading.Thread(target=self._go, name="series", daemon=True)
        self._thread.start()
        return self.doc

    def cancel(self) -> bool:
        if not self.busy():
            return False
        self._cancel.set()
        self.bench.cancel()
        return True

    def _go(self) -> None:
        try:
            self._run()
        finally:
            try:
                if self.on_end:
                    self.on_end()
            finally:
                self.gate.give()                 # whatever failed, even saving: nothing else waits on a dead comparison

    def _run(self) -> None:
        doc = self.doc
        assert doc is not None
        before = None
        outcome = "done"                 # said only once the engine is back, so nobody reads "done" while it reloads
        try:
            before = doc["before"] = self.current()
            for i, c in enumerate(doc["configs"]):
                if self._cancel.is_set():
                    outcome = "cancelled"
                    break
                doc["progress"] = {"index": i, "phase": "engine", "run": None}
                self._save()
                try:
                    doc["served"][i] = self.switch(c, self._cancel.is_set)
                except Exception as exc:  # noqa: BLE001 - a configuration that does not start is said, the next goes on
                    if self._cancel.is_set():            # a cancel while the engine loaded: the load was stopped
                        outcome = "cancelled"
                        break
                    doc["errors"][i] = f"the engine did not start: {exc}"[:400]
                    self._save()
                    continue
                if self._cancel.is_set():
                    outcome = "cancelled"
                    break
                try:
                    run = self.bench.start(doc["mode"], suites=doc["suites"], exactness=doc["exactness"],
                                           engine_name=self.engine_name(), drafts=c["drafts"],
                                           tags={"series": doc["id"], "config": i, "label": label(c)})
                except (RuntimeError, ValueError) as exc:
                    doc["errors"][i] = str(exc)[:400]
                    self._save()
                    continue
                doc["runs"][i] = run["id"]
                doc["progress"] = {"index": i, "phase": "measuring", "run": run["id"]}
                self._save()
                while self.bench.busy():
                    time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001 - the comparison is kept and says why it stopped
            outcome, doc["error"] = "failed", str(exc)[:400]
        finally:
            doc["progress"] = {"index": len(doc["configs"]), "phase": "restoring", "run": None}
            with contextlib.suppress(Exception):      # a record not saved never keeps the engine from going back
                self._save()
            try:
                doc["restored"] = self.switch(before) if before else None
            except Exception as exc:  # noqa: BLE001
                doc["restore_error"] = f"the engine could not be put back: {exc}"[:400]
            doc["status"] = outcome
            doc["progress"] = None
            doc["finished"] = time.time()
            self._save()


def matrix(doc: dict, run_of: Callable[[str], dict | None]) -> dict:
    """The comparison's figures, one row per configuration, with each against the first."""

    runs = [run_of(r) if r else None for r in doc["runs"]]
    first = next((r for r in runs if r and r.get("status") == "done"), None)
    rows = []
    for i, (c, r) in enumerate(zip(doc["configs"], runs, strict=True)):
        o = ((r or {}).get("summary") or {}).get("overall") or {}
        q = (((r or {}).get("summary") or {}).get("quality") or {}).get("overall")
        hw = (r or {}).get("hardware") or {}
        row = {"index": i, "label": label(c), "config": c, "run": (r or {}).get("id"), "status": (r or {}).get("status"),
               "error": doc["errors"][i] or (r or {}).get("error"), "served": doc["served"][i],
               "tps": o.get("tps"), "tps_no_drafts": o.get("tps_no_drafts"), "ttft": o.get("ttft"),
               "exact": o.get("exact"), "checked": o.get("checked"), "j_per_token": hw.get("j_per_token"),
               "right": q.get("passed") if q else None, "items": q.get("items") if q else None,
               "engine_peak_bytes": ((r or {}).get("memory") or {}).get("engine_peak_bytes"), "vs_first": None}
        if r and first and r is not first and r.get("status") == "done":
            try:
                cmp = compare.compare(first, r)
                row["vs_first"] = {"speed": cmp["speed"], "same_text": cmp["same_text"], "pairs": len(cmp["pairs"]),
                                   "findings": cmp["findings"],
                                   **({"quality": cmp["exactness"]} if doc.get("mode") == "quality" else {})}
            except compare.Refused as exc:
                row["vs_first"] = {"refused": str(exc)}
        rows.append(row)
    return {**{k: v for k, v in doc.items() if k != "progress"}, "progress": doc.get("progress"), "rows": rows}
