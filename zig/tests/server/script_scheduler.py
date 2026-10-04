"""The Python server's scheduler for the parity tests: each job replays its script as fake_engine.zig does."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.live import ChunkRate, Meter


def scheduler_class(tokenizer: Any, scripts: dict[str, Any]) -> type:
    """A Scheduler stand-in bound to the fake tokenizer and the shared scripts."""

    class ScriptScheduler:
        def __init__(self, engine: Any, *, lanes: int, eos_ids: frozenset[int], **_: Any) -> None:
            self.engine, self.lanes, self.eos_ids = engine, lanes, eos_ids
            self.active, self.waiting, self.filling = 0, 0, []
            self.decoded, self.prefilled = Meter(), ChunkRate()
            self.cancelled, self.preemptions = 0, 0
            self.session_dir, self.model_id = None, ""
            self.lock = threading.Lock()

        def start(self) -> None:
            pass

        def stop(self, timeout: float = 5.0) -> None:
            pass

        def submit(self, job: Any) -> None:
            with self.lock:
                self.active += 1
            threading.Thread(target=self._run, args=(job,), daemon=True).start()

        def cancel(self, cancellation: Any) -> None:
            cancellation.cancel()

        def _script(self, prompt: list[int]) -> dict[str, Any]:
            text = tokenizer.decode(prompt)
            at = text.rfind("@script=")
            name = "default"
            if at >= 0:
                end = at + 8
                while end < len(text) and (text[end].isascii() and (text[end].isalnum() or text[end] in "_-")):
                    end += 1
                name = text[at + 8:end]
            return scripts.get(name) or scripts["default"]

        def _tokens(self, entry: Any, eos: frozenset[int]) -> list[int]:
            if isinstance(entry, int):
                return [entry]
            if entry == "<EOS>":
                return [min(self.eos_ids)] if eos else []
            return tokenizer.encode(entry, add_special_tokens=False)

        def _run(self, job: Any) -> None:
            script = self._script(job.prompt_ids)
            job.started_at = time.perf_counter()
            job.cached_tokens = int(script.get("cached", 0))
            job.prefilled_at = time.perf_counter()
            stream = job.stream = SimpleNamespace(finish_reason="", rounds=0, drafted=0, accepted=0, min_rows=0,
                                                  prefill_widths=[], prefill_raised=[], proposer=None)
            eos = frozenset() if job.ignore_eos else self.eos_ids
            emitted, reason, done = [], "stop", False
            for chunk in script["chunks"]:
                if script.get("delay_ms"):
                    time.sleep(script["delay_ms"] / 1000)
                if job.cancellation.cancelled:
                    reason, done = "cancelled", True
                    break
                tokens = [t for entry in chunk for t in self._tokens(entry, eos)]
                start = len(emitted)
                for t in tokens:
                    emitted.append(t)
                    if t in eos or (job.stop_check is not None and job.stop_check(emitted)):
                        reason, done = "stop", True
                    elif len(emitted) >= job.max_tokens:
                        reason, done = "length", True
                    if done:
                        break
                landed = len(emitted) - start
                if landed:
                    job.chunks.put(emitted[start:])
                    stream.rounds += 1
                    stream.drafted += landed - 1
                    stream.accepted += landed - 1
                    stream.min_rows = landed if not stream.min_rows else min(stream.min_rows, landed)
                if done:
                    break
            stream.finish_reason = reason
            with self.lock:
                self.active -= 1
                if reason == "cancelled":
                    self.cancelled += 1
                    job.error = RequestCancelled("request cancelled")
            job.finished_at = time.perf_counter()
            job.chunks.put(None)
            job.done.set()

    return ScriptScheduler
