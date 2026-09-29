"""One prompt prefills at a time, a chunk a step, with the live streams' rounds between its chunks (issue #72)."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from tensorfold.server.cancellation import PrefillGuard, RequestCancelled
from tensorfold.server.errors import RequestError


@dataclass
class Filling:
    """The one job whose prompt is prefilling, its prefill steps, and the shared-prefix starts its checkpoints keep."""

    job: Any
    steps: Any
    shared_at: set[int]


class PromptFill:
    """The scheduler's prompt filling: its planned chunks keep each stream's solo bits; rounds get a bounded share."""

    clock = staticmethod(time.perf_counter)      # what chunks and rounds are timed with (tests set their own)

    def _rounds_had_turn(self) -> bool:
        """Whether rounds spent the credit chunks gave them (decode_share of each chunk's time) or fill_rounds ran."""

        return self._credit <= 0.0 or self._rounds_left <= 0

    def _spend_round(self, seconds: float) -> None:
        self._credit -= seconds
        self._rounds_left -= 1

    def _fill(self, abort: BaseException | None = None) -> None:
        """The filling job's next prompt chunk (``abort``: stop between chunks); after the last, its stream starts."""

        filling = self._filling
        started = self.clock()
        fed = getattr(self.engine, "prefill_tokens", 0)
        self.engine.prefill_guard = PrefillGuard(filling.job.cancellation, self.prompt_memory)
        try:
            if abort is None:
                next(filling.steps)
            else:
                filling.steps.throw(abort)
        except StopIteration:
            self._end_fill(filling.job, filling.shared_at, None)
        except Exception as exc:  # noqa: BLE001 - a failed or cancelled prefill ends only its own request
            self._end_fill(filling.job, filling.shared_at, exc)
        finally:
            self.engine.prefill_guard = None
        self.prefilled.add(getattr(self.engine, "prefill_tokens", 0) - fed, self.clock() - started)
        # a debt of the last round carries over, an unspent credit does not
        self._credit = min(self._credit, 0.0) + self.decode_share * (self.clock() - started)
        self._rounds_left = self.fill_rounds

    def _end_fill(self, job: Any, shared_at: set[int], error: BaseException | None) -> None:
        """A prompt's prefill ended: its stream joins the rounds, or its cancellation or error goes to its request."""

        self._filling = None
        try:
            if error is not None:
                raise error
            stream = job.stream
            self._keep_checkpoints(job, shared_at)
            job.cancellation.check()
            job.prefilled_at = time.perf_counter()
            job.cached_tokens = int(stream.cached_tokens)      # 0 when a stored state was not at a chunk start
            if stream.emitted:
                job.chunks.put(list(stream.emitted))
            if stream.finished:
                self._retire(job)
            else:
                self._jobs[stream.stream_id] = job
        except RequestCancelled:
            self._keep_checkpoints(job, shared_at)      # a prefill stopped between chunks: a retry resumes there
            self._discard_job(job)
        except Exception as exc:  # noqa: BLE001 - reported to the waiting request
            job.error = exc.with_traceback(None) if isinstance(exc, RequestError) else exc
            self._keep_checkpoints(job, shared_at)
            print(f"[tensorfold] start failed {job.job_id} cached={job.cached_tokens}: {type(exc).__name__}: {exc}",
                  flush=True)
            self._finish(job)
        finally:
            if self.prompt_memory is not None:
                self.prompt_memory.end()

    def _preempt_filling(self) -> None:
        """Stop a background prefill between chunks for a waiting foreground job; its rerun resumes the progress."""

        filling, self._filling = self._filling, None
        job = filling.job
        try:
            filling.steps.close()                  # the prefill keeps its progress as a checkpoint
        finally:
            job.preempted = True
            self.preemptions += 1
            self._keep_checkpoints(job, filling.shared_at)
            if job.stream is not None:
                self.engine.discard_stream(job.stream)
                job.stream.finish_reason = "preempted"
                job.stream.proposer = None
            if self.prompt_memory is not None:
                self.prompt_memory.end()
            self._finish(job)


__all__ = ["Filling", "PromptFill"]
