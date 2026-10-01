"""Open prompts prefill a chunk a step, the fewest tokens left first, with the live streams' rounds between chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from tensorfold.engine.lane_engine import LaneStream
from tensorfold.server.cancellation import PrefillGuard, RequestCancelled
from tensorfold.server.checkpoints import choose_checkpoints
from tensorfold.server.errors import RequestError


@dataclass(eq=False)
class Filling:
    """An open prompt: its job, its prefill steps, the shared-prefix starts it keeps, and where its fill stands."""

    job: Any
    steps: Any = None          # its prefill steps, from its first chunk (``_start_fill``)
    starts: Any = None         # its prompt's chunk starts
    shared_at: set[int] = field(default_factory=set)
    left: int = 0              # prompt tokens not fed yet
    passed: int = 0            # other prompts' chunks since this one's last
    held: Any = None           # its PromptMemory reservation (None: no prompt memory)


class PromptFill:
    """The scheduler's prompt filling: its planned chunks keep each stream's solo bits; rounds get a bounded share."""

    clock = staticmethod(time.perf_counter)      # what chunks and rounds are timed with (tests set their own)

    def _start_job(self, job: Any) -> None:
        """Open ``job`` and prefill its whole prompt now (the loop's admissions run rounds between chunks)."""

        filling = self._open_job(job)
        while filling is not None and filling in self._fills:
            self._fill(filling)

    def _open_job(self, job: Any) -> Filling | None:
        """Reserve its memory beside the open prompts; its stored prefix and stream come at its first chunk."""

        job.started_at = time.perf_counter()
        self.starts += 1
        self._starting = job
        filling = Filling(job, left=len(job.prompt_ids))
        try:
            job.cancellation.check()
            memory = self.prompt_memory
            if memory is not None:
                getattr(self.engine, "release_rounds", lambda: None)()     # no stream live: rows are nobody's (#95)
                filling.held = memory.begin(len(job.prompt_ids), self._reserved(int(job.max_tokens)),
                                            admit=self.checkpoints is None or job.vision is not None)
                if job.vision is not None:
                    memory.require_workspace(self.engine.model.vision.estimate_workspace_bytes(job.vision))
            # checkpoints sit at the prompt's chunk starts: a shared prefix is kept at the start at or before its end
            starts = filling.starts = self.engine.prompt_chunks(job.prompt_ids)
            filling.shared_at = {starts.floor(n) for n in job.shared_prefix_lens} - {0}
            if self.checkpoints is not None and job.vision is None:
                self._read_disk_block(job.prompt_ids, lambda n: n in starts)
                entry = self.checkpoints.peek(job.prompt_ids, usable=lambda n: n in starts)
                if memory is not None:
                    memory.require(current_cache=None if entry is None else entry.cache, keep=entry)
                filling.left -= 0 if entry is None else len(entry.tokens)
        except Exception as exc:  # noqa: BLE001 - reported to the waiting request, as a failed prefill is
            self._end_fill(filling, exc)
            return None
        self._fills.append(filling)
        return filling

    def _start_fill(self, filling: Filling) -> None:
        """Its first chunk: the longest stored prefix now (another prompt may have stored it), then its stream."""

        job, starts, shared_at = filling.job, filling.starts, filling.shared_at
        cache = None
        cached = 0
        last_prompt: list[int] | None = None
        checkpoints_at: list[int] = []
        if self.checkpoints is not None and job.vision is None:
            usable = lambda n: n in starts
            entry = self.checkpoints.peek(job.prompt_ids, usable=usable)
            take = False
            memory = self.prompt_memory
            if memory is not None:
                # Keep the resumed prefix through admission; use its stored arrays as the working cache if copying cannot fit.
                memory.require(current_cache=None if entry is None else entry.cache, keep=entry)
                take = entry is not None and not memory.fits_now()
            hit = self.checkpoints.match(job.prompt_ids, usable=usable, take=take)
            entry = None        # held through the prefill, a stored prefix evicted for this prompt's copy stays
            if hit is not None:
                cached, cache, last_prompt = hit
                if filling.held is not None:
                    filling.held.cache = cache            # its working cache until the first chunk grows it
                if self.disk_blocks is not None and cached in shared_at:
                    self.disk_blocks.touch(job.prompt_ids[:cached])
            filling.left = len(job.prompt_ids) - cached
            chosen = choose_checkpoints(job.history_len, cached, last_prompt, job.prompt_ids)
            checkpoints_at = sorted(at for at in {*(starts.floor(n) for n in chosen), *shared_at}
                                    if cached < at < len(job.prompt_ids))
        proposer = job.proposer
        if proposer is None and job.drafts and self.proposer_factory is not None:
            proposer = self.proposer_factory()
        stream = LaneStream(
            stream_id=job.job_id,
            prompt_ids=list(job.prompt_ids),
            max_new_tokens=int(job.max_tokens),
            eos_ids=frozenset() if job.ignore_eos else self.eos_ids,
            stop_check=job.stop_check,
            proposer=proposer if job.drafts else None,
            drafts=bool(job.drafts),
            sampling=job.sampling,
            think_budget=int(job.think_budget),
            think_close=tuple(job.think_close),
            think_end=int(job.think_end),
            think_open=bool(job.think_budget),
            call_gate=job.call_gate,
            constraint=job.constraint,
            prompt_data=job.vision,
            retain=job.vision is None,
        )
        if job.label_ids:
            stream.label_ids = tuple(job.label_ids)     # the prefill stops at these logits and draws nothing
        job.stream = stream
        filling.steps = self.engine.begin_stream(stream, cache=cache, cached_tokens=cached,
                                                 checkpoints_at=checkpoints_at)

    def _rounds_had_turn(self) -> bool:
        """Whether rounds spent the credit chunks gave them (decode_share of each chunk's time) or fill_rounds ran."""

        return self._credit <= 0.0 or self._rounds_left <= 0

    def _spend_round(self, seconds: float) -> None:
        self._credit -= seconds
        self._rounds_left -= 1

    def _next_fill(self) -> Filling:
        """Next chunk's prompt: one passed over fill_guard chunks, else foreground, fewest tokens left, opened first."""

        due = [f for f in self._fills if f.passed >= self.fill_guard]
        if due:
            return max(due, key=lambda f: f.passed)         # the most passed over; ties to the one opened first
        return min(self._fills, key=lambda f: (f.job.background, f.left))

    def _chunk_due(self) -> bool:
        """A chunk runs next with no stream live, else once rounds had their share and it fits beside the streams."""

        return bool(self._fills) and (self.engine.active_count == 0 or (self._rounds_had_turn() and self._chunk_fits()))

    def _chunk_fits(self) -> bool:
        """Whether the next chunk fits now: while streams hold more than its admission saw, it waits, not fails."""

        filling = self._next_fill()
        if filling.held is None:
            return True
        self.prompt_memory.focus(filling.held)
        return self.prompt_memory.fits(filling.held.cache)

    def _fill(self, filling: Filling | None = None, abort: BaseException | None = None) -> None:
        """The next chunk of ``filling``, else of the rule's pick (``abort``: stop it between chunks)."""

        filling = filling or self._next_fill()
        started = self.clock()
        fed = getattr(self.engine, "prefill_tokens", 0)
        # several plan chunks a forward only for a lone foreground prompt, with no live stream waiting on it (#72)
        wide = (not getattr(filling.job, "background", False) and len(self._fills) == 1
                and (self.decode_share <= 0 or getattr(self.engine, "active_count", 0) == 0))
        self.engine.prefill_guard = PrefillGuard(filling.job.cancellation, self.prompt_memory, wide=wide)
        if filling.held is not None:
            self.prompt_memory.focus(filling.held)
        try:
            if abort is not None and filling.steps is None:
                raise abort                          # not started: nothing to stop between chunks
            if abort is not None:
                filling.steps.throw(abort)
            else:
                if filling.steps is None:
                    self._start_fill(filling)
                next(filling.steps)
        except StopIteration:
            self._end_fill(filling, None)
        except Exception as exc:  # noqa: BLE001 - a failed or cancelled prefill ends only its own request
            self._end_fill(filling, exc)
        finally:
            self.engine.prefill_guard = None
        fed = getattr(self.engine, "prefill_tokens", 0) - fed
        filling.left -= fed
        self.prefilled.add(fed, self.clock() - started)
        if abort is None:
            for other in self._fills:
                other.passed = 0 if other is filling else other.passed + 1
        # a debt of the last round carries over, an unspent credit does not
        self._credit = min(self._credit, 0.0) + self.decode_share * (self.clock() - started)
        self._rounds_left = self.fill_rounds

    def _end_fill(self, filling: Filling, error: BaseException | None) -> None:
        """A prompt's prefill ended: its stream joins the rounds, or its cancellation or error goes to its request."""

        if filling in self._fills:
            self._fills.remove(filling)
        job, shared_at = filling.job, filling.shared_at
        try:
            if error is not None:
                raise error
            stream = job.stream
            self._keep_checkpoints(job, shared_at)
            scored = getattr(stream, "scored", None)
            if scored is not None:
                job.scored = scored
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
            if filling.held is not None:
                self.prompt_memory.end(filling.held)

    def _preempt_filling(self, filling: Filling) -> None:
        """Stop a background prefill between chunks for a waiting foreground job; its rerun resumes the progress."""

        self._fills.remove(filling)
        job = filling.job
        try:
            if filling.steps is not None:
                filling.steps.close()              # the prefill keeps its progress as a checkpoint
        finally:
            job.preempted = True
            self.preemptions += 1
            self._keep_checkpoints(job, filling.shared_at)
            if job.stream is not None:
                self.engine.discard_stream(job.stream)
                job.stream.finish_reason = "preempted"
                job.stream.proposer = None
            if filling.held is not None:
                self.prompt_memory.end(filling.held)
            self._finish(job)


__all__ = ["Filling", "PromptFill"]
