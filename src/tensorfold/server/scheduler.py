"""The engine thread: requests queued as jobs, admitted into the engine's lanes, stepped round by round."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
from pathlib import Path
import queue
import threading
import time
import traceback
from typing import Any, Callable

from tensorfold.engine.lane_engine import LaneStream
from tensorfold.server.cancellation import Cancellation, RequestCancelled
from tensorfold.server.checkpoints import CheckpointStore, choose_checkpoints
from tensorfold.server.errors import RequestError, RoundError
from tensorfold.server.live import ChunkRate, Meter
from tensorfold.server.prompt_fill import Filling, PromptFill
from tensorfold.server.stream_gate import StreamGate

@dataclass
class ChatJob:
    job_id: str
    prompt_ids: list[int]
    max_tokens: int
    temperature: float
    history_len: int = 0
    # Snapshot system-and-tools blocks for reuse across sessions; history boundaries already include session-specific text.
    shared_prefix_lens: tuple[int, ...] = ()
    # the request's proposer (suffix copies, a drafter model, tool-call structure); None: the engine's default
    proposer: Any = None
    # False ("draft": false): one token a round and no drafts, the serial reference output is checked against
    drafts: bool = True
    # exact_sampling.Sampling for this request (None = greedy)
    sampling: Any = None
    # thinking budget (0: none) and the tokens that close a think block ("\n</think>\n\n"; LaneStream)
    think_budget: int = 0
    think_close: tuple[int, ...] = ()
    think_end: int = -1
    # Background jobs yield to foreground arrivals, and callers restart preempted jobs from the beginning.
    background: bool = False
    preempted: bool = False
    submitted_at: float = field(default_factory=time.perf_counter)
    started_at: float = 0.0
    prefilled_at: float = 0.0
    finished_at: float = 0.0
    cached_tokens: int = 0
    stream: LaneStream | None = None
    error: BaseException | None = None
    chunks: "queue.Queue[list[int] | None]" = field(default_factory=queue.Queue)
    done: threading.Event = field(default_factory=threading.Event)
    cancellation: Cancellation = field(default_factory=Cancellation)
    ignore_eos: bool = False
    stop_check: Callable[[list[int]], bool] | None = None
    call_gate: Any = None                   # tool_choice "required": the answer opens a tool call (LaneStream)
    constraint: Any = None                  # response_format's grammar (engine.grammar.Constraint), or None
    vision: Any = None


class _JobQueue(queue.PriorityQueue):
    """Jobs in arrival order, background jobs after every other job."""

    def __init__(self) -> None:
        super().__init__()
        self._order = itertools.count()

    def put(self, job: ChatJob, block: bool = True, timeout: float | None = None) -> None:
        super().put((1 if job.background else 0, next(self._order), job), block, timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> ChatJob:
        return super().get(block, timeout)[2]

    def foreground_waiting(self) -> bool:
        return self.peek_foreground() is not None

    def peek_foreground(self) -> ChatJob | None:
        with self.mutex:
            return self.queue[0][2] if self.queue and self.queue[0][0] == 0 else None

    def remove(self, cancellation: Cancellation) -> list[ChatJob]:
        with self.mutex:
            removed = [entry[2] for entry in self.queue if entry[2].cancellation is cancellation]
            self.queue[:] = [entry for entry in self.queue if entry[2].cancellation is not cancellation]
            heapq.heapify(self.queue)
            return removed


class Scheduler(PromptFill):
    """Owns the engine on one thread: admits jobs, steps rounds, delivers tokens."""

    def __init__(
        self,
        engine: Any,
        *,
        lanes: int,
        eos_ids: frozenset[int],
        checkpoints: CheckpointStore | None = None,
        proposer_factory: Callable[[], Any] | None = None,
        idle_wait: float = 0.02,
        snapshot_dir: Any = None,
        session_dir: Any = None,
        model_id: str = "",
        admission: Any = None,
        prompt_memory: Any = None,
        decode_share: float = 0.25,
    ) -> None:
        if lanes < 1:
            raise ValueError("lanes must be positive")
        if decode_share < 0:
            raise ValueError("decode_share must be 0 or more")
        # ``engine.memory.Admission``: a job starts beside live streams only while the projected memory fits
        self.admission = admission
        self.snapshot_dir = snapshot_dir
        self.session_dir = session_dir
        self.model_id = model_id
        self.disk_blocks: Any = None
        self.session_blocks: Any = None
        if model_id and (snapshot_dir is not None or session_dir is not None):
            from tensorfold.engine.prefix_snapshots import DiskBlocks

            if snapshot_dir is not None:
                self.disk_blocks = DiskBlocks(Path(snapshot_dir), model_id)
            if session_dir is not None:
                self.session_blocks = DiskBlocks(Path(session_dir), model_id)
        self.engine = engine
        self.lanes = int(lanes)
        self.eos_ids = eos_ids
        self.checkpoints = checkpoints
        self.prompt_memory = prompt_memory
        self.proposer_factory = proposer_factory
        self.idle_wait = float(idle_wait)
        self.slow_round_ms = 1000.0
        self._held: ChatJob | None = None
        self._queue = _JobQueue()
        self.preemptions = 0
        self._jobs: dict[str, ChatJob] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tensorfold-engine", daemon=True)
        self.rounds = 0
        self.failed_rounds = 0
        self.completed = 0
        self.cancelled = 0
        self.starts = 0
        self._starting: ChatJob | None = None
        self._filling: Filling | None = None
        # while a prompt fills, rounds get decode_share of each chunk's time, at most fill_rounds between two chunks
        self.decode_share = float(decode_share)
        self.fill_rounds = 32
        self._credit, self._rounds_left = 0.0, 0
        self._released_at = 0            # ``starts`` when MLX's freed buffers were last handed back
        # Evaluate and save cache arrays on the scheduler thread that owns their streams during shutdown.
        self.on_stop: Callable[[], Any] | None = None
        # streams take memory as they grow: the newest waits, or ends, when the next round's growth can't fit
        self.gate = (StreamGate(prompt_memory, admission.memory.per_token, admission.memory.round_bytes,
                                admission.budget, lanes=self.lanes)
                     if admission is not None and prompt_memory is not None and self.lanes > 1 else None)
        self.stall_s = 120.0            # no round, start or finish while requests wait: dump stacks
        self.stall_prefill_s = 900.0    # the same while one prefill runs
        self._watchdog = threading.Thread(target=self._watch, name="tensorfold-watchdog", daemon=True)
        self.decoded, self.prefilled = Meter(), ChunkRate()       # the live line's decode and prefill tok/s

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()
        if not self._watchdog.is_alive():
            self._watchdog.start()

    def _watch(self) -> None:
        """Dump every thread stack once when waiting requests stall, including prefill chunks; SIGUSR1 also dumps stacks on demand."""

        import faulthandler
        import sys

        mark: tuple[int, int, int, int, bool] | None = None
        since = time.perf_counter()
        dumped = False
        while not self._stop.wait(1.0):
            waiting = self._queue.qsize() + (self._held is not None) + len(self._jobs) + (self._starting is not None)
            starting = self._starting is not None
            now_mark = (self.rounds, self.starts, self.completed, getattr(self.engine, "prefill_chunks", 0), starting)
            now = time.perf_counter()
            if now_mark != mark or not waiting:
                mark, since, dumped = now_mark, now, False
                continue
            limit = self.stall_prefill_s if starting else self.stall_s
            if now - since > limit and not dumped:
                dumped = True
                print(f"[tensorfold] stalled {now - since:.0f}s: queued={self._queue.qsize()} "
                      f"held={self._held is not None} jobs={len(self._jobs)} active={self.engine.active_count} "
                      f"starting={starting}; every thread's stack follows", flush=True)
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def submit(self, job: ChatJob) -> None:
        if self._stop.is_set():
            raise RuntimeError("the scheduler is closed")
        self._queue.put(job)

    def cancel(self, cancellation: Cancellation) -> None:
        cancellation.cancel()
        for job in self._queue.remove(cancellation):
            self._finish_cancelled(job)

    def _finish_cancelled(self, job: ChatJob) -> None:
        if job.done.is_set():
            return
        job.error = RequestCancelled("request cancelled")
        job.proposer = None
        self.cancelled += 1
        self._finish(job)

    def _discard_job(self, job: ChatJob) -> None:
        self._jobs.pop(job.job_id, None)
        if job.stream is not None:
            self.engine.discard_stream(job.stream)
            job.stream.history_checkpoints = []
            job.stream.proposer = None
        self._finish_cancelled(job)

    def _cancel_active(self) -> None:
        if self._held is not None and self._held.cancellation.cancelled:
            self._finish_cancelled(self._held)
            self._held = None
        if self._filling is not None and self._filling.job.cancellation.cancelled:
            self._fill(abort=RequestCancelled("request cancelled"))      # stops between chunks, progress kept
        for job in list(self._jobs.values()):
            if job.cancellation.cancelled:
                self._discard_job(job)

    @property
    def active(self) -> int:
        return len(self._jobs)

    @property
    def waiting(self) -> int:
        """Requests not started yet: queued, or held until they fit."""

        return self._queue.qsize() + (self._held is not None)

    @property
    def filling(self) -> Any:
        """The job whose prompt is prefilling, or None."""

        filling = self._filling
        return None if filling is None else filling.job

    @staticmethod
    def finish_job(job: ChatJob, reason: str = "stop") -> None:
        """Ask the engine to stop a stream at the next round (safe from any thread)."""

        stream = job.stream
        if stream is not None and not stream.finished:
            stream.finish_reason = reason
            stream.finished = True
            stream.finished_at = time.perf_counter()

    # -- the loop -------------------------------------------------------------
    def _run(self) -> None:
        try:
            self._loop()
        finally:
            if self._filling is not None:
                self._fill(abort=RequestCancelled("server stopping"))
            if self.on_stop is not None:
                try:
                    self.on_stop()
                except Exception as exc:  # noqa: BLE001 - the shutdown must finish
                    print(f"[tensorfold] shutdown hook failed: {type(exc).__name__}: {exc}", flush=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._cancel_active()
            self._preempt_background()
            self._admit()
            self._starting = None if self._filling is None else self._filling.job
            self._retire_externally_finished()
            if self._filling is not None and (self.engine.active_count == 0 or self._rounds_had_turn()):
                self._fill()
                continue
            if self.engine.active_count == 0:
                if self._held is None and self._filling is None:
                    self._release_idle()
                    try:
                        self._held = self._queue.get(timeout=self.idle_wait)
                    except queue.Empty:
                        continue
                continue  # _admit starts it
            if self._filling is None:
                self._credit, self._rounds_left = 0.0, 0     # no prompt waits: rounds owe nothing
            if self.gate is not None:
                self._gate_round()
            started = self.clock()
            try:
                landed = self.engine.step()
            except Exception as exc:  # noqa: BLE001 - one bad round must not kill the server
                self.failed_rounds += 1
                traceback.print_exception(exc)
                error_type, message = type(exc).__name__, str(exc)
                exc.__traceback__ = exc.__cause__ = exc.__context__ = None
                for job in list(self._jobs.values()):
                    job.error = RoundError(error_type, message)
                    if job.stream is not None:
                        self.engine.discard_stream(job.stream)
                        job.stream.finish_reason = "error"
                        job.stream.history_checkpoints = []
                        job.stream.proposer = None
                    job.proposer = None
                    self._finish(job)
                self._jobs.clear()
                self.engine.reset()
                job = landed = tokens = None
                continue
            self.rounds += 1
            self._spend_round(self.clock() - started)
            self.decoded.add(sum(len(t) for t in landed.values()))
            self._cancel_active()
            if self.engine.round_stats:
                last = self.engine.round_stats[-1]
                if last.total_ms > self.slow_round_ms:
                    print(f"[tensorfold] slow round {last.total_ms:.0f} ms streams={last.streams} "
                          f"width={last.width} rows={last.rows} forward={last.forward_ms:.0f}", flush=True)
                if len(self.engine.round_stats) > 4096:
                    del self.engine.round_stats[:2048]
            for stream_id, tokens in landed.items():
                job = self._jobs.get(stream_id)
                if job is None:
                    continue
                if tokens:
                    job.chunks.put(list(tokens))
                if job.stream is not None and job.stream.finished:
                    del self._jobs[stream_id]
                    self._retire(job)
            job = landed = tokens = None

    def _release_idle(self) -> None:
        """Once no stream is left and nothing waits, hand MLX's freed buffers back: no round can want them now."""

        if self.prompt_memory is not None and self._released_at != self.starts and self._queue.empty():
            self._released_at = self.starts
            getattr(self.engine, "release_rounds", lambda: None)()
            self.prompt_memory.release_freed()

    def _preempt_background(self) -> None:
        """Release background work until the next foreground request has both a lane and enough memory."""

        waiting = self._held
        if waiting is None or waiting.background:
            waiting = self._queue.peek_foreground()
        if waiting is None or waiting.cancellation.cancelled:
            return
        if self._filling is not None and self._filling.job.background:
            self._preempt_filling()                   # the foreground job waits for the one prompt slot
        for job in self._jobs.values():
            if self.engine.active_count < self.lanes and self._fits(waiting):
                break
            if job.background and not job.preempted and job.stream is not None and not job.stream.finished:
                job.preempted = True
                self.preemptions += 1
                self.engine.discard_stream(job.stream)
                job.stream.finish_reason = "preempted"
                job.stream.history_checkpoints = []

    def _retire_externally_finished(self) -> None:
        for stream_id, job in list(self._jobs.items()):
            if job.stream is not None and job.stream.finished:
                del self._jobs[stream_id]
                self._retire(job)

    def _admit(self) -> None:
        """Admit fitting jobs, one prompt filling at a time in its own chunks; the loop runs rounds between chunks."""

        while self._filling is None and self.engine.active_count < self.lanes:
            job = self._held
            self._held = None
            if job is not None and job.background and self._queue.foreground_waiting():
                self._queue.put(job)          # a request that arrived since goes first
                job = None
            if job is None:
                try:
                    job = self._queue.get_nowait()
                except queue.Empty:
                    return
            if not self._fits(job):
                self._held = job              # waits until a live stream finishes and frees its memory
                return
            if job.cancellation.cancelled:
                self._finish_cancelled(job)
                continue
            if self.decode_share <= 0:
                self._start_job(job)                  # decode_share 0: each prompt whole before any round (0.3.6.2)
                continue
            self._open_job(job)
            if self._filling is not None and (self.engine.active_count == 0 or self._rounds_had_turn()):
                self._fill()                     # a one-chunk prompt starts now, as before

    def _fits(self, job: ChatJob) -> bool:
        """Start alone for prompt admission to validate memory, or beside streams only when prompt memory and admission projections fit."""

        if self.engine.active_count == 0:
            return True
        reply = self._reserved(int(job.max_tokens))
        memory = self.prompt_memory
        if memory is not None and not memory.would_fit(len(job.prompt_ids), reply):
            return False
        if self.admission is None:
            return True
        live = [(n, min(len(j.prompt_ids) + int(j.max_tokens), n + self._reserved(int(j.max_tokens))))
                for j in self._jobs.values() if j.stream is not None and not j.stream.finished
                for n in [len(j.stream.context)]]
        return self.admission.admits(len(job.prompt_ids), len(job.prompt_ids) + reply, live)

    def _reserved(self, reply: int) -> int:
        """The reply tokens a stream holds memory for ahead of its length: the gate's horizon when it guards rounds."""

        return reply if self.gate is None else min(reply, self.gate.horizon)

    def _gate_round(self) -> None:
        """Hold back the newest streams whose growth can't fit, or end the newest when even the oldest can't grow."""

        live = sorted((j for j in self._jobs.values() if j.stream is not None and not j.stream.finished),
                      key=lambda j: j.started_at)
        plan = self.gate.plan([(j.stream.stream_id, len(j.stream.context), len(j.prompt_ids) + int(j.max_tokens))
                               for j in live])
        if set(plan.paused) != self.engine.paused:
            print(f"[tensorfold] memory: {len(plan.paused)} of {len(live)} streams wait for room (newest first)",
                  flush=True)
        self.engine.paused = set(plan.paused)
        for job in (j for j in live if j.stream.stream_id in plan.ended):
            job.error = RequestError(
                f"This server ran out of memory with {len(live)} streams decoding, so the newest (this request, "
                f"after {len(job.stream.emitted)} tokens) was stopped for the older ones to finish. Retry it, "
                "shorten the prompt or max_tokens, or start the server with a smaller --parallel.")
            print(f"[tensorfold] memory: ended {job.job_id}, the newest of {len(live)} streams", flush=True)
            self.engine.discard_stream(job.stream)
            job.stream.finish_reason = "error"
            job.stream.history_checkpoints = []
            del self._jobs[job.stream.stream_id]
            self._finish(job)
            if self.prompt_memory is not None:
                self.prompt_memory.release_freed()      # its arrays are free now: the next plan sees the room

    def _read_disk_block(self, prompt: list[int], usable: Any = None) -> None:
        """Put the longest stored prefix ``usable`` accepts (system blocks, saved conversations) in the store."""

        if self.checkpoints is None or (self.disk_blocks is None and self.session_blocks is None):
            return
        try:
            have = self.checkpoints.longest(prompt, usable)
            found: Any = None
            for blocks, pinned in ((self.disk_blocks, True), (self.session_blocks, False)):
                hit = None if blocks is None else blocks.best(prompt, have if found is None else len(found[1]), usable)
                if hit is not None:
                    found = (hit[0], hit[1], blocks, pinned)
            if found is None:
                return
            from tensorfold.engine.prefix_snapshots import load_snapshot

            started = time.perf_counter()
            if self.prompt_memory is not None and not self.prompt_memory.allow_load(found[0].stat().st_size):
                return
            loaded = load_snapshot(found[0], self.model_id)
            if loaded is None:
                return
            tokens, cache = loaded
            self.checkpoints.insert(tokens, cache, last_prompt=tokens, pinned=found[3])
            found[2].touch(tokens)
            print(f"[tensorfold] read {'system-block' if found[3] else 'conversation'} snapshot tokens={len(tokens)} "
                  f"from disk in {time.perf_counter() - started:.2f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - a bad file costs a prefill, never the request
            print(f"[tensorfold] snapshot read failed: {type(exc).__name__}: {exc}", flush=True)

    def _start_job(self, job: ChatJob) -> None:
        """Open ``job`` and prefill its whole prompt now (the loop's admissions run rounds between chunks)."""

        self._open_job(job)
        while self._filling is not None and self._filling.job is job:
            self._fill()

    def _open_job(self, job: ChatJob) -> None:
        """Find its stored prefix, reserve its memory and make its stream; its prompt then fills a chunk a step."""

        job.started_at = time.perf_counter()
        self.starts += 1
        self._starting = job
        shared_at: set[int] = set()
        try:
            job.cancellation.check()
            memory = self.prompt_memory
            if memory is not None:
                getattr(self.engine, "release_rounds", lambda: None)()     # no stream live: rows are nobody's (#95)
                memory.begin(len(job.prompt_ids), self._reserved(int(job.max_tokens)),
                             admit=self.checkpoints is None or job.vision is not None)
                if job.vision is not None:
                    memory.require_workspace(self.engine.model.vision.estimate_workspace_bytes(job.vision))
            cache = None
            cached = 0
            last_prompt: list[int] | None = None
            checkpoints_at: list[int] = []
            # checkpoints sit at the prompt's chunk starts: a shared prefix is kept at the start at or before its end
            starts = self.engine.prompt_chunks(job.prompt_ids)
            shared_at = {starts.floor(n) for n in job.shared_prefix_lens} - {0}
            if self.checkpoints is not None and job.vision is None:
                usable = lambda n: n in starts
                self._read_disk_block(job.prompt_ids, usable)
                entry = self.checkpoints.peek(job.prompt_ids, usable=usable)
                take = False
                if memory is not None:
                    # Keep the resumed prefix through admission; use its stored arrays as the working cache if copying cannot fit.
                    memory.require(current_cache=None if entry is None else entry.cache, keep=entry)
                    take = entry is not None and not memory.fits_now()
                hit = self.checkpoints.match(job.prompt_ids, usable=usable, take=take)
                entry = None        # held through the prefill, a stored prefix evicted for this prompt's copy stays
                if hit is not None:
                    cached, cache, last_prompt = hit
                    if self.disk_blocks is not None and cached in shared_at:
                        self.disk_blocks.touch(job.prompt_ids[:cached])
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
            job.stream = stream
            steps = self.engine.begin_stream(stream, cache=cache, cached_tokens=cached, checkpoints_at=checkpoints_at)
        except Exception as exc:  # noqa: BLE001 - reported to the waiting request, as a failed prefill is
            self._end_fill(job, shared_at, exc)
            return
        self._filling = Filling(job, steps, shared_at)

    def _keep_checkpoints(self, job: ChatJob, shared_at: set[int]) -> None:
        """Store the prefixes the job's prefill kept (system blocks pinned and saved to disk), once."""

        stream = job.stream
        if stream is None or job.vision is not None:
            return
        kept, stream.history_checkpoints = stream.history_checkpoints, []
        for tokens, snapshot in kept if self.checkpoints is not None else ():
            shared = len(tokens) in shared_at
            self.checkpoints.insert(tokens, snapshot, last_prompt=job.prompt_ids, pinned=shared)
            if self.snapshot_dir is not None and shared:
                self._persist(tokens, snapshot)

    def _persist(self, tokens: list[int], cache: list[Any]) -> None:
        """Write a system-block snapshot to disk once, so a restart does not lose it."""

        from tensorfold.engine.prefix_snapshots import save_snapshot

        try:
            started = time.perf_counter()
            path = save_snapshot(self.snapshot_dir, self.model_id, tokens, cache)
            if path is not None:
                print(f"[tensorfold] saved system-block snapshot tokens={len(tokens)} "
                      f"in {time.perf_counter() - started:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - a full disk must not fail the request
            print(f"[tensorfold] snapshot save failed: {type(exc).__name__}: {exc}", flush=True)

    def _retire(self, job: ChatJob) -> None:
        if job.cancellation.cancelled:
            self._discard_job(job)
            return
        stream = job.stream
        if stream is not None and getattr(stream, "error", None) is not None and job.error is None:
            job.error = stream.error                  # its grammar failed: this request alone answers with the error
        retained = self.engine.finished_caches.pop(job.job_id, None)
        if retained is not None and self.checkpoints is not None and job.vision is None:
            tokens, cache = retained
            if len(tokens) > len(job.prompt_ids):
                self.checkpoints.insert(tokens, cache, last_prompt=job.prompt_ids)
        if stream is not None and stream in self.engine.streams:
            self.engine.streams.remove(stream)
        self.completed += 1
        self._finish(job)

    @staticmethod
    def _finish(job: ChatJob) -> None:
        job.finished_at = time.perf_counter()
        job.chunks.put(None)
        job.done.set()
