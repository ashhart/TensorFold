"""Reserve prompt and reply memory before cache growth or checkpoint copies."""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from typing import Any

from tensorfold.server.errors import RequestError
from tensorfold.server.memory_budget import (PROBE_REPEATS, PROCESS_BYTES, CacheMemory, GIB, budget_ceiling,
                                             cache_nbytes, process_footprint, raise_hint)


def attention_geometry(model: Any) -> tuple[int, int]:
    """Query heads and largest score-row part, from configuration and attention modules."""

    configurations: list[tuple[int, int, bool]] = []
    rows = 144
    seen: set[int] = set()

    def visit(value: Any, depth: int = 0) -> None:
        nonlocal rows
        if depth > 8 or id(value) in seen or value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        seen.add(id(value))
        if hasattr(value, "ndim") and hasattr(value, "nbytes"):
            return
        configured = getattr(value, "num_attention_heads", 0)
        dim = getattr(value, "head_dim", getattr(value, "dims", 0))
        dim = int(dim) if isinstance(dim, int) else 0
        part = getattr(value, "split_rows", 0)
        explicit_parts = isinstance(part, int) and part > 0
        if isinstance(configured, int) and configured > 0:
            configurations.append((configured, dim, explicit_parts))
        if hasattr(value, "q_proj") and hasattr(value, "k_proj"):
            for name in ("heads", "num_heads"):
                count = getattr(value, name, 0)
                if isinstance(count, int) and count > 0:
                    configurations.append((count, dim, explicit_parts))
        if isinstance(part, int):
            rows = max(rows, part)
        children = list(value.values()) if isinstance(value, Mapping) else list(value) \
            if isinstance(value, (list, tuple)) else []
        if hasattr(value, "__dict__"):
            children.extend(vars(value).values())
        for child in children:
            visit(child, depth + 1)

    visit(model)
    fallback = [heads for heads, dim, parts in configurations if parts or dim not in (64, 80, 128)]
    return (max(fallback) if fallback else 0 if configurations else -1), rows


def pass_row_bytes(model: Any) -> int:
    """Bytes the shared expert call holds per row, plus half again for margin; 0 without experts."""

    args = getattr(model, "args", None)
    top, hidden, inner = (int(getattr(args, name, 0) or 0)
                          for name in ("num_experts_per_tok", "hidden_size", "moe_intermediate_size"))
    return 3 * top * (6 * hidden + 8 * inner) // 2 if top and hidden and inner else 0


def probe_tokens(tokenizer: Any) -> list[int]:
    """Real text for the admission's probe: this module's source, as the model's tokenizer reads it."""

    from pathlib import Path

    try:
        return [int(t) for t in tokenizer.encode(Path(__file__).read_text(), add_special_tokens=False)]
    except Exception:  # noqa: BLE001 - a tokenizer that can't: the probe falls back to synthetic ids
        return []


def cached_rows(cache: Any) -> int:
    """Rows a live prompt cache holds (its layers' largest offset); 0 for none."""

    return max((int(getattr(c, "offset", 0) or 0) for c in cache), default=0) if cache is not None else 0


class OpenPrompt:
    """A prompt admitted and still filling: its tokens, its reply's reservation and its working cache once seen."""

    __slots__ = ("prompt", "reply", "cache")

    def __init__(self, prompt: int, reply: int) -> None:
        self.prompt, self.reply, self.cache = int(prompt), int(reply), None


class PromptMemory:
    """One model's profile, learned from a request's first existing prefill chunk."""

    def __init__(self, budget_bytes: int, model: Any, *, runtime: Any = None, store: Any = None,
                 window_tokens: int = 0, overhead_bytes: int = PROCESS_BYTES, bootstrap_bytes: int = 256 * 1024**2,
                 chunk_rows: int = 2048):
        if runtime is None:
            import mlx.core as runtime
        self.runtime, self.store = runtime, store
        self.process_budget = int(budget_bytes)      # the whole process; ``budget`` is MLX's share of it
        self.budget = max(0, int(budget_bytes) - int(overhead_bytes))
        self.bootstrap = int(bootstrap_bytes)
        self.window = int(window_tokens)
        self.chunk_rows = max(1, int(chunk_rows))     # a full prompt chunk: only its peak sizes the workspace
        self.affordable: int | None = None
        self.resumable: int | None = None
        self.carry, self._probe_base = 0, None
        self.stream_per_token = 0      # a live stream's growth a token, beyond its cache (a draft model's context)
        self.heads, self.score_rows = attention_geometry(model)
        self.workspace_per_token = int(getattr(model, "prefill_workspace_per_token", 0) or 0)
        self.pass_row_bytes = pass_row_bytes(model)
        self.profile: CacheMemory | None = None
        self.observed_work = 0
        self.workspace_profiled = False
        self.prompt = self.reply = 0                   # the focused open prompt's (``focus``)
        self.open: list[OpenPrompt] = []               # every admitted prompt still filling, oldest first
        self.current: OpenPrompt | None = None
        self._memory_lock = RLock()

    def memory_snapshot(self, reset_peak: bool = False) -> dict[str, int]:
        with self._memory_lock:
            memory = {"active": int(self.runtime.get_active_memory()),
                      "cache": int(self.runtime.get_cache_memory()), "peak": int(self.runtime.get_peak_memory()),
                      "budget": self.process_budget, "mlx_budget": self.budget}
            footprint = process_footprint()
            if footprint is not None:
                memory["footprint"] = footprint
            if reset_peak and (not self.open or self.workspace_profiled):
                self.runtime.reset_peak_memory()
            return memory

    def release_freed(self) -> None:
        """MLX's cache of freed buffers back to the system; live arrays, retained prefixes among them, stay."""

        with self._memory_lock:
            self.runtime.clear_cache()

    def _used(self) -> int:
        return int(self.runtime.get_active_memory() + self.runtime.get_cache_memory())

    def held(self) -> int:
        """MLX memory no admission can take back: live buffers less retained prefixes (freed buffers count as free)."""

        return max(0, int(self.runtime.get_active_memory()) - (self.store.nbytes if self.store is not None else 0))

    def _reclaim(self, keep: Any = None) -> bool:
        before = self._used()
        self.runtime.clear_cache()
        if self._used() < before:
            return True
        if self.store is not None and self.store.evict_one(keep=keep):
            self.runtime.clear_cache()
            return True
        return False

    def begin(self, prompt: int, reply: int, *, admit: bool = True) -> OpenPrompt:
        """Open a prompt beside those still filling and focus it; a refused one closes again."""

        with self._memory_lock:
            opened = OpenPrompt(prompt, reply)
            self.open.append(opened)
            self.focus(opened)
            self.runtime.reset_peak_memory()
            try:
                if self.profile is None and self.store is not None and self.store._entries:
                    self.observe_cache(self.store._entries[0].cache, workspace=False)
                if admit:
                    self.require()
            except BaseException:
                self.end(opened)
                raise
            return opened

    def focus(self, opened: OpenPrompt | None) -> None:
        """The open prompt whose chunk runs next: every check is its own, beside the other open prompts' growth."""

        with self._memory_lock:
            self.current = opened
            self.prompt, self.reply = (opened.prompt, opened.reply) if opened is not None else (0, 0)

    def end(self, opened: OpenPrompt | None = None) -> None:
        """``opened`` (by default the focused prompt) stopped filling: its reservation goes."""

        with self._memory_lock:
            opened = self.current if opened is None else opened
            self.open = [o for o in self.open if o is not opened]
            if opened is self.current:
                self.focus(None)

    def _work(self, tokens: int) -> int:
        if self.profile is None:
            return self.bootstrap
        # MLX's limit is this budget, so its eval waits on queued work before more old buffers than this pile up
        growth = self.profile.growth_bytes(tokens)
        scores = (self.workspace_per_token * int(tokens) if self.workspace_per_token
                  else 2 * self.score_rows * max(0, self.heads) * int(tokens) * 2)
        return max(self.bootstrap, self.observed_work) + growth + scores

    def need(self, tokens: int, resident: int, *, started: bool = False, rows: int = 0, copies: int = 1) -> int:
        """Bytes ``tokens`` need beside ``resident``, which holds the growth of the ``rows`` already cached."""

        if self.profile is None:
            return int(resident) + self.bootstrap
        return int(resident) + self._held_need(tokens, started, rows, copies) + self._work(tokens)

    def _held_need(self, tokens: int, started: bool, rows: int, copies: int = 1) -> int:
        """What a prompt of ``tokens`` holds between chunks: its caches and a draft model's context past ``rows``."""

        beyond = max(0, self.stream_per_token - self.profile.bytes_per_token)      # a draft model's context
        return ((0 if started else self.carry) + copies * self.profile.cache_bytes(tokens)
                + beyond * max(0, int(tokens) - int(rows)))

    def _others(self, tokens: int) -> int:
        """Bytes the other open prompts still take: their growth to prompt and reply, and a larger chunk workspace."""

        if self.profile is None:
            return 0
        grow, work = 0, self._work(tokens)
        for other in self.open:
            if other is self.current:
                continue
            total, cache = other.prompt + other.reply, other.cache
            held = cache_nbytes(cache) if cache is not None else 0
            grow += self._held_need(total, cache is not None, cached_rows(cache)) - held
            work = max(work, self._work(total))
        return grow + work - self._work(tokens)

    def projected(self, prompt: int, *, current_cache: Any = None, extra_bytes: int = 0) -> int:
        current = cache_nbytes(current_cache) if current_cache is not None else 0
        tokens = int(prompt) + self.reply
        resident = max(0, self._used() - current) + int(extra_bytes) + self._others(tokens)
        return self.need(tokens, resident, started=current_cache is not None, rows=cached_rows(current_cache))

    def require(self, current_cache: Any = None, keep: Any = None) -> None:
        """Reclaim until the prompt fits, never evicting ``keep``; refuse when nothing is left to free."""

        if not self.fits(current_cache, keep=keep):
            raise self._refusal(current_cache)

    def require_workspace(self, size: int) -> None:
        """Reserve image encoder workspace beside the complete prompt and reply cache before encoding."""
        if type(size) is not int or size < 0:
            raise ValueError("workspace size must be a nonnegative byte count")
        with self._memory_lock:
            if not self._make_room(size):
                raise RequestError("image encoding and this prompt exceed the memory budget; reduce image "
                                   "resolution or count, shorten the prompt, or use a smaller checkpoint")

    def fits(self, current_cache: Any = None, *, keep: Any = None) -> bool:
        """Reclaim for the prompt, never ``keep``, only while what is left to free could still make room."""

        return self._make_room(0, current_cache, keep=keep)

    def would_fit(self, prompt: int, reply: int) -> bool:
        """Whether a request would fit now once every retained prefix and freed buffer is released; no side effects."""

        with self._memory_lock:
            saved = self.prompt, self.reply, self.current
            self.prompt, self.reply, self.current = int(prompt), int(reply), None     # beside every open prompt
            try:
                freeable = int(self.runtime.get_cache_memory()) + (self.store.nbytes if self.store is not None else 0)
                return self.projected(self.prompt) - freeable <= self.budget
            finally:
                self.prompt, self.reply, self.current = saved

    def fits_now(self) -> bool:
        """Whether the prompt fits beside every retained prefix, after releasing only freed MLX buffers."""

        if self.projected(self.prompt) <= self.budget:
            return True
        self.runtime.clear_cache()
        return self.projected(self.prompt) <= self.budget

    def _refusal(self, current_cache: Any) -> RequestError:
        current = cache_nbytes(current_cache) if current_cache is not None else 0
        store = self.store.nbytes if self.store is not None else 0
        # what stays once freed buffers and retained prefixes are gone, and the open prompts: the refusal's own terms
        held = max(0, int(self.runtime.get_active_memory()) - store - current)
        started, rows = current_cache is not None, cached_rows(current_cache)
        top = max(0, (self.window or self.prompt + self.reply) - self.reply)
        lo, hi = 0, top
        while lo < hi:
            mid = (lo + hi + 1) // 2
            beside = held + self._others(mid + self.reply)
            if (self.profile is not None
                    and self.need(mid + self.reply, beside, started=started, rows=rows) <= self.budget):
                lo = mid
            else:
                hi = mid - 1
        tokens = self.prompt + self.reply
        needed = self.need(tokens, held + self._others(tokens), started=started, rows=rows)
        return RequestError(f"This request needs about {needed / GIB:.1f} GiB of the {self.budget / GIB:.1f} GiB "
                            f"MLX may use (this server's {self.process_budget / GIB:.1f} GiB memory budget less "
                            f"{(self.process_budget - self.budget) / GIB:.1f} GiB for the rest of the process); it "
                            f"fits up to {lo:,} tokens in the prompt with {self.reply:,} reply tokens. Shorten the "
                            "prompt or max_tokens (the reply is reserved in full), or start the server with a smaller "
                            "--context so clients compact sooner; --drafter none, a smaller or more quantized "
                            "checkpoint, or a Mac with more RAM leaves more room.")

    def before_chunk(self, cache: Any, rows: int) -> None:
        if self.current is not None:
            self.current.cache = cache                 # the other open prompts' checks count what it holds
        self.require(cache if self.profile is not None else None)
        if not self.workspace_profiled:
            # the peak must start after admission freed prefixes, or they would count as workspace
            with self._memory_lock:
                self.runtime.reset_peak_memory()

    def observe_cache(self, cache: Any, *, workspace: bool = True, rows: int | None = None) -> None:
        with self._memory_lock:
            measured = CacheMemory.from_cache(cache)
            if measured.bytes_per_token and self.heads < 0 and not self.workspace_per_token:
                raise RequestError("Cannot size this checkpoint's attention workspace; its configuration must "
                                   "specify num_attention_heads before long prompts can be admitted.")
            if self.profile is None:
                self.profile = measured
            else:
                self.profile = CacheMemory(max(self.profile.fixed_bytes, measured.fixed_bytes),
                                           max(self.profile.bytes_per_token, measured.bytes_per_token),
                                           max(self.profile.step, measured.step),
                                           max(self.profile.entry_bytes_per_token, measured.entry_bytes_per_token))
            if workspace and not self.workspace_profiled:
                work = max(0, int(self.runtime.get_peak_memory()) - int(self.runtime.get_active_memory()))
                # a shorter chunk only raises the floor: its workspace is smaller than a full chunk's
                full = rows is None or int(rows) >= self.chunk_rows
                self.observed_work = work if full else max(self.observed_work, work)
                self.workspace_profiled = full

    def pass_bytes(self, sizes: list[int]) -> int:
        """Memory held beyond the largest chunk: expert pairs or a workspace for the other chunks."""

        if self.pass_row_bytes:
            return (sum(sizes) - max(sizes)) * self.pass_row_bytes
        return (len(sizes) - 1) * max(self.bootstrap, self.observed_work)

    def pass_width(self, cache: Any, sizes: list[int]) -> int:
        """How many consecutive chunks one forward takes within the budget; freed buffers count as free."""

        with self._memory_lock:
            if self.profile is None or len(sizes) < 2:
                return 1
            free = int(self.runtime.get_cache_memory())
            width = len(sizes)
            while width > 1 and self.projected(self.prompt, current_cache=cache,
                                               extra_bytes=self.pass_bytes(sizes[:width])) - free > self.budget:
                width -= 1
            return width

    def pass_room(self, cache: Any, sizes: list[int], extra: int) -> bool:
        """Whether a pass of these chunks fits with ``extra`` bytes more (a larger cache of freed buffers) beside it."""

        with self._memory_lock:
            if self.profile is None:
                return False
            free = int(self.runtime.get_cache_memory())
            return self.projected(self.prompt, current_cache=cache,
                                  extra_bytes=self.pass_bytes(sizes) + int(extra)) - free <= self.budget

    def after_chunk(self, cache: Any, rows: int) -> None:
        if self.current is not None:
            self.current.cache = cache
        self.observe_cache(cache, rows=rows)
        if self._probe_base is not None:          # what a prompt holds between chunks outside its cache
            held = int(self.runtime.get_active_memory()) - cache_nbytes(cache) - self._probe_base
            self.carry = max(self.carry, held)
        self.require(cache)

    def profile_probe(self, engine: Any, tokens: Any = None) -> None:
        """Size the cache, a full chunk's workspace and what a prompt holds between chunks with one probe prompt."""

        from tensorfold.server.cancellation import Cancellation, PrefillGuard

        # real text: a mixture of experts routes it as it routes a prompt (synthetic ids reach fewer experts)
        text = [int(t) for t in tokens or ()] or [1000 + i for i in range(self.chunk_rows + 64)]
        probe = (text * (-(-(self.chunk_rows + 64) // len(text))))[:self.chunk_rows + 64]
        previous, engine.prefill_guard = engine.prefill_guard, PrefillGuard(Cancellation(), self)
        works = []
        try:
            for _ in range(PROBE_REPEATS):         # the worst of a few (memory_budget.PROBE_REPEATS)
                self.runtime.clear_cache()
                self._probe_base, self.workspace_profiled = int(self.runtime.get_active_memory()), False
                engine.prefill_prefix(probe, cache=None, cached_tokens=0)
                works.append(self.observed_work)
            self.observed_work = max(works)
        finally:
            engine.prefill_guard, self._probe_base = previous, None
            self.runtime.clear_cache()

    def sized(self, engine: Any, probes: Any, tokens: Any = None) -> Any:
        """Run ``probes`` (the concurrency measurement), then this admission's own probe on ``tokens``; their result."""

        import gc

        try:
            measured = probes()
            stream = getattr(measured, "memory", None)
            self.stream_per_token = int(getattr(stream, "per_token", 0) or 0)
            release = getattr(engine, "release_rounds", None)
            if release is not None:
                release()                      # the probes' last shared round: no stream keeps rows of it
            gc.collect()                       # arrays the probes left in reference cycles, before anything is sized
            self.profile_probe(engine, tokens)
        except RequestError:
            raise ValueError(self._no_room()) from None
        gc.collect()
        self.runtime.clear_cache()
        return measured

    def _no_room(self) -> str:
        need = self.projected(0)
        hint = raise_hint(need + self.process_budget - self.budget, budget_ceiling(self.runtime))
        return (f"this server's {self.process_budget / GIB:.1f} GiB memory budget ({self.budget / GIB:.1f} GiB for MLX) "
                f"leaves no room for a prompt beside the model: it and one prompt chunk need about "
                f"{need / GIB:.1f} GiB. {hint or 'Serve it'} on a Mac with more memory, without its draft model "
                "(--drafter none), or use a smaller or more quantized checkpoint")

    def spare(self, window: int, work: int = 0) -> int:
        """MLX memory left idle beside what is held now, one ``window``-token request and ``work`` more (a round's)."""

        with self._memory_lock:
            return 0 if self.profile is None else max(0, self.budget - self.need(window, self.held()) - int(work))

    def fit_window(self, window: int, fit: bool) -> tuple[int, bool]:
        """(the context window, whether memory lowered it): omitted, what the budget affords; explicit, it must fit."""

        self.affordable = self.largest_window(window)
        if not self.affordable:
            raise ValueError(self._no_room())
        # with prompts retained, the next turn resumes only if this one's prompt can be kept beside the working cache
        kept = self.largest_window(window, resumable=True) if self.store is not None else None
        self.resumable = kept                 # the longest request whose prompt is kept for the next turn
        resumable = kept or self.affordable
        fitted = bool(fit) and (not window or resumable < window)
        if fitted:
            window = resumable // 1024 * 1024 if resumable >= 1024 else resumable
        elif window > self.affordable:
            raise ValueError(f"a {window:,}-token context window does not fit this server's memory budget: the most "
                             f"one request can use is {self.affordable:,} tokens (prompt plus reply)")
        self.window = int(window)
        return self.window, fitted

    def largest_window(self, limit: int = 0, *, resumable: bool = False) -> int | None:
        """Most prompt-plus-reply tokens one request holds (``resumable``: and keeps its prompt for the next turn)."""

        with self._memory_lock:
            if self.profile is None:
                return None
            retained = self.store.nbytes if self.store is not None else 0
            floor = max(0, int(self.runtime.get_active_memory()) - retained)
            kept = 2 if resumable else 1

            def fits(tokens: int) -> bool:
                return self.need(tokens, floor, copies=kept) <= self.budget

            if not fits(0):
                return 0
            lo, hi = 0, int(limit) if limit > 0 else 1 << 24
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if fits(mid) else (lo, mid - 1)
            return lo

    def _over_store_budget(self, size: int) -> bool:
        store = self.store
        return (store is not None and store.budget_bytes is not None and size > store.budget_bytes
                and not store.admit_oversize)

    def _extra_fits_after_reclaim(self, size: int, *, current_cache: Any = None, keep: Any = None) -> bool:
        """Do not evict prefixes for a copy or a prompt that still cannot fit with every reclaimable buffer gone."""

        store = self.store
        stored = 0 if store is None else store.nbytes - sum(e.nbytes for e in store._entries if e is keep)
        freeable = int(self.runtime.get_cache_memory()) + stored
        return self.projected(self.prompt, current_cache=current_cache, extra_bytes=size) - freeable <= self.budget

    def _make_room(self, size: int, current_cache: Any = None, *, keep: Any = None) -> bool:
        """Reclaim for ``size`` more bytes, first checking at every step that what is left to free could make room."""

        while self.projected(self.prompt, current_cache=current_cache, extra_bytes=size) > self.budget:
            # an eviction that freed less than its entry's size (arrays still held elsewhere) stops the next ones
            if (not self._extra_fits_after_reclaim(size, current_cache=current_cache, keep=keep)
                    or not self._reclaim(keep=keep)):
                return False
        return True

    def allow_checkpoint(self, cache: Any) -> bool:
        size = cache_nbytes(cache)
        if self.store is None or self._over_store_budget(size):
            return False
        return self._make_room(size, cache)

    def allow_load(self, size: int) -> bool:
        return not self._over_store_budget(size) and self._make_room(size)


__all__ = ["OpenPrompt", "PromptMemory", "attention_geometry", "pass_row_bytes", "probe_tokens"]
