"""Reserve prompt and reply memory before cache growth or checkpoint copies."""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from typing import Any

from tensorfold.server.errors import RequestError
from tensorfold.server.memory_budget import (PROCESS_BYTES, CacheMemory, GIB, budget_ceiling, cache_nbytes,
                                             process_footprint, raise_hint)


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


def probe_tokens(tokenizer: Any) -> list[int]:
    """Real text for the admission's probe: this module's source, as the model's tokenizer reads it."""

    from pathlib import Path

    try:
        return [int(t) for t in tokenizer.encode(Path(__file__).read_text(), add_special_tokens=False)]
    except Exception:  # noqa: BLE001 - a tokenizer that can't: the probe falls back to synthetic ids
        return []


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
        self.carry, self._probe_base = 0, None
        self.heads, self.score_rows = attention_geometry(model)
        self.workspace_per_token = int(getattr(model, "prefill_workspace_per_token", 0) or 0)
        self.profile: CacheMemory | None = None
        self.observed_work = 0
        self.workspace_profiled = False
        self.prompt = self.reply = 0
        self._memory_lock = RLock()

    def memory_snapshot(self, reset_peak: bool = False) -> dict[str, int]:
        with self._memory_lock:
            memory = {"active": int(self.runtime.get_active_memory()),
                      "cache": int(self.runtime.get_cache_memory()), "peak": int(self.runtime.get_peak_memory()),
                      "budget": self.process_budget, "mlx_budget": self.budget}
            footprint = process_footprint()
            if footprint is not None:
                memory["footprint"] = footprint
            if reset_peak and (not self.prompt or self.workspace_profiled):
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

    def begin(self, prompt: int, reply: int, *, admit: bool = True) -> None:
        with self._memory_lock:
            self.prompt, self.reply = int(prompt), int(reply)
            self.runtime.reset_peak_memory()
            if self.profile is None and self.store is not None and self.store._entries:
                self.observe_cache(self.store._entries[0].cache, workspace=False)
            if admit:
                self.require()

    def end(self) -> None:
        with self._memory_lock:
            self.prompt = self.reply = 0

    def _work(self, tokens: int) -> int:
        if self.profile is None:
            return self.bootstrap
        # MLX's limit is this budget, so its eval waits on queued work before more old buffers than this pile up
        growth = self.profile.growth_bytes(tokens)
        scores = (self.workspace_per_token * int(tokens) if self.workspace_per_token
                  else 2 * self.score_rows * max(0, self.heads) * int(tokens) * 2)
        return max(self.bootstrap, self.observed_work) + growth + scores

    def projected(self, prompt: int, *, current_cache: Any = None, extra_bytes: int = 0) -> int:
        current = cache_nbytes(current_cache) if current_cache is not None else 0
        resident = max(0, self._used() - current)
        if self.profile is None:
            return resident + int(extra_bytes) + self.bootstrap
        tokens = int(prompt) + self.reply
        return resident + int(extra_bytes) + self.profile.cache_bytes(tokens) + self._work(tokens)

    def require(self, current_cache: Any = None, keep: Any = None) -> None:
        """Reclaim until the prompt fits, never evicting ``keep``; refuse when nothing is left to free."""

        if not self.fits(current_cache, keep=keep):
            raise self._refusal(current_cache)

    def require_workspace(self, size: int) -> None:
        """Reserve image encoder workspace beside the complete prompt and reply cache before encoding."""
        if type(size) is not int or size < 0:
            raise ValueError("workspace size must be a nonnegative byte count")
        with self._memory_lock:
            while self.projected(self.prompt, extra_bytes=size) > self.budget:
                if not self._reclaim():
                    raise RequestError("image encoding and this prompt exceed the memory budget; reduce image "
                                       "resolution or count, shorten the prompt, or use a smaller checkpoint")

    def fits(self, current_cache: Any = None, *, keep: Any = None) -> bool:
        while self.projected(self.prompt, current_cache=current_cache) > self.budget:
            if not self._reclaim(keep=keep):
                return False
        return True

    def would_fit(self, prompt: int, reply: int) -> bool:
        """Whether a request would fit now once every retained prefix and freed buffer is released; no side effects."""

        with self._memory_lock:
            saved = self.prompt, self.reply
            self.prompt, self.reply = int(prompt), int(reply)
            try:
                freeable = int(self.runtime.get_cache_memory()) + (self.store.nbytes if self.store is not None else 0)
                return self.projected(self.prompt) - freeable <= self.budget
            finally:
                self.prompt, self.reply = saved

    def fits_now(self) -> bool:
        """Whether the prompt fits beside every retained prefix, after releasing only freed MLX buffers."""

        if self.projected(self.prompt) <= self.budget:
            return True
        self.runtime.clear_cache()
        return self.projected(self.prompt) <= self.budget

    def _refusal(self, current_cache: Any) -> RequestError:
        top = max(0, (self.window or self.prompt + self.reply) - self.reply)
        lo, hi = 0, top
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.profile is not None and self.projected(mid, current_cache=current_cache) <= self.budget:
                lo = mid
            else:
                hi = mid - 1
        needed = self.projected(self.prompt, current_cache=current_cache)
        return RequestError(f"This request needs about {needed / GIB:.1f} GiB of the {self.budget / GIB:.1f} GiB "
                            f"MLX may use (this server's {self.process_budget / GIB:.1f} GiB memory budget less "
                            f"{(self.process_budget - self.budget) / GIB:.1f} GiB for the rest of the process); it "
                            f"fits up to {lo:,} tokens in the prompt with {self.reply:,} reply tokens. Shorten the "
                            "prompt or max_tokens (the reply is reserved in full), or start the server with a smaller "
                            "--context so clients compact sooner; --drafter none, a smaller or more quantized "
                            "checkpoint, or a Mac with more RAM leaves more room.")

    def before_chunk(self, cache: Any, rows: int) -> None:
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

    def after_chunk(self, cache: Any, rows: int) -> None:
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
        self.runtime.clear_cache()
        self._probe_base, self.workspace_profiled = int(self.runtime.get_active_memory()), False
        try:
            engine.prefill_prefix(probe, cache=None, cached_tokens=0)
        finally:
            engine.prefill_guard, self._probe_base = previous, None
            self.runtime.clear_cache()

    def sized(self, engine: Any, probes: Any, tokens: Any = None) -> Any:
        """Run ``probes`` (the concurrency measurement), then this admission's own probe on ``tokens``; their result."""

        import gc

        try:
            measured = probes()
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

    def fit_window(self, window: int, fit: bool) -> tuple[int, bool]:
        """(the context window, whether memory lowered it): omitted, what the budget affords; explicit, it must fit."""

        self.affordable = self.largest_window(window)
        if not self.affordable:
            raise ValueError(self._no_room())
        # with prompts retained, the next turn resumes only if this one's prompt can be kept beside the working cache
        kept = self.largest_window(window, resumable=True) if self.store is not None else None
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
            floor = max(0, int(self.runtime.get_active_memory()) - retained) + self.carry
            kept = 2 if resumable else 1

            def fits(tokens: int) -> bool:
                return floor + kept * self.profile.cache_bytes(tokens) + self._work(tokens) <= self.budget

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

    def _extra_fits_after_reclaim(self, size: int, *, current_cache: Any = None) -> bool:
        """Do not evict prefixes for a copy that still cannot fit with every reclaimable buffer gone."""

        freeable = int(self.runtime.get_cache_memory()) + (self.store.nbytes if self.store is not None else 0)
        return self.projected(self.prompt, current_cache=current_cache, extra_bytes=size) - freeable <= self.budget

    def _make_room(self, size: int, current_cache: Any = None) -> bool:
        """Reclaim for ``size`` more bytes, first checking at every step that what is left to free could make room."""

        while self.projected(self.prompt, current_cache=current_cache, extra_bytes=size) > self.budget:
            # an eviction that freed less than its entry's size (arrays still held elsewhere) stops the next ones
            if not self._extra_fits_after_reclaim(size, current_cache=current_cache) or not self._reclaim():
                return False
        return True

    def allow_checkpoint(self, cache: Any) -> bool:
        size = cache_nbytes(cache)
        if self.store is None or self._over_store_budget(size):
            return False
        return self._make_room(size, cache)

    def allow_load(self, size: int) -> bool:
        return not self._over_store_budget(size) and self._make_room(size)


__all__ = ["PromptMemory", "attention_geometry", "probe_tokens"]
