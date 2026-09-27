"""Reserve prompt and reply memory before cache growth or checkpoint copies."""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from typing import Any

from tensorfold.server.errors import RequestError
from tensorfold.server.memory_budget import CacheMemory, GIB, cache_nbytes


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


class PromptMemory:
    """One model's profile, learned from a request's first existing prefill chunk."""

    def __init__(self, budget_bytes: int, model: Any, *, runtime: Any = None, store: Any = None,
                 window_tokens: int = 0, overhead_bytes: int = 4 * GIB, bootstrap_bytes: int = 256 * 1024**2):
        if runtime is None:
            import mlx.core as runtime
        self.runtime, self.store = runtime, store
        self.budget = max(0, int(budget_bytes) - int(overhead_bytes))
        self.bootstrap = int(bootstrap_bytes)
        self.window = int(window_tokens)
        self.heads, self.score_rows = attention_geometry(model)
        self.profile: CacheMemory | None = None
        self.observed_work = 0
        self.workspace_profiled = False
        self.prompt = self.reply = 0
        self._memory_lock = RLock()

    def memory_snapshot(self, reset_peak: bool = False) -> dict[str, int]:
        with self._memory_lock:
            memory = {"active": int(self.runtime.get_active_memory()),
                      "cache": int(self.runtime.get_cache_memory()), "peak": int(self.runtime.get_peak_memory())}
            if reset_peak and (not self.prompt or self.workspace_profiled):
                self.runtime.reset_peak_memory()
            return memory

    def _used(self) -> int:
        return int(self.runtime.get_active_memory() + self.runtime.get_cache_memory())

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
        # Queued layer forwards can hold the old KV timelines while growth allocates their replacements.
        growth = self.profile.cache_bytes(tokens)
        scores = 2 * self.score_rows * max(0, self.heads) * int(tokens) * 2
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
        return RequestError(f"This request needs about {needed / GIB:.1f} GiB of MLX memory against a "
                            f"{self.budget / GIB:.1f} GiB budget after reserving process overhead; this server "
                            f"fits up to {lo:,} tokens in the prompt with {self.reply:,} reply tokens. "
                            "Reduce the prompt or max_tokens, disable retained prefixes with --prompt-cache-gib 0, "
                            "or use a smaller or more quantized checkpoint or a Mac with more RAM.")

    def before_chunk(self, cache: Any, rows: int) -> None:
        self.require(cache if self.profile is not None else None)
        if not self.workspace_profiled:
            # the peak must start after admission freed prefixes, or they would count as workspace
            with self._memory_lock:
                self.runtime.reset_peak_memory()

    def observe_cache(self, cache: Any, *, workspace: bool = True) -> None:
        with self._memory_lock:
            measured = CacheMemory.from_cache(cache)
            if measured.bytes_per_token and self.heads < 0:
                raise RequestError("Cannot size this checkpoint's attention workspace; its configuration must "
                                   "specify num_attention_heads before long prompts can be admitted.")
            if self.profile is None:
                self.profile = measured
            else:
                self.profile = CacheMemory(max(self.profile.fixed_bytes, measured.fixed_bytes),
                                           max(self.profile.bytes_per_token, measured.bytes_per_token),
                                           max(self.profile.step, measured.step))
            if workspace and not self.workspace_profiled:
                self.observed_work = max(0, int(self.runtime.get_peak_memory()) -
                                         int(self.runtime.get_active_memory()))
                self.workspace_profiled = True

    def after_chunk(self, cache: Any, rows: int) -> None:
        self.observe_cache(cache)
        self.require(cache)

    def _over_store_budget(self, size: int) -> bool:
        store = self.store
        return (store is not None and store.budget_bytes is not None and size > store.budget_bytes
                and not store.admit_oversize)

    def allow_checkpoint(self, cache: Any) -> bool:
        size = cache_nbytes(cache)
        if self.store is None or self._over_store_budget(size):
            return False
        while self.projected(self.prompt, current_cache=cache, extra_bytes=size) > self.budget:
            if not self._reclaim():
                return False
        return True

    def allow_load(self, size: int) -> bool:
        if self._over_store_budget(size):
            return False
        while self.projected(self.prompt, extra_bytes=size) > self.budget:
            if not self._reclaim():
                return False
        return True


__all__ = ["PromptMemory", "attention_geometry"]
