"""The largest prompt chunk a family offers whose working memory still leaves room for a long context."""

from __future__ import annotations

from typing import Any, Sequence

# tokens of context a larger chunk must leave room for (or the model's window, if smaller)
CONTEXT_FLOOR = 131072


def choose(make_engine: Any, steps: Sequence[int], budget: int, tokens: Sequence[int], window: int = 0) -> int:
    """Probe steps smallest first, a larger one only if its worst case fits; take the largest that leaves the floor."""

    import mlx.core as mx

    from tensorfold.engine.family_common import cache_arrays
    from tensorfold.server.memory_budget import PROBE_REPEATS, CacheMemory
    from tensorfold.server.prompt_memory import PromptMemory

    steps = sorted({int(s) for s in steps})
    small = steps[0]
    if len(steps) == 1:
        return small
    engine = make_engine(small)
    text = [int(t) for t in tokens] or [1000 + i for i in range(small + 64)]
    tighten = getattr(engine.model, "tighten_prefill", None)
    context = min(int(window) or CONTEXT_FLOOR, CONTEXT_FLOOR)
    memory = PromptMemory(int(budget), engine.model, runtime=mx, overhead_bytes=0)   # the server's window terms

    def measure(prober: Any, step: int) -> tuple[int, int, CacheMemory]:
        """Held bytes and workspace of one full ``step`` chunk, and the size of the cache it leaves."""

        probe = (text * (-(-(step + 64) // len(text))))[:step + 64]
        helds, works = [], []
        for _ in range(PROBE_REPEATS):             # the worst of a few: one probe's run-to-run noise can't move it
            mx.synchronize()
            mx.clear_cache()
            helds.append(int(mx.get_active_memory()))
            mx.reset_peak_memory()
            cache = prober.prefill_prefix(probe, cache=None, cached_tokens=0)
            mx.eval(*cache_arrays(cache))
            works.append(max(0, int(mx.get_peak_memory()) - int(mx.get_active_memory())))
            profile = CacheMemory.from_cache(cache)
            del cache
            getattr(prober, "release_rounds", lambda: None)()
            mx.clear_cache()
        return max(helds), max(works), profile

    while True:
        best = 0                                   # the largest step measured to leave the floor; held, work: its probe
        for step in steps:
            # its worst case: what is held, its cache, and a workspace grown at most with the chunk's square
            worst = held + memory.profile.cache_bytes(step + 64) + work * step * step // (best * best) if best else 0
            if worst > memory.budget:              # a probe that could pass the budget never runs
                break
            held, work, memory.profile = measure(engine if step == small else make_engine(step), step)
            memory.observed_work = work
            if memory.need(context, held, copies=2) > memory.budget:     # as largest_window(resumable=True) charges
                break
            best = step
        if best:
            return best
        if tighten is None or not tighten():      # nothing fits: a model that can, keeps less of a chunk in flight
            return small


__all__ = ["CONTEXT_FLOOR", "choose"]
