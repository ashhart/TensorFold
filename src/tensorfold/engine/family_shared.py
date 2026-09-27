"""Batch stream windows and draws while verifying, committing, and rolling back each stream independently."""

from __future__ import annotations

import time
from typing import Any

from tensorfold.engine.family_common import _PROFILE, _ROUND_LOG, cache_arrays


class SharedRounds:
    """Shared rounds for ``FamilyRounds``."""

    def _take_turns(self, live: list[tuple[Any, list[Any]]]) -> list[tuple[Any, list[Any]]]:
        """Select least recently served streams within stream and row limits, never cutting pending or forced rows."""

        order = sorted(range(len(live)), key=lambda i: (self._served.get(live[i][0].stream_id, -1), i))
        chosen: list[int] = []
        rows = 0
        for i in order:
            stream = live[i][0]
            need = min(self.family_width, self.batch_rows, 1 + len(stream.force)) if stream.drafts else 1
            if len(chosen) == self.batch_streams or (chosen and rows + need > self.batch_rows):
                break
            chosen.append(i)
            rows += need
        return [live[i] for i in sorted(chosen)]

    def _land_inflight(self, stream: Any) -> list[int]:
        """Land the queued token without queuing another; resume pipelining when the stream runs alone."""

        import mlx.core as mx

        current = self._inflight.pop(stream.stream_id)
        forced = self._forced_next(stream)
        if forced is not None:
            current = mx.array([forced], dtype=mx.uint32)   # the thinking budget's token, not the sample
        token = int(current.item())
        stream.rounds += 1
        got = stream.commit([token])
        stream.pending = [token]
        self._mode[stream.stream_id] = "verify"
        return got

    def _draw_streams(self, logits: Any, streams: list[tuple[Any, list[int]]]) -> Any:
        """Draw consecutive stream rows using model sampling when available, otherwise GPU sampling with each row's own settings."""

        import mlx.core as mx

        from tensorfold.engine.gpu_sampling import sample_rows

        own = getattr(self.model, "sample_streams", None)
        if callable(own) or callable(getattr(self.model, "sample", None)):
            parts, at = [], 0
            for _, positions in streams:
                parts.append(logits[at:at + len(positions)])
                at += len(positions)
            if callable(own):
                drawn = own(parts, [s for s, _ in streams], [p for _, p in streams])
            else:
                drawn = [self._draw(x, s, p) for x, (s, p) in zip(parts, streams)]
            return mx.concatenate([t if isinstance(t, mx.array) else mx.array([int(v) for v in t], dtype=mx.uint32)
                                   for t in drawn])
        samplings = [sampling for sampling, positions in streams for _ in positions]
        return sample_rows(logits, samplings, [p for _, positions in streams for p in positions])

    def _family_round_streams(self, entries: list[tuple[Any, list[Any]]]) -> dict[str, tuple[list[int], int, int]]:
        """Run stream windows together with each row's independent bits, then verify, commit, roll back, and draft per stream."""

        import mlx.core as mx

        from tensorfold.kernels.qwen.dense.v1.lane_tree import tree_paths

        model = self.model
        started = time.perf_counter()
        plans = []
        for stream, cache in entries:
            kind, drafts, forced, parents = self._plan_window(stream)
            plans.append([stream, cache, stream.cache_len, kind, drafts, forced, parents])

        self._allocate(plans)
        windows = [self._window_tokens(p[0], p[4]) for p in plans]
        lengths = [int(w.shape[0]) for w in windows]
        rows_parents = [self._row_parents(n, p[6]) for p, n in zip(plans, lengths)]
        trees = any(p[6] is not None for p in plans)
        hidden = (model.hidden_rows(windows, [p[1] for p in plans], parents=rows_parents) if trees
                  else model.hidden_rows(windows, [p[1] for p in plans]))
        logits = model.head(hidden)
        logits = logits.reshape(logits.shape[1:])
        offsets = [sum(lengths[:k]) for k in range(len(plans))]
        positions = [[plan[2] + 1 + d for d in tree_paths(rp)[0]] for plan, rp in zip(plans, rows_parents)]
        parts = [self._draw_streams(logits, [(plan[0].sampling, at) for plan, at in zip(plans, positions)])]
        for plan in plans:
            if isinstance(plan[4], mx.array) and int(plan[4].shape[0]):
                parts.append(plan[4].astype(parts[0].dtype))
        built = time.perf_counter()
        values = [int(t) for t in mx.concatenate(parts).tolist()]
        read = time.perf_counter()
        out: dict[str, tuple[list[int], int, int]] = {}
        paths, follows, at, drafts_at = [], [], 0, sum(lengths)
        for plan, n, rp in zip(plans, lengths, rows_parents):
            stream, kind, drafts, forced = plan[0], plan[3], plan[4], plan[5]
            sampled = values[at:at + n]
            if isinstance(drafts, mx.array):
                proposed = values[drafts_at:drafts_at + n - 1]
                drafts_at += n - 1
            else:
                proposed = [int(t) for t in drafts]
            window = [int(stream.pending[-1]), *proposed]
            got, path, cut, follow = self._conclude(stream, kind, forced, sampled, window, rp)
            paths.append(path)
            follows.append(follow)
            out[stream.stream_id] = (got, n, len(path))
            stream.min_rows = n if not stream.min_rows else min(stream.min_rows, n)
            at += n
        keeps = [len(path) for path in paths]
        heads = [(plan, [at + r for r in path], follow)
                 for plan, at, path, follow in zip(plans, offsets, paths, follows)
                 if self.family_mtp and plan[0].drafts]
        hook = getattr(model, "draft_streams", None) or getattr(model, "speculate_streams", None)
        if heads and callable(hook):
            # every stream's head in one forward a depth (the model batches them)
            drafted = hook([p[1] for p, _, _ in heads], [f for _, _, f in heads], [r for _, r, _ in heads],
                           [p[0].cache_len + 1 for p, _, _ in heads], [p[0].sampling for p, _, _ in heads],
                           self._draft_budgets([p[0] for p, _, _ in heads]))
            for (plan, _, _), drafts in zip(heads, drafted):
                self._next[plan[0].stream_id] = drafts
        else:
            for (plan, rows, follow), budget in zip(heads, self._draft_budgets([p[0] for p, _, _ in heads])):
                self._draft_late(plan[0], plan[1], plan[2], follow, rows, budget=budget)
        # Queue heads that read only hidden states before building cache rollback so their GPU work overlaps it.
        model.keep_rows_streams([p[1] for p in plans], tuple(lengths),
                                tuple(len(path) if self._is_prefix(path) else path for path in paths))
        if _PROFILE:
            # diagnostic: the heads' GPU time on its own (one extra sync a round)
            drafted_at = time.perf_counter()
            mx.eval(*[d for d in (self._next.get(p[0].stream_id) for p in plans) if isinstance(d, mx.array)])
            acc = self.__dict__.setdefault("_prof_streams", [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            acc[0] += 1
            acc[1] += sum(lengths)
            acc[2] += (built - started) * 1e3
            acc[3] += (read - built) * 1e3
            acc[4] += (drafted_at - read) * 1e3
            acc[5] += (time.perf_counter() - drafted_at) * 1e3
            acc[6] += len(plans)
            if acc[0] >= 50:
                n = acc[0]
                print(f"[lanes] shared rounds: {acc[6] / n:.1f} streams, {acc[1] / n:.1f} rows, "
                      f"build {acc[2] / n:.2f} ms, wait for the GPU {acc[3] / n:.2f}, after the read {acc[4] / n:.2f}, "
                      f"heads on the GPU "
                      f"{acc[5] / n:.2f} (mean of {n})", flush=True)
                self._prof_streams = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ms = (time.perf_counter() - started) * 1e3
        self._observe_overhead(len(plans), sum(lengths), ms)
        from tensorfold.engine.lane_engine import RoundStats

        self.round_stats.append(RoundStats(
            streams=len(plans), width=max(lengths), rows=sum(lengths), ragged=len(set(lengths)) > 1,
            rollbacks=sum(1 for n, k in zip(lengths, keeps) if k < n), committed=sum(len(v[0]) for v in out.values()),
            forward_ms=ms, finalize_ms=0.0, rollback_ms=0.0, total_ms=ms, started_at=started))
        if _ROUND_LOG:
            with open(_ROUND_LOG, "a") as handle:
                for plan, n, keep in zip(plans, lengths, keeps):
                    handle.write(f"{plan[2]} {plan[3]}@{len(plans)} {n} {keep} {ms:.2f}\n")
        return out

    def round_working_set(self) -> int:
        """Measure extra allocation for the widest shared forward, or return zero when shared forwards are unavailable."""

        if not getattr(self, "family_streams", False):
            return 0
        import mlx.core as mx

        streams = max(1, min(int(self.batch_streams), int(self.batch_rows) // 2))
        base = self.prefill_prefix([1000 + i for i in range(64)], cache=None, cached_tokens=0)
        caches = [self.copy_single_cache(base) for _ in range(streams)]
        mx.eval(*[a for c in caches for a in cache_arrays(c)])
        before = mx.get_active_memory()
        mx.reset_peak_memory()
        mx.eval(self.model.head(self.model.hidden_rows([[2000 + i, 3000 + i] for i in range(streams)], caches)))
        used = max(0, int(mx.get_peak_memory() - before))
        del base, caches
        mx.clear_cache()
        return used
