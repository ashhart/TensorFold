"""GLM's ``--spill-gib`` hooks: both ranks take every spill decision in the same order, rank 0's header first."""

from __future__ import annotations

import time

from tensorfold.server.checkpoints import extends


class GlmSpill:
    """``GlmEngine``'s side of the spill tier (cuda/spill.py): which kept prompts are written, and resuming one read."""

    _SPILL_SKIP = ("ids", "nbytes", "spill_key")
    spill = None                                        # the store; None: the tier is off and every hook returns

    def _spill_setup(self, cfg) -> None:
        """Early in ``__init__``: ranks with other spill settings refused (one would wait forever), staging pinned."""

        mine = [0] * 5 if cfg is None else [1, int(cfg.gib * 1024), int(cfg.highwater * 1000), cfg.min_tokens,
                                            int(cfg.min_free_gib * 1024)]
        every = self._gather_ints(mine)
        if any(list(r) != list(every[0]) for r in every[1:]):
            raise RuntimeError("the ranks were started with different --spill-gib settings: "
                               + ", ".join(f"rank {r} {list(v)}" for r, v in enumerate(every))
                               + "; give every rank the same")
        self.spill_cfg, self.spill, self.spill_jobs = cfg, None, []
        if cfg is not None:
            from tensorfold.cuda.spill_io import preallocate

            preallocate(cfg)

    def _spill_open(self, settings: list[int], drafter_dir=None) -> None:
        """End of ``__init__``: the store, in a folder keyed by what changes the stored bits (``settings`` among it)."""

        if self.spill_cfg is None:
            return
        from tensorfold.cuda.spill import SpillStore
        from tensorfold.cuda.spill_format import build_id, weights_id

        from .decode import Snapshot, _row_views

        torch, world = self.torch, self.comm.world
        device = torch.device("cuda", torch.cuda.current_device())    # this thread's GPU: the I/O threads' streams
        shapes = [(tuple(v.shape[1:]), str(v.dtype)) for v in _row_views(self.e.st, 1, 1)]
        sig = "|".join(["glm-single", build_id(self.model_dir, drafter_dir), f"tp{world}", repr(shapes),
                        repr(list(settings)), str(getattr(self.drafter, "ring", 0))])
        self.spill = SpillStore(self.spill_cfg, rank=self.rank, world=world, device=device, signature=sig,
                                weights=weights_id(self.model_dir, drafter_dir), model_id=self.model_dir.name,
                                share=self._share, gather=self._gather_ints, classes=(Snapshot,))

    def _fits(self, snap, code: list[int]) -> bool:
        """The snapshot's draft caches fit the request's drafters (a kept resume point's or a stored one's)."""

        return self._fit(len(snap.ids), snap.drafter_end, snap.mtp_len, code)

    def _fit(self, n: int, drafter_end: int, mtp_len: int, code: list[int]) -> bool:
        _, mtp, dflash = self._drafters(code)
        return (not dflash or drafter_end == n) and (not mtp or mtp_len >= 0)

    def _spill_cached(self, prompt: list[int], hit, resume: bool, code: list[int]) -> int:
        """Rank 0's header field: -n to read a stored prefix of n tokens longer than ``hit``, else ``hit``'s length."""

        found = None
        if resume and self.spill is not None:          # one whose stored fields fit the request's drafters
            found = self.spill.find(prompt, lambda e: self._fit(e.n, e.fields.get("drafter_end", -1),
                                                                e.fields.get("mtp_len", -1), code))
        if found is not None and (hit is None or found.n > len(hit.ids)):
            return -found.n
        return len(hit.ids) if hit is not None else 0

    def _spill_before_run(self, prompt: list[int], code: list[int], hit, cached: int):
        """Both ranks, after the header: read a stored prefix it names (``cached`` < 0), write early, end copies."""

        if self.spill is None:
            return hit
        if cached < 0:                                  # -cached tokens stored on disk: every rank reads its own
            hit = self._spill_load(list(prompt[:-cached]), code)
        self._spill_behind()
        self._spill_wait()                              # pending copies of the live caches end before they change
        return hit

    def _spill_flushed(self, header: list[int]) -> bool:
        """Rank 1: whether ``header`` was rank 0's shutdown flush (``flush_spill``), then done here alike."""

        if self.spill is None or len(header) != 3 or header[0] != 0 or header[2] != 1:
            return False
        self._do_flush(self._share(None), self.spill.cfg.flush_s)
        return True

    def _spill_rows(self, snap) -> None:
        """Materialize: a snapshot without saved rows is written from views of the live caches."""

        from .decode import _row_views

        if snap.rows is None:
            snap.rows = _row_views(self.e.st, len(snap.ids), max(snap.mtp_len, 0))
            snap._viewed = snap.drafter_end
            if not getattr(self.drafter, "ring", 0):
                snap.drafter_end, snap.drafter_rows = -1, None

    def _spill_wait(self) -> None:
        if self.spill is None:
            return
        for job in self.spill_jobs:
            self.spill.release(job)
        self.spill_jobs = []

    def _key(self, snap) -> tuple:
        """A kept snapshot's ``ids_key``, hashed once (its ids do not change while it is kept)."""

        if getattr(snap, "spill_key", None) is None:
            from tensorfold.cuda.spill_format import ids_key

            snap.spill_key = ids_key(snap.ids)
        return snap.spill_key

    def _writable(self, snap) -> bool:
        """A written copy could resume: its rows are at hand and, with a drafter, it holds the DFlash2 ring's window."""

        n = len(snap.ids)
        if snap.rows is None and self.live[:n] != list(snap.ids):
            return False
        return self.drafter is None or (bool(getattr(self.drafter, "ring", 0)) and snap.drafter_end == n
                                        and snap.drafter_rows is not None)

    def _superseded(self, snap) -> bool:
        """A kept snapshot extends this one and is stored or can be: the Mac spill writes only the longer one."""

        return any(extends(c.ids, snap.ids) and (self.spill.has(c.ids, self._key(c)) or self._writable(c))
                   for c in self.cache)

    def _spill_kept(self, snap, window, end: int) -> None:
        """``_take_over`` saved a live snapshot's rows, which drops its DFlash2 window: written now with that window."""

        if self.spill is None or window is None:
            return
        snap.drafter_rows, snap.drafter_end = window, end
        try:
            self._spill_snap(snap)
        finally:
            snap.drafter_rows, snap.drafter_end = None, -1

    def _spill_snap(self, snap, leaving: bool = False, keep: frozenset = frozenset()) -> bool:
        """Both ranks: write a kept snapshot (``leaving``: dropped; ``keep``: entries it never evicts); True: queued."""

        n = len(snap.ids)
        if self.spill is None or n < self.spill.cfg.min_tokens or self.spill.has(snap.ids, self._key(snap)):
            return False
        if not self._writable(snap) or self._superseded(snap):
            if leaving:
                self.spill.stats.unwritten_evicted += 1
            return False
        window = snap.drafter_rows
        viewed = False
        try:
            job = self.spill.save(snap.ids, snap, skip=self._SPILL_SKIP, materialize=self._spill_rows, keep=keep)
        except Exception as exc:                  # noqa: BLE001 - a spill never fails a request: the prompt is lost
            print(f"[tensorfold] spill rank {self.rank}: not written: {exc!r}", flush=True)
            job = None
        finally:
            if hasattr(snap, "_viewed"):          # the views belong to the live caches: not kept on the snapshot
                snap.rows, snap.drafter_end, snap.drafter_rows = None, snap._viewed, window
                del snap._viewed
                viewed = True
        if job is not None and viewed:            # saved rows stay alive in the job; live caches must wait for it
            self.spill_jobs.append(job)
        return job is not None

    def _spill_behind(self) -> None:
        """Both ranks: past --spill-highwater, write the snapshots ``_remember`` drops next, as rank 0 picks them."""

        hw = self.spill.cfg.highwater
        if hw >= 1.0:
            return
        picks = None
        if self.rank == 0:
            picks = []
            held = self._held_bytes()
            if held >= hw * self.cache_bytes or len(self.cache) >= max(1, self.cache_entries - 1):
                for i, snap in enumerate(self.cache):
                    if len(picks) == 2:
                        break
                    if (snap.rows is None or len(snap.ids) < self.spill.cfg.min_tokens or not self._writable(snap)
                            or self.spill.has(snap.ids, self._key(snap)) or self._superseded(snap)):
                        continue
                    if self.spill.pending_jobs() >= self.spill.cfg.writers:
                        self.spill.stats.deferred += 1
                        break
                    picks.append(i)
        keep = frozenset(self._key(c) for c in self.cache)   # kept snapshots' copies stay (else they take turns)
        for i in self._share(picks):
            if self._spill_snap(self.cache[i], keep=keep):
                self.spill.stats.behind += 1

    def _spill_load(self, ids: list[int], code: list[int]):
        """Both ranks: stored prompt ``ids`` as a kept snapshot; None (both prefill) unless both read it and it fits."""

        from tensorfold.cuda.spill_format import ids_key

        key, ok, fit, snap, meta = ids_key(ids), 0, 0, None, None
        e = self.spill.entry(key)                       # room first: the read's tensors stay within the kept memory
        if e is not None:
            self.spill.touch(e)                         # (the newest entry now: no save made for room evicts it)
        while e is not None and self.cache and self._held_bytes() + e.size > self.cache_bytes:
            self._drop(self.cache[0])
        try:
            snap, meta = self.spill.load(key)
            ok = int(snap is not None and int(meta["n"]) == len(ids))
            if ok:
                snap.ids = list(ids)
                fit = int(self._fits(snap, code))
        except Exception as exc:                        # noqa: BLE001 - a failed read: every rank prefills
            print(f"[tensorfold] spill rank {self.rank}: restore failed: {exc!r}", flush=True)
        both = self._gather_ints([ok, fit])
        if not (both[0][0] and both[1][0]):
            self.spill.stats.load_failed += 1
            self.spill.forget(key)
            return None
        if not (both[0][1] and both[1][1]):             # its fields passed find; the object read does not fit
            self.spill.stats.load_unfit += 1
            self.spill.forget(key)
            return None
        snap.nbytes = sum(r.numel() * r.element_size() for r in (snap.rows or []))
        self._remember(snap)
        return snap if snap in self.cache else None

    def flush_spill(self, budget: float | None = None) -> bool:
        """Rank 0, a clean shutdown: kept snapshots not on disk, newest first, within the budget; True: all written."""

        if self.spill is None:
            return True
        from .decode import row_bytes, snapshot_bytes

        budget = self.spill.cfg.flush_s if budget is None else budget
        st = self.spill.stats
        rate = (st.written_bytes / st.write_s if st.write_s > 1 and st.written_bytes else 0.5 * 2 ** 30)   # (untimed)
        rate *= self.spill.cfg.writers
        picks, t = [], 0.0
        for i in range(len(self.cache) - 1, -1, -1):
            snap = self.cache[i]
            if (len(snap.ids) < self.spill.cfg.min_tokens or self.spill.has(snap.ids, self._key(snap))
                    or not self._writable(snap) or self._superseded(snap)):
                continue
            t += (snapshot_bytes(snap) + (row_bytes(self.e, snap) if snap.rows is None else 0)) / rate
            if t > 0.8 * budget:
                break
            picks.append(i)
        self._ring()
        self._share([0, 0, 1])
        self._share(picks)
        return self._do_flush(picks, budget)

    def _do_flush(self, picks: list[int], budget: float) -> bool:
        t = time.perf_counter()
        done: list = []
        for i in picks:                                 # newest first: the flush's own earlier (newer) picks stay
            self._spill_snap(self.cache[i], keep=frozenset(self._key(c) for c in done))
            done.append(self.cache[i])
        ok = self.spill.drain(max(1.0, budget - (time.perf_counter() - t)))
        print(f"[tensorfold] spill rank {self.rank}: flushed {len(picks)} kept prompts in "
              f"{time.perf_counter() - t:.1f}s{'' if ok else ' (not finished)'}", flush=True)
        return ok
