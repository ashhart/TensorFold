"""A disk tier for the prompt states CUDA engines keep (``--spill-gib``), read back instead of a prefill."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from tensorfold.cuda.spill_format import (
    _TD,
    FORMAT,
    _crc_placeholder,
    _header,
    _read_header_file,
    decode,
    encode,
    ids_key,
    key_name,
)
from tensorfold.cuda.spill_io import Job, LoadJob, SpillIO

WAIT_S = 60.0                          # a read of an entry still being written waits this long for the write
INFLIGHT = 4 << 30                     # bytes of queued writes past which a new one first waits for the oldest


@dataclass
class SpillConfig:
    """``tensorfold serve`` flags of the tier (every rank must be started with the same values)."""

    root: str
    gib: float
    highwater: float = 0.70            # memory occupancy past which states eviction would take next are written early
    min_tokens: int = 8192             # shorter prompts are not written
    min_free_gib: float = 50.0         # no write leaves less disk free (on any rank)
    keep_builds: int = 1
    writers: int = 2                   # writer threads (TF_SPILL_WRITERS), each with two staging buffers
    direct: bool = True                # O_DIRECT where the file system takes it (TF_SPILL_DIRECT=0: buffered)
    flush_s: float = 60.0              # a clean shutdown's budget (TF_SPILL_FLUSH_S)

    def __post_init__(self) -> None:
        if not 0.0 < self.highwater <= 1.0:
            raise ValueError(f"--spill-highwater: a fraction in (0, 1], not {self.highwater}")
        if self.gib <= 0:
            raise ValueError("--spill-gib must be above 0 to turn the tier on")
        if self.writers < 1:
            raise ValueError("TF_SPILL_WRITERS: at least 1")

    @classmethod
    def from_args(cls, args: Any) -> SpillConfig | None:
        gib = float(getattr(args, "spill_gib", 0.0) or 0.0)
        if gib <= 0:
            return None
        root = str(getattr(args, "snapshot_dir", "") or "")
        if not root or root.lower() == "none":
            raise ValueError("--spill-gib needs --snapshot-dir")
        return cls(root=os.path.expanduser(root), gib=gib,
                   highwater=float(getattr(args, "spill_highwater", 0.70)),
                   min_tokens=int(getattr(args, "spill_min_tokens", 8192)),
                   min_free_gib=float(getattr(args, "spill_min_free_gib", 50.0)),
                   keep_builds=int(getattr(args, "spill_keep_builds", 1)),
                   writers=int(os.environ.get("TF_SPILL_WRITERS", "2")),
                   direct=os.environ.get("TF_SPILL_DIRECT", "1") != "0",
                   flush_s=float(os.environ.get("TF_SPILL_FLUSH_S", "60")))


@dataclass
class Entry:
    key: tuple[int, int]
    n: int
    name: str
    size: int = 0                      # exact bytes of the file, agreed by every rank (the largest)
    last: int = 0
    ids: Any = None                    # np.int32 (every rank; rank 0 matches against it)
    fields: dict = field(default_factory=dict)       # the stored object's plain fields: a family's test before a read
    bad: bool = False                  # this rank failed to write it (a read fails here; every rank then forgets it)


@dataclass
class Stats:
    written: int = 0
    write_failed: int = 0
    skipped: int = 0
    full: int = 0
    no_space: int = 0
    deferred: int = 0                  # early writes put off (the writers were busy); the engine counts them
    behind: int = 0                    # early writes queued
    unwritten_evicted: int = 0         # states that left memory without being written
    loaded: int = 0
    load_failed: int = 0
    crc_failed: int = 0                # reads whose data did not match its CRC-32 (the entry goes, a prefill)
    load_unfit: int = 0                # read, but its draft caches did not fit the request (a prefill)
    pruned: int = 0
    save_s: float = 0.0
    write_s: float = 0.0
    load_s: float = 0.0
    copy_wait_s: float = 0.0           # the engine waited for a copy off the device before moving rows
    copy_waits: int = 0
    written_bytes: int = 0
    loaded_bytes: int = 0
    loaded_tokens: int = 0
    restore_ms: list = field(default_factory=list)


def _plain(layers: list) -> dict:
    """A stored object's plain fields (numbers, strings, None) from its JSON layer."""

    return {k: s["v"] for k, s in layers[0]["fields"].items() if "v" in s} if layers else {}


class SpillStore(SpillIO):
    """The tier's files on this rank: an index every rank holds alike, writer threads, reads."""

    def __init__(self, cfg: SpillConfig, *, rank: int = 0, world: int = 1, device: Any = "cuda", signature: str = "",
                 model_id: str = "", share: Callable[[list[int] | None], list[int]] | None = None,
                 gather: Callable[[list[int]], list[list[int]]] | None = None, quiet: bool = False,
                 weights: str = "", classes: Sequence[type] = ()) -> None:
        """``classes``: what ``load`` may decode (a file naming another class fails its read)."""

        self.cfg = cfg
        self.classes = tuple(classes)
        self.rank, self.world = int(rank), int(world)
        self.device = device
        self.model_id = model_id
        self.share, self.gather = share, gather
        self.cap = int(cfg.gib * (1 << 30))
        shared = hashlib.sha256(f"{FORMAT}|{signature}".encode()).digest()
        own = hashlib.sha256(weights.encode()).digest()       # this rank's copy of the weights
        words = [int.from_bytes(h[i:i + 4], "big", signed=True) for h in (shared, own) for i in range(0, 16, 4)]
        if self.world > 1 and gather is not None:             # one directory name on every rank
            rows = gather(words)
            differ = [r for r, row in enumerate(rows) if list(row[:4]) != list(rows[0][:4])]
            if differ:
                print(f"[tensorfold] spill rank {self.rank}: rank(s) {differ} run another build or settings than "
                      "rank 0, so no rank can resume another's stored prompts until they match", flush=True)
            words = [w for row in rows for w in row[:8]]
        self.compat = hashlib.sha256(np.asarray(words, dtype=np.int32).tobytes()).hexdigest()[:16]
        self.top = os.path.join(cfg.root, f"spill-{self.compat}")
        self.dir = os.path.join(self.top, f"rank{self.rank}")
        try:                                                  # files take --snapshot-dir's owner: a server run as
            st = os.stat(cfg.root)                            # root in a container writes them as you
            self.uid, self.gid = st.st_uid, st.st_gid
        except OSError:
            self.uid = self.gid = -1
        self.lock = threading.RLock()
        self.index: dict[tuple, Entry] = {}
        self.pending: dict[tuple, Job] = {}
        self.clock = self.seq = 0
        self.stats = Stats()
        self.direct = bool(cfg.direct) and hasattr(os, "O_DIRECT")
        self.jobs: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._reader: tuple | None = None
        self._dirty = False
        self._last_dir_sync = 0.0
        self._dir_lock = threading.Lock()
        problem, pruned_builds = "", 0
        try:
            for d in (self.top, self.dir):                    # prompts' token ids: this user's only
                os.makedirs(d, mode=0o700, exist_ok=True)
                os.chmod(d, 0o700)
                self._own(d)
            pruned_builds = self._prune_builds()
            self._scan()
        except OSError as exc:
            problem = repr(exc)
        flags = gather([int(bool(problem))]) if self.world > 1 and gather is not None else [[int(bool(problem))]]
        bad = [r for r, row in enumerate(flags) if row[0]]
        if bad:                                               # every rank stops alike: none waits on another
            raise RuntimeError(f"--snapshot-dir {cfg.root}: rank(s) {bad} cannot use it"
                               + (f": {problem}" if problem else ""))
        found = len(self.index)
        dropped = self._reconcile() if self.world > 1 else 0
        if self.rank == 0 and not quiet:
            print(f"[tensorfold] spill: {self.top} ({cfg.gib:g} GiB a rank, early writes past "
                  f"{cfg.highwater:.0%} of memory, from {cfg.min_tokens} tokens, {cfg.writers} writers"
                  f"{', O_DIRECT' if self.direct else ''}, floor {cfg.min_free_gib:g} GiB free): "
                  f"{found - dropped} stored prompts from before this start, {self._total() / (1 << 30):.1f} GiB"
                  f"{f'; pruned {pruned_builds} older builds' if pruned_builds else ''}", flush=True)

    # -- housekeeping --------------------------------------------------------------------------------------------------
    def _own(self, path: str) -> None:
        if self.uid >= 0:
            try:
                os.chown(path, self.uid, self.gid)
            except OSError:
                pass

    def _prune_builds(self) -> int:
        """Remove other compatibility hashes' folders (builds, models, layouts) but the newest ``keep_builds``."""

        try:
            others = [os.path.join(self.cfg.root, d) for d in os.listdir(self.cfg.root)
                      if d.startswith("spill-") and os.path.join(self.cfg.root, d) != self.top]
        except OSError:
            return 0
        others = [d for d in others if os.path.isdir(d)]
        others.sort(key=lambda d: os.stat(d).st_mtime, reverse=True)
        gone = 0
        for d in others[max(0, self.cfg.keep_builds):]:
            mine = os.path.join(d, f"rank{self.rank}")
            if os.path.isdir(mine):
                shutil.rmtree(mine, ignore_errors=True)
                gone += 1
            try:
                os.rmdir(d)                                    # empty once every rank pruned its own
            except OSError:
                pass
        return gone

    def _total(self) -> int:
        with self.lock:
            return sum(e.size for e in self.index.values())

    def disk_free(self) -> int:
        try:
            st = os.statvfs(self.dir)
            return st.f_bavail * st.f_frsize
        except OSError:
            return 0

    # -- the index -----------------------------------------------------------------------------------------------------
    def _scan(self) -> None:
        found = []
        for name in os.listdir(self.dir):
            path = os.path.join(self.dir, name)
            if name.endswith(".partial"):                      # a write cut short: never read
                try:
                    os.remove(path)
                except OSError:
                    pass
                continue
            if name.endswith(".safetensors"):
                try:
                    found.append((os.stat(path).st_mtime, name))
                except OSError:
                    pass
        for _, name in sorted(found):                          # oldest first: recency follows the files' age
            path = os.path.join(self.dir, name)
            try:
                with open(path, "rb") as f:
                    header, data_at = _read_header_file(f)
                    meta = header.get("__metadata__") or {}
                    if meta.get("format") != FORMAT or meta.get("compat") != self.compat:
                        raise ValueError("another format")
                    want = data_at + max([v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"]
                                         or [0])
                    if os.fstat(f.fileno()).st_size != want:
                        raise ValueError("size does not match the header")
                ids = np.asarray(json.loads(meta["tokens"]), dtype=np.int32)
                n = int(meta["n"])
                key = ids_key(ids)
                if len(ids) != n or key_name(key, n) + ".safetensors" != name:
                    raise ValueError("tokens do not match the name")
                e = Entry(key, n, name[:-12], want, ids=ids, fields=_plain(json.loads(meta.get("layers", "[]"))))
                self.clock += 1
                e.last = self.clock
                self.index[key] = e
            except Exception:  # noqa: BLE001 - unreadable, cut or another format: gone
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _reconcile(self) -> int:
        """Every rank: keep entries all ranks hold, in rank 0's order, at the largest size; returns how many went."""

        if self.share is None or self.gather is None:
            return 0
        mine = sorted(self.index.values(), key=lambda e: e.last)
        flat = [v for e in mine for v in (e.key[0], e.key[1], e.n)] if self.rank == 0 else None
        theirs = self.share(flat)
        keys = [(theirs[i], theirs[i + 1], theirs[i + 2]) for i in range(0, len(theirs), 3)]
        keep = []
        if keys:
            held = [int((hi, lo) in self.index and self.index[(hi, lo)].n == n) for hi, lo, n in keys]
            every = self.gather(held)
            keep = [k for i, k in enumerate(keys) if all(r[i] for r in every)]
        sizes = self.gather([self.index[(hi, lo)].size >> 20 for hi, lo, _ in keep]) if keep else []
        want = {(hi, lo) for hi, lo, _ in keep}
        dropped = 0
        for e in list(self.index.values()):
            if e.key not in want:
                self._delete(e)
                dropped += 1
        self.clock = 0
        for i, (hi, lo, _) in enumerate(keep):
            self.clock += 1
            e = self.index[(hi, lo)]
            e.last = self.clock
            e.size = max(e.size, (max(r[i] for r in sizes) + 1) << 20)
        return dropped

    def _delete(self, e: Entry) -> None:
        with self.lock:
            if self.index.get(e.key) is e:
                self.index.pop(e.key, None)
            job = self.pending.get(e.key)
            if job is None or job.entry is not e:             # (one being written: its writer drops the file)
                try:
                    os.remove(os.path.join(self.dir, e.name + ".safetensors"))
                except OSError:
                    pass
            self.stats.pruned += 1

    def touch(self, e: Entry) -> None:
        with self.lock:
            self.clock += 1
            e.last = self.clock

    def forget(self, key: tuple) -> None:
        """Every rank: an entry that failed to read goes (its files are not trusted)."""

        with self.lock:
            e = self.index.get(key)
        if e is not None:
            self._delete(e)

    def has(self, ids: Sequence[int], key: tuple | None = None) -> bool:
        key = ids_key(ids) if key is None else key
        with self.lock:
            e = self.index.get(key)
            return e is not None and e.n == len(ids)      # rank-local states (bad, pending) never decide

    def pending_jobs(self) -> int:
        with self.lock:
            return len(self.pending)

    def find(self, prompt: Sequence[int], usable: Callable[[Entry], bool] | None = None) -> Entry | None:
        """The longest stored prompt that ``prompt`` strictly extends, among those ``usable`` takes."""

        arr = np.asarray(prompt, dtype=np.int32)
        best = None
        with self.lock:
            entries = list(self.index.values())
        for e in entries:
            if e.ids is None or e.bad or e.n >= len(arr) or (best is not None and e.n <= best.n):
                continue
            if (usable is None or usable(e)) and np.array_equal(arr[:e.n], e.ids):
                best = e
        return best

    def _make_room(self, extra: int, keep: frozenset = frozenset()) -> bool:
        """Every rank alike: least recently used entries (not ``keep``'s) go until ``extra`` fits, by the index."""

        with self.lock:                                # (one still being written goes too; its writer drops the file)
            total = self._total() + extra
            if total > self.cap and extra + sum(e.size for e in self.index.values() if e.key in keep) > self.cap:
                return False                           # it cannot fit beside ``keep``'s entries: nothing goes for it
            while total > self.cap:
                old = [e for e in self.index.values() if e.key not in keep]
                if not old:
                    break
                victim = min(old, key=lambda e: (e.last, e.key))
                total -= victim.size
                self._delete(victim)
            return total <= self.cap

    def _throttle(self, size: int) -> None:
        """Past ``INFLIGHT`` queued bytes, wait for this rank's oldest write first (queued writes hold tensors)."""

        while True:
            with self.lock:
                jobs = [j for j in self.pending.values() if not j.done.is_set()]
            if not jobs or sum(j.nbytes for j in jobs) + size <= INFLIGHT:
                return
            jobs[0].done.wait()

    # -- writing -------------------------------------------------------------------------------------------------------
    def save(self, ids: Sequence[int], obj: Any, *, skip: Sequence[str] = (),
             materialize: Callable[[Any], None] | None = None, keep: frozenset = frozenset()) -> Job | None:
        """Every rank, in order: queue ``ids``'s state (None: skipped; ``keep``: not evicted for it); one all-gather."""

        t0 = time.perf_counter()
        n = len(ids)
        if n < self.cfg.min_tokens:
            return None
        key = ids_key(ids)
        with self.lock:
            e = self.index.get(key)
            if e is not None and e.n == n:             # stored or on its way (every rank alike)
                self.touch(e)
                self.stats.skipped += 1
                return None
        items, layer = encode(obj, skip=skip, materialize=materialize)
        items = [(k, t if t.is_contiguous() else t.contiguous()) for k, t in items]
        info = {"format": FORMAT, "compat": self.compat, "model": self.model_id, "n": n,
                "tokens": json.dumps([int(t) for t in ids]), "layers": json.dumps([layer]),
                "saved": f"{time.time():.3f}",
                "crc32": _crc_placeholder(sum(t.numel() * t.element_size() for _, t in items))}
        header, data = _header(items, info)
        size = len(header) + data
        floor = int(self.cfg.min_free_gib * (1 << 30))
        ok = int(self.disk_free() - size >= floor)
        if self.world > 1 and self.gather is not None:
            got = self.gather([ok, (size >> 20) + 1])
            ok = int(all(r[0] for r in got))
            size = max(r[1] for r in got) << 20
        if not ok:
            self.stats.no_space += 1
            return None
        if size > self.cap or not self._make_room(size, keep):   # (larger than the cap: nothing is evicted)
            self.stats.full += 1
            return None
        self._start()
        self._throttle(size)
        name = key_name(key, n)
        entry = Entry(key, n, name, size, ids=np.asarray(ids, dtype=np.int32), fields=_plain([layer]))
        src = None
        if torch.cuda.is_available():
            src = torch.cuda.Event()
            src.record(torch.cuda.current_stream())
        with self.lock:                                    # the index changes here, on every rank alike
            self.seq += 1
            job = Job(entry, os.path.join(self.dir, name + ".safetensors"), header, items, src, self.seq)
            self.index.pop(key, None)
            self.pending[key] = job
            self.clock += 1
            entry.last = self.clock                        # recency is the queue order (every rank alike)
            self.index[key] = entry
        self.jobs.put(job)
        self.stats.save_s += time.perf_counter() - t0
        return job

    def _entry_done(self, job: Job, ok: bool) -> None:
        with self.lock:
            if self.pending.get(job.key) is job:
                self.pending.pop(job.key, None)
            if ok:
                self.stats.written += 1
                if job.key not in self.index:              # dropped after its rename: the file goes too
                    try:
                        os.remove(job.path)
                    except OSError:
                        pass
            elif self.index.get(job.key) is job.entry:
                job.entry.bad = True                       # stays indexed (every rank alike); a read fails here

    # -- reading -------------------------------------------------------------------------------------------------------
    def entry(self, key: tuple) -> Entry | None:
        with self.lock:
            return self.index.get(key)

    def load(self, key: tuple) -> tuple[Any, dict]:
        """(decoded object, metadata) of entry ``key``, read on this thread once a write of it ends; raises if not."""

        t0 = time.perf_counter()
        with self.lock:
            e = self.index.get(key)
            job = self.pending.get(key)
        if job is not None and job.entry is e and not job.done.wait(WAIT_S):
            raise TimeoutError("the write of this prompt did not end in time")
        if e is None or e.bad:
            raise FileNotFoundError(f"rank {self.rank} has no readable stored prompt {key}")
        path = os.path.join(self.dir, e.name + ".safetensors")
        with open(path, "rb") as f:
            header, data_at = _read_header_file(f)
        meta = header.get("__metadata__") or {}
        if meta.get("format") != FORMAT or meta.get("compat") != self.compat:
            raise ValueError(f"{path}: another format or build")
        dests = {name: torch.empty(d["shape"], dtype=_TD[d["dtype"]], device=self.device)
                 for name, d in header.items() if name != "__metadata__"}
        lj = LoadJob(e, path, header, data_at, dests)
        if torch.cuda.is_available():
            lj.after = torch.cuda.Event()
            lj.after.record(torch.cuda.current_stream())      # the new tensors are free to write from here
        self._read(lj)
        if lj.event is not None:
            torch.cuda.current_stream().wait_event(lj.event)
        layers = json.loads(meta.get("layers", "[]"))
        obj = decode(layers[0], dests, self.classes) if layers else None
        self.touch(e)
        dt = time.perf_counter() - t0
        self.stats.loaded += 1
        self.stats.loaded_tokens += e.n
        self.stats.load_s += dt
        self.stats.restore_ms.append(dt * 1000.0)
        if len(self.stats.restore_ms) > 4096:
            del self.stats.restore_ms[:2048]
        return obj, meta

    def info(self) -> dict:
        s = self.stats
        with self.lock:
            entries, pending = len(self.index), len(self.pending)
        lat = sorted(s.restore_ms)
        p = (lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 1)) if lat else (lambda q: 0.0)
        return {"highwater": self.cfg.highwater, "writers": self.cfg.writers, "direct": self.direct,
                "entries": entries, "pending": pending, "stored_gib": round(self._total() / (1 << 30), 2),
                "stored_bytes": self._total(), "written": s.written, "skipped": s.skipped, "full": s.full,
                "no_space": s.no_space, "deferred": s.deferred, "behind": s.behind,
                "unwritten_evicted": s.unwritten_evicted, "write_failed": s.write_failed, "loaded": s.loaded,
                "load_failed": s.load_failed, "crc_failed": s.crc_failed, "load_unfit": s.load_unfit,
                "pruned": s.pruned,
                "save_s": round(s.save_s, 2),
                "write_s": round(s.write_s, 2), "copy_wait_s": round(s.copy_wait_s, 2), "copy_waits": s.copy_waits,
                "written_gib": round(s.written_bytes / (1 << 30), 2), "written_bytes": s.written_bytes,
                "loaded_gib": round(s.loaded_bytes / (1 << 30), 2), "loaded_bytes": s.loaded_bytes,
                "load_s": round(s.load_s, 2), "loaded_tokens": s.loaded_tokens, "restore_ms_p50": p(0.5),
                "restore_ms_p95": p(0.95), "disk_free_bytes": self.disk_free()}


def flush_on_exit(app: Any, budget: float = 60.0) -> None:
    """--spill-gib on CUDA, at SIGTERM or Ctrl-C: the engine serving now writes its kept prompts within ``budget`` s."""

    t = time.perf_counter()
    ok = False
    turns = getattr(app, "_turns", None)
    turns = turns() if callable(turns) else None
    if turns is not None:                       # between requests
        turns.take(False, None)
    try:
        flush = getattr(getattr(app, "engine", None), "flush_spill", None)     # read now: the app may swap engines
        if flush is not None:
            ok = flush(budget) is not False                   # False: the writes did not end within the budget
    finally:
        if turns is not None:
            turns.give()
    print(f"[tensorfold] spill: shutdown flush {'done' if ok else 'not finished'} in {time.perf_counter() - t:.1f}s",
          flush=True)
