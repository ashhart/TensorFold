"""Stored prompt prefixes, their in-memory caches, and conversations saved at shutdown."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable

def longest_common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def extends(longer: list[int], shorter: list[int]) -> bool:
    """Whether ``longer`` continues ``shorter`` (its last token checked first: other conversations fail at once)."""

    n = len(shorter)
    return len(longer) > n > 0 and longer[n - 1] == shorter[-1] and longer[:n] == shorter


def choose_checkpoints(
    history_len: int, cached: int, last_prompt: list[int] | None, prompt: list[int]
) -> list[int]:
    """Keep history and stable-prefix boundaries outside the reused prefix and before the prompt end for differently rendered next turns."""

    candidates = {int(history_len)}
    if last_prompt:
        stable = longest_common_prefix(last_prompt, prompt)
        if 0 < stable < int(history_len) and stable >= int(history_len) // 2:
            candidates.add(stable)
    return sorted(at for at in candidates if int(cached) < at < len(prompt))


@dataclass
class CheckpointEntry:
    tokens: list[int]
    cache: list[Any]
    last_prompt: list[int]
    nbytes: int = 0
    # A system block (loaded from disk, or saved to it): outside the slot count.
    pinned: bool = False
    born: int = 0              # the length of the prompt that stored it: a later turn's is longer


def save_conversations(store: "CheckpointStore", directory: Path, model_id: str, *, keep: int = 2,
                       limit_bytes: int = 10 * 1024**3) -> int:
    """Save up to ``keep`` conversation snapshots within the byte limit; system blocks use their own directory."""

    from tensorfold.engine.prefix_snapshots import save_snapshot

    with store._lock:
        entries = [entry for entry in store._entries if not entry.pinned]   # most recently used first
    # Save longest conversations first, with recency breaking ties, so short background requests cannot displace them.
    # Prompt-side entries (a prefix of their own prompt) go before reply ends: a template that re-renders the reply
    # as history (Qwen3.6 drops the empty think block) never matches a reply end, while a prompt-side entry does and
    # leaves only the reply to prefill again (M4 Pro, Qwen3.6: an 18,419-token turn resumed in 1.2 s, not 27 s).
    def reply_end(entry: CheckpointEntry) -> bool:
        return entry.tokens != entry.last_prompt[:len(entry.tokens)]

    entries.sort(key=lambda entry: (reply_end(entry), -len(entry.tokens)))
    saved = total = 0
    for entry in entries:
        if saved >= keep or total + entry.nbytes > limit_bytes:
            break
        started = time.perf_counter()
        try:
            save_snapshot(directory, model_id, entry.tokens, entry.cache, keep=keep)
        except Exception as exc:  # noqa: BLE001 - a full disk must not hang the shutdown
            print(f"[lanes] conversation save failed: {type(exc).__name__}: {exc}", flush=True)
            break
        saved += 1
        total += entry.nbytes
        print(f"[lanes] saved conversation checkpoint tokens={len(entry.tokens)} "
              f"({entry.nbytes / 1024**3:.1f} GiB) in {time.perf_counter() - started:.1f}s", flush=True)
    return saved


def spill_conversation(entry: CheckpointEntry, directory: Path, model_id: str, *, limit_bytes: int) -> bool:
    """Write an evicted conversation where ``_read_disk_block`` finds it, this model's files kept under ``limit_bytes``."""

    from tensorfold.engine.prefix_snapshots import save_snapshot

    if entry.nbytes > limit_bytes:
        return False
    started = time.perf_counter()
    try:
        save_snapshot(directory, model_id, entry.tokens, entry.cache, keep=1 << 30)   # pruned by bytes below
    except Exception as exc:  # noqa: BLE001 - a full disk costs a later prefill, never the request
        print(f"[tensorfold] conversation spill failed: {type(exc).__name__}: {exc}", flush=True)
        return False
    prune_conversations(directory, model_id, limit_bytes)
    print(f"[tensorfold] spilled conversation tokens={len(entry.tokens)} ({entry.nbytes / 1024**3:.1f} GiB) "
          f"in {time.perf_counter() - started:.2f}s", flush=True)
    return True


def prune_conversations(directory: Path, model_id: str, limit_bytes: int) -> None:
    """Delete this model's oldest saved conversations until the rest fit ``limit_bytes``."""

    from tensorfold.engine.prefix_snapshots import read_metadata

    ours = []
    for path in sorted(directory.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.name.endswith(".partial.safetensors"):
            continue
        try:
            same = str(read_metadata(path).get("model", "")).split("|")[0] == model_id.split("|")[0]
        except Exception:  # noqa: BLE001 - an unreadable file is left alone
            continue
        if same:
            ours.append(path)
    total = 0
    for path in ours:
        total += path.stat().st_size
        if total > limit_bytes:
            path.unlink(missing_ok=True)


class CheckpointStore:
    """LRU caches require strict-prefix hits because GDN state cannot truncate; pinned system blocks bypass slot limits and outlast conversations."""

    def __init__(
        self,
        slots: int,
        copier: Callable[[list[Any]], list[Any]],
        *,
        budget_bytes: int | None = None,
        sizer: Callable[[list[Any]], int] | None = None,
        pinned_slots: int = 3,
        on_evict: Callable[[CheckpointEntry], Any] | None = None,
    ) -> None:
        if slots < 1:
            raise ValueError("slots must be positive")
        if budget_bytes is not None and budget_bytes < 1:
            raise ValueError("budget_bytes must be positive when given")
        self.slots = int(slots)
        self.pinned_slots = int(pinned_slots)
        self.copier = copier
        self.budget_bytes = None if budget_bytes is None else int(budget_bytes)
        self.sizer = sizer
        self._entries: list[CheckpointEntry] = []
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        # set when a memory controller evicts on demand: the newest entry may then exceed the byte budget
        self.admit_oversize = False
        # each evicted conversation no remaining entry extends, outside the lock on the thread that owns the arrays
        self.on_evict = on_evict
        self.spilled = 0

    def _evicted(self, gone: list[CheckpointEntry]) -> None:
        if self.on_evict is None:
            return
        with self._lock:
            remaining = [entry.tokens for entry in self._entries]
        for entry in gone:
            # an older checkpoint of a conversation that moved on continues from the newer entry: no write
            if entry.pinned or any(extends(t, entry.tokens) for t in remaining):
                continue
            try:
                if self.on_evict(entry) is not False:
                    self.spilled += 1
            except Exception as exc:  # noqa: BLE001 - a failed spill costs a later prefill, never the request
                print(f"[tensorfold] eviction hook failed: {type(exc).__name__}: {exc}", flush=True)

    @property
    def nbytes(self) -> int:
        return sum(entry.nbytes for entry in self._entries)

    def _best(self, prompt: list[int], usable: Any) -> CheckpointEntry | None:
        best: CheckpointEntry | None = None
        for entry in self._entries:
            tokens = entry.tokens
            if usable is not None and not usable(len(tokens)):
                continue
            if 0 < len(tokens) < len(prompt) and prompt[: len(tokens)] == tokens:
                if best is None or len(tokens) > len(best.tokens):
                    best = entry
        return best

    def peek(self, prompt: list[int], usable: Any = None) -> CheckpointEntry | None:
        """The entry ``match`` would use, without counting a hit or copying."""

        with self._lock:
            return self._best(prompt, usable)

    def match(self, prompt: list[int], usable: Any = None, *,
              take: bool = False) -> tuple[int, list[Any], list[int]] | None:
        """Return the longest usable strict-prefix hit with a copied cache, or remove and transfer the stored arrays with ``take``."""

        with self._lock:
            best = self._best(prompt, usable)
            if best is None:
                self.misses += 1
                return None
            self.hits += 1
            self._entries.remove(best)
            previous = list(best.last_prompt)
            if take:
                return len(best.tokens), best.cache, previous
            self._entries.insert(0, best)
            best.last_prompt = list(prompt)
            return len(best.tokens), self.copier(best.cache), previous

    def longest(self, prompt: list[int], usable: Any = None) -> int:
        """Length of the longest strict-prefix entry ``usable`` accepts (0 if none); not counted as a hit or miss."""

        with self._lock:
            best = self._best(prompt, usable)
            return len(best.tokens) if best is not None else 0

    def insert(self, tokens: list[int], cache: list[Any], *, last_prompt: list[int],
               pinned: bool = False) -> None:
        if not tokens:
            return
        nbytes = int(self.sizer(cache)) if self.sizer is not None else 0
        oversize = self.budget_bytes is not None and nbytes > self.budget_bytes
        if oversize and not self.admit_oversize:
            return
        with self._lock:
            replaced = [entry for entry in self._entries if entry.tokens == list(tokens)]
            kept = [entry for entry in self._entries if entry.tokens != list(tokens)]
            pinned = pinned or any(entry.pinned for entry in replaced)
            entry = CheckpointEntry(list(tokens), cache, list(last_prompt), nbytes, pinned, len(last_prompt))
            entries = [entry, *kept]
            for extra in [e for e in entries if e.pinned][self.pinned_slots:]:
                extra.pinned = False
            # an oversized newest entry displaces conversations but keeps the pinned system blocks
            limit = self.budget_bytes
            if oversize:
                limit = nbytes + sum(e.nbytes for e in entries[1:] if e.pinned)
            # Never evict the new entry; evict conversations (``_victim``'s order) before system blocks.
            gone: list[CheckpointEntry] = []
            while True:
                over_slots = sum(1 for e in entries if not e.pinned) > self.slots
                over_budget = (limit is not None and len(entries) > 1
                               and sum(e.nbytes for e in entries) > limit)
                if not (over_slots or over_budget):
                    break
                unpinned = [i for i in range(1, len(entries)) if not entries[i].pinned]
                if unpinned:
                    gone.append(entries.pop(self._victim(entries, unpinned)))
                elif over_budget:
                    entries.pop()
                else:
                    break
                self.evictions += 1
            self._entries = entries
        self._evicted(gone)

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _victim(entries: list[CheckpointEntry], candidates: list[int]) -> int:
        """Of ``candidates`` (oldest last): one a later turn's checkpoint continues, oldest first, else the oldest."""

        for i in reversed(candidates):
            entry = entries[i]
            if any(other.born > entry.born and extends(other.tokens, entry.tokens) for other in entries):
                return i
        return candidates[-1]

    def evict_one(self, keep: CheckpointEntry | None = None) -> bool:
        """Free an ordinary prefix (``_victim``'s pick), then a pinned one when memory needs it; never ``keep``."""

        with self._lock:
            candidates = [i for i, entry in enumerate(self._entries) if entry is not keep]
            if not candidates:
                return False
            ordinary = [i for i in candidates if not self._entries[i].pinned]
            gone = self._entries.pop(self._victim(self._entries, ordinary) if ordinary else candidates[-1])
            self.evictions += 1
        self._evicted([gone])
        return True
