"""Per-conversation snapshots: the previous turn's KV (and linear-attention state)
held privately, LRU-capped and byte-budgeted.

Identity is content-based, like SCR on SGLang: a request joins the conversation
whose committed ids share the longest common prefix (>= match_min_tokens) with a
high overall token overlap, subject to an ambiguity margin. No session header is
needed. Snapshots are immutable once built; a stream executing a plan holds the
snapshot object directly, so a later refresh or eviction cannot pull its rows out
from under it (the splice-time token guard covers the rest)."""

from collections import OrderedDict, Counter
from dataclasses import dataclass, field

from .diff import quick_ratio_shared

GIB = 1024 ** 3


@dataclass
class Snapshot:
    ids: list                       # the committed tokens the rows cover: prompt + output
    pos: int                        # rows committed (== len(ids) on a clean turn)
    kv: list                        # per layer (k, v) tensors, rows [0, pos); None on linear layers
    rec: list                       # per layer linear-attention state or None
    conv: list
    bytes: int = 0


@dataclass
class Session:
    sid: str
    ids: list = field(default_factory=list)     # latest committed ids, for matching
    snap: Snapshot | None = None                # latest snapshot, replaced at turn end


class SessionStore:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.sessions: "OrderedDict[str, Session]" = OrderedDict()
        self._counter = 0
        self.stats = {"cold_turns": 0, "matched": 0, "new_sessions": 0,
                      "evicted": 0, "bytes_held": 0}

    # -- identity -----------------------------------------------------------
    def match(self, new_ids) -> str | None:
        """The conversation this prompt continues, by content (SCR `_find_session`)."""
        cfg = self.cfg
        head = new_ids[:cfg.match_window]
        bcount = Counter(new_ids)
        lb = len(new_ids)
        best_sid, best, best_n = None, 0.0, 0
        best_bare = 0.0                       # best ratio over sessions with no snapshot yet
        for sid, sess in reversed(self.sessions.items()):     # newest first
            old = sess.ids
            if not old:
                continue
            n = 0
            for x, y in zip(old[:cfg.match_window], head):
                if x != y:
                    break
                n += 1
            if n < cfg.match_min_tokens:
                continue
            r_pre = n / max(1, len(old))
            r_all = quick_ratio_shared(old, bcount, lb)
            if r_pre < cfg.match_min_prefix_frac and r_all < cfg.match_min_overlap:
                continue
            if sess.snap is None:
                best_bare = max(best_bare, r_all)
                continue
            if r_all > best or (r_all == best and n > best_n):
                best, best_n, best_sid = r_all, n, sid
        if best < cfg.match_min_ratio:
            return None
        if best_bare > best + cfg.match_ambig_margin:
            return None                       # a shorter live conversation could be the real one
        if best_sid is not None:
            self.sessions.move_to_end(best_sid)
            self.stats["matched"] += 1
        return best_sid

    def session_for(self, new_ids) -> tuple:      # -> (session, created)
        sid = self.match(new_ids)
        if sid is None:
            self._counter += 1
            sid = f"s{self._counter:06d}"
            self.sessions[sid] = Session(sid=sid)
            self.stats["new_sessions"] += 1
            self.stats["cold_turns"] += 1
            self._enforce_count()
            self.sessions[sid].ids = list(new_ids)
            return self.sessions[sid], True
        sess = self.sessions[sid]
        return sess, False

    # -- snapshots ----------------------------------------------------------
    def refresh(self, sid: str, snap: Snapshot) -> None:
        """A turn finished: its snapshot becomes the session's (last writer wins)."""
        sess = self.sessions.get(sid)
        if sess is None:
            return
        sess.ids = list(snap.ids)
        sess.snap = snap
        self.sessions.move_to_end(sid)
        self._enforce_count()
        self._enforce_budget()

    def held_bytes(self) -> int:
        return sum(s.snap.bytes for s in self.sessions.values() if s.snap is not None)

    def _enforce_budget(self) -> None:
        budget = int(self.cfg.side_gib * GIB)
        for sid in list(self.sessions.keys()):        # LRU order, oldest first
            if self.held_bytes() <= budget:
                break
            sess = self.sessions.pop(sid)
            self.stats["evicted"] += 1
            if self.cfg.debug:
                from .config import _log
                _log(f"session {sid} evicted for budget (held={self.held_bytes() / GIB:.2f} GiB)")
        self.stats["bytes_held"] = self.held_bytes()

    def _enforce_count(self) -> None:
        while len(self.sessions) > self.cfg.max_sessions:
            self.sessions.popitem(last=False)
            self.stats["evicted"] += 1