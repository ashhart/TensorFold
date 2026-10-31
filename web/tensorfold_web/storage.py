"""What TensorFold Web keeps on disk, and clean-ups scoped by kind and age, each one previewed before it is done.

Kinds: Bench runs (with their results and the comparisons series made of them), conversations, the usage history,
the Monitor's eight hours of readings, and the catalog's models in the Hugging Face cache. A clean-up removes what
is older than a number of days, or everything of that kind; a Bench run still running and the model being served
are never removed. Models go through huggingface_hub's own deletion, which knows how its cache shares files.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

DAY = 86400.0


@dataclass(frozen=True)
class Kind:
    id: str
    label: str
    file: str
    table: str
    column: str            # the time a row is judged by
    keep: str = ""         # rows never removed (a running Bench run)


KINDS = (
    Kind("bench", "Bench runs", "bench.sqlite", "runs", "created", "status = 'running'"),
    Kind("conversations", "Conversations", "chat.sqlite", "conversations", "updated"),   # pinned ones kept, see clean()
    Kind("usage", "Usage history", "usage.sqlite", "usage", "hour"),
    Kind("history", "Monitor readings (8 h)", "history.sqlite", "history", "t"),
)
BY_ID = {k.id: k for k in KINDS}


def _connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=10)
    db.execute("pragma foreign_keys = on")         # a run's results go with it
    return db


def _where(k: Kind, cutoff: float | None) -> tuple[str, tuple]:
    parts, args = [], []
    if cutoff is not None:
        parts.append(f"{k.column} < ?")
        args.append(cutoff)
    if k.keep:
        parts.append(f"not ({k.keep})")
    return (" where " + " and ".join(parts)) if parts else "", tuple(args)


def summary(conf: Path) -> list[dict]:
    out = []
    for k in KINDS:
        p = conf / k.file
        row = {"id": k.id, "label": k.label, "bytes": 0, "count": 0, "oldest": None}
        if p.exists():
            row["bytes"] = sum(f.stat().st_size for f in p.parent.glob(p.name + "*") if f.is_file())   # with -wal
            try:
                with _connect(p) as db:
                    n, oldest = db.execute(f"select count(*), min({k.column}) from {k.table}").fetchone()
                row.update(count=n, oldest=oldest)
            except sqlite3.Error:
                pass
        out.append(row)
    return out


def clean(conf: Path, kind: str, older_than_days: float | None, *, dry_run: bool, now: float | None = None) -> dict:
    """Remove (or count, on a dry run) the rows of ``kind`` older than the days given, or all of them."""

    k = BY_ID.get(kind)
    if k is None:
        raise ValueError(f"unknown kind {kind!r}; one of {', '.join(BY_ID)}")
    if older_than_days is not None and older_than_days <= 0:
        raise ValueError("older_than_days must be above 0, or null for everything")
    cutoff = None if older_than_days is None else (time.time() if now is None else now) - older_than_days * DAY
    p = conf / k.file
    if not p.exists():
        return {"kind": kind, "matched": 0, "removed": 0, "dry_run": dry_run}
    where, args = _where(k, cutoff)
    with _connect(p) as db:
        if k.id == "conversations":                   # a pin lives in the document: a pinned conversation is kept
            kept = [i for i, doc in db.execute(f"select id, doc from conversations{where}", args) if not _pinned(doc)]
            n = len(kept)
        else:
            n = db.execute(f"select count(*) from {k.table}{where}", args).fetchone()[0]
        removed = 0
        if not dry_run and n:
            if k.id == "bench":
                ids = [r[0] for r in db.execute(f"select id from runs{where}", args)]
                removed = db.execute(f"delete from runs{where}", args).rowcount
                _drop_series_of(db, set(ids))
            elif k.id == "conversations":
                removed = db.executemany("delete from conversations where id = ?", [(i,) for i in kept]).rowcount
            else:
                removed = db.execute(f"delete from {k.table}{where}", args).rowcount
            if k.id == "usage" and cutoff is None:
                db.execute("delete from state")    # the baseline goes too: the next reading starts afresh
    if removed:
        with sqlite3.connect(p, timeout=10) as db:
            db.execute("vacuum")                   # give the space back
    return {"kind": kind, "matched": n, "removed": removed, "dry_run": dry_run}


def _pinned(doc: str) -> bool:
    try:
        return bool(json.loads(doc).get("pinned"))
    except (ValueError, AttributeError):
        return False


def _drop_series_of(db: sqlite3.Connection, run_ids: set[str]) -> None:
    """A comparison series is kept while one of its runs is; one whose runs all went goes too."""

    try:
        rows = db.execute("select id, doc from series").fetchall()
    except sqlite3.Error:
        return
    for sid, doc in rows:
        runs = [r for r in (json.loads(doc).get("runs") or []) if r]
        if runs and all(r in run_ids for r in runs):
            db.execute("delete from series where id = ?", (sid,))


# -- models in the Hugging Face cache ---------------------------------------------------------------------------------

def models(cache: Path, catalog_ids: set[str]) -> list[dict]:
    """The catalog's models in the cache, with their size on disk as huggingface_hub counts it."""

    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir(cache)
    except Exception:  # noqa: BLE001 - no cache yet, or one the library cannot read
        return []
    return sorted(({"repo": r.repo_id, "bytes": int(r.size_on_disk), "last_used": r.last_accessed}
                   for r in info.repos if r.repo_type == "model" and r.repo_id in catalog_ids),
                  key=lambda m: -m["bytes"])


def remove_model(cache: Path, repo: str, *, dry_run: bool) -> dict:
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir(cache)
    found = next((r for r in info.repos if r.repo_type == "model" and r.repo_id == repo), None)
    if found is None:
        raise KeyError(repo)
    plan = info.delete_revisions(*[rev.commit_hash for rev in found.revisions])
    # the library finds revisions by commit hash across the whole cache: a fork or a copy sharing one would go too
    own = Path(found.repo_path).resolve()
    stray = [str(x) for x in (*getattr(plan, "repos", ()), *getattr(plan, "snapshots", ()), *getattr(plan, "refs", ()))
             if not Path(x).resolve().is_relative_to(own)]
    covered = any(Path(x).resolve().is_relative_to(own) for x in (*getattr(plan, "repos", ()),
                                                                   *getattr(plan, "snapshots", ())))
    if not covered:
        stray = stray or [f"no file of {repo} itself"]
    if stray:
        raise RuntimeError(f"removing {repo} would also touch {stray[0]} (another model sharing a revision); "
                           "nothing was removed")
    freed = int(plan.expected_freed_size)
    if not dry_run:
        plan.execute()
    return {"repo": repo, "freed": freed, "dry_run": dry_run}
