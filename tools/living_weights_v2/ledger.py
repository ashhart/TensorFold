from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from .schema import Fact, normalized


class LedgerError(RuntimeError):
    pass


def _private(st: os.stat_result, kind) -> bool:
    return kind(st.st_mode) and st.st_uid == os.getuid() and not stat.S_IMODE(st.st_mode) & 0o077


def private_dir(directory: Path) -> tuple[int, int]:
    """Create missing directories 0700; the journal's own directory must be a real directory of ours with mode 0700."""
    if os.name != "posix":
        raise LedgerError("the Living Weights v2 journal requires a POSIX host")
    missing = []
    probe = directory
    while not os.path.lexists(probe):
        missing.append(probe)
        probe = probe.parent
    for item in reversed(missing):
        try:
            item.mkdir(mode=0o700)
        except FileExistsError:
            pass
    st = os.lstat(directory)
    if not _private(st, stat.S_ISDIR):
        found = stat.S_IMODE(st.st_mode)
        raise LedgerError(f"{directory} must be a directory you own with mode 700 (it is {found:o}), "
                          "since it holds the journal; use a private directory of its own")
    return st.st_dev, st.st_ino


def _journal_files(path: Path) -> tuple[Path, Path, Path]:
    return path, Path(str(path) + "-wal"), Path(str(path) + "-shm")


def _check_file(item: Path, st: os.stat_result) -> None:
    if not _private(st, stat.S_ISREG):
        raise LedgerError(f"{item} must be a regular file you own with mode 600 (it is {stat.S_IMODE(st.st_mode):o}), "
                          "since it holds fact text")


def private_journal(path: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    """Check the directory, then create the database at 0600 (or check the one there); return both identities."""
    folder = private_dir(path.parent)
    flags = os.O_RDWR | os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            descriptor = os.open(str(path), flags)
        except OSError as exc:
            raise LedgerError(f"{path} must be a regular file you own with mode 600, not a link") from exc
    try:
        st = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _check_file(path, st)
    for item in _journal_files(path)[1:]:
        try:
            _check_file(item, os.lstat(item))
        except FileNotFoundError:
            pass
    return folder, (st.st_dev, st.st_ino)


def verify_journal(path: Path, identities: tuple[tuple[int, int], tuple[int, int]]) -> None:
    """After SQLite opens it: the same directory and database, and every journal file still private."""
    folder, database = identities
    st = os.lstat(path.parent)
    if (st.st_dev, st.st_ino) != folder or not _private(st, stat.S_ISDIR):
        raise LedgerError(f"{path.parent} changed while the journal opened")
    for item in _journal_files(path):
        try:
            st = os.lstat(item)
        except FileNotFoundError:
            if item == path:
                raise LedgerError(f"{path} changed while the journal opened") from None
            continue
        _check_file(item, st)
        if item == path and (st.st_dev, st.st_ino) != database:
            raise LedgerError(f"{path} changed while the journal opened")


@contextlib.contextmanager
def exclusive_file(path: Path) -> Iterator[None]:
    try:
        import fcntl
    except ImportError as exc:
        raise LedgerError("exclusive locking requires a POSIX host") from exc
    private_dir(path.parent)
    descriptor = os.open(str(path), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    except BlockingIOError as exc:
        raise LedgerError("another Living Weights v2 writer is active") from exc
    finally:
        os.close(descriptor)


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        identities = private_journal(path)  # SQLite gives the WAL and SHM the database file's mode
        self.db = sqlite3.connect(path, timeout=2)
        try:
            verify_journal(path, identities)
            self._create()
            verify_journal(path, identities)
        except BaseException:
            self.db.close()
            raise

    def _create(self) -> None:
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                fact_digest TEXT NOT NULL,
                subject_key TEXT NOT NULL,
                answer TEXT NOT NULL,
                fact_json TEXT NOT NULL,
                manifest_digest TEXT NOT NULL,
                state TEXT NOT NULL,
                receipt_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            );
            CREATE INDEX IF NOT EXISTS entries_model_state ON entries(model, state);
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    def require_safe(self, model: str, fact: Fact) -> None:
        pending = self.db.execute("SELECT id, state FROM entries WHERE model=? AND state IN ('inflight','needs_recovery')", (model,)).fetchall()
        if pending:
            raise LedgerError(f"model has unresolved write records {[int(row['id']) for row in pending]}; recovery required")
        entries = self.db.execute("SELECT answer, fact_digest FROM entries WHERE model=? AND subject_key=? AND state='committed'", (model, fact.key)).fetchall()
        for row in entries:
            if normalized(row["answer"]) != normalized(fact.answer):
                raise LedgerError("contradictory fact about the same subject/relation; explicit delete and independent validation required")
            if row["fact_digest"] == fact.digest:
                raise LedgerError("identical fact already committed; teaching again is not idempotent")
            raise LedgerError("subject/relation already learned with different evidence; review required")

    def retained(self, model: str) -> tuple[Fact, ...]:
        entries = self.db.execute("SELECT fact_json FROM entries WHERE model=? AND state='committed' ORDER BY id", (model,)).fetchall()
        result = []
        for row in entries:
            value = json.loads(row["fact_json"])
            value["questions"] = tuple(value["questions"])
            value["near_misses"] = tuple(value["near_misses"])
            result.append(Fact(**value))
        return tuple(result)

    def begin(self, model: str, manifest_digest: str, fact: Fact) -> int:
        self.require_safe(model, fact)
        with self.db:
            cursor = self.db.execute("INSERT INTO entries(model, fact_digest, subject_key, answer, fact_json, manifest_digest, state) VALUES(?,?,?,?,?,?,?)", (
                model, fact.digest, fact.key, fact.answer, json.dumps(asdict(fact), ensure_ascii=False), manifest_digest, "inflight",
            ))
        return int(cursor.lastrowid)

    def finish(self, entry_id: int, state: str, receipt: dict) -> None:
        if state not in {"committed", "skipped", "rejected", "rolled_back", "needs_recovery"}:
            raise LedgerError("invalid final state")
        with self.db:
            cursor = self.db.execute("UPDATE entries SET state=?, receipt_json=?, updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=? AND state='inflight'", (
                state, json.dumps(receipt, sort_keys=True, ensure_ascii=False), entry_id,
            ))
            if cursor.rowcount != 1:
                raise LedgerError("no matching inflight entry")

    def history(self, model: str | None = None) -> list[dict]:
        if model:
            rows = self.db.execute("SELECT id, model, subject_key, state, receipt_json, created_at FROM entries WHERE model=? ORDER BY id", (model,))
        else:
            rows = self.db.execute("SELECT id, model, subject_key, state, receipt_json, created_at FROM entries ORDER BY id")
        return [{**dict(row), "receipt": json.loads(row["receipt_json"])} for row in rows]
