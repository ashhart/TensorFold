"""Versions of a living copy: the per-fact undo TensorFold does not have.

A version records every file of the copy by content: a file equal to the downloaded checkpoint's is the base's, any
other is kept once in the copy's store (`<name>.versions/blobs/<sha256>`), shared by every version that holds it. A
lesson rewrites the shards it touches (on a GB10 trial, all four, 17 GB), so a version costs what changed since the
versions before it: an APFS clone on a Mac (blocks shared until written), a real copy on Linux. The fact graph and
TensorFold Web's record of lessons go with it. Restoring writes back the files that differ, while the copy is not served.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from . import living

CHUNK = 8 * 1024 * 1024


def _dir(name: str) -> Path:
    return living.meta_dir(name) / "versions"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def _clone(src: Path, dest: Path) -> None:
    """A copy that shares blocks where the file system can (APFS clone on a Mac), a full copy elsewhere."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["cp", "-c", str(src), str(tmp)], capture_output=True, text=True, check=False)
            if out.returncode != 0:                      # no clone here: a real copy, which needs the room
                if src.stat().st_size > shutil.disk_usage(dest.parent).free - 5 * 1024**3:
                    raise RuntimeError(f"copying {src.name} needs {src.stat().st_size / 1024**3:.1f} GB more room")
                shutil.copyfile(src, tmp)
        else:
            shutil.copyfile(src, tmp)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)                      # a failed copy leaves nothing half written


def _files(folder: Path) -> dict[str, Path]:
    return {str(p.relative_to(folder)): p for p in sorted(folder.rglob("*")) if p.is_file() and not p.is_symlink()}


def _base_dir(name: str) -> Path | None:
    meta = living._read(living.meta_dir(name) / "meta.json", {})
    for candidate in (meta.get("source"), ):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    if meta.get("base"):
        try:
            return living.snapshot_of(meta["base"])
        except Exception:  # noqa: BLE001 - the base was removed from the cache: every file is kept in the store
            return None
    return None


def _base_hashes(name: str, base: Path | None) -> dict[str, str]:
    """The base checkpoint's file hashes, computed once and kept beside the versions."""

    if base is None:
        return {}
    cache = _dir(name) / "base.json"
    try:                                               # a file cut by a stop is computed again, not a lesson stopper
        doc = json.loads(cache.read_text()) if cache.is_file() else {}
        if doc.get("path") == str(base) and isinstance(doc.get("files"), dict):
            return doc["files"]
    except (ValueError, OSError):
        pass
    files = {rel: _sha(p.resolve()) for rel, p in
             ((str(p.relative_to(base)), p) for p in sorted(base.rglob("*")) if p.is_file())}
    _private(cache, json.dumps({"path": str(base), "files": files}))
    return files


def manifests(name: str) -> list[dict]:
    d = _dir(name) / "manifests"
    if not d.is_dir():
        return []
    out = [json.loads(p.read_text()) for p in d.glob("*.json")]
    return sorted(out, key=lambda m: m["at"])


def store_bytes(name: str) -> int:
    d = _dir(name) / "blobs"
    return sum(p.stat().st_size for p in d.iterdir() if p.is_file()) if d.is_dir() else 0


def save(name: str, label: str) -> dict:
    """A version of the copy as it is now."""

    folder = living.folder(name)
    if not folder.is_dir():
        raise KeyError(name)
    base = _base_dir(name)
    base_hashes = _base_hashes(name, base)
    previous = (manifests(name) or [None])[-1]
    seen = (previous or {}).get("stat", {})
    blobs = _dir(name) / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    files, stat, new_bytes, to_store = {}, {}, 0, []
    for rel, p in _files(folder).items():
        st = p.stat()
        known = seen.get(rel)
        sha = known[2] if known and known[0] == st.st_size and known[1] == st.st_mtime_ns else _sha(p)
        stat[rel] = [st.st_size, st.st_mtime_ns, sha]
        if base_hashes.get(rel) == sha:
            files[rel] = "base"
            continue
        files[rel] = sha
        if not (blobs / sha).is_file() and sha not in {x[1] for x in to_store}:
            to_store.append((p, sha))
            new_bytes += st.st_size
    # room first: a version that would fill the disk is refused, not half kept (APFS clones cost little until written)
    if platform.system() != "Darwin" and new_bytes > shutil.disk_usage(living.root()).free - 5 * 1024**3:
        raise RuntimeError(f"keeping this version takes {new_bytes / 1024**3:.1f} GB and there is not that much room "
                           "(5 GB kept free); forget older versions in Teach first")
    for p, sha in to_store:
        _clone(p, blobs / sha)
    extras = {}
    for kind, path in (("facts", living.graph_path(name)), ("lessons", living.meta_dir(name) / "lessons.json")):
        if path.is_file():
            extras[kind] = path.read_text()
    m = {"id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4], "label": label[:120], "at": time.time(),
         "files": files, "stat": stat, "extras": extras, "new_bytes": new_bytes,
         "kept": sum(1 for x in json.loads(extras.get("lessons") or "[]") if x.get("outcome") in living.KEPT)}
    (_dir(name) / "manifests").mkdir(parents=True, exist_ok=True)
    _private(_dir(name) / "manifests" / f"{m['id']}.json", json.dumps(m))
    return public(m)


def public(m: dict) -> dict:
    return {k: m[k] for k in ("id", "label", "at", "new_bytes", "kept")} | {
        "changed": sum(1 for v in m["files"].values() if v != "base")}


ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4,12}")     # a version's id as save() makes it; anything else names no file


def _manifest(name: str, vid: str) -> dict:
    if not ID.fullmatch(vid):
        raise KeyError(vid)
    p = _dir(name) / "manifests" / f"{vid}.json"
    if not p.is_file():
        raise KeyError(vid)
    return json.loads(p.read_text())


def _swap_journal(dest: str) -> Path:
    return living.meta_dir(dest) / "restore.json"


def recover() -> list[str]:
    """After a TensorFold Web stopped in the middle of a restore, each copy is brought to one side of it, weights and records
    together, by the phase its journal recorded: "prepared" (the swap may have begun) is undone, "swapped" is
    finished. Records staged for a transaction no journal names, and staging folders, go. Run at start and before a
    restore."""

    base = living.root()
    if not base.is_dir():
        return []
    said = []
    studio = base / ".studio"
    for journal in (studio.glob("*/restore.json") if studio.is_dir() else []):
        dest = journal.parent.name
        target, old, staging = base / dest, base / f".{dest}.old", base / f".{dest}.restore"
        try:
            j = json.loads(journal.read_text())
        except ValueError:
            j = {"phase": "prepared"}
        if j.get("phase") == "swapped":                  # the new weights are in: their records follow
            _apply_records(dest, j)
            said.append(f"{dest}: a restore stopped after its swap; its records are now the version's")
        else:                                            # not swapped yet, or halfway: back to how the copy was
            if old.exists():                             # the copy was moved aside: whatever is in its place is new
                if target.exists():
                    shutil.rmtree(target)
                old.rename(target)
            said.append(f"{dest}: a restore stopped before it finished; the copy is as it was")
        shutil.rmtree(old, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        journal.unlink(missing_ok=True)
    for d in (studio.iterdir() if studio.is_dir() else []):   # staged records no live journal names
        if d.is_dir() and not (d / "restore.json").exists():
            for stray in d.glob("*.json.restore-*"):
                stray.unlink(missing_ok=True)
    for staging in base.glob(".*.restore"):
        shutil.rmtree(staging, ignore_errors=True)
    for old in base.glob(".*.old"):
        name = old.name[1:-4]
        if living.NAME.match(name) and not (base / name).exists():
            old.rename(base / name)
            said.append(f"{name} put back as it was before an interrupted restore")
        else:
            shutil.rmtree(old, ignore_errors=True)
    return said


def _apply_records(dest: str, j: dict) -> None:
    """The records of transaction ``j`` in place: those it staged replace the current ones, those the version did not
    have go; nothing staged by another transaction is touched."""

    d = living.meta_dir(dest)
    for kind in ("facts", "lessons"):
        staged, final = d / f"{kind}.json.restore-{j.get('tx')}", d / f"{kind}.json"
        if staged.exists():
            staged.replace(final)
        elif kind not in j.get("records", []):
            final.unlink(missing_ok=True)


def restore(name: str, vid: str, *, into: str | None = None) -> dict:
    """The copy (or ``into``, another copy of the same base) put back as version ``vid`` was, all at once: every
    source checked first, the whole restored folder built beside it (files kept are linked, files written are checked
    against the version's hashes), then swapped in. A failure before the swap leaves the copy as it was."""

    recover()
    m = _manifest(name, vid)
    dest = into or name
    target = living.folder(dest)
    if not target.is_dir():
        raise KeyError(dest)
    base = _base_dir(name)
    current = _files(target)
    blobs = _dir(name) / "blobs"
    plan: list[tuple[str, Path, str | None]] = []           # (file, source, the hash to check, None if kept)
    need = 0
    for rel, ref in m["files"].items():
        want = m["stat"][rel][2]
        here = current.get(rel)
        # a file is kept as it is only when its bytes are the version's: its hash is read again, since a file
        # rewritten with the same size in the same clock tick looks unchanged to its size and time
        if here is not None and here.stat().st_size == m["stat"][rel][0] and _sha(here) == want:
            plan.append((rel, here, None))
            continue
        if ref == "base":
            if base is None or not (base / rel).exists():
                raise RuntimeError(f"{rel} is the base checkpoint's, which is no longer in the cache; nothing changed")
            src = (base / rel).resolve()
        else:
            src = blobs / ref
            if not src.is_file():
                raise RuntimeError(f"{rel} is missing from the copy's store; nothing changed")
        plan.append((rel, src, want))
        need += src.stat().st_size
    if platform.system() != "Darwin" and need > shutil.disk_usage(living.root()).free - 2 * 1024**3:
        raise RuntimeError(f"restoring writes {need / 1024**3:.1f} GB and there is not that much room; nothing changed")
    staging = living.root() / f".{dest}.restore"
    shutil.rmtree(staging, ignore_errors=True)
    written = 0
    try:
        for rel, src, want in plan:
            out = staging / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if want is None:
                try:
                    os.link(src, out)                        # the same bytes, no copy; one link again after the swap
                except OSError:
                    _clone(src, out)
                continue
            _clone(src, out)
            if _sha(out) != want:
                raise RuntimeError(f"{rel} does not match the version's hash; nothing changed")
            written += out.stat().st_size
        # the records staged under this transaction's id, then a journal: from here a stop is finished or undone
        tx = uuid.uuid4().hex[:12]
        j = {"tx": tx, "version": vid, "records": sorted(m["extras"]), "phase": "prepared"}
        d = living.meta_dir(dest)
        for kind in ("facts", "lessons"):
            if kind in m["extras"]:
                _private(d / f"{kind}.json.restore-{tx}", m["extras"][kind])
        # no stale `.old` before the journal: from here on, an `.old` folder always means the copy was moved there
        old = living.root() / f".{dest}.old"
        shutil.rmtree(old, ignore_errors=True)
        if old.exists():
            raise RuntimeError(f"{old} is left from an earlier restore and cannot be removed; nothing changed")
        _private(_swap_journal(dest), json.dumps(j))
        target.rename(old)
        staging.rename(target)
        _private(_swap_journal(dest), json.dumps({**j, "phase": "swapped"}))
        _apply_records(dest, j)
        _swap_journal(dest).unlink(missing_ok=True)
        shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        recover()                                        # by the journal's phase: undone, or finished if swapped
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"restored": vid, "into": dest, "written_bytes": written}


def _private(path: Path, text: str) -> None:
    from .settings import write_private

    write_private(path, text)


def remove(name: str, vid: str) -> dict:
    if not ID.fullmatch(vid):
        raise KeyError(vid)
    p = _dir(name) / "manifests" / f"{vid}.json"
    if not p.is_file():
        raise KeyError(vid)
    p.unlink()
    used = {v for m in manifests(name) for v in m["files"].values() if v != "base"}
    freed = 0
    for b in (_dir(name) / "blobs").iterdir():
        if b.name not in used:
            freed += b.stat().st_size
            b.unlink()
    return {"removed": vid, "freed_bytes": freed}


def drop_all(name: str) -> None:
    shutil.rmtree(_dir(name), ignore_errors=True)


def current(name: str) -> dict:
    """The copy as it is now, file by file (hash and size), for another machine to fetch what differs from its base;
    hashes are kept between calls and computed again only for files that changed."""

    folder = living.folder(name)
    if not folder.is_dir():
        raise KeyError(name)
    cache_path = _dir(name) / "current.json"
    try:
        seen = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
        if not isinstance(seen, dict):
            seen = {}
    except (ValueError, OSError):
        seen = {}
    files = {}
    for rel, p in _files(folder).items():
        st = p.stat()
        k = seen.get(rel)
        sha = k[2] if k and k[0] == st.st_size and k[1] == st.st_mtime_ns else _sha(p)
        seen[rel] = [st.st_size, st.st_mtime_ns, sha]
        files[rel] = {"sha": sha, "size": st.st_size}
    _private(cache_path, json.dumps({k: v for k, v in seen.items() if k in files}))
    extras = {}
    for kind, path in (("facts", living.graph_path(name)), ("lessons", living.meta_dir(name) / "lessons.json")):
        if path.is_file():
            extras[kind] = path.read_text()
    meta = living._read(living.meta_dir(name) / "meta.json", {})
    return {"name": name, "base": meta.get("base"), "files": files, "extras": extras}


def safe_rel(rel: object) -> bool:
    """A file name inside a copy: relative, forward slashes, no dot segment, nothing hidden or odd."""

    if not isinstance(rel, str) or not rel or len(rel) > 300 or rel.startswith("/") or "\\" in rel or ":" in rel:
        return False
    parts = rel.split("/")
    return all(p and p not in (".", "..") and re.fullmatch(r"[A-Za-z0-9._-]+", p) for p in parts)


def recover_unfinished() -> list[str]:
    """At start only (a copy may be being made otherwise): the copies a stopped TensorFold Web was making or fetching, part
    written, are removed."""

    base = living.root()
    studio = base / ".studio"
    said = []
    # a folder claimed by mkdir and left empty: a create that stopped before its mark (no copy, no record)
    for d in (base.iterdir() if base.is_dir() else []):
        if d.is_dir() and not d.name.startswith(".") and living.NAME.match(d.name) and not any(d.iterdir()):
            with contextlib.suppress(OSError):
                d.rmdir()
                said.append(f"{d.name}: an empty folder left by a copy that never started; it is removed")
    # fetching.json: the mark a TensorFold Web of d7086de wrote while fetching
    marked = {m.parent for g in ("*/unfinished.json", "*/fetching.json") for m in studio.glob(g)} if studio.is_dir() else set()
    for parent in sorted(marked):
        name = parent.name
        try:
            living.remove(name)
        except KeyError:                                 # the copy's folder is already gone: its records go too
            shutil.rmtree(parent, ignore_errors=True)
        except (ValueError, OSError):
            pass
        if (base / name).exists():
            said.append(f"{name}: a copy was left half made and could not be removed; delete it")
        else:
            said.append(f"{name}: a copy was left half made; it is removed")
    return said


def fetch(get_manifest, get_file, new: str, origin: str) -> dict:
    """A copy made here from another machine's: the base checkpoint copied locally, then only the files that differ
    from it fetched, each checked by its hash before it replaces anything."""

    m = get_manifest()
    if not m.get("base"):
        raise ValueError("the other machine does not know which model this copy came from")
    for rel, f in (m.get("files") or {}).items():         # every path judged before anything is made
        if not safe_rel(rel) or not isinstance(f, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha"))):
            raise ValueError(f"the other machine sent a path or hash this TensorFold Web refuses: {rel[:80]!r}")
    living.create(new, m["base"], finish=False)          # marked unfinished until the end: until then, not a copy
    target = living.folder(new)
    fetched = 0
    try:
        base_hashes = _base_hashes(new, _base_dir(new))
        for rel, f in m["files"].items():
            if base_hashes.get(rel) == f["sha"]:
                continue
            dest = target / rel
            if not dest.resolve().is_relative_to(target.resolve()):
                raise ValueError(f"refused path {rel}")
            tmp = dest.with_name(dest.name + ".part")
            dest.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with open(tmp, "wb") as out:
                for chunk in get_file(rel):
                    h.update(chunk)
                    out.write(chunk)
            if h.hexdigest() != f["sha"]:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"{rel} arrived altered (its hash differs); nothing of the copy is kept")
            tmp.replace(dest)
            fetched += f["size"]
        for rel, p in _files(target).items():
            if rel not in m["files"]:
                p.unlink()
        again = get_manifest()                        # a lesson on the other machine meanwhile: its shards moved
        if {k: v.get("sha") for k, v in (again.get("files") or {}).items()} != {k: v["sha"] for k, v in m["files"].items()}:
            raise RuntimeError("the copy changed on the other machine while it was fetched (a lesson?); try again")
        for kind, path in (("facts", living.graph_path(new)), ("lessons", living.meta_dir(new) / "lessons.json")):
            if kind in m["extras"]:
                _private(path, m["extras"][kind])
        meta_path = living.meta_dir(new) / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        meta["fetched_from"] = origin
        _private(meta_path, json.dumps(meta))
        living.unfinished(new).unlink()
    except Exception:
        living.remove(new)
        raise
    return {"copy": new, "fetched_bytes": fetched, "files": len(m["files"])}
