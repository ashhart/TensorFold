"""Chat conversations kept by TensorFold Web: several of them, titled, searchable, exported, each reply with its context.

A conversation is a list of messages and the settings it was held with. An assistant message keeps the model and the
engine that wrote it, the thinking setting and budget, its figures (speed, first token, tokens, drafts, token
digest) and how it ended: done, stopped by the user, cut at the reply limit, an error, or interrupted, which is what a
reply still marked as streaming becomes when the page that streamed it is gone.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

MAX_MESSAGES = 2000
MAX_BYTES = 8 * 1024 * 1024        # one conversation, as JSON
FOLDER_CHARS = 60


def _name_ok(name: object) -> bool:
    """A folder's or a preset's name: 1 to 60 characters, and no slash, since the address that renames or removes it
    carries the name as one segment."""

    return isinstance(name, str) and bool(name.strip()) and len(name.strip()) <= FOLDER_CHARS and \
        "/" not in name and "\\" not in name and name.strip() not in (".", "..")   # a dot segment is dropped from a URL
# the presets the page ships with; a person's own cannot take their names
BUILT_IN_PRESETS = ("Model defaults", "Precise", "Creative", "Code reviewer", "Short answers")
# a preset's sampling, each field null for the model's own value: (kind, low, high)
PRESET_FIELDS = {"temperature": (float, 0.0, 2.0), "max_tokens": (int, 1, 262144), "top_p": (float, 0.0, 1.0),
                 "top_k": (int, 1, 1000), "min_p": (float, 0.0, 1.0), "seed": (int, 0, 2**31 - 1)}


class Taken(Exception):
    """A preset of that name is kept already, and the request did not say to replace it."""


def validate_preset(body: object) -> dict:
    """A preset as the page sends it, checked: a name, a system prompt, the sampling fields, and up to four stops."""

    if not isinstance(body, dict):
        raise ValueError("a preset is an object")  # noqa: TRY004 - one error type for the page
    name = body.get("name")
    if not _name_ok(name):
        raise ValueError(f"a preset's name is 1 to {FOLDER_CHARS} characters, without a slash")
    if name.strip() in BUILT_IN_PRESETS:
        raise ValueError("that name is one of the built-in presets")
    system = body.get("system") or ""
    if not isinstance(system, str) or len(system) > 20000:
        raise ValueError("system must be text of at most 20,000 characters")
    out: dict = {"name": name.strip(), "system": system}
    for field, (kind, low, high) in PRESET_FIELDS.items():
        v = body.get(field)
        if v is not None:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and v != int(v)) \
                    or not low <= v <= high:
                raise ValueError(f"{field} must be {'a whole number' if kind is int else 'a number'} from {low} "
                                 f"to {high}, or null for the model's own")
            v = kind(v)
        out[field] = v
    stop = body.get("stop") or []
    if not isinstance(stop, list) or len(stop) > 4 or not all(isinstance(x, str) and 0 < len(x) <= 64 for x in stop):
        raise ValueError("stop is up to four texts of 1 to 64 characters")
    out["stop"] = stop
    return out


class Conflict(Exception):
    """A save made on a revision that is not the conversation's latest: another page or a replay wrote it since."""

    def __init__(self, current: dict) -> None:
        super().__init__("this conversation was changed elsewhere since this page read it")
        self.current = current


def _title_of(messages: list[dict]) -> str:
    first = next((m.get("content", "") for m in messages if m.get("role") == "user" and m.get("content")), "")
    line = " ".join(str(first).split())
    return (line[:60] + "…") if len(line) > 60 else line or "New conversation"


def validate(body: object) -> dict:
    """The fields a page may set; raises ValueError on the first wrong one."""

    if not isinstance(body, dict):
        raise ValueError("a conversation is a JSON object")  # noqa: TRY004 - one error type for the page
    out: dict = {}
    if "messages" in body:
        msgs = body["messages"]
        if not isinstance(msgs, list) or len(msgs) > MAX_MESSAGES:
            raise ValueError(f"messages must be a list of at most {MAX_MESSAGES}")
        for i, m in enumerate(msgs):
            if not isinstance(m, dict) or m.get("role") not in ("user", "assistant", "system", "tool") or \
                    not isinstance(m.get("content", ""), str):
                raise ValueError(f"message {i + 1} needs a role (user, assistant, system, tool) and text content")
        out["messages"] = msgs
    if "title" in body:
        t = body["title"]
        if t is not None and (not isinstance(t, str) or len(t) > 200):
            raise ValueError("title must be text of at most 200 characters, or null")
        out["title"] = t.strip() if isinstance(t, str) and t.strip() else None
    if "settings" in body:
        if not isinstance(body["settings"], dict):
            raise ValueError("settings must be an object")
        out["settings"] = body["settings"]
    if "pinned" in body:
        if not isinstance(body["pinned"], bool):
            raise ValueError("pinned must be true or false")
        out["pinned"] = body["pinned"]
    if "folder" in body:
        f = body["folder"]
        if f is not None and (not _name_ok(f)):
            raise ValueError(f"folder must be a name of 1 to {FOLDER_CHARS} characters without a slash, or null for none")
        out["folder"] = f.strip() if isinstance(f, str) else None
    if len(json.dumps(out)) > MAX_BYTES:
        raise ValueError("this conversation is too large to keep (8 MB)")
    return out


def settle(doc: dict, *, live: set[str] | None = None) -> dict:
    """A reply left without an ending by a page that is gone is an interrupted reply."""

    if doc["id"] in (live or set()):
        return doc
    for m in doc.get("messages", []):
        if m.get("role") == "assistant" and not m.get("outcome"):
            m["outcome"] = "interrupted"
    return doc


class Conversations:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with self._db() as db:
            db.execute("create table if not exists conversations (id text primary key, created real not null, "
                       "updated real not null, title text, doc text not null)")
            # folders a person made, kept even while empty; a conversation names its folder in its own document
            db.execute("create table if not exists folders (name text primary key, created real not null)")
            # the chat presets a person saved: on TensorFold Web, so every browser and device sees the same ones
            db.execute("create table if not exists presets (name text primary key, created real not null, "
                       "doc text not null)")

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def create(self, body: object) -> dict:
        fields = validate(body)
        now = time.time()
        doc = {"id": uuid.uuid4().hex[:12], "created": now, "updated": now, "messages": fields.get("messages", []),
               "settings": fields.get("settings", {}), "title": fields.get("title"), "rev": 1}
        with self._lock, self._db() as db:
            db.execute("insert into conversations (id, created, updated, title, doc) values (?,?,?,?,?)",
                       (doc["id"], now, now, doc["title"], json.dumps(doc)))
        return self._shown(doc)

    def update(self, cid: str, body: object, *, rev: int | None = None) -> dict:
        """``rev``: the revision the page last read or wrote; a save made on an older one raises :class:`Conflict`
        rather than overwrite what was written since. Messages and settings make a new revision; a title, a pin or a folder
        does not (they are not what a page saving a reply could overwrite)."""

        fields = validate(body)
        with self._lock, self._db() as db:
            row = db.execute("select doc from conversations where id=?", (cid,)).fetchone()
            if row is None:
                raise KeyError(cid)
            old = json.loads(row[0])
            if rev is not None and rev != old.get("rev", 0):
                raise Conflict(self._shown(old))
            # settings merge: a page saving its own (thinking, preset) keeps what a replay recorded (replay_of, target)
            if "settings" in fields:
                fields = {**fields, "settings": {**(old.get("settings") or {}), **fields["settings"]}}
            content = "messages" in fields or "settings" in fields
            # a pin, a folder or a title leaves the conversation where it was in time: "Recent" is about what was said
            doc = {**old, **fields, "updated": time.time() if content else old["updated"]}
            if content:
                doc["rev"] = old.get("rev", 0) + 1
            db.execute("update conversations set updated=?, title=?, doc=? where id=?",
                       (doc["updated"], doc.get("title"), json.dumps(doc), cid))
        return self._shown(doc)

    def get(self, cid: str) -> dict | None:
        with self._db() as db:
            row = db.execute("select doc from conversations where id=?", (cid,)).fetchone()
        return self._shown(json.loads(row[0])) if row else None

    def folders(self) -> list[dict]:
        """The folders made, and each one's count; a folder named by a conversation but not made is listed too."""

        with self._db() as db:
            made = [r[0] for r in db.execute("select name from folders order by lower(name)").fetchall()]
            docs = [json.loads(r[0]) for r in db.execute("select doc from conversations").fetchall()]
        counts: dict[str, int] = {}
        for d in docs:
            if d.get("folder"):
                counts[d["folder"]] = counts.get(d["folder"], 0) + 1
        names = sorted(set(made) | set(counts), key=str.lower)
        return [{"name": n, "count": counts.get(n, 0)} for n in names]

    def make_folder(self, name: object) -> dict:
        if not _name_ok(name):
            raise ValueError(f"a folder's name is 1 to {FOLDER_CHARS} characters, without a slash")
        with self._lock, self._db() as db:
            db.execute("insert or ignore into folders (name, created) values (?, ?)", (name.strip(), time.time()))
        return {"name": name.strip()}

    def _refile(self, db: sqlite3.Connection, old: str, new: str | None) -> int:
        moved = 0
        for cid, d in db.execute("select id, doc from conversations").fetchall():
            doc = json.loads(d)
            if doc.get("folder") == old:
                doc["folder"] = new
                db.execute("update conversations set doc=? where id=?", (json.dumps(doc), cid))
                moved += 1
        return moved

    def remove_folder(self, name: str) -> int:
        """The folder goes; its conversations stay, in no folder. Returns how many were in it."""

        with self._lock, self._db() as db:
            db.execute("delete from folders where name=?", (name,))
            return self._refile(db, name, None)

    def rename_folder(self, old: str, new: object) -> dict:
        if not _name_ok(new):
            raise ValueError(f"a folder's name is 1 to {FOLDER_CHARS} characters, without a slash")
        new = new.strip()
        with self._lock, self._db() as db:
            db.execute("delete from folders where name=?", (old,))
            db.execute("insert or ignore into folders (name, created) values (?, ?)", (new, time.time()))
            self._refile(db, old, new)
        return {"name": new}

    def presets(self) -> list[dict]:
        with self._db() as db:
            rows = db.execute("select doc from presets order by lower(name)").fetchall()
        return [json.loads(d) for (d,) in rows]

    def save_preset(self, body: object, *, replace: bool = False) -> dict:
        p = validate_preset(body)
        with self._lock, self._db() as db:
            if not replace and db.execute("select 1 from presets where name=?", (p["name"],)).fetchone():
                raise Taken(p["name"])
            db.execute("insert or replace into presets (name, created, doc) values (?, ?, ?)",
                       (p["name"], time.time(), json.dumps(p)))
        return p

    def rename_preset(self, old: str, new: object) -> dict:
        with self._lock, self._db() as db:
            row = db.execute("select doc from presets where name=?", (old,)).fetchone()
            if row is None:
                raise KeyError(old)
            p = validate_preset({**json.loads(row[0]), "name": new})
            if p["name"] != old and db.execute("select 1 from presets where name=?", (p["name"],)).fetchone():
                raise Taken(p["name"])
            db.execute("delete from presets where name=?", (old,))
            db.execute("insert into presets (name, created, doc) values (?, ?, ?)",
                       (p["name"], time.time(), json.dumps(p)))
        return p

    def remove_preset(self, name: str) -> bool:
        with self._lock, self._db() as db:
            return db.execute("delete from presets where name=?", (name,)).rowcount > 0

    def delete(self, cid: str) -> bool:
        with self._lock, self._db() as db:
            return db.execute("delete from conversations where id=?", (cid,)).rowcount > 0

    def list(self, query: str = "", limit: int = 200, offset: int = 0) -> list[dict]:
        return self.page(query, limit, offset)[0]

    def page(self, query: str = "", limit: int = 200, offset: int = 0) -> tuple[list[dict], bool]:
        """Pinned first, then most recent, and whether more follow. Every pinned conversation comes on the first page,
        whatever the limit; the limit and ``offset`` count the others, page by page. A query matches the title or any
        message's text, case aside, over every conversation kept."""

        out = []
        q = query.strip().lower()
        with contextlib.closing(self._db()) as db:
            rows = db.execute("select doc from conversations order by "
                              "coalesce(json_extract(doc, '$.pinned'), 0) desc, updated desc").fetchall()
        for (d,) in rows:
            doc = json.loads(d)
            title = doc.get("title") or _title_of(doc.get("messages", []))
            hit = None
            if q:
                if q in title.lower():
                    hit = title
                else:
                    hit = next((m["content"] for m in doc.get("messages", []) if q in m.get("content", "").lower()),
                               None)
                    if hit is None:
                        continue
                    i = hit.lower().index(q)
                    hit = ("…" if i > 40 else "") + hit[max(0, i - 40): i + len(q) + 60] + "…"
            models = [m.get("model") for m in doc.get("messages", []) if m.get("role") == "assistant" and m.get("model")]
            out.append({"id": doc["id"], "title": title, "updated": doc["updated"], "created": doc["created"],
                        "messages": len(doc.get("messages", [])), "model": models[-1] if models else None,
                        "pinned": bool(doc.get("pinned")), "folder": doc.get("folder"),
                        # a replay names its original and what it was written with, so the page can offer the
                        # conversations that asked the same questions, and name each by its model
                        "replay_of": (doc.get("settings") or {}).get("replay_of"),
                        "target": (doc.get("settings") or {}).get("target"),
                        **({"match": hit} if q else {})})
        pins = [x for x in out if x["pinned"]] if offset == 0 else []
        rest = [x for x in out if not x["pinned"]]
        return pins + rest[offset:offset + limit], len(rest) > offset + limit

    @staticmethod
    def _shown(doc: dict) -> dict:
        # a conversation kept before revisions existed reads as revision 0, so its first save is checked too
        return {**doc, "title": doc.get("title") or _title_of(doc.get("messages", [])), "rev": doc.get("rev", 0)}


def markdown(doc: dict) -> str:
    """The conversation as Markdown, each reply with its model and figures."""

    started = time.strftime("%d %B %Y, %H:%M", time.localtime(doc["created"]))
    lines = [f"# {doc['title']}", "",
             f"Exported from TensorFold Web on {time.strftime('%d %B %Y, %H:%M')}; started {started}.", ""]
    for m in doc.get("messages", []):
        if m["role"] == "user":
            lines += ["## You", "", m["content"], ""]
        elif m["role"] == "system":
            lines += ["## System", "", m["content"], ""]
        elif m["role"] == "tool":
            lines += [f"## Tool result: {m.get('name') or 'a tool'}", "", "```json", m["content"], "```", ""]
        else:
            st = m.get("stats") or {}
            facts = [m.get("model") or "the model"]
            counted_here = st.get("measured") == "counted"     # stream events, not tokens: no token figure from them
            if st.get("tps") and not counted_here:
                facts.append(f"{round(st['tps'])} tok/s")
            if st.get("tokens") and not counted_here:
                facts.append(f"{st['tokens']} tokens")
            if m.get("outcome") and m["outcome"] != "done":
                facts.append({"stopped": "stopped", "length": "cut at the reply limit", "error": "error",
                              "interrupted": "interrupted"}.get(m["outcome"], m["outcome"]))
            lines += [f"## {' · '.join(facts)}", ""]
            if m.get("reasoning"):
                lines += ["<details><summary>Thinking</summary>", "", m["reasoning"], "", "</details>", ""]
            lines += [m.get("content") or "", ""]
            for c in m.get("tool_calls") or []:
                lines += [f"Calls `{c.get('name')}({c.get('arguments', '')})`" +
                          (f" (does not match its parameters: {c['check']})" if c.get("check") else ""), ""]
    return "\n".join(lines)
