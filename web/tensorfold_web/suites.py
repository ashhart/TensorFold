"""Custom suites: prompts a user writes for the Bench, kept next to the runs and measured like the built-in ones.

A custom suite is a name and a few prompts, each a user message with an optional system message, optional earlier
turns of the conversation (user and assistant, oldest first) and a reply length. A built-in suite can be copied in as a
starting point.
Its digest covers what is sent, so two runs of the same suite compare only when its prompts did not change. The
exported form is the one imported, so a suite moves between machines as one JSON file.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

KIND = "tensorfold-studio-suite"
MAX_PROMPTS = 20
MAX_TEXT = 40_000                  # characters in one message; long-context work belongs to the prefill sweep
MAX_TURNS = 20                     # earlier turns before a prompt's message
DEFAULT_TOKENS = 300


def validate(doc: object) -> dict:
    """A suite as TensorFold Web keeps it, from what a page or a file sent; raises ValueError naming the first problem."""

    if not isinstance(doc, dict):
        raise ValueError("a suite is a JSON object with a name and prompts")  # noqa: TRY004 - one error type for the page
    name = doc.get("name")
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 60:
        raise ValueError("the suite needs a name of 1 to 60 characters")
    prompts = doc.get("prompts")
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= MAX_PROMPTS:
        raise ValueError(f"a suite has 1 to {MAX_PROMPTS} prompts")
    out, seen = [], set()
    for i, p in enumerate(prompts, 1):
        if not isinstance(p, dict):
            raise ValueError(f"prompt {i} is not an object")  # noqa: TRY004
        user = p.get("user")
        if not isinstance(user, str) or not user.strip() or len(user) > MAX_TEXT:
            raise ValueError(f"prompt {i} needs a message of 1 to {MAX_TEXT:,} characters")
        system = p.get("system") or None
        if system is not None and (not isinstance(system, str) or len(system) > MAX_TEXT):
            raise ValueError(f"prompt {i}: the system message must be text of at most {MAX_TEXT:,} characters")
        history = _history(p.get("history"), i)
        tokens = p.get("max_tokens", DEFAULT_TOKENS)
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 16 <= tokens <= 4096:
            raise ValueError(f"prompt {i}: the reply length must be 16 to 4,096 tokens")
        pid = p.get("id") if isinstance(p.get("id"), str) and re.fullmatch(r"[\w-]{1,40}", p["id"]) else f"p{i}"
        n = i
        while pid in seen:                     # a repeated id gets one not taken: two prompts never share a result
            pid, n = f"p{n}", n + 1
        seen.add(pid)
        out.append({"id": pid, "user": user, "system": system.strip() if system else None, "max_tokens": tokens,
                    **({"history": history} if history else {})})
    about = doc.get("about") if isinstance(doc.get("about"), str) else ""
    return {"name": name.strip(), "about": about.strip()[:500], "prompts": out}


def _history(turns: object, i: int) -> list[dict]:
    if turns in (None, []):
        return []
    if not isinstance(turns, list) or len(turns) > MAX_TURNS:
        raise ValueError(f"prompt {i}: at most {MAX_TURNS} earlier turns")
    out = []
    for t in turns:
        role, content = (t.get("role"), t.get("content")) if isinstance(t, dict) else (None, None)
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip() \
                or len(content) > MAX_TEXT:
            raise ValueError(f"prompt {i}: an earlier turn is a user or assistant message of 1 to {MAX_TEXT:,} characters")
        out.append({"role": role, "content": content})
    return out


def digest(suite: dict) -> str:
    """What is sent, and nothing else: renaming a suite keeps its digest, editing a prompt changes it."""

    # earlier turns count only where there are some, so a suite written before they existed keeps its digest
    sent = [{"id": p["id"], "user": p["user"], "system": p.get("system"), "max_tokens": p["max_tokens"],
             **({"history": p["history"]} if p.get("history") else {})} for p in suite["prompts"]]
    canon = json.dumps(sent, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def as_bench(suite: dict) -> dict:
    """The suite in the Bench's own shape, next to the built-in ones."""

    def messages(p: dict) -> list[dict]:
        sys = [{"role": "system", "content": p["system"]}] if p.get("system") else []
        return [*sys, *p.get("history", []), {"role": "user", "content": p["user"]}]

    return {"id": suite["id"], "name": suite["name"], "custom": True, "digest": digest(suite),
            "prompts": [{"id": p["id"], "messages": messages(p), "max_tokens": p["max_tokens"]}
                        for p in suite["prompts"]]}


def from_builtin(suite: dict) -> dict:
    """A built-in suite (passages already put in) as a custom one to start from: its system message, earlier turns and
    last user message kept apart, named as a copy."""

    prompts = []
    for p in suite["prompts"]:
        msgs = p["messages"]
        system = next((m["content"] for m in msgs if m["role"] == "system"), None)
        turns = [m for m in msgs if m["role"] != "system"]
        prompts.append({"id": p["id"], "user": turns[-1]["content"], "system": system, "max_tokens": p["max_tokens"],
                        "history": [{"role": m["role"], "content": m["content"]} for m in turns[:-1]]})
    return validate({"name": f"{suite['name']} (my copy)"[:60], "about": f"Started from the built-in suite {suite['name']}.",
                     "prompts": prompts})


def export(suite: dict) -> dict:
    return {"kind": KIND, "name": suite["name"], "about": suite.get("about", ""), "digest": digest(suite),
            "prompts": [{k: v for k, v in p.items() if v is not None} for p in suite["prompts"]]}


class CustomSuites:
    """The user's suites, in the Bench's SQLite file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with self._db() as db:
            db.execute("create table if not exists suites (id text primary key, created real not null, doc text not null)")

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def list(self) -> list[dict]:
        with self._db() as db:
            rows = db.execute("select doc from suites order by created").fetchall()
        return [json.loads(d) for (d,) in rows]

    def get(self, sid: str) -> dict | None:
        with self._db() as db:
            row = db.execute("select doc from suites where id=?", (sid,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, doc: object, sid: str | None = None) -> dict:
        """Creates a suite, or replaces the one named by ``sid``; returns it as kept."""

        suite = validate(doc)
        now = time.time()
        with self._lock, self._db() as db:
            if sid is not None:
                row = db.execute("select doc from suites where id=?", (sid,)).fetchone()
                if row is None:
                    raise KeyError(sid)
                suite = {**suite, "id": sid, "created": json.loads(row[0])["created"], "updated": now}
                db.execute("update suites set doc=? where id=?", (json.dumps(suite), sid))
            else:
                suite = {**suite, "id": "custom-" + uuid.uuid4().hex[:8], "created": now, "updated": now}
                db.execute("insert into suites (id, created, doc) values (?,?,?)", (suite["id"], now, json.dumps(suite)))
        return suite

    def delete(self, sid: str) -> bool:
        with self._lock, self._db() as db:
            return db.execute("delete from suites where id=?", (sid,)).rowcount > 0
