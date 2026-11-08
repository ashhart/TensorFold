

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from .living_cards import (
    _kl_cell,
    _read_events,
    card,
    forget_plan,
    graph_summary,
    record,
    why_missed,
)
from .living_teacher import (
    Teacher,
)

PREFIX = "living:"                 # a model id naming a copy, as TensorFold Web's settings keep it

NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")     # no dot: nothing a copy is named can look like another's files

KEPT = ("kept", "partly kept")                       # outcomes that left something in the weights

LEARNABLE = {"nemotron_h"}         # TensorFold Web teaches Nemotron 3.5 Lightning only

ONLY = "teaching works on Nemotron 3.5 Lightning only"

def root() -> Path:
    return Path(os.environ.get("TENSORFOLD_WEB_LIVING") or Path.home() / "tensorfold-living")

def folder(name: str) -> Path:
    if not NAME.match(name):
        raise ValueError("a copy's name is lower case letters, digits, dashes or underscores")
    return root() / name

def meta_dir(name: str) -> Path:

    folder(name)
    d = root() / ".studio" / name
    if not d.is_dir():
        d.mkdir(parents=True, exist_ok=True)
        for x in (root() / ".studio", d):
            x.chmod(0o700)
    # Move migration items independently; a model named <name>.versions must never be treated as version metadata.
    for old, new in ((f"{name}.studio.json", "meta.json"), (f"{name}.facts.json", "facts.json"),
                     (f"{name}.lessons.json", "lessons.json"), (f"{name}.checkset.json", "checkset.json"),
                     (f"{name}.versions", "versions"), (f"{name}.checks", "checks")):
        src = root() / old
        if src.exists() and not (d / new).exists() and not (src / "config.json").exists():
            src.rename(d / new)
    return d

def name_of(model: str | None) -> str | None:
    return model[len(PREFIX):] if model and model.startswith(PREFIX) else None

def _meta_path(name: str) -> Path:
    return meta_dir(name) / "meta.json"

def unfinished(name: str) -> Path:

    return meta_dir(name) / "unfinished.json"

def is_unfinished(name: str) -> bool:
    """The mark, or the one a TensorFold Web of d7086de wrote while fetching (`fetching.json`)."""

    d = meta_dir(name)
    return (d / "unfinished.json").exists() or (d / "fetching.json").exists()

def graph_path(name: str) -> Path:
    return meta_dir(name) / "facts.json"

def _lessons_path(name: str) -> Path:
    return meta_dir(name) / "lessons.json"

def _read(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default

def _write(p: Path, value) -> None:
    from .settings import write_private

    write_private(p, json.dumps(value, indent=1, ensure_ascii=False))

def family_refusal(copy: str) -> str | None:
    """Why TensorFold Web will not teach ``copy``, or None: teaching is Nemotron-only, whatever the engine offers."""

    family = family_of(folder(copy))
    if family in LEARNABLE:
        return None
    return f"{ONLY}; {copy} is a copy of {family or 'a model whose family its config does not name'}"

def family_of(path: Path) -> str | None:
    cfg = _read(path / "config.json", {})
    return cfg.get("model_type") if isinstance(cfg, dict) else None

def copies() -> list[dict]:
    """Every copy under the root: a folder with a config.json (one made by hand is listed too)."""

    out = []
    base = root()
    if not base.is_dir():
        return out
    for p in sorted(base.iterdir()):
        if not p.is_dir() or not (p / "config.json").is_file() or not NAME.match(p.name):
            continue                                  # .studio, staging folders and the like are not copies
        if is_unfinished(p.name):
            continue                                  # half made or fetched: not a copy yet
        meta = _read(_meta_path(p.name), {})
        lessons = _read(_lessons_path(p.name), [])
        out.append({"name": p.name, "model": PREFIX + p.name, "path": str(p), "base": meta.get("base"),
                    "created": meta.get("created") or p.stat().st_mtime, "family": family_of(p),
                    "learnable": family_of(p) in LEARNABLE,
                    "taught": sum(1 for x in lessons if x.get("outcome") in KEPT),
                    "lessons": len(lessons), "bytes": _size(p)})
    return out

def _size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
        except OSError:
            pass
    return total

def snapshot_of(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, local_files_only=True))

def create(name: str, base_repo: str, *, finish: bool = True) -> dict:

    dest = folder(name)
    if not root().is_dir():
        root().mkdir(parents=True, exist_ok=True)
        root().chmod(0o700)                            # learned weights hold what a person taught: theirs alone
    try:
        dest.mkdir(mode=0o700)                         # the one atomic claim: two makers of one name, one folder
    except FileExistsError:
        raise ValueError(f"a copy named {name} already exists") from None
    try:
        _write(unfinished(name), {"base": base_repo, "started": time.time()})
        out = _create(name, base_repo, dest)
        if not finish:
            return out
        unfinished(name).unlink()
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        _drop_mark(name)                               # only what this call wrote: a store left by hand stays
        raise
    return next(c for c in copies() if c["name"] == name)

def _drop_mark(name: str) -> None:
    d = root() / ".studio" / name
    for f in ("unfinished.json", "meta.json"):
        (d / f).unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        d.rmdir()                                      # gone if nothing else (versions, lessons) is kept there

def _empty(dest: Path) -> None:
    for p in dest.iterdir():
        shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()

def _create(name: str, base_repo: str, dest: Path) -> dict:
    try:
        src = snapshot_of(base_repo)
    except Exception as exc:  # not downloaded, or a cache the library cannot read
        raise ValueError(f"{base_repo} is not downloaded on this machine") from exc
    if family_of(src) not in LEARNABLE:
        raise ValueError(f"{base_repo} is not a model TensorFold learns into yet (Nemotron 3.5 Lightning only)")
    need = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())     # through the links: the real sizes
    out = None
    # `src/.` fills the claimed folder (GNU and BSD cp alike) where `src` into an existing folder would nest it
    if platform.system() == "Darwin":                # an APFS clone: no disk until learning writes
        out = subprocess.run(["cp", "-cRL", f"{src}/.", str(dest)], capture_output=True, text=True, check=False)
        if out.returncode != 0:
            _empty(dest)                              # not APFS, or no clones there: a real copy, space checked
    if out is None or out.returncode != 0:
        free = shutil.disk_usage(root()).free
        if need > free - 5 * 1024**3:
            raise ValueError(f"the copy takes {need / 1024**3:.1f} GB and {free / 1024**3:.1f} GB are free here")
        out = subprocess.run(["cp", "-RL", f"{src}/.", str(dest)], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"the copy failed: {out.stderr.strip()[:300]}")
    links = [str(f) for f in dest.rglob("*") if f.is_symlink()]
    if links:                                     # --slide refuses them, and learning must never write through one
        raise RuntimeError(f"the copy still holds links ({links[0]}); nothing was kept")
    _write(_meta_path(name), {"base": base_repo, "created": time.time(), "source": str(src)})
    return {"name": name, "model": PREFIX + name, "path": str(dest), "base": base_repo}

def remove(name: str) -> None:

    dest = folder(name)
    if not dest.is_dir() or not ((dest / "config.json").is_file() or _meta_path(name).is_file()):
        raise KeyError(name)
    shutil.rmtree(dest)
    shutil.rmtree(meta_dir(name), ignore_errors=True)  # its own folder only: no other copy's name can match it

def base_of(name: str, candidates: list[str]) -> str | None:

    meta = _read(_meta_path(name), {})
    if meta.get("base"):
        return meta["base"]
    mine = folder(name) / "tokenizer.json"
    if not mine.is_file():
        return None
    for repo in candidates:
        try:
            src = snapshot_of(repo)
        except Exception:  # noqa: BLE001 - not downloaded here
            src = None
        if src is None:
            continue
        theirs = src / "tokenizer.json"
        if theirs.is_file() and theirs.resolve().read_bytes() == mine.read_bytes():
            _write(_meta_path(name), {**meta, "base": repo, "source": str(src), "created": meta.get("created")
                                      or folder(name).stat().st_mtime})
            return repo
    return None

def lessons(name: str) -> list[dict]:
    return _read(_lessons_path(name), [])

def teachable_here(binary: str | None) -> str | None:
    """Why this machine's engine cannot learn, or None when it can (a Mac, or a GB10, with an engine offering --slide)."""

    if binary is None:
        return "TensorFold's native engine is not installed here"
    from . import native

    try:
        caps = native.capabilities(binary)
    except Exception:  # noqa: BLE001
        return "the engine did not say what it can do"
    if "--slide" not in (caps.get("serve") or {}).get("flags", {}):
        return f"this engine ({caps.get('version')}) does not learn: TensorFold 1.0.7 or later"
    chip = str(caps.get("chip") or "")
    if chip.startswith("nvidia") and chip != "nvidia-sm121":
        return f"TensorFold learns on a GB10 so far, not on {chip}; teach on a GB10 or a Mac and copy the folder here"
    if not (chip.startswith("apple") or chip == "nvidia-sm121"):
        return f"TensorFold does not learn on {chip or 'this machine'} yet"
    return None

PREVIEW_SYSTEM = (
    "You split a text into short standalone facts for a model to learn. Use only what the text states: never infer, "
    "never merge two statements into a new claim, never add a fact about who founded, owns or created something "
    "unless the text says so. Write each fact as one plain sentence that names its subject; replace a pronoun such "
    "as \"it\" or \"its\" by what it stands for. A fact about the person writing the text stays in the first person "
    "as they wrote it (\"My dog is called Ginger.\", \"I was born in Lisbon.\"), and its subject is \"the user\". "
    "For each fact give the one question a person would ask to get it back, naming the subject the way it would be "
    "asked; for a fact about the user, the question is in the first person too (\"What is my dog called?\").")

FACTS_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["facts"], "properties": {
    "facts": {"type": "array", "maxItems": 24, "items": {
        "type": "object", "additionalProperties": False, "required": ["fact", "subject", "question"],
        "properties": {"fact": {"type": "string"}, "subject": {"type": "string"}, "question": {"type": "string"}}}}}}

@dataclass
class Target:
    base: str                 # the engine's base URL, no /v1
    key: str | None
    model: str

    def chat(self, messages: list[dict], *, max_tokens: int, schema: dict | None = None,
             timeout: float = 600.0, thinking: bool = False) -> str:
        body: dict = {"model": self.model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens,
                      "chat_template_kwargs": {"enable_thinking": thinking}, "stream": False}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "facts", "schema": schema,
                                                                              "strict": True}}
        r = httpx.post(trust_env=False, url=self.base + "/v1/chat/completions", json=body, timeout=timeout,
                       headers={"Authorization": f"Bearer {self.key}"} if self.key else {})
        if r.status_code >= 400:
            raise RuntimeError(f"the engine answered {r.status_code}: {r.text[:200]}")
        return (r.json()["choices"][0]["message"].get("content") or "").strip()

def _json_in(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    return json.loads(t[start:end + 1]) if start >= 0 and end > start else {}

def supported(t: Target, text: str, fact: str) -> bool:
    """Whether the text states the fact, as the model reads it: a yes or no, nothing else."""

    reply = t.chat([{"role": "system", "content": "Answer yes or no, nothing else."},
                    {"role": "user", "content": f"Text:\n{text}\n\nDoes the text state this, without inference? "
                                                f"\"{fact}\""}], max_tokens=4)
    return reply.strip().lower().startswith("yes")

_FIRST = re.compile(r"\b(I|my|me|mine|I'm|I've)\b", re.IGNORECASE)

_QUESTION_TURNS = ((r"\bthe user's\b", "my"), (r"\bdoes the user\b", "do I"), (r"\bis the user\b", "am I"),
                   (r"\bhas the user\b", "have I"), (r"\bwas the user\b", "was I"), (r"\bthe user\b", "me"))

def first_person(fact: str, question: str, text: str) -> tuple[str, str]:

    words = {w.lower() for w in re.findall(r"[\w']+", fact)} - {"the", "user", "user's", "is", "a", "an"}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if _FIRST.search(s)]
    best = max(sentences, key=lambda s: len(words & {w.lower() for w in re.findall(r"[\w']+", s)}), default=None)
    their = {w.lower() for w in re.findall(r"[\w']+", best)} if best else set()
    # the sentence stands for the fact when it adds little: a compound sentence holds other facts too
    if best and words & their and len(their - words) <= max(3, len(words)):
        fact = best if best.endswith((".", "!", "?")) else best + "."
    for pattern, turned in _QUESTION_TURNS:
        question = re.sub(pattern, turned, question, flags=re.IGNORECASE)
    return fact, question[:1].upper() + question[1:]

def preview(t: Target, text: str) -> list[dict]:

    raw = t.chat([{"role": "system", "content": PREVIEW_SYSTEM}, {"role": "user", "content": text}],
                 max_tokens=1200, schema=FACTS_SCHEMA)
    try:
        facts = [f for f in (_json_in(raw).get("facts") or []) if isinstance(f, dict) and f.get("fact")]
    except ValueError as exc:
        raise RuntimeError("the model did not list the facts as asked; try a shorter text") from exc
    out = []
    for f in facts[:24]:
        fact = " ".join(str(f["fact"]).split())
        question = " ".join(str(f.get("question") or "").split())
        subject = str(f.get("subject") or "")
        about_user = "user" in subject.lower() or bool(_FIRST.search(fact))
        if about_user:
            fact, question = first_person(fact, question, text)
        out.append({"id": uuid.uuid4().hex[:8], "fact": fact, "subject": subject,
                    "question": question, "about_user": about_user,
                    "supported": supported(t, text, fact),
                    "now": t.chat([{"role": "user", "content": question}], max_tokens=80) if question else None})
    return out

@dataclass
class Item:
    id: str
    fact: str
    question: str = ""
    state: str = "queued"          # queued, learning, kept, partly kept, missed, failed, cancelled, interrupted
    message: str | None = None
    steps: int | None = None
    seconds: float | None = None
    before: str | None = None
    after: str | None = None
    found: list[dict] = field(default_factory=list)   # the facts the engine read in it, each with its own outcome

@dataclass
class Job:
    id: str
    copy: str
    items: list[Item]
    source: str
    version_before: bool = True      # a version of the copy kept before the first fact
    check: bool = True               # the copy's checks asked before and after
    phase: str = "starting"          # saving a version, checking before, learning, checking after, done
    version: str | None = None
    checks: list[str] = field(default_factory=list)
    gate_held: bool = False          # the caller holds the gate (a forgetting); the lesson leaves it alone
    started: float = field(default_factory=time.time)
    finished: float | None = None
    status: str = "running"         # running, done, cancelled, failed
    error: str | None = None

    def to_dict(self) -> dict:
        return {**asdict(self), "items": [asdict(i) for i in self.items]}

_ROUND = re.compile(r"slide: round (\d+): .*?held-out answers (do not come back|come back) token for token, and "
                    r"(do not come back|come back) when asked")

_KEPT_FOR = re.compile(r"slide: (\d+) questions kept for: (.*?)\s*\(")

_CONTRACTIONS = ((r"\bi'm\b", "i am"), (r"\bi've\b", "i have"), (r"\bi'll\b", "i will"), (r"\bi'd\b", "i would"),
                 (r"\bit's\b", "it is"), (r"\bthat's\b", "that is"), (r"\bthey're\b", "they are"),
                 (r"\bwe're\b", "we are"), (r"\bwe've\b", "we have"), (r"\bdon't\b", "do not"),
                 (r"\bdoesn't\b", "does not"), (r"\bisn't\b", "is not"), (r"\baren't\b", "are not"),
                 (r"\bwasn't\b", "was not"), (r"\bcan't\b", "cannot"), (r"\bwon't\b", "will not"))

_PERSON = ((r"\bthe user's\b", "my"), (r"\bthe user is\b", "i am"), (r"\bthe user has\b", "i have"),
           (r"\bthe user was\b", "i was"), (r"\bthe user does\b", "i do"), (r"\bthe user (\w+?)s\b", r"i \1"),
           (r"\bthe user\b", "i"))

def canonical(text: str) -> str:

    t = " ".join(re.sub(r"[^\w' ]+", " ", text.lower().replace("\u2019", "'")).split())
    for short, long in _CONTRACTIONS:                 # "I'm a pilot" and "The user is a pilot" meet at "i am a pilot"
        t = re.sub(short, long, t)
    for pattern, turned in _PERSON:
        t = re.sub(pattern, turned, t)
    return " ".join(t.split())

def matches_approved(approved: str, found: list[dict]) -> bool:

    return len(found) == 1 and canonical(approved) == canonical(str(found[0].get("text") or ""))

def _bounded(r: httpx.Response) -> str:
    from .httpread import error_text

    return error_text(r, limit=2000)

def _journal() -> Path:
    return root() / ".studio" / "lesson.json"

def _safe(ask) -> str | None:
    try:
        return ask()
    except Exception:  # noqa: BLE001 - a question that fails says nothing of the lesson
        return None

_record_lock = threading.Lock()


__all__ = ['FACTS_SCHEMA', 'KEPT', 'LEARNABLE', 'NAME', 'ONLY', 'PREFIX', 'PREVIEW_SYSTEM', 'Item', 'Job', 'Target', 'Teacher', '_kl_cell', '_read_events', 'base_of', 'canonical', 'card', 'copies', 'create', 'family_of', 'family_refusal', 'first_person', 'folder', 'forget_plan', 'graph_path', 'graph_summary', 'is_unfinished', 'lessons', 'matches_approved', 'meta_dir', 'name_of', 'preview', 'record', 'remove', 'root', 'snapshot_of', 'supported', 'teachable_here', 'unfinished', 'why_missed']
