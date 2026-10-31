"""Living Weights: teachable copies of a model, lessons sent to the engine one approved fact at a time, and a record.

TensorFold (1.0.7) teaches a served Nemotron facts into its own weights with `serve <folder> --slide`; it rewrites the
folder's files and nothing undoes a fact, so it is always a copy. TensorFold Web keeps those copies under one folder
(`~/tensorfold-living` unless `TENSORFOLD_WEB_LIVING` says otherwise), each a plain model folder; what TensorFold Web
keeps of a copy is apart, in `.studio/<name>/` (private): `meta.json` (base, creation), `facts.json` (the engine's
fact graph), `lessons.json` (what TensorFold Web sent, the answers before and after, the outcome), its versions and checks.

A text is not sent as is: the model first lists the facts it holds (a preview), each checked against the text, with its
subject, the question it answers and what the model answers now; the user edits and approves them, and only approved
facts are taught, one learn request each. The trial on a GB10
showed why: the engine's own fact finder wrote a claim the text did not make.
"""

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

from . import httpread

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
    """What TensorFold Web keeps of a copy, apart from its model files and private to this user; files kept beside the
    copy by an earlier TensorFold Web (``<name>.studio.json``…) are moved in."""

    folder(name)
    d = root() / ".studio" / name
    if not d.is_dir():
        d.mkdir(parents=True, exist_ok=True)
        for x in (root() / ".studio", d):
            x.chmod(0o700)
    # each item moved on its own, so a migration a stop interrupted goes on at the next call; a folder that is a model
    # (a copy an earlier TensorFold Web allowed to be named "<name>.versions") is never taken for another copy's data
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
    """Present from before a copy's first file until it is complete (made, or fetched from another machine): such a
    copy is neither listed nor served, and a TensorFold Web that starts with one there removes it."""

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
    """A copy of ``base_repo``'s downloaded checkpoint as real files: never links into the Hugging Face cache, which
    learning would rewrite for every snapshot sharing them. An APFS clone on a Mac (no disk until learning writes).

    The copy is marked unfinished until it is complete; ``finish=False`` leaves the mark to a caller that goes on
    writing into it (a fetch)."""

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
    """Removes a copy: a folder of the living root that is one (a model's config, or TensorFold Web's record of it), never
    another folder that happens to sit there under a name of the same shape."""

    dest = folder(name)
    if not dest.is_dir() or not ((dest / "config.json").is_file() or _meta_path(name).is_file()):
        raise KeyError(name)
    shutil.rmtree(dest)
    shutil.rmtree(meta_dir(name), ignore_errors=True)  # its own folder only: no other copy's name can match it


def base_of(name: str, candidates: list[str]) -> str | None:
    """The downloaded model a copy came from: its record, or for a copy made by hand, the candidate whose tokenizer
    (which learning never rewrites) is the same file; then kept in the record."""

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


# -- the preview: the facts a text holds, checked, with what the model answers now --------------------------------------

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
        r = httpx.post(self.base + "/v1/chat/completions", json=body, timeout=timeout,
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
    """A fact about the user as they wrote it, and its question in the first person: TensorFold keys such a fact on
    the reply to "my …" questions, and models tend to rewrite it as "the user's …" whatever they are told."""

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
    """The facts the text holds, each with its subject, its question, whether the text states it, and what the model
    answers to its question now."""

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


# -- teaching: a queue of approved facts, one learn request each, recorded ------------------------------------------------

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


class Teacher:
    """One lesson job at a time, in a thread: each approved fact sent as its own learn request, its answer asked before
    and after. ``hold``/``release`` wrap the job (TensorFold Web's gate: nothing restarts the engine mid-lesson)."""

    def __init__(self, target, hold, release, log=list, served=lambda: None, graph=dict, undo=None) -> None:
        self.target, self.hold, self.release, self.log, self.served = target, hold, release, log, served
        self.graph = graph                     # the engine's fact graph, to see what it did with a stopped lesson
        self.undo = undo                       # undo(copy, version): stop, restore the version, serve again (gate held)
        self.job: Job | None = None
        self._cancel = threading.Event()         # stop after the fact being learned (or before any, during the checks)
        self._abort = threading.Event()          # stop now: the learn request is closed, the engine stops at its next check
        self.recovering: str | None = None       # the copy of a lesson a stopped TensorFold Web left, while it is waited for
        self._thread: threading.Thread | None = None

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, copy: str, facts: list[dict], source: str = "studio", *, version_before: bool = True,
              check: bool = True, gate_held: bool = False) -> Job:
        """``gate_held``: the caller holds the gate already (a forgetting, from its restore to its replay and any
        rollback) and keeps it; the lesson neither takes nor gives it. ``version_before`` is kept for callers and
        ignored: a version is always kept first."""
        if self.busy():
            raise RuntimeError("a lesson is going; wait for it or cancel it")
        items = [Item(id=f.get("id") or uuid.uuid4().hex[:8], fact=" ".join(str(f["fact"]).split()),
                      question=" ".join(str(f.get("question") or "").split()))
                 for f in facts if isinstance(f, dict) and str(f.get("fact") or "").strip()]
        if not items:
            raise ValueError("no approved fact to teach")
        if len(items) > 24:
            raise ValueError("24 facts at most in one lesson (2 to 3 minutes each)")
        if not gate_held:
            self.hold()
        try:                                             # whatever fails before the worker runs gives the gate back
            if self.served() != copy:                   # the preview was made on another copy than the one served now
                raise RuntimeError(f"{copy} is not the copy served now; preview its facts again")
            # a version before every lesson: the only way back if the engine learns something nobody approved
            job = Job(id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6], copy=copy, items=items,
                      source=source, version_before=True, check=check, gate_held=gate_held)
            _journal().parent.mkdir(parents=True, exist_ok=True)
            _write(_journal(), {"job": job.id, "copy": copy, "started": job.started,
                                "items": [{"fact": i.fact, "question": i.question} for i in items]})
            self.job = job
            self._cancel.clear()
            self._abort.clear()
            self._thread = threading.Thread(target=self._go, args=(job,), name="lesson", daemon=True)
            self._thread.start()
        except BaseException:
            try:
                _journal().unlink(missing_ok=True)
            except OSError:
                pass
            finally:
                if not gate_held:
                    self.release()
            raise
        return job

    def cancel(self, now: bool = False) -> bool:
        """Stops the lesson. By default after the fact being learned (the engine finishes or takes out a lesson it
        began); during the checks before it, nothing is learned at all. With ``now``, the learn request is closed, so
        the engine drops the fact at its next check; whatever it had kept of it is rolled back."""

        if not self.busy():
            return False
        self._cancel.set()
        if now:
            self._abort.set()
        return True

    def _go(self, job: Job) -> None:
        from . import living_checks, living_versions

        try:
            t: Target = self.target()
            if job.version_before:
                job.phase = "saving a version"
                job.version = living_versions.save(job.copy, f"before lesson {job.id}")["id"]
            before = None
            if job.check:
                job.phase = "checking before"
                before = living_checks.run(t, job.copy, f"before lesson {job.id}", cancel=self._cancel.is_set,
                                           extra=[asdict(i) for i in job.items])
                job.checks.append(before["id"])
            job.phase = "learning"
            gone = False                                 # the engine stopped, or serves something else, mid-lesson
            for n, it in enumerate(job.items):
                if self._cancel.is_set():
                    it.state = "cancelled"
                    continue
                if it.question:
                    it.before = _safe(lambda q=it.question: t.chat([{"role": "user", "content": q}], max_tokens=80))
                it.state = "learning"
                began = time.time()
                try:
                    self._learn(t, it)
                except Exception as exc:  # noqa: BLE001 - one fact failing does not stop the others
                    it.state, it.message = "failed", str(exc)[:300]
                    gone = self.served() != job.copy     # unless the engine itself is gone: nothing more can be learned
                it.seconds = round(time.time() - began, 1)
                if gone:
                    record(job.copy, job, it)
                    for rest in job.items[n + 1:]:
                        rest.state = "interrupted"
                    raise RuntimeError("the engine stopped while learning: the copy may hold a half-written lesson; "
                                       "restore the version kept before it, or run its checks before trusting it")
                if it.state in ("refused", "cancelled") and self._learned_anyway(it, assume=it.state == "refused"):
                    record(job.copy, job, it)            # written first: a rollback that fails still shows it
                    self._roll_back(job, n)              # something not approved (or not wanted) reached the weights
                    break
                if it.state == "cancelled":              # stopped now, and the engine kept nothing of it
                    record(job.copy, job, it)
                    break
                if it.state == "missed" and not it.message:   # the engine gave no reason: its log has one
                    it.message = why_missed(self.log(), it.fact)
                if it.state in KEPT and it.question:
                    it.after = _safe(lambda q=it.question: t.chat([{"role": "user", "content": q}], max_tokens=80))
                record(job.copy, job, it)
            if job.check and before is not None and any(i.state in KEPT for i in job.items):
                job.phase = "checking after"                 # the very questions asked before, nothing added
                job.checks.append(living_checks.run(t, job.copy, f"after lesson {job.id}", same_as=before,
                                                    cancel=self._cancel.is_set)["id"])
            job.phase = "done"
            if job.status == "running":
                job.status = "cancelled" if self._cancel.is_set() else "done"
        except Exception as exc:  # noqa: BLE001 - the job says why it stopped
            job.status, job.error = "failed", str(exc)[:300]
        finally:
            job.finished = time.time()
            try:
                _journal().unlink(missing_ok=True)
            except OSError:
                pass
            finally:
                if not job.gate_held:
                    self.release()

    def _learned_anyway(self, it: Item, assume: bool = True) -> bool:
        """After a lesson stopped because the engine read something else, or at the user's request: whether the engine
        kept any of it anyway, once it has finished with them (it stops at its next check that the request is gone).
        With nothing announced yet, ``assume`` says the worst (a refusal mid-extraction); a stop asked before the
        engine named a fact looks the fact up by its text instead."""

        ids = {f["id"] for f in it.found}
        if not ids:
            if assume:
                return True                              # nothing to poll: the version is there, the weights may not be
            try:
                facts = (self.graph() or {}).get("facts") or []
            except Exception:  # noqa: BLE001 - an engine that cannot say: assume the worst, the version is there
                return True
            return any(f.get("state") == "learned" and canonical(str(f.get("text") or "")) == canonical(it.fact)
                       for f in facts)
        end = time.time() + 900
        while time.time() < end:
            try:
                facts = [f for f in (self.graph() or {}).get("facts") or [] if f.get("id") in ids]
            except Exception:  # noqa: BLE001 - an engine that cannot say: assume the worst, the version is there
                return True
            if all(f.get("state") not in ("learning", "queued") for f in facts):
                return any(f.get("state") == "learned" for f in facts)
            time.sleep(5)
        return True

    def _roll_back(self, job: Job, upto: int) -> None:
        it = job.items[upto]
        if not job.version or self.undo is None:
            it.message = (it.message or "") + ". The engine kept some of it anyway and no version was kept before this " \
                "lesson: check the copy, or delete it"
            record(job.copy, job, it)
            job.status = "failed"
            return
        job.phase = "rolling back"
        try:
            self.undo(job.copy, job.version)
        except Exception as exc:
            for done in job.items[: upto + 1]:           # the weights hold what was learned: the record says so
                if done.state in ("kept", "partly kept", "refused", "cancelled"):
                    done.message = ((done.message + ". ") if done.message else "") + "The rollback failed (" + \
                        str(exc)[:120] + "): restore the version kept before this lesson"
                    record(job.copy, job, done)
            raise
        # the restore put back the record as it was before the lesson: every item of the lesson is written again
        for done in job.items[: upto + 1]:
            if done.state in ("kept", "partly kept", "refused", "cancelled"):
                stopped = done.state == "cancelled"
                done.state = "rolled back"
                done.message = ("stopped at your request: the engine had kept part of it, so the lesson was undone" if stopped
                                else "the lesson was undone: the engine had learned something that was not approved")
            if done.state != "queued":
                record(job.copy, job, done)
        for rest in job.items[upto + 1:]:
            rest.state = "cancelled"
        job.status = "rolled back"

    def recover(self, graph, alive=lambda: True) -> str | None:
        """A lesson a stopped TensorFold Web left going: the engine may still be learning it. Its unfinished facts are recorded
        as interrupted, and nothing may change the engine or the copy until the engine shows no fact being learned,
        or is not running at all."""

        j = _read(_journal(), None)
        if not j:
            return None
        done = {x["fact"] for x in lessons(j["copy"]) if x.get("job") == j["job"]}
        for x in j["items"]:
            if x["fact"] not in done:
                fake = Job(id=j["job"], copy=j["copy"], items=[], source="interrupted")
                record(j["copy"], fake, Item(id="", fact=x["fact"], question=x.get("question") or "",
                                             state="interrupted", message="TensorFold Web stopped during this lesson; the "
                                             "engine may have kept the fact: run the checks, or restore the version "
                                             "kept before it"))
        self.hold()
        self._cancel.clear()
        self.recovering = j["copy"]

        def wait() -> None:
            try:
                while True:
                    if self._cancel.is_set() or not alive():   # cancelled from Teach, or no engine of ours runs
                        break
                    try:
                        g = graph()
                        facts = g.get("facts") if isinstance(g, dict) else None
                        # only a real graph says nothing is being learned; an error page says nothing at all
                        busy = not isinstance(facts, list) or any(f.get("state") in ("learning", "queued") for f in facts)
                    except httpx.ConnectError:           # nothing listens: idle only if the engine process is gone too
                        busy = alive()
                    except httpx.HTTPStatusError as exc:   # no learner at that address (404), or one we cannot read
                        busy = exc.response.status_code not in (401, 403, 404)
                    except Exception:  # noqa: BLE001 - an engine that answers oddly: ask again
                        busy = True
                    if not busy:
                        break
                    self._cancel.wait(10)
            finally:
                try:
                    _journal().unlink(missing_ok=True)
                except OSError:
                    pass
                finally:
                    self.recovering = None
                    self.release()
        self._thread = threading.Thread(target=wait, name="lesson-recovery", daemon=True)
        self._thread.start()
        return f"a lesson on {j['copy']} was going when TensorFold Web stopped; its unfinished facts are marked interrupted"

    def _learn(self, t: Target, it: Item) -> None:
        """One learn request for one fact; the engine's events give its outcome. If the engine reads the text as
        something else than the approved fact, the request is closed as it starts learning, which stops it."""

        headers = {"Authorization": f"Bearer {t.key}"} if t.key else {}
        opened: list[httpx.Response] = []
        refused = False

        def work() -> None:
            nonlocal refused
            with httpx.stream("POST", t.base + "/v1/slide/learn", json={"text": it.fact, "source": "chat"},
                              headers=headers, timeout=httpx.Timeout(1800.0, connect=10.0)) as r:
                opened.append(r)                         # "stop now" closes it: the engine drops the fact at its next check
                if r.status_code >= 400:
                    raise RuntimeError(f"the engine refused the lesson ({r.status_code}): {_bounded(r)}")
                refused = _read_events(r, it)

        try:
            httpread.until_cancelled(work, self._abort, close=lambda: opened and opened[0].close(), name="lesson")
        except InterruptedError:
            it.state = "cancelled"
            it.message = "stopped at your request; the engine drops what it was learning at its next check"
            return
        if refused:
            return
        kept = [f for f in it.found if f["outcome"] == "kept"]
        missed = [f for f in it.found if f["outcome"] != "kept"]
        it.state = "kept" if kept and not missed else "partly kept" if kept else "missed"
        if not it.found and not it.message:
            it.message = "the engine ended the lesson without an outcome"
        elif missed and not it.message:
            it.message = next((f["message"] for f in missed if f.get("message")), None)
        if len(it.found) == 1 and " ".join(str(it.found[0]["text"]).split()).lower() != it.fact.lower():
            it.message = f"learned as \"{it.found[0]['text']}\"" + (f". {it.message}" if it.message else "")


def _read_events(r: httpx.Response, it: Item) -> bool:
    """The learn stream's events into the item; True when the engine read something else than the approved fact
    (the caller leaves, which closes the request and stops the engine there)."""

    kind = None
    for line in r.iter_lines():
        if line.startswith("event:"):
            kind = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if kind == "learning" and payload.get("id") is not None and \
                all(f["id"] != payload.get("id") for f in it.found):
            it.found.append({"id": payload.get("id"), "text": None, "outcome": "found"})   # never announced
        if kind == "learning" and not matches_approved(it.fact, it.found):
            read = "; ".join(f"\"{f['text']}\"" for f in it.found) or "nothing"
            it.state = "refused"
            it.message = f"the engine read {read} instead of the approved fact; stopped before it learned"
            return True                          # leaving closes the request: the engine stops there
        if kind == "fact":
            it.found.append({"id": payload.get("id"), "text": payload.get("text"), "outcome": "found"})
        elif kind == "learned" or "recalled" in payload:
            f = next((x for x in it.found if x["id"] == payload.get("id")), None)
            if f is None:
                f = {"id": payload.get("id"), "text": it.fact, "outcome": "found"}
                it.found.append(f)
            f["outcome"] = "kept" if payload.get("recalled") else "missed"
            f["message"] = payload.get("message")
            it.steps = (it.steps or 0) + int(payload.get("steps") or 0)
        elif kind == "failed":
            it.message = payload.get("message")
    return False


_ROUND = re.compile(r"slide: round (\d+): .*?held-out answers (do not come back|come back) token for token, and "
                    r"(do not come back|come back) when asked")
_KEPT_FOR = re.compile(r"slide: (\d+) questions kept for: (.*?)\s*\(")


def why_missed(lines: list[str], fact: str | None = None) -> str | None:
    """The engine's reason for the last lesson, read from its log: questions it could not write, or rounds where the
    fact never came back when asked. With ``fact``, only a lesson of that fact counts, never an earlier one's."""

    def mine(line: str) -> bool:
        m = _KEPT_FOR.search(line)
        return bool(m) and (fact is None or canonical(m.group(2)) == canonical(fact))
    since = max((i for i, line in enumerate(lines) if mine(line)), default=None)
    if since is None:
        return None
    tail = lines[since:]
    kept = int(_KEPT_FOR.search(lines[since]).group(1))
    if kept == 0:
        return "the model wrote no clean question to learn this fact from; try another wording"
    rounds = [m for m in (_ROUND.search(x) for x in tail) if m]
    if rounds and all(m.group(3) == "do not come back" for m in rounds):
        return f"it never came back when asked, in {len(rounds)} rounds; nothing of it was kept"
    return None


_CONTRACTIONS = ((r"\bi'm\b", "i am"), (r"\bi've\b", "i have"), (r"\bi'll\b", "i will"), (r"\bi'd\b", "i would"),
                 (r"\bit's\b", "it is"), (r"\bthat's\b", "that is"), (r"\bthey're\b", "they are"),
                 (r"\bwe're\b", "we are"), (r"\bwe've\b", "we have"), (r"\bdon't\b", "do not"),
                 (r"\bdoesn't\b", "does not"), (r"\bisn't\b", "is not"), (r"\baren't\b", "are not"),
                 (r"\bwasn't\b", "was not"), (r"\bcan't\b", "cannot"), (r"\bwon't\b", "will not"))
_PERSON = ((r"\bthe user's\b", "my"), (r"\bthe user is\b", "i am"), (r"\bthe user has\b", "i have"),
           (r"\bthe user was\b", "i was"), (r"\bthe user does\b", "i do"), (r"\bthe user (\w+?)s\b", r"i \1"),
           (r"\bthe user\b", "i"))


def canonical(text: str) -> str:
    """A fact's wording reduced to compare it: lower case, punctuation and spacing gone, and the engine's rewriting of
    the user's own facts ("The user lives in Paris.") turned back to the first person ("i live in paris")."""

    t = " ".join(re.sub(r"[^\w' ]+", " ", text.lower().replace("\u2019", "'")).split())
    for short, long in _CONTRACTIONS:                 # "I'm a pilot" and "The user is a pilot" meet at "i am a pilot"
        t = re.sub(short, long, t)
    for pattern, turned in _PERSON:
        t = re.sub(pattern, turned, t)
    return " ".join(t.split())


def matches_approved(approved: str, found: list[dict]) -> bool:
    """The engine read exactly one fact, the approved one: the same wording once normalised, or its rewriting into
    the third person. Anything else (two facts, another claim, a word added or denied) is not what was approved."""

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


def record(copy: str, job: Job, it: Item) -> None:
    """The copy's record of what was taught: one entry per fact sent."""

    with _record_lock:
        doc = lessons(copy)
        doc.append({"job": job.id, "at": time.time(), "source": job.source, "fact": it.fact, "question": it.question,
                    "outcome": it.state, "message": it.message, "steps": it.steps, "seconds": it.seconds,
                    "before": it.before, "after": it.after, "found": it.found})
        _write(_lessons_path(copy), doc)


def forget_plan(name: str, fact: str, manifests: list[dict]) -> dict:
    """What forgetting one fact takes: the version kept before the lesson that taught it, and the facts kept by later
    lessons, to teach again after restoring it. Facts TensorFold Web did not teach have no lesson to go back before."""

    record = lessons(name)
    kept = [x for x in record if x.get("outcome") in KEPT]
    # the words TensorFold Web taught, else the same fact in other words: the engine's rewriting ("The user has a dog…"),
    # as its graph shows it, or as a lesson recorded it
    taught = next((x for x in kept if x.get("fact") == fact), None) or next(
        (x for x in kept if canonical(fact) in {canonical(str(t)) for t in
                                                [x.get("fact") or "", *(f.get("text") or "" for f in x.get("found") or [])]}),
        None)
    if taught is None:
        raise ValueError("TensorFold Web did not teach this fact, so there is no version from before it; restore an "
                         "earlier version instead")
    version = next((m for m in manifests if m.get("label") == f"before lesson {taught['job']}"), None)
    if version is None:
        raise ValueError("no version was kept before the lesson that taught this fact")
    fact = taught["fact"]
    later, seen = [], set()
    for x in record:                                 # everything kept since that version, but the fact itself
        if x.get("outcome") in KEPT and x.get("at", 0) > version["at"] and x["fact"] != fact and x["fact"] not in seen:
            seen.add(x["fact"])
            later.append({"fact": x["fact"], "question": x.get("question") or ""})
    return {"version": version["id"], "again": later, "fact": fact}


def graph_summary(body: dict) -> list[dict]:
    """The engine's fact graph as rows: each fact with its state, neurons and questions."""

    out = []
    for f in body.get("facts") or []:
        if isinstance(f, dict):
            out.append({"id": f.get("id"), "text": f.get("text"), "state": f.get("state"),
                        "neurons": len(f.get("neurons") or []), "questions": (f.get("questions") or [])[:6],
                        "source": f.get("source")})
    return out


def _kl_cell(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def card(name: str, *, graph: dict, versions: list[dict], runs: list[dict], engine: str | None) -> str:
    """The copy's model card: where it came from, what it was taught and from what, what it kept, its versions and
    what learning cost it, with the warning that taught facts live in its weights."""

    meta = _read(_meta_path(name), {})
    record = lessons(name)
    facts = graph_summary(graph)
    when = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "—"
    lines = [f"# {name}: a living copy of {meta.get('base') or 'an unknown model'}", "",
             f"Made {when(meta.get('created'))}" + (f", fetched from {meta['fetched_from']}" if meta.get("fetched_from")
                                                    else "") + f". Served by {engine or 'TensorFold'} with `--slide`.", "",
             ("Facts taught into its weights (TensorFold Living Weights): they are part of the model files, answer "
              "every client, and cannot be removed one by one except by restoring a version kept before them. Do not "
              "share this folder if a fact is personal or confidential."), "",
             f"## Facts ({sum(1 for f in facts if f['state'] == 'learned')} kept of {len(facts)} tried)", "",
             "| Fact | State | Source | Neurons |", "|---|---|---|---|"]
    lines += [f"| {f['text']} | {f['state']} | {f.get('source') or '—'} | {f['neurons']} |" for f in facts]
    lines += ["", f"## Lessons sent by TensorFold Web ({len(record)})", "", "| When | Fact | Outcome | Answer before | After |",
              "|---|---|---|---|---|"]
    clip = lambda s: " ".join(str(s or "—").split())[:80].replace("|", "/")
    lines += [f"| {when(x.get('at'))} | {clip(x['fact'])} | {x.get('outcome')} | {clip(x.get('before'))} | "
              f"{clip(x.get('after'))} |" for x in record]
    lines += ["", f"## Versions ({len(versions)})", "", "| When | Version | Facts kept |", "|---|---|---|"]
    lines += [f"| {when(v['at'])} | {v['label']} | {v.get('kept', '—')} |" for v in versions]
    if runs:
        lines += ["", "## Measured (Bench)", "", "| When | Run | Facts kept | Speed | Same tokens drafts off | KL |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {when(r['created'])} | {r['mode']} | {r.get('kept', '—')} | "
                  f"{round(r['tps']) if r.get('tps') else '—'} tok/s | "
                  f"{(str(r['exact']) + ' of ' + str(r['checked'])) if r.get('checked') else '—'} | "
                  f"{_kl_cell(r.get('mean_kl'))} |" for r in runs]
    return "\n".join(lines) + "\n"
