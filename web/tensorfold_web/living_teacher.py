"""Living teacher."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict

import httpx

from . import httpread, living


class Teacher:

    def __init__(self, target, hold, release, log=list, served=lambda: None, graph=dict, undo=None) -> None:
        self.target, self.hold, self.release, self.log, self.served = target, hold, release, log, served
        self.graph = graph                     # the engine's fact graph, to see what it did with a stopped lesson
        self.undo = undo                       # undo(copy, version): stop, restore the version, serve again (gate held)
        self.job: living.Job | None = None
        self._cancel = threading.Event()         # stop after the fact being learned (or before any, during the checks)
        # stop now: the learn request is closed, the engine stops at its next check
        self._abort = threading.Event()
        # the copy of a lesson a stopped TensorFold Web left, while it is waited for
        self.recovering: str | None = None
        self._thread: threading.Thread | None = None

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, copy: str, facts: list[dict], source: str = "studio", *, version_before: bool = True,
              check: bool = True, gate_held: bool = False) -> living.Job:
        if self.busy():
            raise RuntimeError("a lesson is going; wait for it or cancel it")
        items = [living.Item(id=f.get("id") or uuid.uuid4().hex[:8], fact=" ".join(str(f["fact"]).split()),
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
            job = living.Job(id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6], copy=copy, items=items,
                      source=source, version_before=True, check=check, gate_held=gate_held)
            living._journal().parent.mkdir(parents=True, exist_ok=True)
            living._write(living._journal(), {"job": job.id, "copy": copy, "started": job.started,
                                "items": [{"fact": i.fact, "question": i.question} for i in items]})
            self.job = job
            self._cancel.clear()
            self._abort.clear()
            self._thread = threading.Thread(target=self._go, args=(job,), name="lesson", daemon=True)
            self._thread.start()
        except BaseException:
            try:
                living._journal().unlink(missing_ok=True)
            except OSError:
                pass
            finally:
                if not gate_held:
                    self.release()
            raise
        return job

    def cancel(self, now: bool = False) -> bool:

        if not self.busy():
            return False
        self._cancel.set()
        if now:
            self._abort.set()
        return True

    def _go(self, job: living.Job) -> None:
        from . import living_checks, living_versions

        try:
            t: living.Target = self.target()
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
                    it.before = living._safe(lambda q=it.question: t.chat([{"role": "user", "content": q}], max_tokens=80))
                it.state = "learning"
                began = time.time()
                try:
                    self._learn(t, it)
                except Exception as exc:  # noqa: BLE001 - one fact failing does not stop the others
                    it.state, it.message = "failed", str(exc)[:300]
                    gone = self.served() != job.copy     # unless the engine itself is gone: nothing more can be learned
                it.seconds = round(time.time() - began, 1)
                if gone:
                    living.record(job.copy, job, it)
                    for rest in job.items[n + 1:]:
                        rest.state = "interrupted"
                    raise RuntimeError("the engine stopped while learning: the copy may hold a half-written lesson; "
                                       "restore the version kept before it, or run its checks before trusting it")
                if it.state in ("refused", "cancelled") and self._learned_anyway(it, assume=it.state == "refused"):
                    living.record(job.copy, job, it)            # written first: a rollback that fails still shows it
                    self._roll_back(job, n)              # something not approved (or not wanted) reached the weights
                    break
                if it.state == "cancelled":              # stopped now, and the engine kept nothing of it
                    living.record(job.copy, job, it)
                    break
                if it.state == "missed" and not it.message:   # the engine gave no reason: its log has one
                    it.message = living.why_missed(self.log(), it.fact)
                if it.state in living.KEPT and it.question:
                    it.after = living._safe(lambda q=it.question: t.chat([{"role": "user", "content": q}], max_tokens=80))
                living.record(job.copy, job, it)
            if job.check and before is not None and any(i.state in living.KEPT for i in job.items):
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
                living._journal().unlink(missing_ok=True)
            except OSError:
                pass
            finally:
                if not job.gate_held:
                    self.release()

    def _learned_anyway(self, it: living.Item, assume: bool = True) -> bool:

        ids = {f["id"] for f in it.found}
        if not ids:
            if assume:
                return True                              # nothing to poll: the version is there, the weights may not be
            try:
                facts = (self.graph() or {}).get("facts") or []
            except Exception:  # noqa: BLE001 - an engine that cannot say: assume the worst, the version is there
                return True
            return any(f.get("state") == "learned" and living.canonical(str(f.get("text") or "")) == living.canonical(it.fact)
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

    def _roll_back(self, job: living.Job, upto: int) -> None:
        it = job.items[upto]
        if not job.version or self.undo is None:
            it.message = (it.message or "") + ". The engine kept some of it anyway and no version was kept before this " \
                "lesson: check the copy, or delete it"
            living.record(job.copy, job, it)
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
                    living.record(job.copy, job, done)
            raise
        # the restore put back the record as it was before the lesson: every item of the lesson is written again
        for done in job.items[: upto + 1]:
            if done.state in ("kept", "partly kept", "refused", "cancelled"):
                stopped = done.state == "cancelled"
                done.state = "rolled back"
                done.message = ("stopped at your request: the engine had kept part of it, so the lesson was undone" if stopped
                                else "the lesson was undone: the engine had learned something that was not approved")
            if done.state != "queued":
                living.record(job.copy, job, done)
        for rest in job.items[upto + 1:]:
            rest.state = "cancelled"
        job.status = "rolled back"

    def recover(self, graph, alive=lambda: True) -> str | None:

        j = living._read(living._journal(), None)
        if not j:
            return None
        done = {x["fact"] for x in living.lessons(j["copy"]) if x.get("job") == j["job"]}
        for x in j["items"]:
            if x["fact"] not in done:
                fake = living.Job(id=j["job"], copy=j["copy"], items=[], source="interrupted")
                living.record(j["copy"], fake, living.Item(id="", fact=x["fact"], question=x.get("question") or "",
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
                    living._journal().unlink(missing_ok=True)
                except OSError:
                    pass
                finally:
                    self.recovering = None
                    self.release()
        self._thread = threading.Thread(target=wait, name="lesson-recovery", daemon=True)
        self._thread.start()
        return f"a lesson on {j['copy']} was going when TensorFold Web stopped; its unfinished facts are marked interrupted"

    def _learn(self, t: living.Target, it: living.Item) -> None:

        headers = {"Authorization": f"Bearer {t.key}"} if t.key else {}
        opened: list[httpx.Response] = []
        refused = False

        def work() -> None:
            nonlocal refused
            with httpx.stream("POST", t.base + "/v1/slide/learn", json={"text": it.fact, "source": "chat"},
                              headers=headers, timeout=httpx.Timeout(1800.0, connect=10.0)) as r:
                # "stop now" closes it: the engine drops the fact at its next check
                opened.append(r)
                if r.status_code >= 400:
                    raise RuntimeError(f"the engine refused the lesson ({r.status_code}): {living._bounded(r)}")
                refused = living._read_events(r, it)

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
