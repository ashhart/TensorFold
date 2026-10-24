"""LLWZ's NoteStore with the server's intake for /v1/notes writes, so a write sequence can be run both ways.

The server takes free text, not the rule book's lessons, so it ports only the deterministic part of
zengine/intake.py (see NOTES-IN-ENGINE.md). Here a lesson marked "http" goes through that same part, and
everything else (the lessons a set's store was built from) through LLWZ's own intake.classify. All the rest
(_apply, add, retire, line, pack, fit, render, delete's rebuild, the export) is LLWZ's code, unchanged.

  verdict    add: ALREADY_KNOWN when a current fact of the same subject has the same normal form (intake.norm,
             ASCII letters lowered), else NEW. change: ALREADY_KNOWN when a current fact of its pair does; else
             it retires its target if that is current (REPLACES for a fact, NEW for a take-back), or, its target
             gone (erased), every current note of the pair. take_back: retires its target if it is a current
             fact (REPLACES), else IMPLIED (nothing to take back, so nothing is written).
  erase      the note's own write, the writes kept out as the same as it, and the take-backs of it.
"""
OPS = {"add": "ASSERT", "change": "CHANGE", "take_back": "RETRACT"}
DEFAULT_TAKEBACK = "I take that back."


def lnorm(intake, text):
    """intake.norm with ASCII letters lowered (the server lowers no other letter)."""
    return "".join(c.lower() if c < "\x80" else c for c in intake.norm(text))


def make(nm, intake):
    """The LiveStore class over LLWZ's notes module `nm` and intake module `intake`."""

    class LiveStore(nm.NoteStore):
        """NoteStore whose HTTP writes take the server's intake; same_of[source] is each write's same_as."""

        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.same_of = {}
            self._gone = []

        def _apply(self, O, lesson, ids, gr, today=None):
            self._O = O
            return super()._apply(O, lesson, ids, gr, today)

        def _target(self, t):
            O = self._O
            if t is None:
                return None
            if "note" in t:
                n = O.notes.get(t["note"])
                return t["note"] if n is not None and not n["lesson"].get("http") else None
            return next((i for i, n in O.notes.items() if n["lesson"].get("http") and n["source"] == t["source"]), None)

        def as_target(self, i):
            n = self._O.notes[i]
            return {"source": n["source"]} if n["lesson"].get("http") else {"note": i}

        def classify(self, store, l, ents=None, sch=None):
            if not l.get("http"):
                return intake.classify(store, l, ents=ents, sch=sch)
            O, op, key = self._O, l["http"], (l["e"], l["a"])
            want = lnorm(intake, l["text"])
            fact = lambda n: n["state"] == nm.CURRENT and n["kind"] == nm.FACT and lnorm(intake, n["text"]) == want
            same = []
            if op == "add":
                same = sorted(i for i, n in O.notes.items() if n["entity"] == l["e"] and fact(n))
            elif op == "change":
                same = sorted(i for i, n in O.notes.items() if (n["entity"], n["attribute"]) == key and fact(n))
            self.same_of[l["id"]] = [self.as_target(i) for i in same]
            self.same_ids = same
            gone, verdict = [], "NEW"
            if same:
                verdict = "ALREADY_KNOWN"
            elif op == "change":
                t = self._target(l.get("target"))
                if t is not None and O.notes[t]["state"] == nm.CURRENT:
                    gone, verdict = [t], "REPLACES" if O.notes[t]["kind"] == nm.FACT else "NEW"
                else:
                    gone = [i for i in O.by_key.get(key, []) if O.notes[i]["state"] == nm.CURRENT]
                    verdict = "REPLACES" if any(O.notes[i]["kind"] == nm.FACT for i in gone) else "NEW"
            elif op == "take_back":
                t = self._target(l.get("target"))
                if t is None or O.notes[t]["state"] != nm.CURRENT or O.notes[t]["kind"] != nm.FACT:
                    verdict = "IMPLIED"
                else:
                    gone, verdict = [t], "REPLACES"
            self._gone = gone
            learn = verdict in ("NEW", "REPLACES")
            return {"verdict": verdict, "learn": learn, "same_as": [O.notes[i]["lesson"]["id"] for i in same],
                    "covered_by": [], "supersedes": [O.notes[i]["lesson"]["id"] for i in gone], "changed": []}

        def no_longer_current(self, sch, ls, new):
            if not new.get("http"):
                return nm.NoteStore.no_longer_current(sch, ls, new)
            O, key = self._O, (new["e"], new["a"])
            cur = [i for i in O.by_key.get(key, []) if O.notes[i]["state"] == nm.CURRENT]
            return [k for k, i in enumerate(cur) if i in self._gone]

        def http_write(self, owner, op, text, told, source, subject="me", target=None):
            """One /v1/notes write: (the log entry, the same_as note ids)."""
            O = self.owner(owner, create=True)
            self._O = O
            if op == "add":
                e, a = subject, f"note:{source}"
            else:
                i = self._target(target)
                e, a = O.notes[i]["entity"], O.notes[i]["attribute"]
            lesson = {"id": source, "http": op, "op": OPS[op], "e": e, "a": a, "text": text, "told": told}
            if target is not None:
                lesson["target"] = target
            self.same_ids = []
            entry = self.write(owner, lesson)
            return entry, list(self.same_ids)

        def erase(self, owner, note_id):
            """The server's erase: (the sources taken out, [(old id, new id)] for the notes that moved)."""
            O = self.owner(owner)
            self._O = O
            x = self.as_target(note_id)
            out = []
            for e in O.log:
                l = e["lesson"]
                own = l.get("http") and e["source"] == x.get("source")
                same = e["verdict"] == "ALREADY_KNOWN" and x in self.same_of.get(e["source"], [])
                takes = l.get("http") == "take_back" and l.get("target") == x
                if own or same or takes:
                    out.append(e["source"])
            before = {n["source"]: i for i, n in O.notes.items() if n["lesson"].get("http")}
            for src in out:
                self.delete(owner, src)
            O = self.owner(owner)
            after = {n["source"]: i for i, n in O.notes.items() if n["lesson"].get("http")}
            moved = sorted((i, after[s]) for s, i in before.items() if s in after and after[s] != i)
            return out, moved

        def told_next(self, owner, today_tick, base_told):
            """The tick the server tells the next write on: after the snapshot's notes and every write kept so far."""
            O = self.owner(owner)
            kept = [e["lesson"]["told"] for e in O.log if e["lesson"].get("http")]
            return max(max([base_told] + kept) + 100, today_tick)

    return LiveStore
