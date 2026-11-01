"""Checks of a living copy: the same questions asked before and after, both thinking modes, and the leak matrix.

A copy's check set holds the questions of the facts it kept (from TensorFold Web's record), near misses and questions of
the user's own, and unrelated questions that should never move. A run asks each with thinking off and on and is kept
(`<name>.checks/<run>.json`); two runs compare question by question. For each kept fact, an answer "recalls" it when it
holds most of the words its fact adds to its question ("Pixel" for "What is my cat called?"), and "leaks" another fact
when it holds that one's words instead: the trial's case of the head of marketing named as the chief executive.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from . import living
from .settings import write_private

UNRELATED = ["What is the capital of Australia?", "What is 17 times 23?", "Name three primary colours.",
             "At what temperature does water boil at sea level, in Celsius?", "Who wrote Hamlet?",
             "What is the chemical symbol for gold?", "Which is the largest planet of the solar system?",
             "Translate 'good morning' into French."]
STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to", "my", "i", "me", "its", "it", "and",
        "for", "by", "with", "what", "who", "when", "where", "which", "how", "does", "do", "did", "called", "named",
        "name", "every", "each", "has", "have", "this", "that", "from", "as"}
THINK_TOKENS = 1500


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and (len(w) > 1 or w.isdigit())}


def answer_words(fact: str, question: str) -> list[str]:
    """What a fact adds to its question: the words a recalled answer must hold."""

    return sorted(_words(fact) - _words(question))


def _set_path(name: str) -> Path:
    return living.meta_dir(name) / "checkset.json"


def _runs_dir(name: str) -> Path:
    return living.meta_dir(name) / "checks"


def undone(name: str, before: list[dict], after: list[dict]) -> None:
    """After a restore: the facts it took out join the checks as "undone", so the next run shows they are gone."""

    kept_after = {x["fact"] for x in after if x.get("outcome") in living.KEPT}
    gone = [x for x in before if x.get("outcome") in living.KEPT and x["fact"] not in kept_after and x.get("question")]
    if not gone:
        return
    cs = check_set(name)
    have = {o["q"] for o in cs["own"]}
    extra = [{"q": x["question"], "kind": "undone", "fact": x["fact"], "expect": answer_words(x["fact"], x["question"])}
             for x in gone if x["question"] not in have]
    set_own(name, cs["own"] + extra, cs["unrelated"])


def check_set(name: str) -> dict:
    """The facts kept (from the record), the user's own questions, and the unrelated ones."""

    own = json.loads(_set_path(name).read_text()) if _set_path(name).is_file() else {"questions": []}
    facts, seen = [], set()
    for x in living.lessons(name):
        if x.get("outcome") in living.KEPT and x.get("question") and x["fact"] not in seen:
            seen.add(x["fact"])
            facts.append({"q": x["question"], "kind": "fact", "fact": x["fact"],
                          "expect": answer_words(x["fact"], x["question"])})
    # a fact taken out and taught again since is a fact again, not an undone one
    taught = {f["fact"] for f in facts}
    own_now = [o for o in own["questions"] if not (o.get("kind") == "undone" and o.get("fact") in taught)]
    return {"facts": facts, "own": own_now, "unrelated": own.get("unrelated", UNRELATED)}


def set_own(name: str, questions: list[dict], unrelated: list[str] | None = None) -> dict:
    def one(q: dict) -> dict:
        kind = q.get("kind") if q.get("kind") in ("near", "undone") else "own"
        out = {"q": " ".join(str(q.get("q", "")).split())[:300], "kind": kind}
        if kind == "undone":                              # a fact a restore took out: it must not come back
            out |= {"fact": str(q.get("fact") or ""), "expect": [str(w) for w in q.get("expect") or []][:12]}
        return out
    clean = [one(q) for q in questions if isinstance(q, dict) and str(q.get("q", "")).strip()][:60]
    doc = {"questions": clean, "unrelated": [u for u in (unrelated or UNRELATED) if u.strip()][:30]}
    from .settings import write_private

    write_private(_set_path(name), json.dumps(doc, indent=1, ensure_ascii=False))
    return check_set(name)


def run(t: living.Target, name: str, label: str, *, thinking: bool = True, cancel=None,
        extra: list[dict] | None = None, same_as: dict | None = None) -> dict:
    """Every question of the set, thinking off (and on), kept as a run. ``extra``: the facts a lesson is about to
    teach (or just taught), asked too, so the run before a lesson and the run after it hold the same questions."""

    if same_as is not None:                              # the run after a lesson asks exactly what the run before did
        rows = [{k: r[k] for k in ("q", "kind", "fact", "expect") if k in r} for r in same_as["rows"]]
        thinking = bool(same_as.get("thinking"))
        extra = None
    cs = check_set(name)
    if same_as is None:
        rows = [{"q": f["q"], "kind": "fact", "fact": f["fact"], "expect": f["expect"]} for f in cs["facts"]]
    have = {r["q"] for r in rows}
    for x in extra or []:
        if x.get("question") and x["question"] not in have:
            have.add(x["question"])
            rows.append({"q": x["question"], "kind": "fact", "fact": x["fact"],
                         "expect": answer_words(x["fact"], x["question"])})
    if same_as is None:
        rows += [{"q": o["q"], "kind": o["kind"], **({"fact": o["fact"], "expect": o["expect"]}
                                                    if o["kind"] == "undone" else {})} for o in cs["own"]]
        rows += [{"q": u, "kind": "unrelated"} for u in cs["unrelated"]]
    for r in rows:
        if cancel is not None and cancel():
            break
        r["off"] = _ask(t, r["q"], False)
        if thinking:
            r["on"] = _ask(t, r["q"], True)
    out = {"id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4], "label": label[:120], "at": time.time(),
           "thinking": thinking, "rows": rows}
    out["summary"] = summarise(out)
    _runs_dir(name).mkdir(parents=True, exist_ok=True)
    write_private(_runs_dir(name) / f"{out['id']}.json", json.dumps(out, ensure_ascii=False))   # the person's facts
    return out


def _ask(t: living.Target, q: str, thinking: bool) -> str | None:
    try:
        body_tokens = THINK_TOKENS if thinking else 160
        reply = t.chat([{"role": "user", "content": q}], max_tokens=body_tokens, thinking=thinking, timeout=120.0)
        return reply
    except Exception as exc:  # noqa: BLE001 - a question that fails is said in its cell
        return f"[error: {str(exc)[:120]}]"


NEGATION = re.compile(r"\b(not|no|never|isn't|isnt|aren't|don't|dont|doesn't|didn't|didnt|haven't|hasn't|wasn't|weren't|"
                      r"won't|wouldn't|couldn't|cannot|can't|unknown|unsure|none|neither|nor|nobody|nothing)\b")


def recalls(answer: str | None, expect: list[str]) -> bool:
    """Most of the words the fact adds are in the answer, and none of them is denied ("not Pixel"). A heuristic: the
    TensorFold Web reads words, not meaning, and says so."""

    if not answer or not expect or answer.startswith("[error:"):
        return False
    # a word counts in a sentence that does not deny: "Pixel is not my cat" does not recall Pixel, "Not Ginger. It is
    # Pixel." does
    plain = answer.lower().replace("\u2019", "'").replace("\u2018", "'")     # curly apostrophes read as straight
    sentences = [s for s in re.split(r"(?<=[.!?;])\s+|\n+", plain) if s.strip()]
    affirmed = set().union(*(_words(s) for s in sentences if not NEGATION.search(s))) if sentences else set()
    hits = [w for w in expect if w in affirmed]
    return len(hits) * 2 >= len(expect) and bool(hits)


def summarise(run_doc: dict) -> dict:
    """Recall per mode, empty replies, and the leak matrix: which fact's words answer which fact's question."""

    facts = [r for r in run_doc["rows"] if r["kind"] == "fact"]
    out: dict = {"facts": len(facts), "modes": {}}
    for mode in ("off", "on"):
        if not any(mode in r for r in facts) and mode == "on" and not run_doc.get("thinking"):
            continue
        asked = [r for r in facts if mode in r]              # a run cut short asked some facts only
        recalled = sum(1 for r in asked if recalls(r.get(mode), r["expect"]))
        empty = sum(1 for r in run_doc["rows"] if mode in r and not (r.get(mode) or "").strip())
        errors = sum(1 for r in run_doc["rows"] if (r.get(mode) or "").startswith("[error:"))
        leaks = []
        for r in run_doc["rows"]:                        # any answer, recalled or not, near miss or unrelated
            for other in facts:
                if other is r or not other["expect"]:
                    continue
                own = set(r.get("expect") or []) | _words(r["q"])   # an answer restating its question leaks nothing
                if set(other["expect"]) - own and recalls(r.get(mode), sorted(set(other["expect"]) - own)):
                    leaks.append({"question": r["q"], "answered_with": other["fact"]})
        back = sum(1 for r in run_doc["rows"] if r["kind"] == "undone" and recalls(r.get(mode), r.get("expect", [])))
        out["modes"][mode] = {"recalled": recalled, "asked": len(asked), "empty": empty, "errors": errors,
                              "leaks": leaks, "undone_back": back}
    out["undone"] = sum(1 for r in run_doc["rows"] if r["kind"] == "undone")
    return out


def runs(name: str) -> list[dict]:
    d = _runs_dir(name)
    if not d.is_dir():
        return []
    docs = [json.loads(p.read_text()) for p in d.glob("*.json")]
    return sorted(({k: v for k, v in x.items() if k != "rows"} for x in docs), key=lambda x: x["at"])


ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4,12}")     # a check's id as run() makes it (also `against`)


def get(name: str, rid: str) -> dict:
    if not ID.fullmatch(rid):
        raise KeyError(rid)
    p = _runs_dir(name) / f"{rid}.json"
    if not p.is_file():
        raise KeyError(rid)
    return json.loads(p.read_text())


def compare(a: dict, b: dict) -> list[dict]:
    """Question by question, what changed from run ``a`` to run ``b`` in each mode."""

    before = {r["q"]: r for r in a["rows"]}
    out = []
    for r in b["rows"]:
        old = before.get(r["q"])
        row = {"q": r["q"], "kind": r["kind"], "fact": r.get("fact")}
        for mode in ("off", "on"):
            if mode not in r:
                continue
            row[mode] = {"before": (old or {}).get(mode), "after": r.get(mode),
                         "changed": old is None or (old.get(mode) or "").strip() != (r.get(mode) or "").strip()}
            if r["kind"] in ("fact", "undone"):
                row[mode]["recalled"] = recalls(r.get(mode), r["expect"])
        out.append(row)
    return out
