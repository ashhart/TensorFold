"""The server's who-first note lookup (zig/src/server/notes/who.zig) in Python, on LLWZ's NoteStore: the tests' answers.

A question's subjects are the store's entity phrases it holds (retrieve()'s subject matching), the user, and the
entities a current name fact calls a word of it. The user is a subject when the question says I, me, mine or myself
(not the "me" of "tell me", "give me", "show me"), or "my X" where X is no word of any entity's name and is a part of
a stored attribute's name or a word of one of the user's current notes. A name several entities hold names those of
the user's circle ("me", "my ...", "the <word>" and the entities their current notes name), or else all of them with
a weight below 1. The candidates are the subjects' notes (every state for a question about the past) and one hop to
the entities those notes name; an excluded note is never a candidate and its link is never followed. They are ranked
by retrieve()'s scores times the weights, then packed and fitted as retrieve() packs and fits. No subject: no block.

  lookup(nm, ns, owner, question, cap=None, past=None, exclude=())   nm: LLWZ's notes module
"""
import heapq
import math
import re

FIRST = re.compile(r"\b(?:I|me|mine|myself)\b")
TELL_ME = re.compile(r"(?i)\b(?:tell|give|show)\s+me\b")
MY = re.compile(r"(?i)\bmy\s+([a-z']+)")
WORD = re.compile(r"[A-Za-z]+")
LETTERS = re.compile(r"[a-z]+")
KEY_SPLIT = re.compile(r"[_\- \t\r\n]+")
CIRCLE = re.compile(r"^(?:me|my .+|the [a-z]+)$")


def name_value(subject, text):
    """alias.zig's nameValue: the name a name fact's text gives its subject, or None."""
    if len(text) <= len(subject) or text[:len(subject)].lower() != subject.lower():
        return None
    rest, value = text[len(subject):], None
    for prefix in ("'s name is ", " is named ", " is called "):
        if rest[:len(prefix)].lower() == prefix:
            value = rest[len(prefix):]
            break
    if value is None:
        return None
    v = value.strip(" \t")
    v = v[:-1] if v.endswith(".") else v
    v = v[:-4] if len(v) >= 4 and v[-4:].lower() == " now" else v
    b = v.encode()
    if not b or len(b) > 96:
        return None
    if any(not (65 <= c <= 90 or 97 <= c <= 122) and c not in (32, 45, 39) and c < 0x80 for c in b) or b.count(32) > 3:
        return None
    for bad in ("not", "unknown", "no longer", "don't know"):
        if v.lower() == bad or (len(v) > len(bad) and v[:len(bad)].lower() == bad and v[len(bad)] == " "):
            return None
    return v


def scores(nm, ns, O, qids, cap, past, exclude):
    """NoteStore._lookup up to pack(): retrieve()'s scores after its hop, without the excluded notes."""
    cfg = ns.cfg
    qkeys = [ns.key(O, i) for i in qids]
    cands, dfs, N = ns.postings(O, [nm.CURRENT, "history"] if past else [nm.CURRENT])
    score = {}
    for g in ns.grams(O, qids):
        df = sum(nm._post_len(ix.get(g)) for ix in dfs)
        if not df:
            continue
        w = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
        for ix in cands:
            for i in nm._post_ids(ix.get(g)):
                score[i] = score.get(i, 0.0) + w
    if cfg["subject"] > 0 and N:
        top = math.log(1.0 + (N - 0.5) / 1.5) * cfg["subject"]
        for e in ns.subjects(O, qkeys):
            for i in O.ent_cur.get(e, ()):
                score[i] = score.get(i, 0.0) + top
            if past:
                for k in O.ent_keys.get(e, ()):
                    for i in O.by_key[k]:
                        score[i] = score.get(i, 0.0) + top
    for i in exclude:
        score.pop(i, None)
    if not score:
        return score
    notes = O.notes
    order = lambda i: (-score[i], -notes[i]["told"], i)  # noqa: E731
    room = max(1, (cap - ns.fixed()["head"]) // 4)
    direct = heapq.nsmallest(4 * room, score, key=order) if len(score) > 8 * room else sorted(score, key=order)
    if cfg["min_rel"] > 0:
        floor = score[direct[0]] * cfg["min_rel"]
        direct = [i for i in direct if score[i] >= floor]
    for i in direct[:cfg["hop_from"]]:
        n = notes[i]
        s, near = score[i] * cfg["hop_weight"], []
        if cfg["hop"] != "none":
            near.append(O.ent_cur.get(n["entity"], ()))
            if cfg["hop"] == "subject+value" and n["value"] in O.ent_cur and n["value"] != n["entity"]:
                near.append(O.ent_cur[n["value"]])
        if past and cfg["hop_history"]:
            near.append(O.by_key.get((n["entity"], n["attribute"]), ()))
        for js in near:
            for j in js:
                if j not in exclude and s > score.get(j, 0.0):
                    score[j] = s
    return score


def circle(O):
    out = {e for e in O.ent_cur if CIRCLE.match(e)}
    for e in list(out):
        for i in O.ent_cur.get(e, ()):
            v = O.notes[i]["value"]
            if isinstance(v, str) and v in O.ent_cur:
                out.add(v)
    return out


def names(nm, ns, O):
    """Current name facts by the keys of " " + the name their text gives: entities."""
    by = {}
    for (e, a), ids in O.by_key.items():
        if a != "name":
            continue
        for i in ids:
            n = O.notes[i]
            v = name_value(e, n["text"]) if n["state"] == nm.CURRENT and n["kind"] == nm.FACT else None
            k = tuple(ns.key(O, t) for t in ns.tok.encode(" " + v)) if v is not None else ()
            if 0 < len(k) <= 8:
                by.setdefault(k, set()).add(e)
    return by


def speaker(ns, O, question):
    if FIRST.search(TELL_ME.sub(" ", question)):
        return True
    ent_words = {w for e in O.ent_cur for w in LETTERS.findall(e.lower())}
    key_parts = {t.lower() for n in O.notes.values() for t in KEY_SPLIT.split(n["attribute"]) if t}
    own = {w for i in O.ent_cur.get("me", ()) for w in LETTERS.findall(ns.line(O, i).lower())}
    for m in MY.finditer(question):
        x = m.group(1).lower().split("'")[0]
        if x and x not in ent_words and (x in key_parts or x in own):
            return True
    return False


def subjects(nm, ns, O, question, wts):
    keys = [ns.key(O, i) for i in ns.tok.encode(question)]
    out = set(ns.subjects(O, keys))
    if speaker(ns, O, question) and (O.ent_cur.get("me") or O.ent_keys.get("me")):
        out.add("me")
    by, mine = names(nm, ns, O), None
    for w in WORD.findall(question):
        cs = by.get(tuple(ns.key(O, t) for t in ns.tok.encode(" " + w)), set())
        if len(cs) <= 1:
            out |= cs
            continue
        mine = circle(O) if mine is None else mine
        if cs & mine:
            out |= cs & mine
            continue
        E = max(len(O.ent_cur), len(cs))
        wgt = math.log(1.0 + (E - len(cs) + 0.5) / (len(cs) + 0.5)) / math.log(1.0 + (E - 0.5) / 1.5)
        out |= cs
        for e in cs:
            wts[e] = min(wts.get(e, 1.0), wgt)
    return out


def lookup(nm, ns, owner, question, cap=None, past=None, exclude=()):
    """{"ids", "block", "tokens", "candidates", "past"}: the who-first block for one question of one owner."""
    O = ns.owner(owner)
    cap = ns.cfg["cap"] if cap is None else cap
    past = nm.is_past(question) if past is None else past
    exclude = frozenset(exclude)
    res = {"ids": [], "block": "", "tokens": 0, "candidates": 0, "past": past}
    wts = {}
    subj = subjects(nm, ns, O, question, wts)
    if not subj:
        return res
    score = scores(nm, ns, O, ns.tok.encode(question), cap, past, exclude)

    def notes_of(e):
        got = set(O.ent_cur.get(e, ()))
        if past:
            for k in O.ent_keys.get(e, ()):
                got |= set(O.by_key[k])
        return got - exclude

    own = set()
    for e in subj:
        own |= notes_of(e)
    cand = set(own)
    for i in own:
        v = O.notes[i]["value"]
        if isinstance(v, str) and v in O.ent_cur and v not in subj:
            if O.notes[i]["entity"] in wts:
                wts[v] = min(wts.get(v, 1.0), wts[O.notes[i]["entity"]])
            cand |= notes_of(v)
    notes = O.notes
    sc = lambda i: score.get(i, 0.0) * wts.get(notes[i]["entity"], 1.0)  # noqa: E731
    order = sorted(cand, key=lambda i: (-sc(i), -notes[i]["told"], i))
    chosen, block, ntok = ns.fit(O, ns.pack(O, order, cap, past), cap)
    cur = sorted((i for i in chosen if notes[i]["state"] == nm.CURRENT), key=lambda i: notes[i]["told"])
    hist = sorted((i for i in chosen if notes[i]["state"] != nm.CURRENT), key=lambda i: notes[i]["told"])
    res.update(ids=cur + hist, block=block, tokens=ntok, candidates=len(cand))
    return res
