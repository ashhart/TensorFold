"""The server's who-first note lookup (zig/src/server/notes/who.zig) in Python, on LLWZ's NoteStore."""
import heapq
import math
import re

FIRST = re.compile(r"\b(?:I|me|mine|myself)\b", re.ASCII)
TELL_ME = re.compile(r"(?i)\b(?:tell|give|show)\s+me\b", re.ASCII)
MY = re.compile(r"(?i)\bmy\s+([a-z']+)", re.ASCII)
LETTERS = re.compile(r"[A-Za-z]+")
KEY_SPLIT = re.compile(r"[_\- \t\r\n]+")
CIRCLE = re.compile(r"me|my [^\n]+|the [a-z]+")
NAME_LETTER = "A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u04ff\u1e00-\u1eff"  # Latin, Greek, Cyrillic
NAME_RUN = re.compile(f"[{NAME_LETTER}]+(?:-[{NAME_LETTER}]+)*")
CHAIN = 4  # a name is up to this many runs of name letters, one space apart
LOOSE = "note:"  # the attribute prefix of a note the server wrote without sorting it


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
    """NoteStore._lookup up to pack(): retrieve()'s scores after its hop and its pool, without the excluded notes."""
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
        return score, set()
    notes = O.notes
    order = lambda i: (-score[i], -notes[i]["told"], i)  # noqa: E731
    room = max(1, (cap - ns.fixed()["head"]) // 4)
    direct = heapq.nsmallest(4 * room, score, key=order) if len(score) > 8 * room else sorted(score, key=order)
    if cfg["min_rel"] > 0:
        floor = score[direct[0]] * cfg["min_rel"]
        direct = [i for i in direct if score[i] >= floor]
    pool = set(direct)
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
                if j not in exclude:
                    if s > score.get(j, 0.0):
                        score[j] = s
                    pool.add(j)
    return score, pool


def circle(O):
    out = {e for e in O.ent_cur if CIRCLE.fullmatch(e)}
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
    """I, me, mine or myself; or "my X" with X no word of an entity's name, a key part or a word of the user's notes."""
    if FIRST.search(TELL_ME.sub(" ", question)):
        return True
    xs = {m.group(1).split("'")[0].lower() for m in MY.finditer(question)} - {""}
    xs -= {w.lower() for e in O.ent_cur for w in LETTERS.findall(e)}
    if not xs:
        return False
    key_parts = {t.lower() for n in O.notes.values() for t in KEY_SPLIT.split(n["attribute"]) if t and t.isascii()}
    own = {w.lower() for i in O.ent_cur.get("me", ()) for w in LETTERS.findall(ns.line(O, i))}
    return bool(xs & (key_parts | own))


def subjects(nm, ns, O, question, wts):
    keys = [ns.key(O, i) for i in ns.tok.encode(question)]
    out = set(ns.subjects(O, keys))
    if speaker(ns, O, question) and (O.ent_cur.get("me") or O.ent_keys.get("me")):
        out.add("me")
    by, mine = names(nm, ns, O), None
    runs = [(m.start(), m.end(), tuple(ns.key(O, t) for t in ns.tok.encode(" " + m.group())))
            for m in NAME_RUN.finditer(question)] if by else []
    for k in range(len(runs)):
        got = ()
        for j in range(k, max(-1, k - CHAIN), -1):
            if j < k and not (runs[j][1] + 1 == runs[j + 1][0] and question[runs[j][1]] == " "):
                break
            got = runs[j][2] + got
            if len(got) > 8:
                break
            cs = by.get(got, set())
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
    score, pool = scores(nm, ns, O, ns.tok.encode(question), cap, past, exclude)
    notes = O.notes
    loose = {i for i in pool if notes[i]["entity"] == "me" and notes[i]["attribute"].startswith(LOOSE)}
    if not subj and not loose:
        return res

    def notes_of(e):
        got = set(O.ent_cur.get(e, ()))
        if past:
            for k in O.ent_keys.get(e, ()):
                got |= set(O.by_key[k])
        return got - exclude

    own = set()
    for e in subj:
        own |= notes_of(e)
    cand = own | loose
    for i in own:
        v = notes[i]["value"]
        if isinstance(v, str) and v in O.ent_cur and v not in subj:
            if notes[i]["entity"] in wts:
                wts[v] = min(wts.get(v, 1.0), wts[notes[i]["entity"]])
            cand |= notes_of(v)
    sc = lambda i: score.get(i, 0.0) * wts.get(notes[i]["entity"], 1.0)  # noqa: E731
    order = sorted(cand, key=lambda i: (-sc(i), -notes[i]["told"], i))
    chosen, block, ntok = ns.fit(O, ns.pack(O, order, cap, past), cap)
    cur = sorted((i for i in chosen if notes[i]["state"] == nm.CURRENT), key=lambda i: notes[i]["told"])
    hist = sorted((i for i in chosen if notes[i]["state"] != nm.CURRENT), key=lambda i: notes[i]["told"])
    res.update(ids=cur + hist, block=block, tokens=ntok, candidates=len(cand))
    return res
