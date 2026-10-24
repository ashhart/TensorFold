"""Writes the who-first lookup's test fixtures (a made-up byte-tokenizer store and Python's answers from who_ref.py)."""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fixture  # noqa: E402
import live_intake  # noqa: E402
import who_ref  # noqa: E402

OWNER = "u"
USAGE = """who_bytes_snapshot.json: the made-up store (LLWZ's notes_export.py, fixture.py's byte tokenizer).
who_bytes_cases.json: per case the who-first answer and retrieve()'s ids.
who_bytes_live.json: WRITES as /v1/notes adds on that store, and per LIVE case the who-first answer after them.
l6_bytes_who.json: the who-first answer for each case of l6_bytes_cases.json (the L6 store must export unchanged)."""
STORE = [  # (entity, attribute, value, text); a phrase of the byte tokenizer is at most 8 keys: " my son" is 7
    ("my son", "name", "Ivo", "My son's name is Ivo."),
    ("my son", "eyes", "hazel", "My son has hazel eyes."),
    ("my aunt", "name", "Vera", "My aunt's name is Vera."),
    ("my aunt", "home_town", "Riga", "My aunt lives in Riga."),
    ("my son", "town_like", "my aunt", "My son lives in the same town as my aunt."),
    ("the van", "owner", "my son", "The van belongs to my son."),
    ("the van", "paint", "green", "The van is painted green."),
    ("me", "likes", "plums", "I like plums."),
    ("my kid", "likes", "figs", "My kid likes figs."),
    ("me", "eyes", "grey", "I have grey eyes."),
    ("Rafa's dog", "name", "Ivo", "Rafa's dog is called Ivo."),
    ("Rafa's cat", "name", "Pip", "Rafa's cat is called Pip."),
    ("Mira's cat", "name", "Pip", "Mira's cat is called Pip."),
    ("Mira's cat", "likes", "fish", "Mira's cat likes fish."),
    ("my aunt", "home_town", "Tartu", "My aunt lives in Tartu now."),
    ("the van", "van_tax", "paid", "The van's tax is paid."),
    ("me", "hobby", "bees", "I keep bees."),
    ("Mira's cat", "pal", "Rafa's dog", "Mira's cat plays with Rafa's dog."),
    ("Lee's hen", "name", "Bo", "Lee's hen is called Bo."),
    ("Ada's hen", "name", "Bo", "Ada's hen is called Bo."),
    ("my kid", "pet", "Lee's hen", "My kid feeds Lee's hen."),
    ("the cow", "name", "Max", "The cow is called Max."),
    ("Ada's dog", "name", "Max", "Ada's dog is called Max."),
    ("my kids", "likes", "kites", "My kids like kites."),
    ("Uma's cat", "name", "Zo\u00eb", "Uma's cat is called Zo\u00eb."),
    ("Ida's ox", "name", "Ea-Li", "Ida's ox is called Ea-Li."),
    ("Ida's yak", "name", "Al Ko", "Ida's yak is called Al Ko."),
]
# (question, past, cap, exclude)
CASES = [
    ("What colour are my cousin's eyes?", None, None, ()),
    ("What does my son like?", None, None, ()),
    ("Where is my son's home?", None, None, ()),
    ("Where is my son's home?", None, None, (5,)),
    ("Where does the owner of the van live?", None, None, ()),
    ("What colour are my eyes?", None, None, ()),
    ("Tell me the colour of my eyes.", None, None, ()),
    ("Tell me about Rafa.", None, None, ()),
    ("What is my town?", None, None, ()),
    ("Is the van mine?", None, None, ()),
    ("Do I like figs?", None, None, ()),
    ("What colour are Ivo's eyes?", None, None, ()),
    ("What does Pip like?", None, None, ()),
    ("Where did my aunt live before?", None, None, ()),
    ("Where does my aunt live?", True, None, ()),
    ("Where does my aunt live?", None, None, (15,)),
    ("Who owns the van?", None, 400, ()),
    ("What is the weather like?", None, None, ()),
    ("Is my van taxed?", None, None, ()),
    ("Where are my bees?", None, None, ()),
    ("What is my town's name?", None, None, ()),
    ("Does Pip play with Rafa's dog, and what does my kid like?", None, 404, ()),
    ("What does Bo eat?", None, None, ()),
    ("What does Max eat?", None, None, ()),
    ("What do my kids like?", None, None, ()),
    ("What does Zo\u00eb like?", None, None, ()),
    ("Is Ea-Li big?", None, None, ()),
    ("Is Al Ko big?", None, None, ()),
    ("Is Al  Ko big?", None, None, ()),
    ("What colour are my\u00a0eyes?", None, None, ()),
    ("\u00e9my eyes", None, None, ()),
]
# /v1/notes adds on the made-up store: the server files them under the user (or `subject`) as "note:SOURCE"
WRITES = [
    {"text": "Rex sleeps in the barn."},
    {"text": "My dog's name is Rex."},
    {"text": "Lumi likes the red ball."},
    {"text": "Rafa's dog sleeps in a box.", "subject": "Rafa's dog"},
]
LIVE = [
    ("Where does Rex sleep?", None, None, ()),
    ("What does Lumi like?", None, None, ()),
    ("Where does Ivo sleep?", None, None, ()),
    ("What colour are my cousin's eyes?", None, None, ()),
    ("Where does Rex sleep?", None, None, (28,)),
]


def write_store(ns):
    for k, (e, a, v, t) in enumerate(STORE):
        r = ns.write(OWNER, {"id": f"m{k + 1}", "op": "ASSERT", "e": e, "a": a, "v": v, "text": t,
                             "told": 739901000100 + 100 * k})
        if r["note"] != k + 1:
            raise SystemExit(f"lesson {k + 1} was not written as note {k + 1}: {r['verdict']}")
    return ns


def schema_of():
    return {a: {"multi": a == "likes", "applies": ["self", "person", "animal", "vehicle", "object", "thing"],
                "domain": [], "extra": [], "topic": []} for a in {x[1] for x in STORE}}


def answer(nm, ns, owner, q, past, cap, ex):
    w = who_ref.lookup(nm, ns, owner, q, cap=cap, past=past, exclude=ex)
    sha = hashlib.sha256(w["block"].encode()).hexdigest() if w["block"] else None
    return {"question": q, "past": past, "cap": cap, "exclude": list(ex), "past_read": w["past"], "ids": w["ids"],
            "tokens": w["tokens"], "candidates": w["candidates"], "block_sha256": sha}


def live(nm, nx, tok, snap):
    """WRITES through the server's intake on the made-up store, then each LIVE case's who-first answer."""
    import intake  # noqa: E402  (LLWZ's, beside notes.py)
    import lwlogic as L  # noqa: E402
    ns = write_store(live_intake.make(nm, intake)(tok, schema=schema_of()))
    again, _v = nx.snapshot(ns, OWNER, meta={"store": "who made-up", "tokenizer": "bytes"})
    if again != snap:
        raise SystemExit("the made-up store written with the server's intake exports otherwise")
    base_told = max(n["told"] for n in snap["notes"])
    today = L.tick(fixture.TODAY)
    writes = []
    for k, w in enumerate(WRITES):
        told = ns.told_next(OWNER, today, base_told)
        e, _same = ns.http_write(OWNER, "add", w["text"], told, f"w{k + 1}", w.get("subject", "me"))
        writes.append(dict(w, id=e["note"], told=told, verdict=e["verdict"]))
    cases = []
    for q, past, cap, ex in LIVE:
        c = answer(nm, ns, OWNER, q, past, cap, ex)
        c["retrieve_ids"] = ns.retrieve(OWNER, q, cap=cap, past=past, exclude=ex)["ids"]
        cases.append(c)
    return {"today": fixture.TODAY, "writes": writes, "cases": cases}


def main():
    raw = argparse.RawDescriptionHelpFormatter
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=raw)
    ap.add_argument("llwz", help="an LLWZ checkout (zengine/ and examples/)")
    ap.add_argument("--out", default=fixture.OUT)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.llwz, "zengine"))
    import notes as nm  # noqa: E402  (LLWZ's)
    import notes_export as nx  # noqa: E402

    tok = fixture.bytes_tok(nm)
    ns = write_store(nm.NoteStore(tok, schema=schema_of()))
    snap, _v = nx.snapshot(ns, OWNER, meta={"store": "who made-up", "tokenizer": "bytes"})
    cases = []
    for q, past, cap, ex in CASES:
        c = answer(nm, ns, OWNER, q, past, cap, ex)
        c["retrieve_ids"] = ns.retrieve(OWNER, q, cap=cap, past=past, exclude=ex)["ids"]
        cases.append(c)
    written = live(nm, nx, tok, snap)
    S = nm.load_set(os.path.join(args.llwz, "examples", "L6-seed1.json"))
    teach = next(p for p in S["phases"] if p["name"] == "teach")
    l6, _w = nm.set_store(S, teach["store"], tok)
    l6_snap, _v = nx.snapshot(l6, "user", meta={"set": S["id"], "phase": "teach", "tokenizer": "bytes"})
    with open(os.path.join(args.out, "l6_bytes_snapshot.json")) as f:
        if f.read() != json.dumps(l6_snap, separators=(",", ":")) + "\n":
            raise SystemExit("the L6 teach store does not export as l6_bytes_snapshot.json: another LLWZ?")
    with open(os.path.join(args.out, "l6_bytes_cases.json")) as f:
        l6_cases = json.load(f)
    l6_who = [answer(nm, l6, "user", c["question"], c["past"], c["cap"], c["exclude"]) for c in l6_cases]
    fixture.dump(os.path.join(args.out, "who_bytes_snapshot.json"), snap)
    fixture.dump(os.path.join(args.out, "who_bytes_cases.json"), cases)
    fixture.dump(os.path.join(args.out, "who_bytes_live.json"), written)
    fixture.dump(os.path.join(args.out, "l6_bytes_who.json"), l6_who)
    empty = sum(1 for c in l6_who if not c["ids"])
    print(f"{args.out}: {len(snap['notes'])} made-up notes, {len(cases)} cases, {len(written['cases'])} after "
          f"{len(WRITES)} writes; L6 {len(l6_who)} cases, {empty} empty")


if __name__ == "__main__":
    main()
