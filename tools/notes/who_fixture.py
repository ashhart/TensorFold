"""The who-first lookup's test fixtures: a made-up store under the byte tokenizer, and Python's answers (who_ref.py).

  python3 tools/notes/who_fixture.py LLWZ_DIR [--out zig/src/server/notes/testdata]

who_bytes_snapshot.json  a small made-up store (a son, an aunt, a van, a kid, three pets and the user), exported by
                         LLWZ's notes_export.py with the byte tokenizer of fixture.py
who_bytes_cases.json     its cases: per question the who-first answer and retrieve()'s
l6_bytes_who.json        the who-first answer for each case of l6_bytes_cases.json, on l6_bytes_snapshot.json's store
                         (which must come out byte for byte as the file in the tree)
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fixture  # noqa: E402
import who_ref  # noqa: E402

OWNER = "u"
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
]


def made_up_store(nm, tok):
    schema = {a: {"multi": a == "likes", "applies": ["self", "person", "animal", "vehicle", "object", "thing"],
                  "domain": [], "extra": [], "topic": []} for a in {x[1] for x in STORE}}
    ns = nm.NoteStore(tok, schema=schema)
    for k, (e, a, v, t) in enumerate(STORE):
        r = ns.write(OWNER, {"id": f"m{k + 1}", "op": "ASSERT", "e": e, "a": a, "v": v, "text": t,
                             "told": 739901000100 + 100 * k})
        if r["note"] != k + 1:
            raise SystemExit(f"lesson {k + 1} was not written as note {k + 1}: {r['verdict']}")
    return ns


def answer(nm, ns, owner, q, past, cap, ex):
    w = who_ref.lookup(nm, ns, owner, q, cap=cap, past=past, exclude=ex)
    sha = hashlib.sha256(w["block"].encode()).hexdigest() if w["block"] else None
    return {"question": q, "past": past, "cap": cap, "exclude": list(ex), "past_read": w["past"], "ids": w["ids"],
            "tokens": w["tokens"], "candidates": w["candidates"], "block_sha256": sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("llwz")
    ap.add_argument("--out", default=fixture.OUT)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.llwz, "zengine"))
    import notes as nm  # noqa: E402  (LLWZ's)
    import notes_export as nx  # noqa: E402

    tok = fixture.bytes_tok(nm)
    ns = made_up_store(nm, tok)
    snap, _v = nx.snapshot(ns, OWNER, meta={"store": "who made-up", "tokenizer": "bytes"})
    cases = []
    for q, past, cap, ex in CASES:
        c = answer(nm, ns, OWNER, q, past, cap, ex)
        r = ns.retrieve(OWNER, q, cap=cap, past=past, exclude=ex)
        c["retrieve_ids"] = r["ids"]
        cases.append(c)
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
    fixture.dump(os.path.join(args.out, "l6_bytes_who.json"), l6_who)
    empty = sum(1 for c in l6_who if not c["ids"])
    print(f"{args.out}: {len(snap['notes'])} made-up notes, {len(cases)} cases; L6 {len(l6_who)} cases, {empty} empty")


if __name__ == "__main__":
    main()
