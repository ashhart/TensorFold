"""Goldens for `zig build test-grammar`: a reference grammar implementation over xgrammar 0.2.8, walked step by step."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def reference_modules(repo: str, commit: str):
    """The reference grammar.py / structured.py at `commit`, importable (``.server``: tool fixes off, its default)."""

    root = Path(tempfile.mkdtemp(prefix="grammar-golden-"))
    spark = root / "tensorfold" / "families" / "glm5_next" / "spark"
    ds = root / "tensorfold" / "families" / "deepseek_v41" / "cuda"
    for d in (spark, ds):
        d.mkdir(parents=True)
    for d in (root / "tensorfold", root / "tensorfold" / "families", spark.parent, ds.parent):
        (d / "__init__.py").write_text("")
    for d in (spark, ds):
        (d / "__init__.py").write_text("")

    def show(path: str) -> str:
        return subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], check=True, capture_output=True,
                              text=True).stdout

    (spark / "grammar.py").write_text(show("src/tensorfold/families/glm5_next/spark/grammar.py"))
    (spark / "server.py").write_text("def tool_fixes(raw=None):\n    return frozenset()\n")
    (ds / "structured.py").write_text(show("src/tensorfold/families/deepseek_v41/cuda/structured.py"))
    sys.path.insert(0, str(root))
    from tensorfold.families.deepseek_v41.cuda import structured
    from tensorfold.families.glm5_next.spark import grammar as gm
    return gm, structured


def digest(words: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(words, dtype="<i4").tobytes()).hexdigest()[:16]


def vocab_digest(enc: list[str]) -> str:
    h = hashlib.sha256()
    for s in enc:
        b = s.encode()
        h.update(len(b).to_bytes(4, "little"))
        h.update(b)
    return h.hexdigest()[:16]


def allowed(bits_row: np.ndarray, vocab: int) -> np.ndarray:
    u = bits_row.view(np.uint32)
    ids = np.nonzero(((u[:, None] >> np.arange(32, dtype=np.uint32)) & 1).reshape(-1)[:vocab])[0]
    return ids


def synthetic_tokenizer(path: Path) -> None:
    """A small byte-level BPE tokenizer.json: 256 byte symbols, some words, eos/think/tool-markup added tokens."""

    from tokenizers.pre_tokenizers import ByteLevel

    alphabet = ByteLevel.alphabet()
    byte_map = sorted(alphabet)
    words = ["true", "false", "null", "name", "city", "Paris", "yes", "no", "\":", "{\"", "\"}", ", ", "  ", "12",
             "Ġthe", "Ġ{", "invoke", "Ġcalls", "parameter", "ĊĊ", "Ċ", "=\"", "\">", "</", "get", "_weather"]
    vocab = {s: i for i, s in enumerate(byte_map)}
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
    added = []
    for content in ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<think>", "</think>", "｜DSML｜", "<｜User｜>"]:
        added.append({"id": len(vocab) + len(added), "content": content, "single_word": False, "lstrip": False,
                      "rstrip": False, "normalized": False, "special": True})
    doc = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
           "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
           "post_processor": None,
           "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
           "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                     "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                     "vocab": vocab, "merges": []}}
    path.write_text(json.dumps(doc, ensure_ascii=False))


WEATHER = {"type": "object", "properties": {"city": {"type": "string", "maxLength": 12},
                                            "unit": {"enum": ["c", "f"]},
                                            "days": {"type": "integer", "minimum": 1, "maximum": 7}},
           "required": ["city", "unit"], "additionalProperties": False}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "w", "parameters": WEATHER}},
         {"type": "function", "function": {"name": "note", "parameters": {"type": "object", "properties": {
             "text": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2}},
             "required": ["text"]}}},
         {"type": "function", "function": {"name": "loose", "strict": False, "parameters": {"type": "object"}}}]


def cases(gm):
    out = []

    def body(**kw):
        return kw

    out.append(("json_object", body(response_format={"type": "json_object"}), False))
    out.append(("json_schema", body(response_format={"type": "json_schema", "json_schema": {"name": "w", "schema": WEATHER}}), False))
    out.append(("array", body(guided_json={"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3}), False))
    out.append(("regex", body(guided_regex=r"\d{3}-[a-z]{2,5}( ok)?"), False))
    out.append(("choice", body(guided_choice=["yes", "no", "maybe \"so\"", "été"]), False))
    out.append(("ebnf", body(guided_grammar='root ::= expr\nexpr ::= term ("+" term)*\nterm ::= [0-9]+ | "(" expr ")"'), False))
    out.append(("tools_required", body(tools=TOOLS, tool_choice="required"), False))
    out.append(("tools_required_single", body(tools=TOOLS, tool_choice="required", parallel_tool_calls=False), False))
    out.append(("tools_named", body(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "note"}}), False))
    strict = [dict(TOOLS[0], function=dict(TOOLS[0]["function"], strict=True)), TOOLS[1]]
    out.append(("tools_strict_auto", body(tools=strict), False))
    out.append(("json_schema_thinking", body(response_format={"type": "json_schema", "json_schema": {"schema": WEATHER}}), True))
    out.append(("tools_required_thinking", body(tools=TOOLS, tool_choice="required"), True))
    return out


BAD = [("bad_schema", {"guided_json": {"type": "object", "properties": {"a": {"type": "nosuch"}}}}),
       ("bad_regex", {"guided_regex": "(ab"}),
       ("bad_ebnf", {"guided_grammar": "root ::= undefined_rule"}),
       ("named_missing", {"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "nope"}}})]


def walk(gm, grammars, spec, active, rng: random.Random, steps: int, vocab: int, eos: int, think_end: int):
    """Drafted reply rounds under Constraint: scratch-matcher draft chains (some wrong), cut, fill, then advance."""

    bound = gm.Bound(spec, grammars.compile(spec), think_end, active)
    con = grammars.constraint(bound)
    hist: list[int] = []
    recs = []
    pending = rng.randrange(vocab)
    for step in range(steps):
        if con.finished:
            break
        scratch = clone(grammars, bound, hist)
        n = rng.choice([1, 2, 3, 4, 6, 9])
        drafts = []
        for i in range(n - 1):
            if rng.random() < 0.12:
                drafts.append(rng.randrange(vocab))          # a wrong draft (cut unless the grammar takes it)
                break
            t = pick(rng, scratch, con.words, vocab, eos, think_end, step + i)
            drafts.append(t)
            try:
                scratch.advance([t])
            except gm.GrammarError:
                break
        tokens = [pending] + drafts
        win = con.cut(tokens)
        keep = len(win.tokens)
        stop = stop_at(con, tokens, keep, think_end)
        con.fill(win)
        rows = []
        for j in range(len(win.rows)):
            rows.append([digest(win.bits[j]), int(len(allowed(win.bits[j], vocab)))])
        acc = rng.randrange(keep)
        if win.rows and acc >= win.rows[0]:
            ok = allowed(win.bits[acc - win.rows[0]], vocab)
            bonus = eos if eos in ok and rng.random() < 0.15 else int(ok[rng.randrange(min(len(ok), 400))])
        else:
            bonus = think_end if step >= 2 and rng.random() < 0.4 else rng.randrange(vocab)
        adv = [int(t) for t in tokens[1:acc + 1]] + [int(bonus)]
        con.advance(adv)
        hist.extend(adv)
        recs.append({"window": [int(t) for t in tokens], "keep": keep, "first": win.rows[0] if win.rows else keep,
                     "stop": stop, "rows": rows, "advance": adv})
        pending = bonus
    return recs, con


def clone(grammars, bound, hist):
    """A constraint at the committed path `hist` (replayed)."""

    c = grammars.constraint(bound)
    c.advance(hist)
    return c


def stop_at(con, tokens, keep, think_end) -> bool:
    """Whether the cut ended at a draft that completes the grammar (it and every later row are cut)."""

    if keep >= len(tokens):
        return False
    m = con.m
    active = con.active
    acc = 0
    try:
        for t in tokens[1:keep]:
            if active:
                m.accept_token(t)
                acc += 1
            else:
                active = t == think_end
        t = tokens[keep]
        if not active or t in con.never:
            return False
        if not m.accept_token(t):
            return False
        acc += 1
        return bool(m.is_terminated())
    finally:
        if acc:
            m.rollback(acc)


def pick(rng, scratch, words, vocab, eos, think_end, unthink) -> int:
    """A token the scratch matcher takes next (sometimes the stop token); before </think>, any token or </think>."""

    if not scratch.active:
        return think_end if unthink >= 3 and rng.random() < 0.5 else rng.randrange(vocab)
    if scratch.m.is_terminated():
        return eos
    bits = np.full((1, words), -1, dtype=np.int32)
    scratch.m.fill_next_token_bitmask(bits, 0)
    ok = allowed(bits[0], vocab)
    if eos in ok and rng.random() < 0.15:
        return eos
    if len(ok) == 0:
        return eos
    # prefer short ASCII tokens so walks reach the grammar's end
    return int(ok[rng.randrange(min(len(ok), 400))]) if rng.random() < 0.7 else int(ok[rng.randrange(len(ok))])


def main() -> None:
    ap = argparse.ArgumentParser(description=(
        "Write test-grammar goldens: the tokenizer views (vocab type, prefix space, vocabulary and special-id digests, "
        "think ids); per case the spec, the tools spec's structural tag JSON, the think state and each step's window, "
        "cut (keep, first constrained row, stop), per-row bitmask digest (first 16 hex of sha256 over the int32 words) "
        "and allowed count, and advanced tokens; and the refusal message of each bad spec. Needs xgrammar==0.2.8, "
        "tokenizers and numpy."))
    ap.add_argument("--reference-repo", required=True,
                    help="checkout of a Python serving implementation (structured.py and grammar.py)")
    ap.add_argument("--commit", required=True, help="reference commit to read structured.py and grammar.py from")
    ap.add_argument("--tokenizer", help="dir with tokenizer.json")
    ap.add_argument("--synthetic", action="store_true",
                    help="use a small synthetic tokenizer (written as synthetic/tokenizer.json beside --out)")
    ap.add_argument("--vocab-size", type=int, default=129280, help="logits columns (ignored with --synthetic)")
    ap.add_argument("--eos", type=int, default=1, help="stop token id (ignored with --synthetic)")
    ap.add_argument("--steps", type=int, default=24, help="most rounds walked per case")
    ap.add_argument("--seed", type=int, default=7, help="walk RNG seed")
    ap.add_argument("--out", required=True, help="golden JSON to write")
    a = ap.parse_args()
    gm, structured = reference_modules(a.reference_repo, a.commit)
    import xgrammar as xgr

    out = Path(a.out)
    if a.synthetic:
        tok_dir = out.parent / "synthetic"
        tok_dir.mkdir(parents=True, exist_ok=True)
        synthetic_tokenizer(tok_dir / "tokenizer.json")
        vocab = 288
        eos = 1 + json.loads((tok_dir / "tokenizer.json").read_text())["added_tokens"][0]["id"]
    else:
        tok_dir = Path(a.tokenizer)
        vocab, eos = a.vocab_size, a.eos
    grammars = structured.Dsv41Grammars.from_model(tok_dir, vocab, (eos,))
    views = {}
    path = tok_dir / "tokenizer.json"
    for name, keep in (("text", ()), ("tools", structured.DSML_TOKENS)):
        enc, backend, t_open, t_end = gm._vocab(path, vocab, keep)
        meta = xgr.TokenizerInfo._detect_metadata_from_hf(backend)
        views[name] = {"vocab_type": int(meta["vocab_type"].value), "add_prefix_space": bool(meta["add_prefix_space"]),
                       "vocab": vocab_digest(enc), "never_digest": hashlib.sha256(json.dumps(sorted(int(t) for t in grammars.special[name])).encode()).hexdigest()[:16],
                       "never_count": len(grammars.special[name])}
    rng = random.Random(a.seed)
    recs = []
    for name, body, thinking in cases(gm):
        spec = structured.Host.spec(structured.Host.__new__(structured.Host), body)
        tag = None
        if spec.kind == "tools":
            d = json.loads(spec.text)
            tag = xgr.get_model_structural_tag(structured.MODEL_TAG, tools=d["tools"], tool_choice=d["tool_choice"],
                                               reasoning="disabled", max_whitespace_cnt=gm.BLANKS,
                                               parallel_tool_calls=bool(d["parallel_tool_calls"])).model_dump_json(indent=None)
        active = not thinking
        steps, con = walk(gm, grammars, spec, active, rng, a.steps, vocab, eos, grammars.think_end)
        recs.append({"name": name, "kind": spec.kind, "text": spec.text, "tag": tag, "active": active,
                     "steps": steps, "finished": bool(con.finished)})
        print(f"{name}: {len(steps)} steps, {sum(len(s.get('rows', [])) for s in steps)} rows, finished "
              f"{con.finished}", file=sys.stderr)
    bad = []
    for name, body in BAD:
        try:
            spec = structured.Host.spec(structured.Host.__new__(structured.Host), body)
            grammars.compile(spec)
            msg = None
        except ValueError as exc:
            msg = str(exc)
        bad.append({"name": name, "kind": spec.kind, "text": spec.text, "field": spec.field, "message": msg})
    doc = {"xgrammar": "0.2.8", "reference": a.commit, "vocab_size": vocab, "eos": eos, "think_open": grammars.think_open,
           "think_end": grammars.think_end, "words": (vocab + 31) // 32, "views": views, "cases": recs, "bad": bad,
           "tokenizer": "synthetic" if a.synthetic else "model"}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=None, separators=(",", ":")) + "\n")
    print(f"wrote {out} ({out.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
