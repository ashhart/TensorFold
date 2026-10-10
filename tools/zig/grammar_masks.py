"""libtfgrammar's masks against xgrammar's own matcher (the Python package, same release) on a fixed grammar set.

The structured-output gate's first part (#564): for each grammar, walk tokens through both matchers and compare
every step's allowed-token bitmask, accept result and termination. libtfgrammar is driven through its C ABI with
ctypes, as the server loads it; xgrammar through ``TokenizerInfo.from_huggingface``, as the Python engine built it.

    python tools/zig/grammar_masks.py --model DIR --lib zig-out/native/lib/libtfgrammar.dylib

Needs xgrammar==0.2.8 (zig/build/grammar.zig's pinned tag), transformers and torch.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import random
import sys
from pathlib import Path

import numpy as np

BLANKS = 32
OBJECT = '{"type": "object"}'
KINDS = ("json", "json_schema", "regex", "choice", "grammar")

# (name, kind, text): every kind the server parses, with nesting, alternatives, escapes and counted repeats.
CASES = [
    ("json object", "json", ""),
    ("schema: flat", "json_schema", json.dumps({"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]})),
    ("schema: nested", "json_schema", json.dumps({
        "type": "object",
        "properties": {
            "city": {"type": "string", "maxLength": 12},
            "days": {"type": "array", "items": {"type": "object", "properties": {"t": {"type": "number"}, "rain": {"type": "boolean"}}, "required": ["t", "rain"]}, "minItems": 1, "maxItems": 3},
            "unit": {"enum": ["C", "F"]},
        },
        "required": ["city", "days", "unit"],
    })),
    ("schema: anyOf", "json_schema", json.dumps({"anyOf": [{"type": "integer"}, {"type": "object", "properties": {"ok": {"const": True}}, "required": ["ok"]}]})),
    ("regex: date", "regex", r"\d{4}-\d{2}-\d{2}"),
    ("regex: email", "regex", r"[a-z0-9._]{1,16}@[a-z]{2,10}\.(com|org|net)"),
    ("regex: unicode", "regex", r"(héllo|日本語|🙂){1,3}!"),
    ("choice", "choice", json.dumps(["positive", "negative", "neutral"])),
    ("choice: quotes", "choice", json.dumps(['say "yes"', "back\\slash", "tab\tend"])),
    ("ebnf: arithmetic", "grammar", 'root ::= expr\nexpr ::= term (("+" | "-") term)*\nterm ::= [0-9]+ | "(" expr ")"'),
    ("ebnf: sql-ish", "grammar", 'root ::= "SELECT " cols " FROM " name (" WHERE " name "=" [0-9]+)?\ncols ::= name (", " name)*\nname ::= [a-z_]{1,8}'),
]


class Shim:
    """libtfgrammar through its C ABI (zig/src/grammar/tf_grammar.cc)."""

    def __init__(self, path: str) -> None:
        lib = ctypes.CDLL(path)
        p, i, s = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        sig = {
            "tfg_open": (p, [ctypes.c_char_p, s, i, ctypes.POINTER(ctypes.c_int32), i, ctypes.c_char_p, s]),
            "tfg_words": (i, [p]),
            "tfg_compile": (p, [p, i, ctypes.c_char_p, s, ctypes.c_char_p, s]),
            "tfg_free": (None, [p]),
            "tfg_matcher": (p, [p]),
            "tfg_matcher_free": (None, [p]),
            "tfg_accept": (i, [p, ctypes.c_int32]),
            "tfg_rollback": (i, [p, i]),
            "tfg_terminated": (i, [p]),
            "tfg_fill": (i, [p, ctypes.POINTER(ctypes.c_int32), i]),
        }
        for name, (res, args) in sig.items():
            f = getattr(lib, name)
            f.restype, f.argtypes = res, args
        self.lib = lib

    def open(self, tokenizer: bytes, vocab: int, stops: list[int]):
        err = ctypes.create_string_buffer(512)
        arr = (ctypes.c_int32 * len(stops))(*stops)
        h = self.lib.tfg_open(tokenizer, len(tokenizer), vocab, arr, len(stops), err, 512)
        if not h:
            raise RuntimeError(err.value.decode())
        return h

    def compile(self, h, kind: str, text: str):
        err = ctypes.create_string_buffer(512)
        raw = text.encode()
        g = self.lib.tfg_compile(h, KINDS.index(kind), raw, len(raw), err, 512)
        if not g:
            raise RuntimeError(err.value.decode())
        return g


def reference(compiler, kind: str, text: str):
    """The compiled grammar as tensorfold/engine/grammar.py compiled it."""

    if kind in ("json", "json_schema"):
        return compiler.compile_json_schema(OBJECT if kind == "json" else text, max_whitespace_cnt=BLANKS)
    if kind == "regex":
        return compiler.compile_regex(text)
    if kind == "choice":
        return compiler.compile_grammar("root ::= " + " | ".join(json.dumps(v, ensure_ascii=False) for v in json.loads(text)))
    return compiler.compile_grammar(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="checkpoint folder (tokenizer.json, config.json)")
    ap.add_argument("--lib", required=True, help="libtfgrammar (zig build grammar)")
    ap.add_argument("--walks", type=int, default=3, help="walks a grammar, each from its own seed")
    ap.add_argument("--steps", type=int, default=48, help="most steps a walk")
    args = ap.parse_args()

    import torch  # noqa: F401  (xgrammar's bitmask helpers)
    import xgrammar as xgr
    from transformers import PreTrainedTokenizerFast

    model = Path(args.model)
    config = json.loads((model / "config.json").read_text())
    text_config = config.get("text_config") or {}
    vocab = int(text_config.get("vocab_size") or config["vocab_size"])
    eos = text_config.get("eos_token_id", config.get("eos_token_id"))
    stops = [int(eos)] if isinstance(eos, int) else [int(t) for t in eos]
    tok = PreTrainedTokenizerFast(tokenizer_file=str(model / "tokenizer.json"))
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab, stop_token_ids=stops)
    compiler = xgr.GrammarCompiler(info, max_threads=8, cache_limit_bytes=256 << 20)
    shim = Shim(args.lib)
    h = shim.open((model / "tokenizer.json").read_bytes(), vocab, stops)
    words = shim.lib.tfg_words(h)
    want_words = xgr.allocate_token_bitmask(1, vocab).shape[1]
    if words != want_words:
        print(f"FAIL: libtfgrammar fills {words} words a row, xgrammar {want_words}")
        return 1

    steps = mismatches = 0
    for name, kind, text in CASES:
        g = shim.compile(h, kind, text)
        compiled = reference(compiler, kind, text)
        case_steps = case_bad = 0
        for walk in range(args.walks):
            rng = random.Random(f"{name}/{walk}")
            m = shim.lib.tfg_matcher(g)
            ref = xgr.GrammarMatcher(compiled)
            ours = np.zeros(words, dtype=np.int32)
            theirs = xgr.allocate_token_bitmask(1, vocab)
            taken = 0
            for _ in range(args.steps):
                if shim.lib.tfg_fill(m, ours.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), words) != 0:
                    raise RuntimeError(f"{name}: tfg_fill failed")
                ref.fill_next_token_bitmask(theirs, 0)
                want = theirs[0].numpy()
                case_steps += 1
                if not np.array_equal(ours, want):
                    case_bad += 1
                    diff = np.nonzero(np.unpackbits(np.bitwise_xor(ours, want).view(np.uint8), bitorder="little"))[0]
                    print(f"  {name} walk {walk} step {taken}: {len(diff)} tokens differ, first {diff[:5].tolist()}")
                allowed = np.nonzero(np.unpackbits(want.view(np.uint8), bitorder="little")[:vocab])[0]
                if len(allowed) == 0:
                    break
                # mostly an allowed token; now and then a disallowed one, which both must reject and stay put
                if rng.random() < 0.15:
                    t = rng.randrange(vocab)
                else:
                    stop_ok = [s for s in stops if s in set(allowed.tolist())]
                    t = stop_ok[0] if stop_ok and taken > args.steps * 2 // 3 else int(allowed[rng.randrange(len(allowed))])
                a, b = shim.lib.tfg_accept(m, t), int(ref.accept_token(t))
                if a != b:
                    case_bad += 1
                    print(f"  {name} walk {walk} step {taken}: token {t} accepted {a} by libtfgrammar, {b} by xgrammar")
                    break
                taken += a
                # a rollback now and then, as a window's walk does
                if a and taken > 1 and rng.random() < 0.2:
                    if shim.lib.tfg_rollback(m, 1) != 0:
                        raise RuntimeError(f"{name}: tfg_rollback failed")
                    ref.rollback(1)
                    taken -= 1
                if shim.lib.tfg_terminated(m) != int(ref.is_terminated()):
                    case_bad += 1
                    print(f"  {name} walk {walk} step {taken}: terminated differs")
                    break
                if ref.is_terminated():
                    break
            shim.lib.tfg_matcher_free(m)
        shim.lib.tfg_free(g)
        steps += case_steps
        mismatches += case_bad
        print(f"{'ok  ' if case_bad == 0 else 'FAIL'} {name}: {case_steps} steps")
    # a grammar xgrammar cannot compile is refused by both, with a message
    for kind, text in (("regex", "(unclosed"), ("grammar", "root ::= missing_rule"), ("json_schema", "{not json")):
        try:
            shim.compile(h, kind, text)
            print(f"FAIL {kind} {text!r}: libtfgrammar compiled it")
            mismatches += 1
        except RuntimeError as exc:
            print(f"ok   refused {kind} {text!r}: {str(exc).splitlines()[0][:80]}")
    print(f"{'PASS' if mismatches == 0 else 'FAIL'}: {steps} steps over {len(CASES)} grammars, {mismatches} mismatches")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
