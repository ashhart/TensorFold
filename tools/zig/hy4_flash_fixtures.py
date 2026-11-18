#!/usr/bin/env python3
"""Rebuild header.txt (`CHECKPOINT`) or mini_tokenizer.json with tokenizer_cases.json (`--tokenizer CHECKPOINT OUT`)."""

import glob
import json
import re
import struct
import sys

KEEP_LAYERS = {"0", "1", "2"}


def headers(root):
    for path in sorted(glob.glob(f"{root}/*.safetensors")):
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            for name, meta in json.loads(f.read(n)).items():
                if name != "__metadata__":
                    yield name, meta["dtype"], meta["shape"]


def keep(name, last_expert):
    m = re.match(r"model\.(layers|dfly_layers)\.(\d+)\.", name)
    if m and (m.group(2) not in KEEP_LAYERS or (m.group(1) == "dfly_layers" and m.group(2) != "0")):
        return False
    m = re.match(r"vit\.layers\.(\d+)\.", name)
    if m and m.group(1) != "0":
        return False
    m = re.search(r"\.experts\.(\d+)\.", name)
    return not m or m.group(1) in ("0", last_expert)


CASES = [
    "Hello, world!",
    "def fib(n):\n    a, b = 0, 1\n    return a",
    "The year 2026 has 365 days; 17 * 23 = 391.",
    "\u4f60\u597d\uff0c\u4e16\u754c",
    "  spaces\tand\ttabs  ",
    "caf\u00e9 na\u00efve \u00fcber",
]


def pieces(tok, text):
    return [p for p, _ in tok.pre_tokenizer.pre_tokenize_str(text)] if tok.pre_tokenizer else [text]


def mini_tokenizer(root, out):
    """Keeps every merge whose joined form occurs in a case, so each case encodes exactly as the full tokenizer."""
    from tokenizers import Tokenizer

    full = json.load(open(f"{root}/tokenizer.json", encoding="utf-8"))
    tok = Tokenizer.from_str(json.dumps(full))
    prompts = json.load(open(f"{out}/prompt_cases.json", encoding="utf-8"))
    texts = CASES + [p["text"] for p in prompts]
    added = {a["content"]: a for a in full["added_tokens"]}
    used_added = sorted({c for t in texts for c in added if c in t}, key=lambda c: added[c]["id"])
    plain = list(texts)
    for c in used_added:
        plain = [x for t in plain for x in t.split(c)]
    words = {p for t in plain for p in pieces(tok, t)}
    vocab = full["model"]["vocab"]
    merges = full["model"]["merges"]
    keep_merges = []
    for m in merges:
        a, b = m if isinstance(m, list) else m.split(" ", 1)
        if any(a + b in w for w in words):
            keep_merges.append(m)
    keep = {k for k in vocab if len(k) == 1}
    for m in keep_merges:
        a, b = m if isinstance(m, list) else m.split(" ", 1)
        keep |= {a, b, a + b}
    mini = dict(full)
    small_vocab = {k: v for k, v in vocab.items() if k in keep}
    small_vocab.update({c: added[c]["id"] for c in used_added})  # the library renumbers added tokens missing from the vocab
    mini["model"] = dict(full["model"], vocab=small_vocab, merges=keep_merges)
    mini["added_tokens"] = [added[c] for c in used_added]
    small = Tokenizer.from_str(json.dumps(mini))
    cases = []
    for t in texts[: len(CASES)]:
        ids = tok.encode(t, add_special_tokens=False).ids
        assert small.encode(t, add_special_tokens=False).ids == ids, t
        cases.append({"text": t, "ids": ids})
    for p in prompts:
        assert small.encode(p["text"], add_special_tokens=False).ids == p["ids"], p["name"]
    with open(f"{out}/mini_tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(mini, f, ensure_ascii=False, separators=(",", ":"))
    with open(f"{out}/tokenizer_cases.json", "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)


def main():
    if sys.argv[1] == "--tokenizer":
        return mini_tokenizer(sys.argv[2], sys.argv[3])
    rows = list(headers(sys.argv[1]))
    last = str(max(int(m.group(1)) for n, _, _ in rows if (m := re.search(r"\.experts\.(\d+)\.", n))))
    for name, dtype, shape in sorted(rows):
        if keep(name, last):
            print(name, dtype, "x".join(map(str, shape)) or "scalar")


if __name__ == "__main__":
    main()
