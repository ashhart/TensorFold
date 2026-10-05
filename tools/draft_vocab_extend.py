"""Extend a draft vocabulary with every id whose text holds characters of a script.

`tools/draft_vocab.py` ranks ids by frequency over a corpus, which needs a corpus in the language
you care about. This tool needs none: it walks the tokenizer's ids, decodes each one, and adds the
ids whose text contains a character of the requested script to an existing list. That is what fixes
a workload whose language the shipped list does not cover -- a token outside the draft list can
never be drafted, so a list built for English and code leaves e.g. Chinese drafting nothing at all.

  python tools/draft_vocab_extend.py TOKENIZER_JSON BASE.txt OUT.txt --script cjk

Both the base list and the result are plain text, one id per line, sorted.
"""

from __future__ import annotations

import argparse
import unicodedata

from tokenizers import Tokenizer

SCRIPTS: dict[str, tuple[tuple[int, int], ...]] = {
    "cjk": ((0x3000, 0x303F), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
            (0xFF00, 0xFFEF), (0x20000, 0x2FA1F)),
    "hangul": ((0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7A3)),
    "kana": ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F)),
    "cyrillic": ((0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "thai": ((0x0E00, 0x0E7F),),
}


def in_scripts(text: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    """Whether ``text`` holds a character of any of the given code point ranges."""

    if not text:
        return False
    if "CJK" in unicodedata.name(text[0], "") and any(a <= 0x4E00 <= b for a, b in ranges):
        return True
    for char in text:
        point = ord(char)
        if any(a <= point <= b for a, b in ranges):
            return True
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer", help="the checkpoint's tokenizer.json")
    p.add_argument("base", help="an existing draft vocabulary to extend (tools/draft_vocab.py writes one)")
    p.add_argument("out")
    p.add_argument("--script", action="append", required=True, choices=sorted(SCRIPTS),
                   help="a script whose ids to add; repeat for more")
    args = p.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    vocab = tok.get_vocab_size(with_added_tokens=False)
    with open(args.base, encoding="utf-8") as handle:
        base = {int(line) for line in handle if line.strip()}
    ranges = tuple(r for name in args.script for r in SCRIPTS[name])

    added: set[int] = set()
    for index in range(vocab):
        # byte-level BPE: a token is a byte blob, so a character test only works after decoding
        if in_scripts(tok.decode([index], skip_special_tokens=False), ranges):
            added.add(index)
    ids = sorted(base | added)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("".join(f"{i}\n" for i in ids))
    print(f"{args.base}: {len(base)} ids + {len(added - base)} {'/'.join(args.script)} ids "
          f"= {len(ids)} of {vocab} ({len(ids) / vocab:.1%}) -> {args.out}")


if __name__ == "__main__":
    main()
