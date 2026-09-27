"""A draft vocabulary for a draft head: the most frequent token ids over some text, plus the lowest ids.

Drafts never need to be exact (the target verifies every one), so a head can score a subset of the vocabulary and
read proportionally fewer weights. A token outside the list can never be drafted, which costs speed, never
correctness. Build it from public text only: the list ships with the package.

  python tools/draft_vocab.py TOKENIZER_JSON OUT.txt --size 32768 --min-count 10 'corpus/**/*.py' 'docs/**/*.md'
"""

from __future__ import annotations

import argparse
import collections
import glob
import json

from tokenizers import Tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer")
    p.add_argument("out")
    p.add_argument("patterns", nargs="+")
    p.add_argument("--size", type=int, default=32768)
    p.add_argument("--keep-below", type=int, default=1024, help="always keep ids below this (special tokens)")
    p.add_argument("--max-bytes", type=int, default=2_000_000, help="skip files larger than this")
    p.add_argument("--min-count", type=int, default=10,
                   help="leave out ids seen fewer times (rare names and one-off words); the id prefix fills the rest")
    p.add_argument("--added-tokens", action="store_true", help="always keep the tokenizer's added (special) tokens")
    args = p.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    counts: collections.Counter = collections.Counter()
    files = sorted({f for pat in args.patterns for f in glob.glob(pat, recursive=True)})
    total = used = 0
    for path in files:
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read(args.max_bytes + 1)
        except (UnicodeDecodeError, OSError, IsADirectoryError):
            continue
        if not text or len(text) > args.max_bytes:
            continue
        ids = tok.encode(text).ids
        counts.update(ids)
        total += len(ids)
        used += 1
    keep = set(range(args.keep_below))
    if args.added_tokens:
        keep |= {t["id"] for t in json.loads(open(args.tokenizer).read())["added_tokens"]}
    for tid, count in counts.most_common():
        if len(keep) >= args.size or count < args.min_count:
            break
        keep.add(tid)
    fill = iter(range(args.keep_below, tok.get_vocab_size()))
    while len(keep) < args.size:                       # the id prefix fills a short list
        keep.add(next(fill))
    ids = sorted(keep)
    covered = sum(c for t, c in counts.items() if t in keep)
    with open(args.out, "w") as handle:
        handle.write("\n".join(str(i) for i in ids) + "\n")
    print(json.dumps({"files": used, "tokens": total, "size": len(ids), "coverage": round(covered / max(1, total), 4),
                      "from_counts": sum(1 for t, c in counts.items() if c >= args.min_count and t in keep)}))


if __name__ == "__main__":
    main()
