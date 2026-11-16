#!/usr/bin/env python3
"""Cut prompt token files p<N>.json (raw ids, no special tokens) from the tree's docs/**/*.md via a running server's /tokenize.

usage: mkprompts.py PORT TREE_DIR OUT_DIR N...; the docs are read in path order, receipts excluded; one prefix per size.
"""
import json
import sys
import urllib.request
from pathlib import Path


def tok(port, text):
    body = {"model": "local-model", "prompt": text, "add_special_tokens": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["tokens"]


def main():
    port, tree, out, sizes = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), [int(x) for x in sys.argv[4:]]
    ids = []
    for f in sorted((tree / "docs").rglob("*.md")):
        if "receipts" in f.parts:
            continue
        ids += tok(port, f.read_text(errors="replace"))
        if len(ids) >= max(sizes):
            break
    for n in sizes:
        (out / f"p{n}.json").write_text(json.dumps(ids[:n]))
        print(n, len(ids[:n]))


if __name__ == "__main__":
    main()
