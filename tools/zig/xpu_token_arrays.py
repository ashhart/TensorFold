"""Exact token arrays as /v1/completions request bodies: the first L tokens of a public text."""
import json
import os
import sys
from tokenizers import Tokenizer

# usage: xpu_token_arrays.py CHECKPOINT_DIR TEXT_FILE OUT_DIR [LENGTH ...]   (default 2048 8192 32768 65536 131072)
ck, text, out = sys.argv[1:4]
lens = [int(x) for x in sys.argv[4:]] or [2048, 8192, 32768, 65536, 131072]
os.makedirs(out, exist_ok=True)
ids = Tokenizer.from_file(f"{ck}/tokenizer.json").encode(open(text).read()[:3_000_000]).ids
assert len(ids) >= max(lens), len(ids)
for n in lens:
    body = {"model": "x", "prompt": ids[:n], "max_tokens": 8, "temperature": 0, "ignore_eos": True}
    json.dump(body, open(f"{out}/cold_{n}.json", "w"))
    print(n, os.path.getsize(f"{out}/cold_{n}.json"))
