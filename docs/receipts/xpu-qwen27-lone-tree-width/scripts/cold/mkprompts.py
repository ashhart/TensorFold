#!/usr/bin/env python3
"""Writes the prompts of the cold-start series: short (34 tokens), long (7,588), p28946 and p121022.
usage: mkprompts.py CORP OUTDIR

Same texts as scripts/bench_tree.py of the lone-tree-width receipt (cells 0, 8192, 32768 and 131072): the first
(depth - 300) * 3.8729666924864445 characters of corp_code (the calibration request's 5,179 tokens for 20,000
characters), framed as bench_tree.py frames them. prompt_sha256 prefixes: long 829dddadac0f329f (the
xpu-qwen27-attn-exp receipt's 7,588 tokens), p28946 0bd40816987651da, p121022
60986a50b4990e56. Nothing is sent to a server.
"""
import hashlib, os, sys

CORP = open(sys.argv[1], errors="ignore").read()
OUT = sys.argv[2]
ASK = "Review the code above in detail, then write one new, complete Python function in the same style."
SHORT = ("Write a complete Python function that parses a CSV file and prints the average of each numeric column, "
         "with comments.")
TPC = 20000 / (5179 - 15)
def deep(depth):
    return "Here is the material:\n\n" + CORP[:int((depth - 300) * TPC)] + "\n\n" + ASK
PROMPTS = (("short", SHORT, "cf916455e20cf240"), ("long", deep(8192), "829dddadac0f329f"),
           ("p28946", deep(32768), "0bd40816987651da"), ("p121022", deep(131072), "60986a50b4990e56"))
os.makedirs(OUT, exist_ok=True)
bad = 0
for name, text, want in PROMPTS:
    open(f"{OUT}/{name}.txt", "w").write(text)
    got = hashlib.sha256(text.encode()).hexdigest()
    print(name, len(text), "chars", got[:16], "ok" if got.startswith(want) else "WRONG")
    bad += not got.startswith(want)
if bad:
    sys.exit("a prompt is not the receipts': wrong corp_code.txt (expects sha256 31b2f903b4a2ea01...)")
