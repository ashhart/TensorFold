#!/usr/bin/env python3
"""Build the fidelity prompts (text) from a TensorFold checkout: python fid_prompts.py CKPT REPO OUT.json"""
import json, sys, glob
from transformers import AutoTokenizer
ck, repo, out = sys.argv[1:4]
t = AutoTokenizer.from_pretrained(ck)
rd = lambda p: open(f"{repo}/{p}", encoding="utf-8").read()
docs = sorted(glob.glob(f"{repo}/docs/*.md") + glob.glob(f"{repo}/docs/recipes/*.md"))
big = "\n\n".join(open(p, encoding="utf-8").read() for p in docs)

def cut(text, n):
    ids = t(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= n, (len(ids), n)
    return t.decode(ids[:n])

P = {}
P["code-python"] = cut(rd("tools/pr_check.py"), 1500)
P["code-zig"] = cut(rd("zig/src/server/fields.zig"), 1500)
P["prose"] = cut(rd("README.md"), 900)
P["math"] = ("Problem: Let f(x) = x^3 - 6x^2 + 11x - 6. Find all real roots of f, then compute the definite integral of f "
             "from 0 to 3, and finally prove that the sum of the squares of the roots equals 14.\n\nSolution:\nStep 1.")
P["multilingual"] = ("Der Zug von München nach Berlin fährt um 8:15 Uhr ab. 東京から大阪までは新幹線で約二時間半かかります。"
                     "La tour Eiffel mesure trois cent trente mètres. Fasse a tradução destas três frases para o inglês:\n1.")
P["doc-8k"] = cut(big, 8000 - 12) + "\n\nIn one paragraph, the document above says:"
P["doc-32k"] = cut(big * 3 if len(t(big, add_special_tokens=False)["input_ids"]) < 32100 else big, 32000 - 12) + "\n\nIn one paragraph, the document above says:"
json.dump(P, open(out, "w"))
for k, v in P.items():
    print(k, len(t(v)["input_ids"]))
