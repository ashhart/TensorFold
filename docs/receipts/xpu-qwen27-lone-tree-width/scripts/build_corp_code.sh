#!/bin/bash
# Rebuilds inputs/corp_code.txt: the top-level *.py of the Ubuntu 26.04 python3.14 standard library (3.14.4-1ubuntu0.2),
# concatenated without separator in the order of a punctuation- and case-blind sort of the file names, cut to 1,500,000
# bytes. usage: build_corp_code.sh OUTFILE      expects sha256 31b2f903b4a2ea01728927e53436d924994c93cf3114c7b7f35e58d906961e30
set -e
python3 -I - "$1" <<"PY"
import hashlib, os, re, sys
root = "/usr/lib/python3.14"
names = sorted((f for f in os.listdir(root) if f.endswith(".py")), key=lambda f: (re.sub("[^a-z0-9]", "", f.lower()), f))
data = b"".join(open(os.path.join(root, f), "rb").read() for f in names)[:1500000]
open(sys.argv[1], "wb").write(data)
print(hashlib.sha256(data).hexdigest(), len(data), "bytes from", len(names), "files")
PY
