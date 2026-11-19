#!/bin/bash
# Rebuild the historical corpus with build_corp_code.sh OUTFILE; source and expected hash are in the receipt.
set -e
python3 -I - "$1" <<"PY"
import hashlib, os, re, sys
root = "/usr/lib/python3.14"
names = sorted((f for f in os.listdir(root) if f.endswith(".py")), key=lambda f: (re.sub("[^a-z0-9]", "", f.lower()), f))
data = b"".join(open(os.path.join(root, f), "rb").read() for f in names)[:1500000]
open(sys.argv[1], "wb").write(data)
print(hashlib.sha256(data).hexdigest(), len(data), "bytes from", len(names), "files")
PY
