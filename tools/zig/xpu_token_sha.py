"""The two token digests of this repository for one token list: the CLI's (and reports') and the server's."""
# usage: xpu_token_sha.py REPORT.json | IDS.txt ("1,2,3" or space separated); prints 12 hex digits of each definition
import hashlib
import json
import sys

text = open(sys.argv[1]).read()
try:
    ids = json.loads(text)["tokens"]
except (ValueError, KeyError, TypeError):
    ids = [int(x) for x in text.replace(",", " ").split()]
cli = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12]
server = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:12]
print(f"{len(ids)} tokens; cli/report (sha256 of the JSON list) {cli}; server token_sha (sha256 of ids joined by commas) {server}")
